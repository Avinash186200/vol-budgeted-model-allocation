"""Candidate construction methods and the buy-and-hold book accounting.

Every month-end ``m`` each method builds a book from daily returns up to and
including the last day of month ``m``. The book is held, without trading,
through every day of month ``m + 1``; the weights drift with the returns and the
next rebalance trades from the drifted weights. That is the only place a cost is
charged.

The methods are the ones the rest of this repository already implements:

- ``equal_weight``, ``vol_parity``, ``min_variance`` and ``erc`` come from
  ``volbudget.min_variance.weights`` and are scaled to the same predicted vol;
- ``vb_optimiser`` is ``volbudget.optimizer.optimize`` (the vol-budgeted,
  long-only optimiser of the main package), fed with each model's trailing mean
  return instead of the per-model ML forecast, which would be far too slow to
  refit inside this loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..min_variance.backtest import prepare_daily
from ..min_variance.covariance import ledoit_wolf_cov
from ..min_variance.weights import equal_weight, erc, inverse_vol, min_variance, scale_to_vol
from ..optimizer import Budget, default_budget, optimize

METHODS: tuple[str, ...] = ("equal_weight", "vol_parity", "min_variance", "erc", "vb_optimiser")


@dataclass
class SelectorConfig:
    """Settings for the whole selector study. Fixed before any run; only the model
    hyperparameters in ``walkforward`` are chosen, and only on the development period.

    Attributes:
        cov_days: Daily look-back for the Ledoit-Wolf covariance. A model is active
            at month-end ``m`` only if it has no missing day in this window.
        mu_days: Look-back for the trailing mean return fed to ``vb_optimiser``.
        min_models: Fewest active models needed to build a book.
        target_vol: Predicted annual vol every scaled method is set to.
        cost_bps: Cost per unit of one-way turnover, in basis points.
        horizon: ``h``, the number of months the target looks ahead.
        embargo: Extra months dropped after the purge, between the newest training
            label and the decision month.
        min_train: Fewest labelled training rows before the first prediction.
        retrain_every: Months between refits (the models are refit on an expanding
            window; between refits the last fitted model is reused).
        holdout_months: Length of the final held-out period, in decision months.
        feature_days: Short, medium and long look-backs for the regime features.
    """

    cov_days: int = 252
    mu_days: int = 252
    min_models: int = 5
    target_vol: float = 0.06
    cost_bps: float = 5.0
    horizon: int = 1
    embargo: int = 1
    min_train: int = 36
    retrain_every: int = 3
    holdout_months: int = 60
    feature_days: tuple[int, int, int] = (21, 63, 126)


@dataclass
class Market:
    """A daily panel cut into months.

    Attributes:
        X: ``days x models`` daily returns, NaN before a model starts.
        dates: The daily index.
        months: Month-end labels, one per month in the panel.
        ends: Row position of the last day of each month.
        cols: Model names.
    """

    X: np.ndarray
    dates: pd.DatetimeIndex
    months: pd.DatetimeIndex
    ends: np.ndarray
    cols: list[str]

    def days_of(self, m: int) -> slice:
        """Row positions of the days in month ``m``."""
        start = 0 if m == 0 else int(self.ends[m - 1]) + 1
        return slice(start, int(self.ends[m]) + 1)


def make_market(daily: pd.DataFrame) -> Market:
    """Build a ``Market`` from a daily return panel (leading zeros mean "not live yet")."""
    d = prepare_daily(daily)
    per = d.index.to_period("M")
    change = np.flatnonzero(per[1:] != per[:-1])
    ends = np.append(change, len(d) - 1)
    months = pd.DatetimeIndex(per[ends].to_timestamp(how="end").normalize())
    return Market(X=d.to_numpy(), dates=pd.DatetimeIndex(d.index), months=months, ends=ends,
                  cols=list(d.columns))


def active_window(mk: Market, m: int, days: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``(end - days, end]`` of month-end ``m`` and the models with no gap in them."""
    end = int(mk.ends[m])
    W = mk.X[max(0, end - days + 1):end + 1]
    if len(W) < days:
        return W, np.array([], dtype=int)
    return W, np.flatnonzero(~np.isnan(W).any(axis=0))


def method_weights(mk: Market, m: int, cfg: SelectorConfig, budget: Budget | None = None
                   ) -> dict | None:
    """Every candidate book at month-end ``m``, from data up to that day only.

    Returns ``{"w": {method: full-length weights}, "S": active covariance,
    "act": active positions}``, or None when too few models are live.
    """
    W, act = active_window(mk, m, cfg.cov_days)
    if len(act) < cfg.min_models:
        return None
    S = ledoit_wolf_cov(W[:, act])
    if S is None:
        return None
    tv, n = cfg.target_vol, len(mk.cols)
    sub = {
        "equal_weight": scale_to_vol(equal_weight(S), S, tv),
        "vol_parity": scale_to_vol(inverse_vol(S), S, tv),
        "min_variance": scale_to_vol(min_variance(S), S, tv),
        "erc": scale_to_vol(erc(S), S, tv),
    }
    out = {}
    for k, v in sub.items():
        full = np.zeros(n)
        full[act] = v
        out[k] = full
    budget = budget or default_budget(mk.cols)
    S_full = np.zeros((n, n))
    S_full[np.ix_(act, act)] = S
    mu = np.zeros(n)
    end = int(mk.ends[m])
    mu[act] = np.nanmean(mk.X[max(0, end - cfg.mu_days + 1):end + 1][:, act], axis=0) * 252
    active = np.zeros(n, dtype=bool)
    active[act] = True
    out["vb_optimiser"] = optimize(mu, S_full, active, mk.cols, budget)
    return {"w": out, "S": S, "act": act}


def hold(w: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily returns of a book bought at ``w`` and held through the rows of ``R``.

    Each position grows with its own cumulative return; whatever is not invested
    (``1 - sum(w)``, or the proceeds of a short) sits in cash at 0%. Works for
    negative weights. Returns the daily book returns and the drifted end weights.
    """
    G = np.cumprod(1.0 + np.nan_to_num(R), axis=0)
    V = 1.0 + (G - 1.0) @ w
    r = V / np.concatenate([[1.0], V[:-1]]) - 1.0
    return r, w * G[-1] / V[-1]


@dataclass
class BookResult:
    """Daily gross and net returns of one book, plus its monthly turnover."""

    gross: pd.Series
    net: pd.Series
    turnover: pd.Series   # one-way turnover traded at each decision month-end


def run_book(mk: Market, weights: dict[int, np.ndarray], cost_bps: float, borrow: float = 0.0
             ) -> BookResult:
    """Hold ``weights[m]`` through month ``m + 1`` for each decision month ``m`` (in order).

    Turnover at ``m`` is ``sum |w_m - drifted w_{m-1}|``; the cost is charged on
    the first day of month ``m + 1``. ``borrow`` is an annual rate charged daily on
    the short side (``sum max(-w, 0)`` at the start of the month).
    """
    gross, net, turn, idx = [], [], {}, []
    prev = None
    for m in sorted(weights):
        if m + 1 >= len(mk.ends):
            break
        w = weights[m]
        sl = mk.days_of(m + 1)
        r, drifted = hold(w, mk.X[sl])
        t = float(np.abs(w - (prev if prev is not None else 0.0)).sum())
        rn = r - borrow / 252.0 * float(np.clip(-w, 0, None).sum())
        rn[0] -= t * cost_bps / 1e4
        gross.append(r)
        net.append(rn)
        idx.append(mk.dates[sl])
        turn[mk.months[m]] = t
        prev = drifted
    ix = idx[0].append(idx[1:]) if idx else pd.DatetimeIndex([])
    return BookResult(gross=pd.Series(np.concatenate(gross) if gross else [], index=ix, dtype=float),
                      net=pd.Series(np.concatenate(net) if net else [], index=ix, dtype=float),
                      turnover=pd.Series(turn, dtype=float))


@dataclass
class MethodPanel:
    """Every method's book at every month-end, and its realised daily returns.

    ``weights[m][k]`` is method ``k``'s book at month-end ``m``; ``books[k]`` holds
    the method run as a fixed strategy from the first buildable month.
    """

    weights: dict[int, dict[str, np.ndarray]]
    books: dict[str, BookResult]
    first: int


def build_method_panel(mk: Market, cfg: SelectorConfig, budget: Budget | None = None) -> MethodPanel:
    """Compute every method's book at every month-end where enough models are live."""
    weights: dict[int, dict[str, np.ndarray]] = {}
    for m in range(len(mk.ends) - 1):
        got = method_weights(mk, m, cfg, budget)
        if got is not None:
            weights[m] = got["w"]
    if not weights:
        raise ValueError("no month has enough live models to build a book")
    books = {k: run_book(mk, {m: w[k] for m, w in weights.items()}, cfg.cost_bps) for k in METHODS}
    return MethodPanel(weights=weights, books=books, first=min(weights))
