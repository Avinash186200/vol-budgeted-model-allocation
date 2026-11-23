"""Daily walk-forward backtest of the minimum-variance constructions.

The out-of-sample history is cut into ``n_windows`` back-to-back test windows of
``oos_days`` each (window 0 is the oldest). Inside a window:

  1. Every ``refresh`` trading days, at row position ``t``, build weights from
     rows ``< t`` only.
  2. Hold them from row ``t + exec_lag`` for ``refresh`` days (a T+2 execution lag
     by default), so the weights never see the day they trade on.
  3. Charge ``cost_bps`` per unit of one-way turnover on the first day the new
     weights are held.

Every construction is scaled to the same predicted portfolio vol under one
shared covariance, so the comparison, the blend and the cost charge are all on
the same footing. The cap check uses that same covariance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..optimizer import Budget, default_budget
from .covariance import ledoit_wolf_cov
from .weights import (
    cap_ratios,
    caps_for,
    constrained_min_variance,
    equal_weight,
    inverse_vol,
    min_variance,
    project_to_caps,
    scale_to_vol,
)

BASE = ["equal_weight", "vol_parity", "vol_parity_capped", "minvar_ens", "blend", "blend_capped"]


@dataclass
class MVConfig:
    """Backtest settings, fixed before any run. None of them is tuned on the test windows.

    ``alpha`` (the min-variance share of the blend) is the value the original
    research settled on after looking at all its test windows. It is kept as the
    headline only to reproduce that design; ``alpha_grid`` is used to choose
    alpha properly on the search windows and check it on the hold-out windows.
    """

    oos_days: int = 504
    n_windows: int = 5
    refresh: int = 5
    exec_lag: int = 2
    cov_window: int = 504
    min_active_obs: int = 252
    min_models: int = 5
    mv_windows: tuple[int, ...] = (252, 504, 756)
    alpha: float = 0.75
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    target_vol: float = 0.06
    cost_bps: float = 5.0
    search_windows: int = 2


def prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Mark leading zeros (before a model's first non-zero day) as NaN; keep interior zeros."""
    out = daily.sort_index().astype(float).copy()
    for c in out.columns:
        nz = out[c].fillna(0).ne(0).to_numpy()
        first = int(np.argmax(nz)) if nz.any() else len(out)
        out.iloc[:first, out.columns.get_loc(c)] = np.nan
    return out


def _clean_window(X: np.ndarray, t: int, win: int, act: np.ndarray) -> np.ndarray:
    R = X[max(0, t - win):t][:, act]
    return R[~np.isnan(R).any(axis=1)]


def blend_name(a: float) -> str:
    return f"blend_a{a:.2f}"


def weights_at(X: np.ndarray, t: int, cols: list[str], cfg: MVConfig, budget: Budget) -> dict:
    """All constructions at rebalance position ``t``, using rows ``< t`` only.

    Returns a dict with full-length weight vectors per strategy, the shared
    covariance ``S`` of the active models, their positions ``act`` and the caps.
    """
    n = len(cols)
    names = BASE + [blend_name(a) for a in cfg.alpha_grid]
    zero = {k: np.zeros(n) for k in names}
    hist = X[max(0, t - cfg.cov_window):t]
    active = (~np.isnan(hist)).sum(axis=0) >= cfg.min_active_obs
    act = np.where(active)[0]
    if len(act) < cfg.min_models:
        return {"w": zero, "S": None, "act": act, "caps": None}
    covs = [c for c in (ledoit_wolf_cov(_clean_window(X, t, win, act)) for win in cfg.mv_windows)
            if c is not None]
    if not covs:
        return {"w": zero, "S": None, "act": act, "caps": None}
    S = np.mean(covs, axis=0)       # one covariance for scaling, vol parity and the cap check
    caps = caps_for(budget, cols, act)
    tv = cfg.target_vol

    ew = scale_to_vol(equal_weight(S), S, tv)
    vp = scale_to_vol(inverse_vol(S), S, tv)
    mv = scale_to_vol(np.mean([min_variance(C) for C in covs], axis=0), S, tv)
    cmv = constrained_min_variance(S, caps)
    sub = {
        "equal_weight": ew,
        "vol_parity": vp,
        "vol_parity_capped": project_to_caps(vp, S, caps),
        "minvar_ens": mv,
        "blend": scale_to_vol(cfg.alpha * mv + (1 - cfg.alpha) * vp, S, tv),
        "blend_capped": project_to_caps(cfg.alpha * scale_to_vol(cmv, S, tv) + (1 - cfg.alpha) * vp, S, caps),
    }
    for a in cfg.alpha_grid:
        sub[blend_name(a)] = scale_to_vol(a * mv + (1 - a) * vp, S, tv)
    full = {}
    for k, w in sub.items():
        f = np.zeros(n)
        f[act] = w
        full[k] = f
    return {"w": full, "S": S, "act": act, "caps": caps}


@dataclass
class WindowResult:
    """One test window."""

    gross: pd.DataFrame                 # daily gross returns, one column per strategy
    net: pd.DataFrame                   # daily returns after costs
    turnover: dict[str, float]          # one-way turnover per year
    worst_caps: dict[str, dict[str, float]] = field(default_factory=dict)  # strategy -> level -> worst ratio


def run_window(daily: pd.DataFrame, end: int, cfg: MVConfig, budget: Budget) -> WindowResult:
    """Backtest the test window ``[end - oos_days, end)`` (row positions)."""
    cols = list(daily.columns)
    X = daily.to_numpy(dtype=float)
    oos = end - cfg.oos_days
    names = BASE + [blend_name(a) for a in cfg.alpha_grid]
    prev = {k: None for k in names}
    turn = {k: 0.0 for k in names}
    worst = {k: {"model": 0.0, "group": 0.0, "port": 0.0}
             for k in ("vol_parity", "vol_parity_capped", "minvar_ens", "blend", "blend_capped")}
    gross = {k: [] for k in names}
    net = {k: [] for k in names}
    dates = []
    for t in range(oos - cfg.exec_lag, end - cfg.exec_lag, cfg.refresh):
        out = weights_at(X, t, cols, cfg, budget)
        W = out["w"]
        cost = {}
        for k in names:
            dw = 0.0 if prev[k] is None else float(np.abs(W[k] - prev[k]).sum())
            turn[k] += dw
            cost[k] = dw * cfg.cost_bps / 1e4
            prev[k] = W[k]
        if out["S"] is not None:
            for k in worst:
                r = cap_ratios(W[k][out["act"]], out["S"], out["caps"])
                for lvl in r:
                    worst[k][lvl] = max(worst[k][lvl], r[lvl])
        first = True
        for d in range(t + cfg.exec_lag, min(t + cfg.exec_lag + cfg.refresh, end)):
            if d < oos:
                continue
            r = np.nan_to_num(X[d])
            dates.append(daily.index[d])
            for k in names:
                g = float(W[k] @ r)
                gross[k].append(g)
                net[k].append(g - (cost[k] if first else 0.0))
            first = False
    years = len(dates) / 252.0
    idx = pd.DatetimeIndex(dates, name="date")
    return WindowResult(
        gross=pd.DataFrame(gross, index=idx),
        net=pd.DataFrame(net, index=idx),
        turnover={k: turn[k] / years if years else float("nan") for k in names},
        worst_caps=worst,
    )


def window_ends(n_rows: int, cfg: MVConfig) -> list[int]:
    """End positions of the test windows, oldest first. Raises if history is too short."""
    ends = [n_rows - i * cfg.oos_days for i in range(cfg.n_windows)][::-1]
    first_oos = ends[0] - cfg.oos_days
    if first_oos - cfg.exec_lag < max(cfg.mv_windows) // 2:
        raise ValueError("not enough history before the first test window; reduce n_windows or oos_days")
    return ends


def run_min_variance(daily: pd.DataFrame, cfg: MVConfig = MVConfig(),
                     budget: Budget | None = None) -> list[WindowResult]:
    """Run every test window. ``daily`` is a raw panel (leading zeros = not live yet)."""
    panel = prepare_daily(daily)
    budget = budget or default_budget(list(panel.columns))
    return [run_window(panel, e, cfg, budget) for e in window_ends(len(panel), cfg)]
