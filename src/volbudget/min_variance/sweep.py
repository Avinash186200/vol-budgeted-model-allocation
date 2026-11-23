"""The approaches the original research ruled out, rebuilt with a proper search/hold-out split.

Two parts:

* ``construction_sweep``: covariance estimator x look-back x objective, every spec
  scored per test window against vol parity. Specs are ranked on the search
  windows only; those that beat vol parity there are scored on the hold-out
  windows, with a paired block bootstrap and a Bonferroni threshold for the
  number of specs tried.
* ``overlay_grid``: time-varying leverage on the vol parity stream, keyed off
  its own trailing vol (a vol-managed overlay) or off trailing market vol
  (market timing). Same split: choose on search, report hold-out.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .backtest import MVConfig, _clean_window, prepare_daily, window_ends
from .covariance import ESTIMATORS, sample_cov
from .evaluation import block_bootstrap_sharpe_diff, sharpe
from .weights import (
    equal_weight,
    erc,
    inverse_var,
    inverse_vol,
    kelly_diag,
    max_diversification,
    min_variance,
    scale_to_vol,
)

OBJECTIVES = {
    "equal": lambda S, mu: equal_weight(S),
    "inv_var": lambda S, mu: inverse_var(S),
    "erc": lambda S, mu: erc(S),
    "min_var": lambda S, mu: min_variance(S),
    "max_div": lambda S, mu: max_diversification(S),
    "kelly_diag": kelly_diag,
}


def construction_sweep(daily: pd.DataFrame, cfg: MVConfig = MVConfig(),
                       windows: tuple[int, ...] = (252, 504),
                       estimators: tuple[str, ...] = ("sample", "lw", "oas", "ewma60"),
                       objectives: tuple[str, ...] = tuple(OBJECTIVES)) -> dict[str, pd.DataFrame]:
    """Daily net returns of every spec, one DataFrame per test window (oldest first).

    The reference ``VP`` is inverse vol on the sample covariance of ``cfg.cov_window`` days.
    Every spec is scaled to ``cfg.target_vol`` under its own covariance and pays
    ``cfg.cost_bps`` on turnover.
    """
    panel = prepare_daily(daily)
    X = panel.to_numpy(float)
    n = X.shape[1]
    specs = ["VP"] + [f"{e}_{w}:{o}" for w, e, o in product(windows, estimators, objectives)]
    out = {}
    for wi, end in enumerate(window_ends(len(panel), cfg)):
        oos = end - cfg.oos_days
        prev = {s: None for s in specs}
        rets = {s: [] for s in specs}
        dates = []
        for t in range(oos - cfg.exec_lag, end - cfg.exec_lag, cfg.refresh):
            hist = X[max(0, t - cfg.cov_window):t]
            act = np.where((~np.isnan(hist)).sum(axis=0) >= cfg.min_active_obs)[0]
            W = {s: np.zeros(n) for s in specs}
            if len(act) >= cfg.min_models:
                S_vp = sample_cov(_clean_window(X, t, cfg.cov_window, act))
                if S_vp is not None:
                    W["VP"][act] = scale_to_vol(inverse_vol(S_vp), S_vp, cfg.target_vol)
                for w in windows:
                    R = _clean_window(X, t, w, act)
                    mu = R.mean(axis=0) * 252 if len(R) else None
                    for e in estimators:
                        S = ESTIMATORS[e](R)
                        if S is None:
                            continue
                        for o in objectives:
                            W[f"{e}_{w}:{o}"][act] = scale_to_vol(OBJECTIVES[o](S, mu), S, cfg.target_vol)
            cost = {}
            for s in specs:
                cost[s] = 0.0 if prev[s] is None else float(np.abs(W[s] - prev[s]).sum()) * cfg.cost_bps / 1e4
                prev[s] = W[s]
            first = True
            for d in range(t + cfg.exec_lag, min(t + cfg.exec_lag + cfg.refresh, end)):
                if d < oos:
                    continue
                r = np.nan_to_num(X[d])
                dates.append(panel.index[d])
                for s in specs:
                    rets[s].append(float(W[s] @ r) - (cost[s] if first else 0.0))
                first = False
        out[wi] = pd.DataFrame(rets, index=pd.DatetimeIndex(dates, name="date"))
    return out


def judge_specs(streams: dict[int, pd.DataFrame], n_search: int, n_boot: int = 1000) -> pd.DataFrame:
    """Rank specs on the search windows; score the ones that beat VP there on the hold-out windows."""
    keys = sorted(streams)
    search, hold = keys[:n_search], keys[n_search:]
    sh = pd.DataFrame({k: streams[k].apply(sharpe) for k in keys}).T     # rows = windows
    specs = [c for c in sh.columns if c != "VP"]
    vp_s, vp_h = sh.loc[search, "VP"].mean(), sh.loc[hold, "VP"].mean()
    thr = 0.05 / max(len(specs), 1)
    rows = []
    for s in specs:
        row = {"spec": s, "search": sh.loc[search, s].mean() - vp_s, "holdout": sh.loc[hold, s].mean() - vp_h,
               "p_le_0": np.nan, "verdict": "not a finalist"}
        if row["search"] > 0:
            a = np.concatenate([streams[k][s].to_numpy() for k in hold])
            b = np.concatenate([streams[k]["VP"].to_numpy() for k in hold])
            bs = block_bootstrap_sharpe_diff(a, b, n_boot=n_boot)
            row["p_le_0"] = bs["p_le_0"]
            row["verdict"] = ("robust" if row["holdout"] > 0 and bs["p_le_0"] < thr
                              else "nominal" if row["holdout"] > 0 and bs["p_le_0"] < 0.05
                              else "noise")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("search", ascending=False).reset_index(drop=True)


def _causal_scale(raw: pd.Series) -> np.ndarray:
    """Normalise a leverage series by its expanding mean known the day before, so the
    average leverage is about 1 without using any future value."""
    norm = raw.expanding().mean().shift(1)
    return (raw / norm).replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy()


def vol_managed(stream: pd.Series, lookback: int, power: int) -> pd.Series:
    """Scale day t by ``1 / sigma_{t-1}^power``, sigma = trailing vol of the stream itself."""
    sig = stream.rolling(lookback).std().shift(1)
    raw = 1.0 / sig.replace(0, np.nan) ** power
    return stream * _causal_scale(raw)


def market_timed(stream: pd.Series, market: pd.Series, lookback: int, kind: str) -> pd.Series:
    """Scale day t by a function of trailing market vol known at t-1.

    ``inv_vol``: 1 / market vol. ``regime``: 1.3 when market vol is below its
    trailing one-year median, else 0.7.
    """
    mvol = market.rolling(lookback).std().shift(1)
    if kind == "regime":
        med = mvol.rolling(252).median().shift(1)
        lev = pd.Series(np.where(mvol <= med, 1.3, 0.7), index=market.index).where(med.notna(), 1.0)
        return stream * lev.reindex(stream.index).fillna(1.0).to_numpy()
    raw = (1.0 / mvol.replace(0, np.nan)).reindex(stream.index)
    return stream * _causal_scale(raw)


def overlay_grid(vp_streams: list[pd.Series], market: pd.Series | None, n_search: int) -> pd.DataFrame:
    """Score every overlay spec per window; rows = specs, columns = search/holdout Sharpe minus VP."""
    specs = {f"volman_p{p}_{lb}": (lambda s, p=p, lb=lb: vol_managed(s, lb, p))
             for p, lb in product((1, 2), (21, 63))}
    if market is not None:
        for kind, lb in product(("inv_vol", "regime"), (21, 63)):
            specs[f"mkt_{kind}_{lb}"] = (lambda s, kind=kind, lb=lb: market_timed(s, market, lb, kind))
    base = np.array([sharpe(s) for s in vp_streams])
    rows = []
    for name, fn in specs.items():
        sc = np.array([sharpe(fn(s)) for s in vp_streams]) - base
        rows.append({"spec": name, "search": sc[:n_search].mean(), "holdout": sc[n_search:].mean()})
    return pd.DataFrame(rows).sort_values("search", ascending=False).reset_index(drop=True)
