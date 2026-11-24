"""Point-in-time regime and state features, and the next-period target.

The feature row dated month-end ``m`` uses daily returns up to and including the
last day of month ``m`` and nothing later. The target row dated ``m`` uses only
months ``m + 1 .. m + h``. ``tests/test_selector.py`` checks both by changing the
data after a date and confirming that nothing dated earlier moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .methods import METHODS, Market, MethodPanel, SelectorConfig


def _ew_daily(mk: Market, lo: int, hi: int) -> np.ndarray:
    """Equal-weight average of the live models on each day in rows ``[lo, hi)``."""
    block = mk.X[lo:hi]
    live = ~np.isnan(block)
    n = live.sum(axis=1)
    return np.where(n > 0, np.nansum(block, axis=1) / np.maximum(n, 1), 0.0)


def _avg_corr(R: np.ndarray) -> float:
    """Mean off-diagonal correlation of the columns of ``R`` (no missing values)."""
    if R.shape[1] < 2:
        return float("nan")
    C = np.corrcoef(R, rowvar=False)
    k = C.shape[0]
    return float((C.sum() - np.trace(C)) / (k * (k - 1)))


def _top_pc_share(R: np.ndarray) -> float:
    """Share of total variance on the first principal component (the "absorption ratio").

    Unlike the average correlation it does not cancel when some models load
    positively and others negatively on the same common factor.
    """
    ev = np.linalg.eigvalsh(np.cov(R, rowvar=False))
    return float(ev[-1] / ev.sum()) if ev.sum() > 0 else float("nan")


def _sharpe(r: np.ndarray) -> float:
    s = r.std()
    return float(r.mean() / s * np.sqrt(252)) if len(r) > 1 and s > 0 else 0.0


def regime_features(mk: Market, panel: MethodPanel, cfg: SelectorConfig) -> pd.DataFrame:
    """One row per month-end with every feature computable from data up to that day.

    Columns:
        vol_s, vol_m: annualised vol of the equal-weight book over the short and
            medium look-backs (21 and 63 days by default).
        vol_ratio: short vol over the long-look-back (126-day) vol.
        avg_corr: mean pairwise correlation of the models with no gap over the
            medium look-back.
        top_pc_share: share of those models' variance on the first principal
            component over the same window.
        dispersion: cross-sectional std of those models' medium-look-back returns.
        trend: equal-weight return over the long look-back divided by its vol.
        rel_<method>: each method's trailing medium-look-back Sharpe (net of costs,
            from its realised returns) minus the average over all methods.

    Rows that cannot be filled (too little history) are dropped.
    """
    ds, dm, dl = cfg.feature_days
    net = pd.DataFrame({k: panel.books[k].net for k in METHODS})
    net_dates = net.index.to_numpy()
    rows = {}
    for m in range(len(mk.ends)):
        end = int(mk.ends[m])
        if end + 1 < dl:
            continue
        ew = _ew_daily(mk, end + 1 - dl, end + 1)
        Wm = mk.X[end + 1 - dm:end + 1]
        Wm = Wm[:, ~np.isnan(Wm).any(axis=0)]
        if Wm.shape[1] < 3:
            continue
        hist = net.iloc[:int(np.searchsorted(net_dates, mk.dates[end].to_datetime64(), side="right"))]
        if len(hist) < dm:
            continue
        tail = hist.iloc[-dm:]
        sh = np.array([_sharpe(tail[k].to_numpy()) for k in METHODS])
        vol_l = ew.std() * np.sqrt(252)
        row = {
            "vol_s": ew[-ds:].std() * np.sqrt(252),
            "vol_m": ew[-dm:].std() * np.sqrt(252),
            "vol_ratio": ew[-ds:].std() * np.sqrt(252) / vol_l if vol_l > 0 else 1.0,
            "avg_corr": _avg_corr(Wm),
            "top_pc_share": _top_pc_share(Wm),
            "dispersion": float(Wm.sum(axis=0).std()),
            "trend": float(ew.sum() / (ew.std() * np.sqrt(dl))) if ew.std() > 0 else 0.0,
        }
        for k, s in zip(METHODS, sh - sh.mean()):
            row[f"rel_{k}"] = float(s)
        rows[mk.months[m]] = row
    out = pd.DataFrame(rows).T
    out.index.name = "date"
    return out.dropna()


def forward_scores(mk: Market, panel: MethodPanel, cfg: SelectorConfig) -> pd.DataFrame:
    """Each method's realised risk-adjusted net return over months ``m + 1 .. m + h``.

    The score is the mean over the standard deviation of the method's daily net
    returns in that span (a per-day Sharpe ratio; the scale does not matter for the
    ranking). Rows whose span runs past the end of the panel are left out.
    """
    h = cfg.horizon
    net = pd.DataFrame({k: panel.books[k].net for k in METHODS})
    rows = {}
    for m in range(len(mk.ends) - h):
        lo = mk.days_of(m + 1).start
        hi = mk.days_of(m + h).stop
        d0, d1 = mk.dates[lo], mk.dates[hi - 1]
        if d0 < net.index[0] or d1 > net.index[-1]:
            continue
        span = net.loc[d0:d1]
        sd = span.std()
        rows[mk.months[m]] = (span.mean() / sd.where(sd > 0)).to_dict()
    out = pd.DataFrame(rows).T
    out.index.name = "date"
    return out[list(METHODS)].dropna()


def best_method(scores: pd.DataFrame) -> pd.Series:
    """The target: the index (0..K-1) of the method with the best forward score."""
    return pd.Series(np.argmax(scores.to_numpy(), axis=1), index=scores.index, name="target")
