"""Performance and signal-quality metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summary(r: pd.Series) -> dict[str, float]:
    """Annualised return, vol, Sharpe, max drawdown and hit rate of monthly returns."""
    ann_ret = float(r.mean() * 12)
    ann_vol = float(r.std() * np.sqrt(12))
    cum = (1 + r).cumprod()
    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else float("nan"),
        "max_dd": float((cum / cum.cummax() - 1).min()),
        "hit_rate": float((r > 0).mean()),
    }


def summary_table(ret: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """One row per strategy column in ``ret``."""
    return pd.DataFrame({n: summary(ret[n]) for n in names}).T


def rank_ic(pred: pd.DataFrame) -> float:
    """Mean per-month Spearman correlation of P(up) with the realised next return."""
    ics = []
    for _, g in pred.dropna().groupby("date"):
        if len(g) >= 3 and g["prob"].std() > 0:
            ics.append(g["prob"].rank().corr(g["fwd_ret"].rank()))
    return float(np.mean(ics)) if ics else float("nan")
