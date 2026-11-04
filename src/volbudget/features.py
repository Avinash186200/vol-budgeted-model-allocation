"""Per-model, cross-sectional and macro features.

Timing convention: the row dated month-end t holds features that are known at
the close of month t, and the target is the sign of the return in month t+1.
Model features use returns up to and including month t. Macro features are
shifted by ``macro_lag`` months to respect publication lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANK_VARS = ["ret_3m", "ret_12m", "vol_12m", "sharpe_12m"]


def model_features(s: pd.Series) -> pd.DataFrame:
    """Trailing features for one model's monthly return series (data <= t)."""
    f = pd.DataFrame(index=s.index)
    f["ret_1m"] = s
    f["ret_3m"] = s.rolling(3).mean()
    f["ret_6m"] = s.rolling(6).mean()
    f["ret_12m"] = s.rolling(12).mean()
    f["vol_6m"] = s.rolling(6).std() * np.sqrt(12)
    f["vol_12m"] = s.rolling(12).std() * np.sqrt(12)
    f["sharpe_12m"] = s.rolling(12).mean() / (s.rolling(12).std() + 1e-8) * np.sqrt(12)
    f["win_rate_12m"] = (s > 0).astype(float).where(s.notna()).rolling(12).mean()
    cum = (1 + s.fillna(0)).cumprod()
    f["drawdown_12m"] = cum / cum.rolling(12, min_periods=1).max() - 1
    return f


def build_panels(monthly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute model features plus cross-sectional percentile ranks for all models."""
    per = {c: model_features(monthly[c]) for c in monthly.columns}
    for var in RANK_VARS:
        panel = pd.DataFrame({c: per[c][var] for c in monthly.columns})
        ranks = panel.rank(axis=1, pct=True)
        if var == "vol_12m":
            ranks = 1.0 - ranks
        for c in monthly.columns:
            per[c][f"cs_rank_{var}"] = ranks[c]
    return per


def align_macro(macro: pd.DataFrame, index: pd.DatetimeIndex, macro_lag: int = 1) -> pd.DataFrame:
    """Shift macro features by their publication lag and align to ``index``."""
    if macro is None or macro.empty:
        return pd.DataFrame(index=index)
    m = macro.sort_index().resample("ME").last().shift(macro_lag)
    m = m.reindex(index).ffill()
    m.columns = [f"mac_{c}" for c in m.columns]
    return m


def feature_matrix(
    monthly: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    macro_aligned: pd.DataFrame,
    col: str,
) -> pd.DataFrame:
    """Date-indexed feature matrix for one model, with ``target`` and ``fwd_ret``.

    ``target`` is NaN where next month's return is unknown, so no label is
    invented for the last month or for months before inception.
    """
    X = pd.concat([panels[col], macro_aligned], axis=1)
    fwd = monthly[col].shift(-1)
    X["fwd_ret"] = fwd
    X["target"] = (fwd > 0).astype(float).where(fwd.notna())
    X = X[monthly[col].notna()]
    return X
