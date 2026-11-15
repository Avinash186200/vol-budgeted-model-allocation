"""
data_prep.py — load raw daily returns, clean, resample to monthly.
"""
import pandas as pd
import numpy as np
from config import RAW_RETURNS

def load_monthly_returns() -> pd.DataFrame:
    """
    Returns a DataFrame indexed by month-end date,
    columns = ['model 1' .. 'model 62'], values = monthly decimal returns.

    Rules:
      - Leading zeros (pre-inception) → NaN
      - Interior zeros (flat / non-trading days) → 0.0  (no change)
      - Monthly return = compound of daily returns within the month
      - Months where model has not yet incepted → NaN
    """
    df = pd.read_excel(RAW_RETURNS).rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    model_cols = [c for c in df.columns if c != "date"]

    # ── 1. mark pre-inception zeros as NaN ──────────────────────────────────
    for c in model_cols:
        first_nonzero = df[c].ne(0).idxmax()
        if df.loc[first_nonzero, c] == 0:
            # model never trades — leave all as NaN
            df[c] = np.nan
        else:
            df.loc[:first_nonzero - 1, c] = np.nan

    # ── 2. resample to monthly: compound daily returns ───────────────────────
    df = df.set_index("date")
    monthly = (df + 1).resample("ME").prod() - 1

    # ── 3. months with all-NaN → NaN (model not yet active that month) ───────
    # if every daily value in the month was NaN, prod gives 1.0 → should be NaN
    # track which months had at least one non-NaN daily return
    active = df.notna().resample("ME").sum() > 0
    monthly[~active] = np.nan

    monthly.index.name = "date"
    print(f"Monthly returns: {monthly.shape[0]} months x {monthly.shape[1]} models")
    print(f"Date range: {monthly.index[0].date()} to {monthly.index[-1].date()}")
    return monthly


if __name__ == "__main__":
    m = load_monthly_returns()
    print(m.tail())
