"""
volmanage_experiment.py -- two untried, PRINCIPLED levers that work on VP's own
vol axis (not the failed cross-sectional reshaping):

  (A) VP-EWMA   : VP but with EWMA (RiskMetrics lambda=0.94) per-strategy vol in
                  the 1/sigma denominator, instead of trailing 756d realized vol.
                  Standard, untuned lambda -> not overfit.
  (B) Vol-managed overlay (Moreira-Muir 2017): keep VP's cross-section, but scale
                  the whole book each day by k / sigma_hat_{t-1}^p, where
                  sigma_hat is trailing realized vol of the VP return stream.
                  p=2 (variance, the JF spec) and p=1 (vol) tested; lookbacks
                  21d (monthly, the paper) and 63d. k set so mean scale = 1
                  (Sharpe is invariant to k -- only the TIME-VARIATION matters).

Pre-registered primary: (B) p=2, 21d. Others reported as secondary.
Success = beat VP mean Sharpe (<real figure removed>) across all 5 windows. Reuses exact
production VP path (sanity: VP mean Sharpe must be ~<real figure removed>).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from backtest_har_rv import load_daily_returns
from features import get_spy_daily
from daily_pipeline import factor_covariance_daily
from benchmark_vol_parity import vol_parity_weights
from config import MODEL_VOL_CAPS
from config_daily import (
    OOS_DAYS, SIGNAL_REFRESH_EVERY_D, EXEC_LAG_D, COV_WINDOW_D, MIN_ACTIVE_OBS_D,
)

OFFSETS = [0, 501, 1002, 1503, 2004]
EWMA_LAMBDA = 0.94  # RiskMetrics standard


def sharpe(r):
    r = np.asarray(r, float)
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return (r.mean() * 252) / (r.std() * np.sqrt(252))


def vol_managed(ret_vp, lookback, power):
    """Scale day t by k / sigma_hat_{t-1}^power; sigma_hat = trailing realized
    vol ending t-1 (causal). k = mean(scale) so average gross is unchanged."""
    s = pd.Series(ret_vp)
    sig = s.rolling(lookback).std().shift(1)          # causal, known before day t
    raw = 1.0 / np.power(sig.replace(0, np.nan), power)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    scale = raw / raw.mean()                            # normalize -> mean 1
    managed = (scale.fillna(1.0).values) * s.values
    return managed


def run_window(daily_ret_full, spy_px, off):
    daily_ret = daily_ret_full.iloc[:len(daily_ret_full) - off] if off else daily_ret_full
    spy_daily_ret = spy_px.pct_change().reindex(daily_ret.index)
    model_cols = list(daily_ret.columns)
    col_to_idx = {c: int(c.split(" ")[1]) for c in model_cols}
    n_total = len(daily_ret.index)
    OOS_START = n_total - OOS_DAYS
    refresh_t = list(range(OOS_START - EXEC_LAG_D, n_total - EXEC_LAG_D, SIGNAL_REFRESH_EVERY_D))

    # precompute EWMA vol (annualized) for all strategies, all dates
    ewma_var = daily_ret.ewm(alpha=1 - EWMA_LAMBDA, min_periods=60).var()
    ewma_sig_ann = np.sqrt(ewma_var) * np.sqrt(252)

    ret_vp, ret_ewma = [], []
    for t_pos in refresh_t:
        active = np.array([
            daily_ret[c].iloc[max(0, t_pos - COV_WINDOW_D): t_pos].notna().sum() >= MIN_ACTIVE_OBS_D
            for c in model_cols
        ])
        cov_slice = daily_ret.iloc[max(0, t_pos - COV_WINDOW_D): t_pos]
        spy_slice = spy_daily_ret.reindex(cov_slice.index)
        cov_756 = factor_covariance_daily(cov_slice, spy_slice)
        sigma_756 = np.sqrt(np.maximum(np.diag(cov_756), 0))
        w_vp = vol_parity_weights(sigma_756, active, model_cols)

        # VP-EWMA: same form, EWMA vol denominator (value as-of t_pos-1)
        sig_e = ewma_sig_ann.iloc[t_pos - 1].values
        w_ewma = np.zeros(len(model_cols))
        for i in range(len(model_cols)):
            if active[i] and np.isfinite(sig_e[i]) and sig_e[i] > 1e-6:
                w_ewma[i] = MODEL_VOL_CAPS[col_to_idx[model_cols[i]]] / sig_e[i]

        eff_start = t_pos + EXEC_LAG_D
        eff_end = min(eff_start + SIGNAL_REFRESH_EVERY_D, n_total)
        for d in range(eff_start, eff_end):
            if d < OOS_START:
                continue
            r = daily_ret.iloc[d].fillna(0).values
            ret_vp.append(float(w_vp @ r))
            ret_ewma.append(float(w_ewma @ r))

    out = {"off": off, "n": len(ret_vp), "vp": sharpe(ret_vp), "vp_ewma": sharpe(ret_ewma)}
    # vol-managed overlays on the VP stream
    for p in (2, 1):
        for lb in (21, 63):
            out[f"vm_p{p}_{lb}"] = sharpe(vol_managed(ret_vp, lb, p))
    return out, ret_vp


def main():
    print("Loading data ...")
    daily_ret_full = load_daily_returns()
    spy_px = get_spy_daily()
    rows = []
    for off in OFFSETS:
        r, _ = run_window(daily_ret_full, spy_px, off)
        rows.append(r)
        print(f"  off={off:<5d}  VP {r['vp']:.3f} | VP-EWMA {r['vp_ewma']:.3f} | "
              f"VM(p2,21) {r['vm_p2_21']:.3f} | VM(p2,63) {r['vm_p2_63']:.3f} | "
              f"VM(p1,21) {r['vm_p1_21']:.3f} | VM(p1,63) {r['vm_p1_63']:.3f}")

    df = pd.DataFrame(rows)
    cols = ["vp", "vp_ewma", "vm_p2_21", "vm_p2_63", "vm_p1_21", "vm_p1_63"]
    print("\n=== mean Sharpe across 5 windows (VP must be ~<real figure removed>) ===")
    for c in cols:
        m = df[c].mean()
        beat = int((df[c] > df["vp"]).sum())
        tag = "  <-- PRIMARY" if c == "vm_p2_21" else ""
        print(f"  {c:<10s} mean {m:.3f}   delta {m - df['vp'].mean():+.3f}   beats VP {beat}/5{tag}")
    df.to_csv("results/volmanage_experiment_summary.csv", index=False)
    print("\nDONE -> results/volmanage_experiment_summary.csv")


if __name__ == "__main__":
    main()