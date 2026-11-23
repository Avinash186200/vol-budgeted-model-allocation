"""
erc_experiment.py -- DISCIPLINED RISK-SIDE EXPERIMENT (pre-registered).

Hypothesis (single, primary): Vol Parity (VP) is correlation-BLIND -- it sets
w_i = cap_i / sigma_i and ignores the off-diagonal covariance, so it
over-allocates risk to the menu's persistent tight subclusters (models
56/58/59/60, 27/35/39 -- confirmed stable in what_did_not_work.md sec 3). A
correlation-AWARE Equal-Risk-Contribution (ERC, "true" risk parity) portfolio
built on a Ledoit-Wolf SHRUNK covariance should diversify away that redundancy
-> lower realized vol at similar return -> higher Sharpe, robustly.

Why this is defensible (not overfitting):
  * ERC + Ledoit-Wolf is PARAMETER-FREE (shrinkage intensity is auto-estimated;
    ERC has no tunable knobs). Nothing is fit to the OOS windows.
  * Evaluated on ALL 5 non-overlapping windows; success = beat VP on the MEAN
    Sharpe (not any single cherry-picked window).
  * Long-only, no return forecasting -- pure risk model.

Method: reproduces the EXACT production VP path (factor_covariance_daily ->
sigma_756 -> vol_parity_weights, T+2 lag, refresh=5, OOS_DAYS=501) so the VP
column must match the validated mean Sharpe ~<real figure removed> -- that is the harness
sanity check. ERC uses a Ledoit-Wolf covariance of the same 756d daily slice.
Sharpe is leverage-invariant, so only the weight DIRECTION matters here.

Runs in <1 min (no ML/GARCH).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from backtest_har_rv import load_daily_returns
from features import get_spy_daily
from daily_pipeline import factor_covariance_daily
from benchmark_vol_parity import vol_parity_weights
from config import MODEL_VOL_CAPS
from config_daily import (
    OOS_DAYS, SIGNAL_REFRESH_EVERY_D, EXEC_LAG_D, COV_WINDOW_D, MIN_ACTIVE_OBS_D,
)

OFFSETS = [0, 501, 1002, 1503, 2004]


def erc_weights(Sigma, iters=20000, tol=1e-12):
    """Equal Risk Contribution (long-only, sums to 1) via the convergent
    multiplicative fixed point w_i <- b_i / (Sigma w)_i (b=1/n). At the fixed
    point each strategy's risk contribution w_i*(Sigma w)_i is equal."""
    n = Sigma.shape[0]
    b = np.ones(n) / n
    d = np.diag(Sigma).copy()
    d[d <= 0] = np.nanmax(d) if np.isfinite(np.nanmax(d)) else 1.0
    w = 1.0 / np.sqrt(d)
    w = w / w.sum()
    for _ in range(iters):
        Sw = Sigma @ w
        Sw = np.where(Sw <= 1e-18, 1e-18, Sw)
        w_new = b / Sw
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        # light damping for stability on ill-conditioned matrices
        w = 0.5 * w + 0.5 * w_new
    return w


def sharpe(r):
    r = np.asarray(r, float)
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return (r.mean() * 252) / (r.std() * np.sqrt(252))


def run_window(daily_ret_full, spy_px, off):
    daily_ret = daily_ret_full.iloc[:len(daily_ret_full) - off] if off else daily_ret_full
    spy_daily_ret = spy_px.pct_change().reindex(daily_ret.index)
    model_cols = list(daily_ret.columns)
    n_total = len(daily_ret.index)
    OOS_START = n_total - OOS_DAYS
    refresh_t = list(range(OOS_START - EXEC_LAG_D, n_total - EXEC_LAG_D, SIGNAL_REFRESH_EVERY_D))

    ret_vp, ret_erc = [], []
    for t_pos in refresh_t:
        active = np.array([
            daily_ret[c].iloc[max(0, t_pos - COV_WINDOW_D): t_pos].notna().sum() >= MIN_ACTIVE_OBS_D
            for c in model_cols
        ])
        cov_slice = daily_ret.iloc[max(0, t_pos - COV_WINDOW_D): t_pos]
        spy_slice = spy_daily_ret.reindex(cov_slice.index)

        # ---- production VP (exact reproduction) ----
        cov_756 = factor_covariance_daily(cov_slice, spy_slice)
        sigma_756 = np.sqrt(np.maximum(np.diag(cov_756), 0))
        w_vp = vol_parity_weights(sigma_756, active, model_cols)

        # ---- Variant F: ERC on Ledoit-Wolf shrunk cov of active strategies ----
        act_idx = np.where(active)[0]
        w_erc = np.zeros(len(model_cols))
        if len(act_idx) >= 2:
            R = cov_slice.iloc[:, act_idx].dropna(axis=0, how="any").values
            if R.shape[0] > R.shape[1] + 5:
                Sig = LedoitWolf().fit(R).covariance_ * 252.0
                w_a = erc_weights(Sig)
                w_erc[act_idx] = w_a

        eff_start = t_pos + EXEC_LAG_D
        eff_end = min(eff_start + SIGNAL_REFRESH_EVERY_D, n_total)
        for d in range(eff_start, eff_end):
            if d < OOS_START:
                continue
            r = daily_ret.iloc[d].fillna(0).values
            ret_vp.append(float(w_vp @ r))
            ret_erc.append(float(w_erc @ r))

    return {
        "off": off,
        "start": daily_ret.index[OOS_START].date(),
        "end": daily_ret.index[-1].date(),
        "n": len(ret_vp),
        "vp_sharpe": sharpe(ret_vp),
        "vp_vol": np.std(ret_vp) * np.sqrt(252),
        "erc_sharpe": sharpe(ret_erc),
        "erc_vol": np.std(ret_erc) * np.sqrt(252),
    }


def main():
    print("Loading data ...")
    daily_ret_full = load_daily_returns()
    spy_px = get_spy_daily()
    rows = []
    for off in OFFSETS:
        r = run_window(daily_ret_full, spy_px, off)
        rows.append(r)
        print(f"  off={off:<5d} {r['start']}..{r['end']}  "
              f"VP Sharpe {r['vp_sharpe']:.3f} (vol {r['vp_vol']*100:.2f}%)   "
              f"ERC Sharpe {r['erc_sharpe']:.3f} (vol {r['erc_vol']*100:.2f}%)")

    df = pd.DataFrame(rows)
    vp_m = df["vp_sharpe"].mean()
    erc_m = df["erc_sharpe"].mean()
    n_beat = int((df["erc_sharpe"] > df["vp_sharpe"]).sum())
    print("\n=== RESULT (primary: ERC-shrunk vs VP) ===")
    print(f"  VP  mean Sharpe = {vp_m:.3f}   per-window: {df['vp_sharpe'].round(3).tolist()}")
    print(f"  ERC mean Sharpe = {erc_m:.3f}   per-window: {df['erc_sharpe'].round(3).tolist()}")
    print(f"  ERC beats VP in {n_beat}/5 windows;  mean delta = {erc_m - vp_m:+.3f}")
    print(f"  VP  mean vol = {df['vp_vol'].mean()*100:.2f}%   ERC mean vol = {df['erc_vol'].mean()*100:.2f}%")
    print(f"\n  SANITY CHECK: VP mean Sharpe should be ~<real figure removed> (matches validated run). "
          f"Got {vp_m:.3f}.")
    df.to_csv("results/erc_experiment_summary.csv", index=False)
    print("\nDONE -> results/erc_experiment_summary.csv")


if __name__ == "__main__":
    main()