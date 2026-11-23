"""
holdout_experiment.py -- DISCIPLINED holdout validation of the vol-targeting
overlay (the only lever showing a persistent positive flicker).

Method (anti-overfitting):
  SEARCH windows  = {0, 501}      -> choose the single best overlay spec here.
  HOLDOUT windows = {1002,1503,2004} (untouched) -> evaluate that ONE spec.
  Accept ONLY if it beats VP out-of-sample on the holdout.

Overlay: managed_t = alpha * (k/sigma_hat_{t-1}^power) * vp_t + (1-alpha) * vp_t
where sigma_hat = trailing realized vol of the VP stream (causal, lag 1),
k normalizes mean scale to 1. Grid: power{1,2} x lookback{21,42,63,126} x
alpha{0.5,1.0} = 16 specs. Reuses the completed run's ret_vp streams (no re-run).
SE on a per-set mean Sharpe is large (~0.3-0.4) given few windows -- reported.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from itertools import product

SEARCH = [0, 501]
HOLDOUT = [1002, 1503, 2004]
RES = "results"


def sharpe(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return (r.mean() * 252) / (r.std() * np.sqrt(252))


def overlay(vp, lb, power, alpha):
    s = pd.Series(vp)
    sig = s.rolling(lb).std().shift(1)
    raw = (1.0 / np.power(sig.replace(0, np.nan), power)).replace([np.inf, -np.inf], np.nan)
    scale = (raw / raw.mean()).fillna(1.0).values
    return alpha * (scale * s.values) + (1 - alpha) * s.values


def load_vp():
    vp = {}
    for off in SEARCH + HOLDOUT:
        suffix = f"_oos{off}" if off else ""
        df = pd.read_csv(f"{RES}/daily_wf_returns{suffix}.csv", index_col=0, parse_dates=True)
        vp[off] = df["ret_vp"].astype(float).values
    return vp


def mean_sharpe(vp, offs, spec=None):
    vals = []
    for o in offs:
        series = vp[o] if spec is None else overlay(vp[o], *spec)
        vals.append(sharpe(series))
    return float(np.mean(vals)), [round(v, 3) for v in vals]


def main():
    vp = load_vp()
    grid = list(product((1, 2), (21, 42, 63, 126), (0.5, 1.0)))  # power, lb, alpha

    # ---- SEARCH: pick best spec on {0,501} ----
    vp_search, _ = mean_sharpe(vp, SEARCH)
    results = []
    for spec in grid:
        m, _ = mean_sharpe(vp, SEARCH, spec)
        results.append((m, spec))
    results.sort(key=lambda x: -x[0])
    best_m, best_spec = results[0]
    p, lb, a = best_spec
    print("=== SEARCH set {0,501} ===")
    print(f"  VP mean Sharpe          = {vp_search:.3f}")
    print(f"  BEST overlay spec       = power={p}, lookback={lb}d, alpha={a}  -> {best_m:.3f} "
          f"(search delta {best_m - vp_search:+.3f})")
    print(f"  (top 3 search specs: " +
          "; ".join(f"p{s[1][0]}/{s[1][1]}d/a{s[1][2]}={s[0]:.3f}" for s in results[:3]) + ")")

    # ---- HOLDOUT: evaluate that ONE spec on untouched windows ----
    vp_hold, vp_hold_pw = mean_sharpe(vp, HOLDOUT)
    ov_hold, ov_hold_pw = mean_sharpe(vp, HOLDOUT, best_spec)
    print("\n=== HOLDOUT set {1002,1503,2004}  (the honest test) ===")
    print(f"  VP mean Sharpe          = {vp_hold:.3f}   per-window {vp_hold_pw}")
    print(f"  chosen-overlay Sharpe   = {ov_hold:.3f}   per-window {ov_hold_pw}")
    print(f"  HOLDOUT delta           = {ov_hold - vp_hold:+.3f}")
    verdict = "PASS (beats VP out-of-sample)" if ov_hold > vp_hold else "FAIL (does not beat VP OOS)"
    print(f"  VERDICT: {verdict}")

    # ---- full 5-window context ----
    vp_all, _ = mean_sharpe(vp, SEARCH + HOLDOUT)
    ov_all, ov_all_pw = mean_sharpe(vp, SEARCH + HOLDOUT, best_spec)
    print("\n=== full 5-window context (chosen spec) ===")
    print(f"  VP all-5 mean = {vp_all:.3f} | overlay all-5 mean = {ov_all:.3f} "
          f"(delta {ov_all - vp_all:+.3f}) per-window {ov_all_pw}")
    print("\n  CAVEAT: per-set mean Sharpe SE is ~0.3-0.4 (2-3 windows). "
          "A holdout delta below that is not a reliable edge.")


if __name__ == "__main__":
    main()