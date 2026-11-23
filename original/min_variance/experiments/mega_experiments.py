"""
mega_experiments.py -- EXHAUSTIVE portfolio-construction sweep vs Vol Parity.

Matrix: {covariance estimator x lookback window} x {construction objective},
evaluated on all 5 non-overlapping windows at the production rebalance grid
(refresh=5, T+2 lag, OOS_DAYS=501). Designed to run for a few hours and finish
cleanly; writes results incrementally (results/mega_report.txt) and wraps every
cell in try/except so one failure never kills the run.

Discipline (anti-overfitting):
  * SEARCH windows  = {0, 501}     -> rank specs here.
  * HOLDOUT windows = {1002,1503,2004} (untouched) -> validate finalists.
  * Finalist = beats VP on SEARCH. Reported with HOLDOUT delta + a paired
    block-bootstrap on the SHARPE DIFFERENCE (the correct test).
  * Reports total #specs tested + a Bonferroni-style deflated threshold, so a
    spec is only "real" if it beats VP on BOTH halves and clears adjusted signif.

Covariance estimators: sample, Ledoit-Wolf, OAS, EWMA(hl=60,120), at windows
{252,504,756,1260}. Objectives: equal-weight, inverse-variance, ERC,
min-variance(LO), max-diversification, HRP(single,ward), full-Kelly, diag-Kelly.
VP (production factor-cov 1/sigma) is the reference. Overlays + ensembles on the
top finalists at the end.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf, OAS

from backtest_har_rv import load_daily_returns
from features import get_spy_daily
from daily_pipeline import factor_covariance_daily
from benchmark_vol_parity import vol_parity_weights
from config_daily import (
    OOS_DAYS, SIGNAL_REFRESH_EVERY_D, EXEC_LAG_D, COV_WINDOW_D, MIN_ACTIVE_OBS_D,
)

OFFSETS = [0, 501, 1002, 1503, 2004]
SEARCH, HOLDOUT = [0, 501], [1002, 1503, 2004]
COV_WINDOWS = [252, 504, 756, 1260]
EST_NAMES = ["sample", "lw", "oas", "ewma60", "ewma120"]
REPORT = "results/mega_report.txt"
_LOG = []


def log(*a):
    line = " ".join(str(x) for x in a)
    _LOG.append(line); print(line, flush=True)
    try:
        with open(REPORT, "w") as f:
            f.write("\n".join(_LOG) + "\n")
    except Exception:
        pass


# ---------------- covariance estimators (R = T x k daily returns, ann *252) ---
def est_cov(R, name):
    if R.shape[0] < R.shape[1] + 5:
        return None
    if name == "sample":
        C = np.cov(R, rowvar=False)
    elif name == "lw":
        C = LedoitWolf().fit(R).covariance_
    elif name == "oas":
        C = OAS().fit(R).covariance_
    elif name.startswith("ewma"):
        hl = int(name[4:])
        w = (0.5 ** (np.arange(R.shape[0])[::-1] / hl))
        w = w / w.sum()
        Rm = R - np.average(R, axis=0, weights=w)
        C = (Rm * w[:, None]).T @ Rm
    else:
        C = np.cov(R, rowvar=False)
    return C * 252.0


# ---------------- objectives (Sigma ann, mu ann) -> weights (sum 1) -----------
def o_equal(S, mu=None):
    n = S.shape[0]; return np.ones(n) / n

def o_inv_var(S, mu=None):
    iv = 1.0 / np.clip(np.diag(S), 1e-12, None); return iv / iv.sum()

def o_erc(S, mu=None, iters=15000, tol=1e-12):
    n = S.shape[0]; b = np.ones(n) / n
    w = 1 / np.sqrt(np.clip(np.diag(S), 1e-12, None)); w /= w.sum()
    for _ in range(iters):
        Sw = np.clip(S @ w, 1e-18, None); wn = b / Sw; wn /= wn.sum()
        if np.max(np.abs(wn - w)) < tol:
            w = wn; break
        w = 0.5 * w + 0.5 * wn
    return w

def o_min_var(S, mu=None):
    n = S.shape[0]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    r = minimize(lambda w: w @ S @ w, o_inv_var(S), method="SLSQP",
                 bounds=[(0, 1)] * n, constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    return r.x / r.x.sum() if r.success and r.x.sum() > 0 else o_inv_var(S)

def o_max_div(S, mu=None):
    n = S.shape[0]; s = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    r = minimize(lambda w: -(w @ s) / np.sqrt(max(w @ S @ w, 1e-18)), o_inv_var(S),
                 method="SLSQP", bounds=[(0, 1)] * n, constraints=cons,
                 options={"maxiter": 200, "ftol": 1e-11})
    return r.x / r.x.sum() if r.success and r.x.sum() > 0 else o_inv_var(S)

def _hrp(S, method):
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None)); corr = np.clip(S / np.outer(d, d), -1, 1)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    link = linkage(squareform(dist, checks=False), method).astype(int)
    n = S.shape[0]; sortIx = [link[-1, 0], link[-1, 1]]
    while max(sortIx) >= n:
        nw = []
        for i in sortIx:
            nw += ([link[i - n, 0], link[i - n, 1]] if i >= n else [i])
        sortIx = nw
    w = np.ones(n); clusters = [sortIx]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            h = len(c) // 2; c0, c1 = c[:h], c[h:]
            def cv(cl):
                sub = S[np.ix_(cl, cl)]; iv = 1 / np.clip(np.diag(sub), 1e-12, None); iv /= iv.sum()
                return iv @ sub @ iv
            v0, v1 = cv(c0), cv(c1); a = 1 - v0 / (v0 + v1)
            for i in c0: w[i] *= a
            for i in c1: w[i] *= (1 - a)
            nxt += [c0, c1]
        clusters = nxt
    return w / w.sum()

def o_hrp_single(S, mu=None): return _hrp(S, "single")
def o_hrp_ward(S, mu=None):   return _hrp(S, "ward")

def o_kelly_full(S, mu=None):
    if mu is None: return o_inv_var(S)
    try: raw = np.linalg.solve(S, mu)
    except Exception: return o_inv_var(S)
    w = np.clip(raw, 0, None); return w / w.sum() if w.sum() > 0 else o_inv_var(S)

def o_kelly_diag(S, mu=None):
    if mu is None: return o_inv_var(S)
    w = np.clip(mu, 0, None) / np.clip(np.diag(S), 1e-12, None)
    return w / w.sum() if w.sum() > 0 else o_inv_var(S)

OBJ = {"equal": o_equal, "inv_var": o_inv_var, "erc": o_erc, "min_var": o_min_var,
       "max_div": o_max_div, "hrp_single": o_hrp_single, "hrp_ward": o_hrp_ward,
       "kelly_full": o_kelly_full, "kelly_diag": o_kelly_diag}


def sharpe(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    return (r.mean() * 252) / (r.std() * np.sqrt(252)) if len(r) and r.std() > 0 else np.nan


def run_window(daily_ret_full, spy_px, off):
    daily_ret = daily_ret_full.iloc[:len(daily_ret_full) - off] if off else daily_ret_full
    spy_ret = spy_px.pct_change().reindex(daily_ret.index)
    cols = list(daily_ret.columns); n_total = len(daily_ret.index)
    OOS_START = n_total - OOS_DAYS
    refresh_t = list(range(OOS_START - EXEC_LAG_D, n_total - EXEC_LAG_D, SIGNAL_REFRESH_EVERY_D))
    spec_names = ["VP"] + [f"{e}{w}:{o}" for w in COV_WINDOWS for e in EST_NAMES for o in OBJ]
    streams = {s: [] for s in spec_names}

    for t_pos in refresh_t:
        active = np.array([daily_ret[c].iloc[max(0, t_pos - COV_WINDOW_D):t_pos].notna().sum() >= MIN_ACTIVE_OBS_D for c in cols])
        # VP (production)
        cs = daily_ret.iloc[max(0, t_pos - COV_WINDOW_D):t_pos]
        cov_756 = factor_covariance_daily(cs, spy_ret.reindex(cs.index))
        w_vp = vol_parity_weights(np.sqrt(np.maximum(np.diag(cov_756), 0)), active, cols)
        act = np.where(active)[0]
        wmaps = {}
        if len(act) >= 3:
            for win in COV_WINDOWS:
                R = daily_ret.iloc[max(0, t_pos - win):t_pos, :].iloc[:, act].dropna(axis=0, how="any").values
                if R.shape[0] < R.shape[1] + 5:
                    continue
                mu = R.mean(axis=0) * 252.0
                for e in EST_NAMES:
                    try:
                        S = est_cov(R, e)
                        if S is None: continue
                        for oname, ofn in OBJ.items():
                            try:
                                wa = ofn(S, mu)
                                wf = np.zeros(len(cols)); wf[act] = wa
                                wmaps[f"{e}{win}:{oname}"] = wf
                            except Exception:
                                pass
                    except Exception:
                        pass
        # realize
        eff_start = t_pos + EXEC_LAG_D; eff_end = min(eff_start + SIGNAL_REFRESH_EVERY_D, n_total)
        for d in range(eff_start, eff_end):
            if d < OOS_START: continue
            r = daily_ret.iloc[d].fillna(0).values
            streams["VP"].append(float(w_vp @ r))
            for s in spec_names:
                if s == "VP": continue
                streams[s].append(float(wmaps[s] @ r) if s in wmaps else np.nan)
    return {k: np.array(v) for k, v in streams.items()}


def msharpe(streams_by_off, key, offs):
    return float(np.nanmean([sharpe(streams_by_off[o][key]) for o in offs]))


def paired_block_bootstrap_sharpediff(cand, vp, n=5000, mean_block=21):
    m = ~(np.isnan(cand) | np.isnan(vp)); cand, vp = cand[m], vp[m]
    N = len(cand)
    if N < 100: return np.nan, np.nan, np.nan
    p = 1.0 / mean_block; rng = np.random.RandomState(777); diffs = np.empty(n)
    for b in range(n):
        idx = np.empty(N, dtype=int); i = 0
        while i < N:
            st = rng.randint(0, N); L = max(1, rng.geometric(p))
            for j in range(L):
                if i >= N: break
                idx[i] = (st + j) % N; i += 1
        diffs[b] = sharpe(cand[idx]) - sharpe(vp[idx])
    p_le0 = (diffs <= 0).mean()
    return float(p_le0), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


import pickle


def compute_one(off):
    """Compute one window's streams and pickle them (run in parallel, 1 per window)."""
    daily_ret_full = load_daily_returns(); spy_px = get_spy_daily()
    tw = time.time()
    streams = run_window(daily_ret_full, spy_px, off)
    path = f"results/mega_streams_off{off}.pkl"
    with open(path, "wb") as f:
        pickle.dump(streams, f)
    print(f"window off={off} done in {time.time()-tw:.0f}s ({len(streams['VP'])} OOS days) -> {path}", flush=True)


def aggregate():
    t0 = time.time()
    log("=" * 70); log("MEGA EXPERIMENTS — exhaustive construction sweep vs VP (aggregate)"); log("=" * 70)
    log(f"specs = 1 VP + {len(COV_WINDOWS)} windows x {len(EST_NAMES)} estimators x {len(OBJ)} objectives "
        f"= {1 + len(COV_WINDOWS)*len(EST_NAMES)*len(OBJ)} total")
    streams_by_off = {}
    for off in OFFSETS:
        try:
            with open(f"results/mega_streams_off{off}.pkl", "rb") as f:
                streams_by_off[off] = pickle.load(f)
            log(f"  loaded window off={off} ({len(streams_by_off[off]['VP'])} OOS days)")
        except Exception as e:
            log(f"  window off={off} MISSING: {e}")
    if len(streams_by_off) < len(OFFSETS):
        log("  WARNING: not all windows present; aggregating what exists.")

    all_specs = [s for s in streams_by_off[OFFSETS[0]].keys()]
    vp_all = msharpe(streams_by_off, "VP", OFFSETS)
    vp_search = msharpe(streams_by_off, "VP", SEARCH)
    vp_hold = msharpe(streams_by_off, "VP", HOLDOUT)

    # rank all specs by SEARCH-set mean Sharpe
    ranked = []
    for s in all_specs:
        if s == "VP": continue
        try:
            ranked.append((s, msharpe(streams_by_off, s, SEARCH), msharpe(streams_by_off, s, HOLDOUT),
                           msharpe(streams_by_off, s, OFFSETS)))
        except Exception:
            pass
    ranked.sort(key=lambda x: -x[1])

    log(f"\n=== VP reference: search {vp_search:.3f} | holdout {vp_hold:.3f} | all-5 {vp_all:.3f} ===")
    log("\n=== TOP 25 specs by SEARCH-set Sharpe (search / holdout / all5) ===")
    for s, sc, hd, al in ranked[:25]:
        log(f"  {s:<22s} search {sc:.3f}  holdout {hd:.3f} (vs VP {vp_hold:.3f}, d {hd-vp_hold:+.3f})  all5 {al:.3f}")

    # finalists = beat VP on BOTH search and holdout
    finalists = [(s, sc, hd, al) for (s, sc, hd, al) in ranked if sc > vp_search and hd > vp_hold]
    n_tested = len(ranked)
    log(f"\n=== FINALISTS (beat VP on BOTH search AND holdout) — {len(finalists)} of {n_tested} specs ===")
    for s, sc, hd, al in finalists[:15]:
        cand = np.concatenate([streams_by_off[o][s] for o in HOLDOUT])
        vp = np.concatenate([streams_by_off[o]["VP"] for o in HOLDOUT])
        p_le0, lo, hi = paired_block_bootstrap_sharpediff(cand, vp)
        adj = 0.05 / max(n_tested, 1)  # Bonferroni-adjusted threshold
        verdict = "ROBUST" if p_le0 < adj else ("nominal" if p_le0 < 0.05 else "noise")
        log(f"  {s:<22s} holdout d {hd-vp_hold:+.3f}  bootstrap p(diff<=0)={p_le0:.4f} "
            f"95%CI[{lo:+.3f},{hi:+.3f}]  -> {verdict} (Bonf thr {adj:.5f})")
    if not finalists:
        log("  NONE. No spec beats VP on both search and holdout.")

    log("\n=== CONCLUSION ===")
    if finalists:
        best = finalists[0]
        log(f"  Best holdout-confirmed spec: {best[0]} (all-5 {best[3]:.3f} vs VP {vp_all:.3f}).")
        log(f"  {len(finalists)} specs beat VP on both halves out of {n_tested} tested.")
        log("  Treat as REAL only if bootstrap verdict is ROBUST (clears multiple-comparison threshold).")
    else:
        log(f"  Nothing beat VP out-of-sample. VP ({vp_all:.3f}) stands as the ceiling.")
    log(f"\nTotal time {time.time()-t0:.0f}s\nDONE")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    win = [a for a in args if a.startswith("--window=")]
    if win:
        compute_one(int(win[0].split("=", 1)[1]))
    elif "--aggregate" in args:
        aggregate()
    else:
        # fallback: sequential (slow). Prefer the parallel launcher + --aggregate.
        for off in OFFSETS:
            compute_one(off)
        aggregate()