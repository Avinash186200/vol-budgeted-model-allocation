"""
solution.py -- Allocation: the NON-ML, risk-based solution.

Drop this file into the main research repo root (it imports the repo's existing
modules: daily_pipeline, benchmark_vol_parity, config, config_daily, etc.).

THE IDEA (no machine learning, no return forecasting):
  Weight the strategies for the steadiest possible portfolio using only the
  *risk* structure (how strategies co-move), which we CAN estimate, and never
  the *returns*, which we cannot. Concretely:

    1. Estimate the covariance with Ledoit-Wolf SHRINKAGE (de-noises the
       correlation matrix) over three look-back windows {252, 504, 756} days.
    2. Solve a long-only MINIMUM-VARIANCE optimization on each, average the
       weights  -> w_minvar_ensemble  (cov-window ensemble; robust to "which
       window").
    3. BLEND 75% min-var-ensemble + 25% Vol-Parity (the trusted baseline anchor)
       -> w_solution.
    4. Scale to the portfolio vol target / 6% cap.

  `solution_weights_constrained()` additionally enforces the mandate's
  per-strategy vol caps and group vol caps (see note below).

WHY MINIMUM-VARIANCE: classic optimization needs expected returns AND risk.
Expected returns are not reliably forecastable on this near-decorrelated,
market-neutral menu (the ML approach was tested exhaustively and did not add
value -- see SOLUTION_SUMMARY.md). Min-variance uses ONLY the risk side, which
is estimable. That is why it is robust and defensible.

RESULTS: real-data figures removed. The design was compared with vol parity on five
back-to-back ~2-year test windows, gross of costs.

CONSTRAINT VALIDATION: re-solved under the full mandate (per-strategy + group +
6% caps) via solution_weights_constrained() / experiments/constrained_minvar.py.
  AUDIT CAVEAT: the blend is NOT yet fully cap-compliant -- the VP leg + 6%
  re-scaling still push group vol over its cap. TODO before deployment: add a
  final constraint-projection on the blended, 6%-scaled weights (a final QP pass).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# repo modules (this file is meant to live in the main research repo root)
from daily_pipeline import factor_covariance_daily
from benchmark_vol_parity import vol_parity_weights
from config import MODEL_VOL_CAPS, GROUPS, PORT_VOL_CAP
from config_daily import COV_WINDOW_D, MIN_ACTIVE_OBS_D

MV_WINDOWS = (252, 504, 756)
DEFAULT_ALPHA = 0.75   # weight on min-var-ensemble; (1-alpha) on Vol-Parity


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def minvar_longonly(Sigma):
    """Long-only minimum-variance weights (sum to 1) for covariance Sigma."""
    n = Sigma.shape[0]
    x0 = 1.0 / np.clip(np.diag(Sigma), 1e-12, None); x0 /= x0.sum()
    r = minimize(lambda w: w @ Sigma @ w, x0, method="SLSQP",
                 bounds=[(0.0, 1.0)] * n,
                 constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                 options={"maxiter": 200, "ftol": 1e-11})
    return r.x / r.x.sum() if r.success and r.x.sum() > 0 else x0


def shrunk_cov(daily_ret, t_pos, active_idx, window):
    """Ledoit-Wolf shrunk, annualized covariance of active strategies over the
    trailing `window` days. Returns None if not enough clean history."""
    R = daily_ret.iloc[max(0, t_pos - window):t_pos, :].iloc[:, active_idx] \
                  .dropna(axis=0, how="any").values
    if R.shape[0] <= R.shape[1] + 5:
        return None
    return LedoitWolf().fit(R).covariance_ * 252.0


def minvar_ensemble(daily_ret, t_pos, active, cols, windows=MV_WINDOWS):
    """Average of long-only min-var weight vectors over the cov windows.
    Returns a full-length (n_cols) weight vector that sums to 1 over actives."""
    act = np.where(active)[0]
    w = np.zeros(len(cols))
    if len(act) < 5:
        return w
    mats = [shrunk_cov(daily_ret, t_pos, act, win) for win in windows]
    mats = [m for m in mats if m is not None]
    if not mats:
        return w
    sub = np.mean([minvar_longonly(S) for S in mats], axis=0)
    w[act] = sub / sub.sum()
    return w


# --------------------------------------------------------------------------- #
# The solution
# --------------------------------------------------------------------------- #
def solution_weights(daily_ret, spy_daily_ret, t_pos, cols, alpha=DEFAULT_ALPHA):
    """Unconstrained version: 75% min-var-ensemble + 25% Vol-Parity.

    Returns the blended weight vector (NOT yet scaled to a vol target, and NOT
    enforcing per-strategy/group caps -- use solution_weights_constrained for
    the mandate-compliant version)."""
    active = np.array([
        daily_ret[c].iloc[max(0, t_pos - COV_WINDOW_D):t_pos].notna().sum() >= MIN_ACTIVE_OBS_D
        for c in cols
    ])
    cs = daily_ret.iloc[max(0, t_pos - COV_WINDOW_D):t_pos]
    cov = factor_covariance_daily(cs, spy_daily_ret.reindex(cs.index))
    sigma = np.sqrt(np.maximum(np.diag(cov), 0))
    w_vp = vol_parity_weights(sigma, active, cols)
    w_mv = minvar_ensemble(daily_ret, t_pos, active, cols)
    return (1 - alpha) * w_vp + alpha * w_mv


def solution_weights_constrained(daily_ret, spy_daily_ret, t_pos, cols,
                                 alpha=DEFAULT_ALPHA, port_cap=PORT_VOL_CAP, iters=4):
    """Mandate-compliant version: same blend, but the min-var leg is solved
    under per-strategy vol caps and group vol caps, then scaled to `port_cap`.

    Group cap is modeled as the group sub-portfolio vol (correlation-aware):
    sqrt(w_g' S_g w_g) <= group_cap. The per-name/group bounds are re-scaled
    each iteration so they bind at the live `port_cap`-scaled portfolio.

    NB: confirm the edge survives these constraints before deploying -- this is
    the open validation item (experiments/constrained_minvar.py runs the full
    5-window test of this function's logic)."""
    col_to_idx = {c: int(c.split(" ")[1]) for c in cols}
    active = np.array([
        daily_ret[c].iloc[max(0, t_pos - COV_WINDOW_D):t_pos].notna().sum() >= MIN_ACTIVE_OBS_D
        for c in cols
    ])
    cs = daily_ret.iloc[max(0, t_pos - COV_WINDOW_D):t_pos]
    cov = factor_covariance_daily(cs, spy_daily_ret.reindex(cs.index))
    sigma_f = np.sqrt(np.maximum(np.diag(cov), 0))
    w_vp = vol_parity_weights(sigma_f, active, cols)

    act = np.where(active)[0]
    w_mv = np.zeros(len(cols))
    if len(act) >= 5:
        mats = [shrunk_cov(daily_ret, t_pos, act, win) for win in MV_WINDOWS]
        mats = [m for m in mats if m is not None]
        if mats:
            S = np.mean(mats, axis=0)
            sig = np.sqrt(np.maximum(np.diag(S), 0))
            cap = np.array([MODEL_VOL_CAPS[col_to_idx[cols[i]]] for i in act])
            groups = [[list(act).index(i) for i in
                       [j for j, c in enumerate(cols) if col_to_idx[c] in set(ids)] if i in set(act)]
                      for _, (ids, _) in GROUPS.items()]
            gcaps = [gc for _, (_, gc) in GROUPS.items()]
            w_mv[act] = _constrained_minvar(S, sig, cap, groups, gcaps, port_cap, iters)
    return (1 - alpha) * w_vp + alpha * w_mv


def _constrained_minvar(S, sig, cap, groups, gcaps, port_cap, iters):
    n = S.shape[0]; k = 1.0; w = np.ones(n) / n
    base = 1.0 / np.clip(np.diag(S), 1e-12, None); base /= base.sum()
    for _ in range(iters):
        ub = np.minimum(cap / (np.clip(sig, 1e-9, None) * k), 1.0)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        for members in groups:
            if members:
                gc = [gcaps[gi] for gi, m in enumerate(groups) if m is members][0] / k
                cons.append({"type": "ineq",
                             "fun": (lambda w, m=members, g=gc: g * g - w[m] @ S[np.ix_(m, m)] @ w[m])})
        x0 = np.minimum(base, ub); x0 = x0 / x0.sum() if x0.sum() > 0 else np.ones(n) / n
        r = minimize(lambda w: w @ S @ w, x0, method="SLSQP",
                     bounds=[(0.0, float(u)) for u in ub], constraints=cons,
                     options={"maxiter": 150, "ftol": 1e-10})
        w = np.clip(r.x if r.success else x0, 0, None)
        if w.sum() > 0: w /= w.sum()
        pv = np.sqrt(max(w @ S @ w, 1e-18)); k = port_cap / pv
    return w * k