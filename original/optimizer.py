"""
optimizer.py — vol-budgeted QP optimizer with factor covariance model and weight shrinkage.

Key structural improvements over naive Ledoit-Wolf:
  1. Factor covariance  Sigma = B * Sigma_F * B' + D
       Factor 1    : SPY (market) — economically anchored
       Factors 2..K: top K-1 PCs of market-residual returns
     Removes the ~58 noise dimensions (effective rank ≈ 4) that cause
     MVO error-maximisation under plain LW covariance.
  2. Weight shrinkage toward equal weight
       w_final = (1 - alpha) * w_EW + alpha * w_CPO
     Prevents concentration when ensemble probability estimates are uncertain.
     alpha = 0 → pure EW; alpha = 1 → pure CPO.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import PORT_VOL_CAP, MODEL_VOL_CAPS, GROUPS

K_FACTORS     = 5     # market + 4 statistical PCs  (same as colleague)
SHRINK_ALPHA  = 0.80  # 80% CPO, 20% equal-weight blend
MIN_IVAR      = 1e-6  # floor on idiosyncratic variance (numerical stability)


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR COVARIANCE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def factor_covariance(
    monthly_ret_slice: pd.DataFrame,
    spy_monthly_ret: pd.Series,
    k: int = K_FACTORS,
) -> np.ndarray:
    """
    Fit a factor covariance model on a window of monthly returns.

    Sigma = B @ Sigma_F @ B.T + D   (annualised)

    B        : (n_strats, k) loadings
    Sigma_F  : (k, k) factor return covariance (annualised)
    D        : (n_strats, n_strats) diagonal idiosyncratic (annualised)

    Returns annualised full covariance matrix.
    """
    R = monthly_ret_slice.fillna(0).values          # (T, N)
    T, N = R.shape
    k = min(k, T - 1, N)

    # ── Factor 1: SPY (market) ────────────────────────────────────────────────
    spy = spy_monthly_ret.reindex(monthly_ret_slice.index).fillna(0).values
    spy = spy.reshape(-1, 1)                         # (T, 1)

    denom = (spy ** 2).sum()
    if denom < 1e-12:
        # no SPY data — fall back to PCA-only
        beta_mkt = np.zeros((1, N))
    else:
        beta_mkt = (spy.T @ R) / denom              # (1, N)

    resid = R - spy @ beta_mkt                       # (T, N)

    # ── Factors 2..k: PCA on market residuals ─────────────────────────────────
    n_pc = max(1, k - 1)
    if T > n_pc + 1 and N > n_pc:
        pca = PCA(n_components=n_pc, random_state=42)
        pca.fit(resid)
        pc_scores = pca.transform(resid)             # (T, n_pc)
        pc_load   = pca.components_.T                # (N, n_pc)
    else:
        pc_scores = np.zeros((T, 0))
        pc_load   = np.zeros((N, 0))

    # ── Factor matrix ─────────────────────────────────────────────────────────
    F = np.hstack([spy, pc_scores])                  # (T, k_actual)
    B = np.hstack([beta_mkt.T, pc_load])             # (N, k_actual)

    # ── Factor covariance (annualised monthly) ────────────────────────────────
    Sigma_F = np.cov(F.T) * 12 if F.shape[1] > 1 else np.array([[np.var(F) * 12]])

    # ── Idiosyncratic variance ────────────────────────────────────────────────
    fitted = F @ B.T                                 # (T, N)
    resid2 = R - fitted
    D_diag = np.maximum(np.var(resid2, axis=0) * 12, MIN_IVAR)
    D      = np.diag(D_diag)

    # ── Reconstruct full Sigma ────────────────────────────────────────────────
    Sigma = B @ Sigma_F @ B.T + D
    # symmetrise and ridge for numerical stability
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma += np.eye(N) * 1e-8

    return Sigma


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _scale_to_port_vol(w: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    """Uniform scale-down until portfolio annualised vol <= cap."""
    pv = np.sqrt(max(float(w @ cov @ w), 0))
    return w * (cap / pv) if pv > cap else w


def _equal_weight(active: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(active)
    w = np.zeros(n)
    if active.sum() == 0:
        return w
    w[active] = 1.0 / active.sum()
    return _scale_to_port_vol(w, cov, PORT_VOL_CAP)


def _vol_parity(active: np.ndarray, cov: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    n = len(active)
    w = np.zeros(n)
    inv = np.where(active & (sigma > 1e-6), 1.0 / sigma, 0.0)
    if inv.sum() == 0:
        return w
    w = inv / inv.sum()
    return _scale_to_port_vol(w, cov, PORT_VOL_CAP)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def optimize_cpo(
    mu: np.ndarray,
    cov: np.ndarray,
    sigma_ann: np.ndarray,
    active: np.ndarray,
    model_cols: list[str],
    shrink_alpha: float = SHRINK_ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the vol-budgeted CPO allocation and apply weight shrinkage.

    Objective : maximise w'mu  (mu = ensemble prob - 0.5, centred signal)
    Constraints:
      w_i >= 0                              (long-only)
      w_i * sigma_i <= cap_i               (per-model vol budget)
      sqrt(w_g' Cov_g w_g) <= cap_g        (group vol)
      sqrt(w'  Cov   w)    <= PORT_VOL_CAP (portfolio vol 6%)

    Post-optimisation:
      w_final = (1 - shrink_alpha) * w_EW + shrink_alpha * w_CPO

    Returns (w_cpo_raw, w_cpo_shrunk, w_ew)
    """
    col_to_idx = {c: int(c.split(" ")[1]) for c in model_cols}
    n          = len(model_cols)
    idx_active = np.where(active)[0]
    n_a        = len(idx_active)

    w_ew = _equal_weight(active, cov)

    if n_a == 0:
        z = np.zeros(n)
        return z, z, w_ew

    mu_a   = mu[active]
    cov_a  = cov[np.ix_(idx_active, idx_active)]
    sig_a  = sigma_ann[active]

    # ── per-model upper bounds from vol budget ────────────────────────────────
    local = {j: col_to_idx[model_cols[i]] for j, i in enumerate(idx_active)}
    bounds = []
    for j in range(n_a):
        cap = MODEL_VOL_CAPS[local[j]]
        ub  = cap / sig_a[j] if sig_a[j] > 1e-6 else 0.0
        bounds.append((0.0, min(ub, 1.0)))

    # ── group vol constraints sqrt(w_g'Cov_g w_g) <= cap_g ───────────────────
    group_cons = []
    for g_id, (g_models, g_cap) in GROUPS.items():
        g_local = [j for j, i in enumerate(idx_active)
                   if col_to_idx[model_cols[i]] in g_models]
        if not g_local:
            continue
        gl = np.array(g_local)
        cg = cov_a[np.ix_(gl, gl)]
        def _gc(w, gl=gl, cg=cg, gc=g_cap):
            wg = w[gl]
            return gc ** 2 - float(wg @ cg @ wg)
        group_cons.append({"type": "ineq", "fun": _gc})

    # ── portfolio vol constraint ──────────────────────────────────────────────
    def _pv(w):
        return PORT_VOL_CAP ** 2 - float(w @ cov_a @ w)
    group_cons.append({"type": "ineq", "fun": _pv})

    # ── capital constraint: long-only, no leverage ────────────────────────────
    def _sum_cap(w):
        return 1.0 - w.sum()           # w.sum() <= 1  (can hold cash)
    group_cons.append({"type": "ineq", "fun": _sum_cap})

    # ── solve: maximise w'μ (return objective) ────────────────────────────────
    # Sharpe objective is theoretically cleaner but converges to near-EW when
    # signals are weak (avg_prob ≈ 0.55).  Return objective + sum≤1 lets the
    # optimizer hold cash in low-conviction months, giving better OOS Sharpe.
    w0 = np.clip(np.full(n_a, 1.0 / n_a),
                 [b[0] for b in bounds], [b[1] for b in bounds])

    res = minimize(
        lambda w: -float(w @ mu_a),
        w0,
        jac=lambda w: -mu_a,
        method="SLSQP",
        bounds=bounds,
        constraints=group_cons,
        options={"ftol": 1e-9, "maxiter": 600},
    )

    w_full = np.zeros(n)
    if res.success or res.status in (0, 9):
        w_full[idx_active] = np.maximum(res.x, 0)
    else:
        w_full = _scale_to_port_vol(w_ew.copy(), cov, PORT_VOL_CAP)

    # ── dynamic shrinkage toward EW ──────────────────────────────────────────
    # shrink_alpha is passed from backtest; when caller passes None, use fixed
    # SHRINK_ALPHA.  Confidence-based alpha: trust CPO more when ensemble is
    # further from 0.5 (high conviction), stay near EW when uncertain.
    alpha = shrink_alpha if shrink_alpha is not None else SHRINK_ALPHA
    w_shrunk = alpha * w_full + (1.0 - alpha) * w_ew

    return w_full, w_shrunk, w_ew
