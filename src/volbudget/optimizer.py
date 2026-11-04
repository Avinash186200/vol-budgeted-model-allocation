"""Factor covariance and the vol-budgeted long-only allocator."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA


@dataclass
class Budget:
    """Three-tier volatility budget (all annualised).

    Attributes:
        port_cap: Portfolio vol cap.
        model_cap: Default per-model cap on ``w_i * sigma_i``.
        model_caps: Optional per-model overrides keyed by column name.
        groups: Mapping group name -> (list of columns, group vol cap).
    """

    port_cap: float = 0.06
    model_cap: float = 0.01
    model_caps: dict[str, float] = field(default_factory=dict)
    groups: dict[str, tuple[list[str], float]] = field(default_factory=dict)


def default_budget(cols: list[str], n_groups: int = 4) -> Budget:
    """Split the models into ``n_groups`` contiguous groups with a 3% vol cap each."""
    chunks = np.array_split(np.array(cols), n_groups)
    groups = {f"g{i + 1}": (list(ch), 0.03) for i, ch in enumerate(chunks) if len(ch)}
    return Budget(groups=groups)


def factor_covariance(R: pd.DataFrame, market: pd.Series | None, k: int = 4) -> np.ndarray:
    """Annualised ``B Sigma_F B' + D`` from monthly returns.

    Factor 1 is the market (if given); the rest are principal components of
    the market residuals. Missing returns are filled with each column's mean
    over the window, so pre-inception gaps do not shrink a model's variance.
    """
    R = R.apply(lambda s: s.fillna(s.mean())).fillna(0.0)
    X = R.to_numpy()
    T, N = X.shape
    X = X - X.mean(axis=0)
    if market is not None and len(market):
        m = market.reindex(R.index).fillna(0.0).to_numpy()
        m = (m - m.mean()).reshape(-1, 1)
        beta = (m.T @ X) / max(float((m ** 2).sum()), 1e-12)
    else:
        m = np.zeros((T, 1))
        beta = np.zeros((1, N))
    resid = X - m @ beta
    n_pc = max(0, min(k - 1, T - 2, N - 1))
    if n_pc > 0:
        pca = PCA(n_components=n_pc, random_state=0).fit(resid)
        F = np.hstack([m, pca.transform(resid)])
        B = np.hstack([beta.T, pca.components_.T])
    else:
        F, B = m, beta.T
    Sf = np.atleast_2d(np.cov(F.T)) * 12
    D = np.maximum((X - F @ B.T).var(axis=0) * 12, 1e-6)
    S = B @ Sf @ B.T + np.diag(D)
    return 0.5 * (S + S.T) + 1e-10 * np.eye(N)


def scale_to_cap(w: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    """Scale ``w`` down uniformly until predicted vol <= cap."""
    pv = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return w * (cap / pv) if pv > cap else w


def equal_weight(active: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    """Equal weight across active models, scaled to the portfolio cap."""
    w = active.astype(float)
    return scale_to_cap(w / max(w.sum(), 1), cov, cap)


def vol_parity(active: np.ndarray, cov: np.ndarray, cap: float) -> np.ndarray:
    """Inverse-vol weights across active models, scaled to the portfolio cap."""
    sig = np.sqrt(np.diag(cov))
    inv = np.where(active & (sig > 1e-6), 1.0 / sig, 0.0)
    if inv.sum() == 0:
        return inv
    return scale_to_cap(inv / inv.sum(), cov, cap)


def optimize(
    mu: np.ndarray, cov: np.ndarray, active: np.ndarray, cols: list[str], budget: Budget
) -> np.ndarray:
    """Maximise ``w' mu`` subject to the vol budget, long-only and sum(w) <= 1.

    Falls back to scaled equal weight if SLSQP fails.
    """
    n = len(cols)
    idx = np.where(active)[0]
    w = np.zeros(n)
    if len(idx) == 0:
        return w
    sig = np.sqrt(np.diag(cov))[idx]
    ca = cov[np.ix_(idx, idx)]
    mua = mu[idx]
    bounds = [(0.0, min(1.0, budget.model_caps.get(cols[i], budget.model_cap) / max(s, 1e-6)))
              for i, s in zip(idx, sig)]
    cons = [
        {"type": "ineq", "fun": lambda x: budget.port_cap ** 2 - x @ ca @ x},
        {"type": "ineq", "fun": lambda x: 1.0 - x.sum()},
    ]
    pos = {cols[i]: j for j, i in enumerate(idx)}
    for members, cap in budget.groups.values():
        g = np.array([pos[c] for c in members if c in pos], dtype=int)
        if len(g):
            cg = ca[np.ix_(g, g)]
            cons.append({"type": "ineq", "fun": lambda x, g=g, cg=cg, cap=cap: cap ** 2 - x[g] @ cg @ x[g]})
    x0 = np.array([b[1] for b in bounds]) * 0.1
    res = minimize(lambda x: -x @ mua, x0, jac=lambda x: -mua, method="SLSQP",
                   bounds=bounds, constraints=cons, options={"ftol": 1e-10, "maxiter": 500})
    if not res.success:
        return equal_weight(active, cov, budget.port_cap)
    w[idx] = np.clip(res.x, 0, None)
    return w
