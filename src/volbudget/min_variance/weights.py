"""Risk-only portfolio constructions and the vol-budget projection.

Nothing in this module uses an expected return, except ``kelly_diag``, which is
kept only as a contrast in the sweep. All functions work on the covariance of
the *active* models and return long-only weights.

Weights that "sum to 1" are directions only. ``scale_to_vol`` turns a direction
into a book with a given predicted volatility, which is how every leg is put on
the same footing before blending.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..optimizer import Budget


# --------------------------------------------------------------------------- #
# Directions (sum to 1)
# --------------------------------------------------------------------------- #
def _normalise(w: np.ndarray) -> np.ndarray:
    s = float(w.sum())
    return w / s if s > 0 else np.ones_like(w) / len(w)


def equal_weight(S: np.ndarray) -> np.ndarray:
    """1/N."""
    return np.ones(S.shape[0]) / S.shape[0]


def inverse_vol(S: np.ndarray) -> np.ndarray:
    """Vol parity: weight proportional to 1/sigma_i. Ignores correlations."""
    return _normalise(1.0 / np.sqrt(np.clip(np.diag(S), 1e-12, None)))


def inverse_var(S: np.ndarray) -> np.ndarray:
    """Weight proportional to 1/sigma_i^2 (min variance if all correlations were zero)."""
    return _normalise(1.0 / np.clip(np.diag(S), 1e-12, None))


def min_variance(S: np.ndarray) -> np.ndarray:
    """Long-only minimum variance: minimise ``w'Sw`` with ``w >= 0`` and ``sum(w) = 1``.

    Falls back to inverse variance if SLSQP does not converge.
    """
    n = S.shape[0]
    x0 = inverse_var(S)
    res = minimize(lambda w: w @ S @ w, x0, jac=lambda w: 2 * S @ w, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones_like(w)},),
                   options={"maxiter": 300, "ftol": 1e-12})
    if not res.success or res.x.sum() <= 0:
        return x0
    return _normalise(np.clip(res.x, 0, None))


def erc(S: np.ndarray, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Equal risk contribution: every ``w_i (S w)_i`` is the same.

    Cyclical coordinate descent on ``0.5 w'Sw - sum(log w_i) / n``, whose minimiser
    has equal risk contributions; each coordinate step is the positive root of a
    quadratic. On the synthetic panel it converges about fifty times faster than
    the damped fixed point used in the original research.
    """
    n = S.shape[0]
    b = 1.0 / n
    d = np.clip(np.diag(S), 1e-18, None)
    w = inverse_vol(S)
    for _ in range(iters):
        w_old = w.copy()
        for i in range(n):
            c = S[i] @ w - d[i] * w[i]
            w[i] = (-c + np.sqrt(c * c + 4 * d[i] * b)) / (2 * d[i])
        if np.max(np.abs(_normalise(w) - _normalise(w_old))) < tol:
            break
    return _normalise(w)


def max_diversification(S: np.ndarray) -> np.ndarray:
    """Maximise ``w'sigma / sqrt(w'Sw)``, long-only."""
    n = S.shape[0]
    s = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    res = minimize(lambda w: -(w @ s) / np.sqrt(max(w @ S @ w, 1e-18)), inverse_var(S),
                   method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                   options={"maxiter": 300, "ftol": 1e-12})
    return _normalise(np.clip(res.x, 0, None)) if res.success else inverse_var(S)


def kelly_diag(S: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Diagonal Kelly ``mu_i / sigma_i^2`` with negatives set to 0. Uses trailing mean returns."""
    w = np.clip(mu, 0, None) / np.clip(np.diag(S), 1e-12, None)
    return _normalise(w) if w.sum() > 0 else inverse_var(S)


# --------------------------------------------------------------------------- #
# Vol budget
# --------------------------------------------------------------------------- #
@dataclass
class Caps:
    """A vol budget expressed on the active models (all annualised).

    Attributes:
        model: Per-model cap on ``w_i * sigma_i``, one entry per active model.
        groups: List of (positions within the active set, group vol cap).
        port: Portfolio vol cap.
    """

    model: np.ndarray
    groups: list[tuple[np.ndarray, float]]
    port: float


def caps_for(budget: Budget, cols: list[str], active_idx: np.ndarray) -> Caps:
    """Translate a column-keyed ``Budget`` into ``Caps`` on the active subset."""
    names = [cols[i] for i in active_idx]
    model = np.array([budget.model_caps.get(c, budget.model_cap) for c in names])
    pos = {c: j for j, c in enumerate(names)}
    groups = []
    for members, cap in budget.groups.values():
        g = np.array([pos[c] for c in members if c in pos], dtype=int)
        if len(g):
            groups.append((g, float(cap)))
    return Caps(model=model, groups=groups, port=float(budget.port_cap))


def scale_to_vol(w: np.ndarray, S: np.ndarray, target: float) -> np.ndarray:
    """Scale ``w`` so its predicted vol under ``S`` equals ``target``."""
    pv = float(np.sqrt(max(w @ S @ w, 0.0)))
    return w * (target / pv) if pv > 0 else w


def cap_ratios(w: np.ndarray, S: np.ndarray, caps: Caps) -> dict[str, float]:
    """Worst usage of each cap level (1.0 = exactly at the cap, above 1 = breach)."""
    sig = np.sqrt(np.clip(np.diag(S), 0, None))
    model = float(np.max(w * sig / caps.model)) if len(w) else 0.0
    group = max((float(np.sqrt(max(w[g] @ S[np.ix_(g, g)] @ w[g], 0.0))) / c for g, c in caps.groups),
                default=0.0)
    port = float(np.sqrt(max(w @ S @ w, 0.0))) / caps.port
    return {"model": model, "group": group, "port": port}


def _cap_constraints(S: np.ndarray, caps: Caps) -> list[dict]:
    """SLSQP inequality constraints (with gradients) for the portfolio and group vol caps."""
    cons = [{"type": "ineq", "fun": lambda w: caps.port ** 2 - w @ S @ w, "jac": lambda w: -2 * S @ w}]
    for g, c in caps.groups:
        Sg = S[np.ix_(g, g)]

        def gfun(w, g=g, Sg=Sg, c=c):
            return c ** 2 - w[g] @ Sg @ w[g]

        def gjac(w, g=g, Sg=Sg):
            out = np.zeros_like(w)
            out[g] = -2 * Sg @ w[g]
            return out

        cons.append({"type": "ineq", "fun": gfun, "jac": gjac})
    return cons


def project_to_caps(w0: np.ndarray, S: np.ndarray, caps: Caps) -> np.ndarray:
    """Closest long-only weights to ``w0`` that respect every cap under ``S``.

    Solves ``min ||w - w0||^2`` subject to ``0 <= w_i sigma_i <= cap_i``, every
    group vol <= its cap and portfolio vol <= ``caps.port``. All constraints are
    positively homogeneous, so ``w0`` scaled down by its worst cap ratio is always
    feasible; that is the starting point and the fallback.
    """
    w0 = np.clip(w0, 0, None)
    worst = max(cap_ratios(w0, S, caps).values())
    feasible = w0 / worst if worst > 1 else w0.copy()
    if worst <= 1 + 1e-9:
        return feasible
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    ub = caps.model / sig
    cons = _cap_constraints(S, caps)
    res = minimize(lambda w: np.sum((w - w0) ** 2), feasible, jac=lambda w: 2 * (w - w0),
                   method="SLSQP", bounds=[(0.0, float(u)) for u in ub], constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-14})
    if not res.success:
        return feasible
    w = np.clip(res.x, 0, None)
    # SLSQP can end a hair outside a quadratic constraint; scale back onto the feasible set
    worst = max(cap_ratios(w, S, caps).values())
    return w / worst if worst > 1 else w


def constrained_min_variance(S: np.ndarray, caps: Caps) -> np.ndarray:
    """Minimum variance re-solved under the caps, as an absolute book.

    Long-only minimum variance has a dual form: the book with the largest gross
    exposure ``sum(w)`` for a given portfolio vol is the min-variance direction
    (the KKT conditions are the same). Written that way, the per-model and group
    caps go straight in as absolute limits:

        maximise sum(w)  s.t.  w >= 0,  w_i sigma_i <= cap_i,
                               group vol <= group cap,  portfolio vol <= port cap.

    With no binding model or group cap this returns the long-only min-variance
    direction scaled to the portfolio cap. The original research instead solved
    a sum-to-1 problem with caps divided by a scale factor that it re-estimated in
    a loop. That problem can be infeasible when the group caps are tight (on the
    synthetic panel SLSQP often failed to converge in my checks), and
    the code then fell back to the solver's starting point.
    """
    n = S.shape[0]
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    start = inverse_var(S)
    start = start / max(cap_ratios(start, S, caps).values())   # feasible: every cap ratio <= 1
    res = minimize(lambda w: -w.sum(), start, jac=lambda w: -np.ones(n), method="SLSQP",
                   bounds=[(0.0, float(u)) for u in caps.model / sig], constraints=_cap_constraints(S, caps),
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None) if res.success else start
    worst = max(cap_ratios(w, S, caps).values())
    return w / worst if worst > 1 else w
