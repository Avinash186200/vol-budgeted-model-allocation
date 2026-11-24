"""A long-short optimiser with a borrow cost, written as one convex problem (cvxpy).

Each weight is split into a long part and a short part, ``w = w_long - w_short``
with both parts non-negative. The long part earns the sleeve's return; the short
part earns minus the return and also pays ``borrow`` a year. Because the short
part costs money, the solver never holds both parts of the same sleeve at once
(that would pay borrow for no change in ``w``), so the split is the usual
positive/negative decomposition at the optimum. ``tests/test_longshort.py`` checks
that numerically.

At month-end ``m``::

    maximise   mu' w  -  borrow * sum(w_short)  -  turnover_penalty * ||w - w_prev||_1
    subject to w_long, w_short >= 0
               sum(w_long + w_short)         <= gross_max          (gross exposure)
               net_min <= sum(w_long - w_short) <= net_max          (net exposure)
               (w_long + w_short)_i * sigma_i <= model_cap         (per-sleeve risk)
               || L_g' w_g ||_2              <= group_cap          (group vol, L_g L_g' = S_g)
               || L' w ||_2                  <= port_cap           (portfolio vol)

``mu`` is each sleeve's trailing mean return (the same signal ``vb_optimiser``
uses), ``S`` the Ledoit-Wolf covariance and ``w_prev`` last month's book after
drift. The solver is pinned to Clarabel. cvxpy is an optional dependency: ``pip install -e ".[longshort]"``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..optimizer import Budget, default_budget
from .methods import Market, SelectorConfig, active_window, hold
from ..min_variance.covariance import ledoit_wolf_cov


@dataclass
class LongShortConfig:
    """Limits and costs of the long-short book (annual units where relevant).

    Attributes:
        gross_max: Cap on ``sum |w|``.
        net_min, net_max: Bounds on ``sum w``.
        borrow: Annual cost of the short side, per unit of short weight.
        turnover_penalty: Penalty per unit of one-way turnover in the objective,
            in annual-return units (it is a regulariser, not the cost charged in
            the backtest, which is ``SelectorConfig.cost_bps``).
        allow_short: False gives the same problem with ``w_short = 0`` (long-only).
    """

    gross_max: float = 1.5
    net_min: float = 0.0
    net_max: float = 1.0
    borrow: float = 0.02
    turnover_penalty: float = 0.002
    allow_short: bool = True


def _chol(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    return np.linalg.cholesky(S + 1e-12 * np.trace(S) / len(S) * np.eye(len(S)))


def solve_long_short(mu: np.ndarray, S: np.ndarray, w_prev: np.ndarray, model_cap: np.ndarray,
                     groups: list[tuple[np.ndarray, float]], port_cap: float, ls: LongShortConfig
                     ) -> dict[str, np.ndarray] | None:
    """Solve the problem above for one month on the active sleeves.

    Returns ``{"w", "w_long", "w_short"}`` or None if the solver fails.
    """
    import cvxpy as cp

    n = len(mu)
    sig = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    wl = cp.Variable(n, nonneg=True)
    ws = cp.Variable(n, nonneg=True)
    w = wl - ws
    cons = [cp.sum(wl + ws) <= ls.gross_max,
            cp.sum(w) >= ls.net_min, cp.sum(w) <= ls.net_max,
            cp.multiply(wl + ws, sig) <= model_cap,
            cp.norm(_chol(S).T @ w, 2) <= port_cap]
    for g, cap in groups:
        cons.append(cp.norm(_chol(S[np.ix_(g, g)]).T @ w[g], 2) <= cap)
    if not ls.allow_short:
        cons.append(ws == 0)
    obj = cp.Maximize(mu @ w - ls.borrow * cp.sum(ws) - ls.turnover_penalty * cp.norm(w - w_prev, 1))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate") or wl.value is None:
        return None
    a, b = np.clip(wl.value, 0, None), np.clip(ws.value, 0, None)
    return {"w": a - b, "w_long": a, "w_short": b}


def long_short_weights(mk: Market, months: list[int], cfg: SelectorConfig, ls: LongShortConfig,
                       budget: Budget | None = None) -> dict[int, np.ndarray]:
    """Run the optimiser month by month, feeding each solve last month's drifted book.

    Uses the same active-model rule, covariance window and trailing-mean signal as
    the long-only methods, and the same vol budget as ``vb_optimiser`` (per-model,
    group and portfolio caps). If a solve fails, the drifted book is kept.
    """
    budget = budget or default_budget(mk.cols)
    n = len(mk.cols)
    out: dict[int, np.ndarray] = {}
    prev = np.zeros(n)
    for t in months:
        W, act = active_window(mk, t, cfg.cov_days)
        w_new = prev.copy()
        S = ledoit_wolf_cov(W[:, act]) if len(act) >= cfg.min_models else None
        if S is not None:
            end = int(mk.ends[t])
            mu = np.nanmean(mk.X[max(0, end - cfg.mu_days + 1):end + 1][:, act], axis=0) * 252
            names = [mk.cols[i] for i in act]
            mcap = np.array([budget.model_caps.get(c, budget.model_cap) for c in names])
            pos = {c: j for j, c in enumerate(names)}
            groups = []
            for members, cap in budget.groups.values():
                g = np.array([pos[c] for c in members if c in pos], dtype=int)
                if len(g):
                    groups.append((g, float(cap)))
            sol = solve_long_short(mu, S, prev[act], mcap, groups, budget.port_cap, ls)
            if sol is not None:
                w_new = np.zeros(n)
                w_new[act] = sol["w"]
        out[t] = w_new
        if t + 1 < len(mk.ends):
            _, prev = hold(w_new, mk.X[mk.days_of(t + 1)])
    return out
