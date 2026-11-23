"""Scoring: per-window Sharpe, win counts, block bootstrap and search/hold-out selection."""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from .backtest import WindowResult


def sharpe(r: np.ndarray | pd.Series) -> float:
    """Annualised Sharpe of daily returns (no risk-free rate; the book is self-financing)."""
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))


def window_sharpes(results: list[WindowResult], kind: str = "net") -> pd.DataFrame:
    """Rows = test windows (oldest first), columns = strategies."""
    return pd.DataFrame([getattr(w, kind).apply(sharpe) for w in results]).rename_axis("window")


def summary(results: list[WindowResult], names: list[str], base: str = "vol_parity") -> pd.DataFrame:
    """Mean gross and net Sharpe across windows, windows won against ``base``, turnover and worst caps."""
    g = window_sharpes(results, "gross")
    n = window_sharpes(results, "net")
    rows = {}
    for k in names:
        caps = [w.worst_caps.get(k) for w in results if k in w.worst_caps]
        rows[k] = {
            "gross_sharpe": g[k].mean(),
            "net_sharpe": n[k].mean(),
            "wins_vs_vp": int((n[k] > n[base]).sum()) if k != base else np.nan,
            "turnover_yr": float(np.mean([w.turnover[k] for w in results])),
            "worst_model_cap": max((c["model"] for c in caps), default=np.nan),
            "worst_group_cap": max((c["group"] for c in caps), default=np.nan),
        }
    return pd.DataFrame(rows).T


def sign_test_p(wins: int, n: int) -> float:
    """One-sided binomial p-value of at least ``wins`` wins out of ``n`` under a fair coin."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def block_bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, mean_block: int = 21,
                                seed: int = 0) -> dict[str, float]:
    """Paired stationary block bootstrap of ``sharpe(a) - sharpe(b)``.

    Both series are resampled with the same blocks, so their correlation is kept.
    Returns the observed difference, P(diff <= 0) and a 95% interval.

    The days inside one test window are not independent draws of the allocator's
    skill, so this interval is optimistic; read it next to the per-window win count.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    N = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(N / mean_block * 3)) + 1
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, N, n_blocks)
        lens = rng.geometric(1.0 / mean_block, n_blocks)
        idx = np.concatenate([np.arange(s, s + L) for s, L in zip(starts, lens)])[:N] % N
        diffs[i] = sharpe(a[idx]) - sharpe(b[idx])
    return {"diff": sharpe(a) - sharpe(b), "p_le_0": float((diffs <= 0).mean()),
            "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}


def select_on_search(scores: pd.DataFrame, candidates: list[str], n_search: int) -> dict:
    """Pick the candidate with the best mean score on the first ``n_search`` windows only.

    ``scores`` has one row per window (oldest first). The hold-out windows are
    never looked at during the choice; they are only scored afterwards.
    """
    search, hold = scores.iloc[:n_search], scores.iloc[n_search:]
    best = max(candidates, key=lambda c: search[c].mean())
    return {"choice": best, "search": float(search[best].mean()), "holdout": float(hold[best].mean()),
            "holdout_scores": hold[best].tolist()}


def net_of_cost_sharpe(gross: np.ndarray, turnover_yr: float, bps: float) -> float:
    """Approximate net Sharpe: ``(annual return - turnover * cost) / annual vol``."""
    g = np.asarray(gross, float)
    vol = g.std() * np.sqrt(252)
    return float((g.mean() * 252 - turnover_yr * bps / 1e4) / vol) if vol > 0 else float("nan")
