"""Covariance estimators on a window of daily returns.

Every estimator takes a ``T x k`` array of daily returns with no missing values
and returns an annualised ``k x k`` covariance (daily times 252). They return
``None`` when the window is too short to estimate ``k`` columns reliably.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import OAS, LedoitWolf

TRADING_DAYS = 252


def _too_short(R: np.ndarray) -> bool:
    return R.ndim != 2 or R.shape[0] <= R.shape[1] + 5


def sample_cov(R: np.ndarray) -> np.ndarray | None:
    """Plain sample covariance, annualised."""
    if _too_short(R):
        return None
    return np.atleast_2d(np.cov(R, rowvar=False)) * TRADING_DAYS


def ledoit_wolf_cov(R: np.ndarray) -> np.ndarray | None:
    """Ledoit-Wolf shrinkage towards a scaled identity, annualised.

    The shrinkage intensity is estimated from the data, so there is no knob to tune.
    """
    if _too_short(R):
        return None
    return LedoitWolf().fit(R).covariance_ * TRADING_DAYS


def oas_cov(R: np.ndarray) -> np.ndarray | None:
    """Oracle Approximating Shrinkage, annualised."""
    if _too_short(R):
        return None
    return OAS().fit(R).covariance_ * TRADING_DAYS


def ewma_cov(R: np.ndarray, halflife: float = 60.0) -> np.ndarray | None:
    """Exponentially weighted covariance (most recent row weighted most), annualised."""
    if _too_short(R):
        return None
    w = 0.5 ** (np.arange(R.shape[0])[::-1] / halflife)
    w = w / w.sum()
    Rm = R - np.average(R, axis=0, weights=w)
    return (Rm * w[:, None]).T @ Rm * TRADING_DAYS


ESTIMATORS = {
    "sample": sample_cov,
    "lw": ledoit_wolf_cov,
    "oas": oas_cov,
    "ewma60": lambda R: ewma_cov(R, 60.0),
}
