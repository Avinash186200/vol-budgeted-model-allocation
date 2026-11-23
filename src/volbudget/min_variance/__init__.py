"""Non-ML, risk-only allocation: Ledoit-Wolf minimum variance, blended with vol parity.

A follow-up to the forecast-driven allocator in the parent package. It uses no
return forecast at all, only the covariance of the trading models, and is run
at a daily cadence with a T+2 execution lag.
"""
from .backtest import MVConfig, run_min_variance, weights_at
from .evaluation import block_bootstrap_sharpe_diff, select_on_search, sharpe, summary, window_sharpes
from .weights import Caps, cap_ratios, constrained_min_variance, min_variance, project_to_caps

__all__ = [
    "MVConfig", "run_min_variance", "weights_at",
    "block_bootstrap_sharpe_diff", "select_on_search", "sharpe", "summary", "window_sharpes",
    "Caps", "cap_ratios", "constrained_min_variance", "min_variance", "project_to_caps",
]
