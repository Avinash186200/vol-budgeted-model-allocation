"""Conditional allocation selector: pick (or blend) a construction method each month.

An ML classifier reads point-in-time regime features (trailing vol, correlation,
dispersion, trend, recent method performance) and predicts which of five
construction methods will have the best risk-adjusted return next month. The
package also holds a long-short optimiser with a borrow cost (cvxpy, optional)
to compare with the long-only methods.
"""
from .evaluation import book_stats, compare, stats_table
from .features import best_method, forward_scores, regime_features
from .methods import METHODS, Market, SelectorConfig, build_method_panel, hold, make_market, run_book
from .walkforward import build_dataset, develop, run_holdout, train_rows, walk_forward_proba

__all__ = [
    "METHODS", "Market", "SelectorConfig", "build_method_panel", "hold", "make_market", "run_book",
    "best_method", "forward_scores", "regime_features",
    "build_dataset", "develop", "run_holdout", "train_rows", "walk_forward_proba",
    "book_stats", "compare", "stats_table",
]
