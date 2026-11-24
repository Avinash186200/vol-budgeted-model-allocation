"""Tests for the cvxpy long-short optimiser (volbudget.selector.longshort). Skipped without cvxpy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cvxpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volbudget import make_synthetic  # noqa: E402
from volbudget.selector import SelectorConfig, make_market  # noqa: E402
from volbudget.selector.longshort import LongShortConfig, long_short_weights, solve_long_short  # noqa: E402

TOL = 1e-5


def _problem(n=8, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.01, (500, n)) + rng.normal(0, 0.006, (500, 1)) * rng.uniform(-0.5, 1.5, n)
    S = np.cov(A, rowvar=False) * 252
    mu = rng.normal(0.0, 0.05, n)
    groups = [(g, 0.03) for g in np.array_split(np.arange(n), 2)]
    return mu, S, np.full(n, 0.02), groups


def _solve(ls, w_prev=None, **kw):
    mu, S, mcap, groups = _problem(**kw)
    w_prev = np.zeros(len(mu)) if w_prev is None else w_prev
    return solve_long_short(mu, S, w_prev, mcap, groups, 0.06, ls), (mu, S, mcap, groups)


def test_every_limit_holds():
    ls = LongShortConfig(gross_max=1.2, net_min=0.1, net_max=0.8)
    sol, (mu, S, mcap, groups) = _solve(ls)
    w = sol["w"]
    sig = np.sqrt(np.diag(S))
    assert np.abs(w).sum() <= ls.gross_max + TOL
    assert ls.net_min - TOL <= w.sum() <= ls.net_max + TOL
    assert (np.abs(w) * sig <= mcap + TOL).all()
    assert np.sqrt(w @ S @ w) <= 0.06 + TOL
    for g, cap in groups:
        assert np.sqrt(w[g] @ S[np.ix_(g, g)] @ w[g]) <= cap + TOL


def test_never_long_and_short_the_same_sleeve():
    sol, _ = _solve(LongShortConfig(borrow=0.01))
    assert np.minimum(sol["w_long"], sol["w_short"]).max() < 1e-6
    assert (sol["w_short"] > 1e-4).any()                      # it does short something here


def test_a_high_borrow_cost_removes_the_shorts():
    sol, _ = _solve(LongShortConfig(borrow=5.0))
    assert sol["w_short"].max() < 1e-6


def test_long_only_switch():
    sol, _ = _solve(LongShortConfig(allow_short=False))
    assert sol["w"].min() > -1e-7


def test_negative_signal_is_shorted_when_borrow_is_cheap():
    S = np.diag([0.01, 0.01, 0.01])
    mu = np.array([0.10, 0.05, -0.20])
    sol = solve_long_short(mu, S, np.zeros(3), np.full(3, 0.05), [], 0.08,
                           LongShortConfig(borrow=0.01, net_min=-1.0))
    assert sol["w"][2] < -0.1 and sol["w"][0] > 0.1


def test_turnover_penalty_shrinks_trading():
    mu, S, mcap, groups = _problem()
    prev = solve_long_short(mu, S, np.zeros(len(mu)), mcap, groups, 0.06, LongShortConfig())["w"]
    mu2 = mu[::-1].copy()
    trades = []
    for pen in (0.0, 0.01, 1.0):
        w = solve_long_short(mu2, S, prev, mcap, groups, 0.06, LongShortConfig(turnover_penalty=pen))["w"]
        trades.append(np.abs(w - prev).sum())
    assert trades[0] >= trades[1] - TOL >= trades[2] - 2 * TOL
    assert trades[2] < 1e-4


def test_monthly_books_ignore_data_after_the_month_end():
    daily = make_synthetic(n_models=8, n_months=48, seed=3).daily
    cfg = SelectorConfig(cov_days=126, mu_days=126)
    mk = make_market(daily)
    months = list(range(10, 16))
    base = long_short_weights(mk, months, cfg, LongShortConfig())
    cut = mk.months[13] + pd.Timedelta(hours=23)
    pert = daily.copy()
    later = pert.index > cut
    pert.loc[later] = np.where(pert.loc[later].to_numpy() != 0, 0.05, 0.0)
    again = long_short_weights(make_market(pert), months, cfg, LongShortConfig())
    for t in months:
        if t <= 13:
            assert np.allclose(base[t], again[t], atol=1e-7), t
    assert not np.allclose(base[15], again[15])
