"""Tests for the minimum-variance allocator (volbudget.min_variance)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volbudget import make_synthetic  # noqa: E402
from volbudget.data import make_independent_panel  # noqa: E402
from volbudget.optimizer import Budget, default_budget  # noqa: E402
from volbudget.min_variance import MVConfig, run_min_variance, weights_at  # noqa: E402
from volbudget.min_variance.backtest import prepare_daily, window_ends  # noqa: E402
from volbudget.min_variance.covariance import ESTIMATORS, ledoit_wolf_cov  # noqa: E402
from volbudget.min_variance.evaluation import (  # noqa: E402
    block_bootstrap_sharpe_diff,
    net_of_cost_sharpe,
    select_on_search,
    sharpe,
    sign_test_p,
    window_sharpes,
)
from volbudget.min_variance.sweep import judge_specs, construction_sweep, market_timed, vol_managed  # noqa: E402
from volbudget.min_variance.weights import (  # noqa: E402
    Caps,
    cap_ratios,
    caps_for,
    constrained_min_variance,
    erc,
    inverse_vol,
    min_variance,
    project_to_caps,
    scale_to_vol,
)

SMALL = MVConfig(oos_days=126, n_windows=3, refresh=10, cov_window=252, min_active_obs=126,
                 mv_windows=(126, 252), search_windows=1)


def _cov(seed=0, n=8):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.01, (400, n)) + rng.normal(0, 0.006, (400, 1)) * rng.uniform(0, 1.5, n)
    return ledoit_wolf_cov(A)


def _caps(n, model=0.01, port=0.06, gcap=0.03, n_groups=2):
    groups = [(g, gcap) for g in np.array_split(np.arange(n), n_groups)]
    return Caps(model=np.full(n, model), groups=groups, port=port)


# --- covariance and constructions -------------------------------------------------------------------------
def test_estimators_are_symmetric_psd_and_refuse_short_windows():
    R = np.random.default_rng(1).normal(0, 0.01, (300, 6))
    for name, fn in ESTIMATORS.items():
        S = fn(R)
        assert np.allclose(S, S.T), name
        assert np.linalg.eigvalsh(S).min() > -1e-12, name
        assert fn(R[:8]) is None, name


def test_min_variance_diagonal_is_inverse_variance():
    S = np.diag([0.01, 0.04, 0.09])
    w = min_variance(S)
    iv = 1 / np.diag(S)
    assert np.allclose(w, iv / iv.sum(), atol=1e-5)


def test_min_variance_is_long_only_and_beats_vol_parity_in_sample():
    S = _cov()
    w = min_variance(S)
    assert (w >= -1e-12).all() and abs(w.sum() - 1) < 1e-9
    vp = inverse_vol(S)
    assert w @ S @ w <= vp @ S @ vp + 1e-12


def test_erc_equalises_risk_contributions():
    S = _cov(2)
    w = erc(S)
    rc = w * (S @ w)
    assert np.allclose(rc, rc.mean(), rtol=1e-4)


def test_projection_fixes_a_breaching_blend():
    """The original's open item: a blend scaled to the portfolio cap breaches model and group caps."""
    S = _cov(3)
    n = S.shape[0]
    caps = _caps(n, model=0.012, gcap=0.035)
    blend = scale_to_vol(0.75 * min_variance(S) + 0.25 * inverse_vol(S), S, caps.port)
    assert max(cap_ratios(blend, S, caps).values()) > 1.0          # breaches before
    w = project_to_caps(blend, S, caps)
    r = cap_ratios(w, S, caps)
    assert (w >= 0).all() and max(r.values()) <= 1 + 1e-6          # compliant after, same covariance


def test_projection_leaves_a_compliant_book_alone():
    S = _cov(4)
    caps = _caps(S.shape[0], model=1.0, gcap=1.0, port=1.0)
    w0 = scale_to_vol(inverse_vol(S), S, 0.05)
    assert np.allclose(project_to_caps(w0, S, caps), w0)


def test_constrained_min_variance_is_compliant_and_matches_unconstrained_when_caps_are_loose():
    S = _cov(5)
    n = S.shape[0]
    tight = constrained_min_variance(S, _caps(n, model=0.012, gcap=0.035))
    assert max(cap_ratios(tight, S, _caps(n, model=0.012, gcap=0.035)).values()) <= 1 + 1e-6
    loose_caps = _caps(n, model=1.0, gcap=1.0, port=0.06)
    loose = constrained_min_variance(S, loose_caps)
    ref = scale_to_vol(min_variance(S), S, 0.06)
    assert np.allclose(loose, ref, atol=2e-3 * ref.max())


def test_caps_for_maps_columns_to_active_positions():
    cols = ["model 1", "model 2", "model 3", "model 4"]
    b = Budget(model_caps={"model 3": 0.005}, groups={"a": (["model 1", "model 3"], 0.02)})
    c = caps_for(b, cols, np.array([1, 2, 3]))
    assert np.allclose(c.model, [0.01, 0.005, 0.01])
    assert len(c.groups) == 1 and list(c.groups[0][0]) == [1]


# --- timing ---------------------------------------------------------------------------------------------
def test_leading_zeros_become_nan_interior_zeros_stay():
    idx = pd.bdate_range("2020-01-01", periods=10)
    d = prepare_daily(pd.DataFrame({"m": [0, 0, 0.01, 0, 0.02, 0, 0, 0, 0, 0.01]}, index=idx, dtype=float))
    assert d["m"].isna().sum() == 2 and d["m"].iloc[3] == 0


def test_weights_at_t_ignore_rows_from_t_onwards():
    p = make_synthetic(10, 60, 0.0, seed=2)
    X = prepare_daily(p.daily).to_numpy(float)
    cols = list(p.daily.columns)
    b = default_budget(cols, 2)
    t = 900
    base = weights_at(X, t, cols, SMALL, b)["w"]
    bumped = X.copy()
    bumped[t:] = bumped[t:] * -5 + 0.03
    after = weights_at(bumped, t, cols, SMALL, b)["w"]
    for k in base:
        np.testing.assert_allclose(base[k], after[k], err_msg=k)


def test_returns_start_after_the_execution_lag():
    """A shock on the rebalance day and the next exec_lag - 1 days is never booked into the first window."""
    p = make_synthetic(8, 60, 0.0, seed=4)
    daily = p.daily.copy()
    ends = window_ends(len(daily), SMALL)
    first_oos = ends[0] - SMALL.oos_days
    base = run_min_variance(daily, SMALL)[0].gross
    shocked = daily.copy()
    shocked.iloc[first_oos - SMALL.exec_lag: first_oos] += 0.5     # days before the window: never realised
    after = run_min_variance(shocked, SMALL)[0].gross
    # weights may react to the shock only from the next rebalance, so the first hold period is unchanged
    hold = SMALL.refresh
    pd.testing.assert_frame_equal(base.iloc[:hold], after.iloc[:hold])
    assert base.index[0] == daily.index[first_oos]


def test_windows_are_back_to_back_and_do_not_overlap():
    p = make_synthetic(8, 60, 0.0, seed=5)
    res = run_min_variance(p.daily, SMALL)
    for a, b in zip(res[:-1], res[1:]):
        assert a.gross.index.max() < b.gross.index.min()
    assert all(len(w.gross) == SMALL.oos_days for w in res)


def test_costs_lower_net_returns_by_turnover():
    p = make_synthetic(8, 60, 0.0, seed=6)
    free = run_min_variance(p.daily, MVConfig(**{**SMALL.__dict__, "cost_bps": 0.0}))
    paid = run_min_variance(p.daily, MVConfig(**{**SMALL.__dict__, "cost_bps": 10.0}))
    for f, c in zip(free, paid):
        pd.testing.assert_frame_equal(f.gross, c.gross)
        assert (c.net.sum() <= f.net.sum() + 1e-12).all()
        assert c.net["minvar_ens"].sum() < f.net["minvar_ens"].sum()


def test_blend_capped_respects_every_cap_in_the_backtest():
    p = make_synthetic(12, 60, 0.0, seed=7)
    res = run_min_variance(p.daily, SMALL)
    for w in res:
        for k in ("blend_capped", "vol_parity_capped"):
            assert max(w.worst_caps[k].values()) <= 1 + 1e-6, k


def test_too_little_history_raises():
    p = make_synthetic(6, 24, 0.0, seed=1)
    with pytest.raises(ValueError):
        run_min_variance(p.daily, MVConfig())


# --- evaluation --------------------------------------------------------------------------------------------
def test_sharpe_and_net_cost():
    r = np.random.default_rng(0).normal(0.001, 0.01, 2000)
    assert abs(sharpe(r) - r.mean() / r.std() * np.sqrt(252)) < 1e-12
    assert net_of_cost_sharpe(r, 0.0, 5) == pytest.approx(sharpe(r))
    assert net_of_cost_sharpe(r, 10.0, 5) < sharpe(r)


def test_bootstrap_identical_streams_and_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0005, 0.01, 1500)
    same = block_bootstrap_sharpe_diff(a, a, n_boot=200)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
    b = rng.normal(0.0005, 0.01, 1500)
    noise = block_bootstrap_sharpe_diff(a, b, n_boot=300, seed=3)
    assert noise["lo"] < noise["diff"] < noise["hi"]
    assert block_bootstrap_sharpe_diff(a, b, n_boot=300, seed=3) == noise      # deterministic
    better = block_bootstrap_sharpe_diff(a + 0.002, a, n_boot=300)              # same noise, higher mean
    assert better["lo"] > 0 and better["p_le_0"] == 0.0


def test_sign_test():
    assert sign_test_p(5, 5) == pytest.approx(1 / 32)
    assert sign_test_p(0, 5) == 1.0


def test_selection_never_looks_at_holdout():
    scores = pd.DataFrame({"a": [1.0, 1.0, 0.0, 0.0], "b": [0.5, 0.5, 9.0, 9.0]})
    sel = select_on_search(scores, ["a", "b"], n_search=2)
    assert sel["choice"] == "a" and sel["holdout"] == 0.0
    scores.loc[2:, "a"] = -9.0                                  # change only the hold-out
    assert select_on_search(scores, ["a", "b"], n_search=2)["choice"] == "a"


# --- overlays and sweep ----------------------------------------------------------------------------------
def test_overlays_are_causal():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=600)
    s = pd.Series(rng.normal(0.0004, 0.01, 600), index=idx)
    m = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    t = 400
    for fn in (lambda x, mk: vol_managed(x, 21, 2), lambda x, mk: market_timed(x, mk, 21, "inv_vol"),
               lambda x, mk: market_timed(x, mk, 21, "regime")):
        base = fn(s, m)
        s2, m2 = s.copy(), m.copy()
        s2.iloc[t:] *= 5
        m2.iloc[t:] *= 5
        pd.testing.assert_series_equal(base.iloc[:t], fn(s2, m2).iloc[:t])


def test_sweep_runs_and_judges_on_search_first():
    p = make_synthetic(8, 60, 0.0, seed=8)
    streams = construction_sweep(p.daily, SMALL, windows=(126,), estimators=("lw",),
                                 objectives=("equal", "min_var"))
    assert set(streams[0].columns) == {"VP", "lw_126:equal", "lw_126:min_var"}
    judged = judge_specs(streams, n_search=1, n_boot=50)
    assert set(judged["verdict"]) <= {"robust", "nominal", "noise", "not a finalist"}
    assert ((judged["search"] > 0) == (judged["verdict"] != "not a finalist")).all()


# --- the noise control -------------------------------------------------------------------------------------
def test_independent_equal_sharpe_panel_gives_min_variance_no_edge():
    """With no correlation structure and equal Sharpe ratios there is nothing for min-variance to find."""
    daily = make_independent_panel(10, n_days=1400, seed=3)
    cfg = MVConfig(oos_days=252, n_windows=3, refresh=10, cov_window=252, min_active_obs=126,
                   mv_windows=(126, 252), search_windows=1)
    ws = window_sharpes(run_min_variance(daily, cfg), "net")
    assert (ws["blend"] - ws["vol_parity"]).mean() < 0.1


def test_script_smoke(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_min_variance

    assert run_min_variance.main(["--models", "8", "--months", "60", "--windows", "3", "--oos-days", "126",
                                  "--boot", "50", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "window_net_sharpe.csv").exists()
