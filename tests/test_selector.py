"""Tests for the conditional allocation selector (volbudget.selector)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volbudget import make_synthetic  # noqa: E402
from volbudget.selector import (  # noqa: E402
    METHODS,
    SelectorConfig,
    build_dataset,
    build_method_panel,
    develop,
    forward_scores,
    hold,
    make_market,
    regime_features,
    run_book,
    train_rows,
    walk_forward_proba,
)
from volbudget.selector.evaluation import book_stats  # noqa: E402
from volbudget.selector.methods import method_weights  # noqa: E402
from volbudget.selector.walkforward import oracle_weights, selector_weights  # noqa: E402

SMALL = SelectorConfig(cov_days=126, mu_days=126, min_train=20, holdout_months=18, retrain_every=6)


def _daily(seed=3, n=8, months=96, strength=2.0):
    return make_synthetic(n_models=n, n_months=months, seed=seed, regime_strength=strength).daily


@pytest.fixture(scope="module")
def small():
    daily = _daily()
    mk = make_market(daily)
    panel = build_method_panel(mk, SMALL)
    ds = build_dataset(mk, panel, SMALL)
    return daily, mk, panel, ds


def _perturb_after(daily: pd.DataFrame, date, seed=99) -> pd.DataFrame:
    """Replace every return strictly after ``date`` with noise (keeping pre-inception zeros at zero)."""
    out = daily.copy()
    rng = np.random.default_rng(seed)
    later = out.index > date
    noise = rng.normal(0, 0.02, (int(later.sum()), out.shape[1]))
    out.loc[later] = np.where(out.loc[later].to_numpy() != 0, noise, 0.0)
    return out


# --- synthetic generator ----------------------------------------------------------------------------------
def test_regime_off_leaves_the_existing_generator_unchanged():
    a = make_synthetic(n_models=6, n_months=40, seed=5)
    b = make_synthetic(n_models=6, n_months=40, seed=5, regime_strength=0.0)
    pd.testing.assert_frame_equal(a.daily, b.daily)
    assert a.regime is None and b.regime is None


def test_regime_mode_adds_a_persistent_two_state_path_and_keeps_inception_zeros():
    off = make_synthetic(n_models=6, n_months=120, seed=5)
    on = make_synthetic(n_models=6, n_months=120, seed=5, regime_strength=1.0)
    s = on.regime.to_numpy()
    assert set(np.unique(s)) <= {0, 1} and 0 < s.mean() < 1
    assert (s[1:] != s[:-1]).mean() < 0.25            # spells, not coin flips
    pd.testing.assert_frame_equal(on.macro, off.macro)  # the regime draws use their own stream
    assert ((off.daily == 0) == (on.daily == 0)).all().all()


def test_planted_effect_makes_the_best_method_depend_on_the_state():
    p = make_synthetic(n_models=24, n_months=180, seed=7, regime_strength=2.0)
    cfg = SelectorConfig()
    mk = make_market(p.daily)
    panel = build_method_panel(mk, cfg)
    y = forward_scores(mk, panel, cfg).idxmax(axis=1)
    state = p.regime.reindex(y.index)
    mv_share = {s: float((y[state == s] == "min_variance").mean()) for s in (0, 1)}
    assert mv_share[1] > mv_share[0] + 0.2


# --- book accounting ----------------------------------------------------------------------------------------
def test_hold_matches_a_hand_computed_buy_and_hold_book_including_shorts():
    R = np.array([[0.01, -0.02], [0.03, 0.01], [-0.01, 0.02]])
    w = np.array([0.6, -0.3])
    r, drift = hold(w, R)
    G = np.cumprod(1 + R, axis=0)
    V = 1 + (G - 1) @ w
    assert np.allclose(np.cumprod(1 + r), V)
    assert np.allclose(drift, w * G[-1] / V[-1])
    r0, d0 = hold(np.zeros(2), R)
    assert np.allclose(r0, 0) and np.allclose(d0, 0)


def test_run_book_charges_cost_on_turnover_from_the_drifted_book(small):
    _, mk, panel, ds = small
    months = ds.dev[:3]
    w = {t: panel.weights[t]["vol_parity"] for t in months}
    bk = run_book(mk, w, cost_bps=10.0)
    _, drift0 = hold(w[months[0]], mk.X[mk.days_of(months[0] + 1)])
    expected = np.abs(w[months[1]] - drift0).sum()
    assert np.isclose(bk.turnover.iloc[1], expected)
    first_day = mk.dates[mk.days_of(months[1] + 1).start]
    assert np.isclose(bk.gross[first_day] - bk.net[first_day], expected * 10.0 / 1e4)
    assert np.isclose(bk.turnover.iloc[0], np.abs(w[months[0]]).sum())   # start from cash


def test_scaled_methods_hit_the_target_vol_and_the_optimiser_is_long_only(small):
    _, mk, _, ds = small
    got = method_weights(mk, ds.dev[0], SMALL)
    S, act = got["S"], got["act"]
    for k in ("equal_weight", "vol_parity", "min_variance", "erc"):
        w = got["w"][k][act]
        assert np.isclose(np.sqrt(w @ S @ w), SMALL.target_vol)
        assert (w >= -1e-12).all()
    assert (got["w"]["vb_optimiser"] >= 0).all() and got["w"]["vb_optimiser"].sum() <= 1 + 1e-6


# --- point in time ------------------------------------------------------------------------------------------
def test_method_weights_and_features_ignore_data_after_the_month_end(small):
    daily, mk, panel, ds = small
    m = ds.dev[5]
    pert = _perturb_after(daily, mk.months[m] + pd.Timedelta(hours=23))
    mk2 = make_market(pert)
    panel2 = build_method_panel(mk2, SMALL)
    for t in [x for x in panel.weights if x <= m]:
        for k in METHODS:
            assert np.allclose(panel.weights[t][k], panel2.weights[t][k]), (t, k)
    F1 = regime_features(mk, panel, SMALL)
    F2 = regime_features(mk2, panel2, SMALL)
    early = F1.index[F1.index <= mk.months[m]]
    pd.testing.assert_frame_equal(F1.loc[early], F2.loc[early])
    assert not F1.loc[F1.index > mk.months[m]].equals(F2.loc[F2.index > mk.months[m]])


def test_target_uses_only_the_next_h_months(small):
    _, mk, panel, _ = small
    cfg = SelectorConfig(**{**SMALL.__dict__, "horizon": 2})
    sc = forward_scores(mk, panel, cfg)
    d = sc.index[10]
    m = int(np.flatnonzero(mk.months == d)[0])
    lo, hi = mk.days_of(m + 1).start, mk.days_of(m + 2).stop
    for k in METHODS:
        span = panel.books[k].net.loc[mk.dates[lo]:mk.dates[hi - 1]]
        assert np.isclose(sc.loc[d, k], span.mean() / span.std())


@pytest.mark.parametrize("h,e", [(1, 0), (1, 1), (3, 1), (2, 2)])
def test_purge_and_embargo_keep_every_training_target_realised(h, e):
    cfg = SelectorConfig(horizon=h, embargo=e)
    idx = pd.Index(range(100))
    for t in (30, 57, 99):
        rows = train_rows(idx, t, cfg)
        assert rows.max() + h <= t - e           # the newest label ended at least e months before t
        assert len(rows) == t - h - e + 1        # and nothing older is dropped


def test_predictions_do_not_change_when_unrealised_targets_change(small):
    _, _, _, ds = small
    t = ds.dev[8]
    base = walk_forward_proba(ds, [t], "logit", 0.3, SMALL)
    y2 = ds.y.copy()
    unrealised = y2.index[y2.index + SMALL.horizon + SMALL.embargo > t]
    y2.loc[unrealised] = (y2.loc[unrealised] + 2) % len(METHODS)
    ds2 = type(ds)(F=ds.F, y=y2, scores=ds.scores, dev=ds.dev, hold=ds.hold, features=ds.features)
    pd.testing.assert_frame_equal(base, walk_forward_proba(ds2, [t], "logit", 0.3, SMALL))


def test_development_never_reads_the_held_out_period(small):
    daily, mk, panel, ds = small
    dev = develop(mk, panel, ds, SMALL)
    last_dev_return_month = mk.months[ds.dev[-1] + 1]
    pert = _perturb_after(daily, last_dev_return_month + pd.Timedelta(hours=23))
    mk2 = make_market(pert)
    panel2 = build_method_panel(mk2, SMALL)
    ds2 = build_dataset(mk2, panel2, SMALL)
    assert ds2.dev == ds.dev and ds2.hold == ds.hold
    dev2 = develop(mk2, panel2, ds2, SMALL)
    pd.testing.assert_frame_equal(dev.table, dev2.table)
    assert dev.choices == dev2.choices and dev.best_fixed == dev2.best_fixed


def test_split_is_in_time_order_with_no_overlap(small):
    _, _, _, ds = small
    assert len(ds.hold) == SMALL.holdout_months
    assert max(ds.dev) < min(ds.hold)
    assert ds.dev == sorted(ds.dev) and ds.hold == sorted(ds.hold)


# --- from probabilities to books ----------------------------------------------------------------------------
def test_hard_soft_and_temperature(small):
    _, _, panel, ds = small
    t = ds.dev[0]
    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    probs = pd.DataFrame([p], index=[t], columns=list(METHODS))
    W = np.vstack([panel.weights[t][k] for k in METHODS])
    assert np.allclose(selector_weights(panel, probs, "hard")[t], W[2])
    assert np.allclose(selector_weights(panel, probs, "soft", 1.0)[t], p @ W)
    cold = selector_weights(panel, probs, "soft", 0.05)[t]
    hot = selector_weights(panel, probs, "soft", 50.0)[t]
    assert np.abs(cold - W[2]).max() < 1e-3
    assert np.abs(hot - W.mean(axis=0)).max() < 0.05 * np.abs(W).max()


def test_unseen_classes_get_zero_probability(small):
    _, _, _, ds = small
    probs = walk_forward_proba(ds, ds.dev[:3], "forest", 10, SMALL)
    assert np.allclose(probs.sum(axis=1), 1.0)
    seen = set(ds.y.loc[train_rows(ds.y.index, ds.dev[0], SMALL)].unique())
    for j, k in enumerate(METHODS):
        if j not in seen:
            assert probs.iloc[0][k] == 0.0


def test_oracle_picks_the_realised_best_method(small):
    _, _, panel, ds = small
    w = oracle_weights(panel, ds, ds.dev[:5])
    for t, v in w.items():
        assert np.allclose(v, panel.weights[t][METHODS[int(ds.y.loc[t])]])


def test_book_stats_cost_drag_matches_turnover(small):
    _, mk, panel, ds = small
    w = {t: panel.weights[t]["min_variance"] for t in ds.dev}
    bk = run_book(mk, w, cost_bps=20.0)
    st = book_stats(bk)
    assert np.isclose(st["cost_drag_bp"], st["turnover_yr"] * 20.0, rtol=1e-6)
    assert st["gross_sharpe"] >= st["net_sharpe"]


# --- multi-seed summary ------------------------------------------------------------------------------------
def _record(seed, diffs, p, rr=5.0):
    from volbudget.selector.evaluation import SELECTORS
    ns = {"erc": 4.0, "regime_rule": rr, "long_short": 3.0, "long_only_cvx": 3.5}
    vs = {}
    for k, d in zip(SELECTORS, diffs):
        ns[k] = 4.0 + d
        vs[k] = {"diff": d, "lo": d - 1, "hi": d + 1, "p_le_0": p, "p_le_0_b63": min(1.0, 2 * p),
                 "lo_b63": d - 2, "hi_b63": d + 2}
    return {"seed": seed, "best_fixed_dev": "erc", "best_fixed_hindsight": "erc", "net_sharpe": ns,
            "turnover_yr": {}, "vs_dev_best": vs}


def test_summarise_seeds_counts_wins_and_bonferroni_significance():
    from volbudget.selector.evaluation import summarise_seeds
    recs = [_record(1, [0.5, 1.5, -0.2, 0.1], 0.001), _record(2, [-0.5, 0.2, 0.3, 0.4], 0.02),
            _record(3, [1.0, 0.9, -0.1, 2.0], 0.3)]
    sm = summarise_seeds(recs)
    lh = sm["selectors"]["logit_hard"]
    assert lh["median_diff"] == 0.5 and lh["min_diff"] == -0.5 and lh["max_diff"] == 1.0
    assert np.isclose(lh["share_beats"], 2 / 3, atol=1e-4)
    assert np.isclose(lh["share_significant"], 1 / 3, atol=1e-4)     # only p = 0.001 < 0.0125
    assert np.isclose(lh["share_significant_b63"], 1 / 3, atol=1e-4)   # 0.002 < 0.0125
    assert np.isclose(lh["share_significant_12"], 1 / 3, atol=1e-4)    # 0.001 < 0.05 / 12
    assert np.isclose(lh["share_significant_12_b63"], 1 / 3, atol=1e-4)  # 0.002 < 0.05 / 12
    assert np.isclose(sm["selectors"]["forest_soft"]["share_beats_regime_rule"], 1 / 3, atol=1e-4)  # 6.0 > 5.0 once
    assert sm["share_long_short_beats_long_only"] == 0.0


def test_code_hash_is_stable_and_threads_are_pinned():
    import os
    from volbudget.selector.repro import code_hash, pin_threads
    assert code_hash() == code_hash() and len(code_hash()) == 16
    env = pin_threads({"OMP_NUM_THREADS": "4"})
    assert env["OPENBLAS_NUM_THREADS"] == "1" and env["MKL_NUM_THREADS"] == "1" and env["OMP_NUM_THREADS"] == "4"
    assert os.environ.get("OPENBLAS_NUM_THREADS")              # set by conftest before numpy is imported


def test_committed_multiseed_summary_matches_its_records():
    import json
    from volbudget.selector.evaluation import summarise_seeds
    path = Path(__file__).resolve().parents[1] / "results" / "selector" / "multiseed.json"
    if not path.exists():
        pytest.skip("multi-seed file not generated")
    data = json.loads(path.read_text())
    assert 7 not in data["seeds"]                          # the seed used to design the planted factor
    for mode, block in data["modes"].items():
        assert [r["seed"] for r in block["records"]] == data["seeds"], mode
        assert summarise_seeds(block["records"]) == block["summary"], mode
