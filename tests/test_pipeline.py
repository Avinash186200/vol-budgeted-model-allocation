import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volbudget import Config, daily_to_monthly, default_budget, make_synthetic, run_backtest  # noqa: E402
from volbudget.features import align_macro, build_panels  # noqa: E402
from volbudget.metrics import rank_ic  # noqa: E402
from volbudget.optimizer import factor_covariance, optimize  # noqa: E402

SMALL = Config(oos_months=10, min_train=24, cov_window=24)


def _panel(signal=0.0, seed=1, n=6, months=60):
    p = make_synthetic(n_models=n, n_months=months, signal_strength=signal, seed=seed)
    return daily_to_monthly(p.daily), p.macro, p.market


def test_pre_inception_is_nan_and_interior_zero_kept():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    s = np.zeros(len(idx))
    s[30:] = 0.001
    s[45] = 0.0
    m = daily_to_monthly(pd.DataFrame({"model 1": s}, index=idx))
    assert np.isnan(m.iloc[0, 0])
    assert m.iloc[-1, 0] > 0


def test_features_at_t_ignore_future_returns():
    monthly, macro, _ = _panel()
    t = monthly.index[40]
    base = build_panels(monthly)
    bumped = monthly.copy()
    bumped.loc[bumped.index > t] += 0.5
    after = build_panels(bumped)
    for c in monthly.columns:
        pd.testing.assert_frame_equal(base[c].loc[:t], after[c].loc[:t])


def test_macro_is_lagged():
    _, macro, _ = _panel()
    al = align_macro(macro, macro.index, macro_lag=1)
    assert np.isnan(al.iloc[0, 0])
    assert al.iloc[5]["mac_macro_a"] == macro.iloc[4]["macro_a"]


def test_prediction_unchanged_when_future_is_perturbed():
    monthly, macro, market = _panel()
    cfg = Config(oos_months=3, min_train=24, cov_window=24)
    _, w1, p1 = run_backtest(monthly, macro, market, cfg)
    first = p1["date"].min()
    bumped = monthly.copy()
    bumped.loc[bumped.index > first] *= -3  # every month after the rebalance date
    _, w2, p2 = run_backtest(bumped, macro, market, cfg)
    a = p1[p1.date == first].set_index("model")["prob"]
    b = p2[p2.date == first].set_index("model")["prob"]
    pd.testing.assert_series_equal(a, b)


def test_optimizer_respects_budget():
    monthly, _, market = _panel(n=8)
    win = monthly.iloc[-36:]
    cov = factor_covariance(win, market)
    cols = list(monthly.columns)
    b = default_budget(cols, n_groups=2)
    mu = np.linspace(0.2, -0.1, len(cols))
    w = optimize(mu, cov, np.ones(len(cols), bool), cols, b)
    sig = np.sqrt(np.diag(cov))
    assert (w >= -1e-9).all() and w.sum() <= 1 + 1e-6
    assert (w * sig <= b.model_cap + 1e-6).all()
    assert np.sqrt(w @ cov @ w) <= b.port_cap + 1e-6
    for members, cap in b.groups.values():
        g = [cols.index(c) for c in members]
        assert np.sqrt(w[g] @ cov[np.ix_(g, g)] @ w[g]) <= cap + 1e-6


def test_noise_has_no_ic_and_signal_is_recovered():
    noise = rank_ic(run_backtest(*_panel(signal=0.0, n=8, months=80, seed=3), SMALL)[2])
    sig = rank_ic(run_backtest(*_panel(signal=0.03, n=8, months=80, seed=3), SMALL)[2])
    assert abs(noise) < 0.2
    assert sig > noise + 0.1


def test_make_synthetic_data_script_roundtrip(tmp_path):
    """CSVs written by the script load back to the same monthly panel, macro and market series."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import make_synthetic_data

    from volbudget.data import load_daily_csv

    assert make_synthetic_data.main(["--models", "4", "--months", "24", "--out", str(tmp_path)]) == 0
    p = make_synthetic(4, 24, 0.01, 7)
    pd.testing.assert_frame_equal(daily_to_monthly(load_daily_csv(tmp_path / "returns.csv")),
                                  daily_to_monthly(p.daily), check_freq=False)
    macro = pd.read_csv(tmp_path / "macro.csv", parse_dates=["date"]).set_index("date")
    market = pd.read_csv(tmp_path / "market.csv", parse_dates=["date"]).set_index("date")["market"]
    assert np.allclose(macro.to_numpy(), p.macro.to_numpy())
    assert np.allclose(market.to_numpy(), p.market.to_numpy())
