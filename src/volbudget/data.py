"""Data loading, daily-to-monthly compounding, and a synthetic generator.

The real input was a daily return panel of trading models with staggered
inception dates. The synthetic generator reproduces that shape: one ``date``
column plus ``model 1`` .. ``model N`` columns of daily decimal returns, with
leading zeros before each model starts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound a daily return panel into month-end returns.

    Leading zeros (before a model's first non-zero day) are treated as
    pre-inception and become NaN. Interior zeros are flat days and stay 0.
    Months with no live day are NaN.

    Args:
        daily: DataFrame indexed by date, one column per model.

    Returns:
        DataFrame indexed by month-end, same columns, monthly decimal returns.
    """
    daily = daily.sort_index().astype(float).copy()
    for c in daily.columns:
        nz = daily[c].fillna(0).ne(0).to_numpy()
        if not nz.any():
            daily[c] = np.nan
            continue
        first = int(np.argmax(nz))
        daily.iloc[:first, daily.columns.get_loc(c)] = np.nan
    monthly = (daily + 1).resample("ME").prod(min_count=1) - 1
    active = daily.notna().resample("ME").sum() > 0
    monthly = monthly.where(active)
    monthly.index.name = "date"
    return monthly


def load_daily_csv(path: str) -> pd.DataFrame:
    """Load a daily return CSV with a ``date`` column and one column per model."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df


@dataclass
class SyntheticPanel:
    """Container for a synthetic dataset."""

    daily: pd.DataFrame       # daily model returns (leading zeros pre-inception)
    macro: pd.DataFrame       # monthly macro features, dated by observation month
    market: pd.Series         # monthly market return (factor for covariance)
    market_daily: pd.Series | None = None  # daily market return (used by the min-variance overlays)
    regime: pd.Series | None = None        # monthly hidden state (0 calm, 1 stress) when the regime mode is on


def make_synthetic(
    n_models: int = 24,
    n_months: int = 156,
    signal_strength: float = 0.0,
    seed: int = 7,
    regime_strength: float = 0.0,
) -> SyntheticPanel:
    """Generate a daily return panel with a factor structure and optional signal.

    Each model has a small positive drift, loads on one market factor and three
    latent factors, and starts on a random month in the first third of the
    sample. Two monthly macro series are produced. When ``signal_strength`` > 0,
    the mean return of every model in month t+1 depends on ``macro_a`` dated
    month t-1, i.e. the value that is public at the end of month t under a
    one-month publication lag. With ``signal_strength=0`` there is no signal.

    Args:
        n_models: Number of models (columns).
        n_months: Number of months.
        signal_strength: Size of the planted macro effect (monthly return per
            unit of the standardised macro series).
        seed: RNG seed.
        regime_strength: Size of the planted regime effect (see below). 0, the
            default, switches the regime mode off and leaves every other output
            unchanged.

    Regime mode (``regime_strength > 0``): a hidden monthly state follows a
    two-state Markov chain (calm and stress; a calm spell lasts about 12 months
    and a stress spell about 8 on average). Every model loads on an extra common
    factor, most of them positively (loadings uniform on [-0.6, 1.4]). In calm
    months that factor has a positive drift and low vol; in stress months its vol
    is 2.5 times higher and its drift turns negative. Books with a net exposure to
    the factor (equal weight, vol parity, the return-chasing optimiser) earn its
    premium in calm months and lose in stress months, while minimum variance can
    hedge most of it out by mixing positive and negative loaders. So the best
    construction method depends on the state, and the state is persistent and
    visible in trailing vol and in how much variance the first principal
    component explains. The regime draws use their own RNG stream, so the rest of
    the panel is the same as with ``regime_strength=0`` and the same seed.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2008-01-31", periods=n_months, freq="ME")
    days = pd.bdate_range(months[0] - pd.offsets.MonthBegin(1), months[-1])

    macro_a = np.zeros(n_months)
    for t in range(1, n_months):
        macro_a[t] = 0.8 * macro_a[t - 1] + rng.normal(0, 0.6)
    macro_b = rng.normal(0, 1, n_months)
    macro = pd.DataFrame({"macro_a": macro_a, "macro_b": macro_b}, index=months)
    macro.index.name = "date"

    month_of_day = days.to_period("M").to_timestamp("M")
    m_pos = pd.Index(months).get_indexer(month_of_day)

    n_days = len(days)
    mkt_d = rng.normal(0.0003, 0.010, n_days)
    lat_d = rng.normal(0, 0.004, (n_days, 3))
    beta_m = rng.normal(0.02, 0.03, n_models)
    beta_l = rng.normal(0, 1, (n_models, 3))
    drift = rng.uniform(0.00005, 0.0003, n_models)
    idio = rng.uniform(0.002, 0.006, n_models)
    sens = rng.choice([-1.0, 1.0], n_models) * rng.uniform(0.5, 1.0, n_models)

    # planted signal: month t+1 mean depends on macro dated t-1
    lagged = np.full(n_months, 0.0)
    lagged[2:] = macro_a[:-2]
    sig_daily = signal_strength * lagged[m_pos] / 21.0

    R = (drift[None, :]
         + mkt_d[:, None] * beta_m[None, :]
         + lat_d @ beta_l.T
         + rng.normal(0, 1, (n_days, n_models)) * idio[None, :]
         + sig_daily[:, None] * sens[None, :])

    regime = None
    if regime_strength > 0:
        reg_d, regime = _regime_component(n_months, m_pos, n_models, regime_strength, seed)
        R = R + reg_d
        regime = pd.Series(regime, index=months, name="regime")
        regime.index.name = "date"

    start = rng.integers(0, n_months // 3, n_models)
    start[0] = 0
    for i in range(n_models):
        R[m_pos < start[i], i] = 0.0

    daily = pd.DataFrame(R, index=days, columns=[f"model {i + 1}" for i in range(n_models)])
    daily.index.name = "date"
    market = (pd.Series(mkt_d, index=days) + 1).resample("ME").prod() - 1
    market.name = "market"
    market_daily = pd.Series(mkt_d, index=days, name="market")
    market_daily.index.name = "date"
    return SyntheticPanel(daily=daily, macro=macro, market=market, market_daily=market_daily, regime=regime)


def _regime_component(n_months: int, m_pos: np.ndarray, n_models: int, strength: float,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Daily returns of the regime factor times each model's loading, and the monthly state path.

    Calm (state 0): factor drift ``+20 bp * strength`` a day, daily vol 0.4%.
    Stress (state 1): drift ``-12 bp * strength`` a day, daily vol 1.0%.
    Monthly switch probabilities: calm to stress 8%, stress to calm 12%.
    """
    rng = np.random.default_rng([seed, 2026])
    state = np.zeros(n_months, dtype=int)
    for t in range(1, n_months):
        p_switch = 0.08 if state[t - 1] == 0 else 0.12
        state[t] = 1 - state[t - 1] if rng.random() < p_switch else state[t - 1]
    s_day = state[m_pos]
    mean = np.where(s_day == 1, -0.0012, 0.0020) * strength
    vol = np.where(s_day == 1, 0.010, 0.004)
    f = mean + vol * rng.normal(0, 1, len(m_pos))
    load = rng.uniform(-0.6, 1.4, n_models)
    return f[:, None] * load[None, :], state


def make_independent_panel(n_models: int = 24, n_days: int = 3400, sharpe: float = 1.0,
                           seed: int = 11) -> pd.DataFrame:
    """A control panel: independent models with different vols but the same Sharpe.

    With no correlation structure and equal Sharpe ratios, vol parity is already
    the best long-only mix, so a risk-only optimiser should add nothing here. The
    min-variance tests and the docs use it as the "nothing to find" case.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=n_days)
    vol = rng.uniform(0.002, 0.008, n_models)
    drift = sharpe / np.sqrt(252) * vol
    R = drift[None, :] + rng.normal(0, 1, (n_days, n_models)) * vol[None, :]
    out = pd.DataFrame(R, index=days, columns=[f"model {i + 1}" for i in range(n_models)])
    out.index.name = "date"
    return out
