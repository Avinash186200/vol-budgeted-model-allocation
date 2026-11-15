"""
features.py — build monthly macro + per-strategy features.

Macro sources  : feats.csv (139 validated features used in VIXY/SPY models)
                 + Finnhub ETFs (SPY, TLT, HYG, GLD, UUP, LQD, SHV + 8 sectors)
Per-strategy   : computed from the monthly returns panel

Key features (validated by colleague's MI analysis):
  - Rate change + level      (top predictor)
  - SPY realized vol 21d/63d (top 2 predictor — was missing in v1)
  - Sector dispersion        (top 5 predictor — was missing in v1)
  - SPY return, drawdown, trend
  - Credit (HYG/LQD), gold, oil, dollar
All features lagged 1 month before use — no look-ahead.
"""
import warnings; warnings.filterwarnings("ignore")
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import requests, time, urllib3
urllib3.disable_warnings()

import pandas as pd
import numpy as np
from pathlib import Path
from config import MACRO_PARQUET, FINNHUB_KEY, FEATS_CSV

# ── Finnhub tickers ────────────────────────────────────────────────────────
CORE_TICKERS   = ["SPY", "TLT", "HYG", "GLD", "UUP", "LQD", "SHV"]
SECTOR_TICKERS = ["XLK", "XLF", "XLE", "XLI", "XLY", "XLP", "XLV", "XLU"]
ALL_TICKERS    = CORE_TICKERS + SECTOR_TICKERS

START_ISO = "1996-01-01"


def _finnhub_daily(sym: str) -> pd.Series | None:
    end_ts   = int(time.time())
    start_ts = int(pd.Timestamp(START_ISO).timestamp())
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={"symbol": sym, "resolution": "D",
                    "from": start_ts, "to": end_ts, "token": FINNHUB_KEY},
            timeout=30, verify=False
        ).json()
    except Exception as e:
        print(f"  {sym}: request error ({e})")
        return None
    if r.get("s") != "ok":
        print(f"  {sym}: no data")
        return None
    idx = pd.to_datetime(r["t"], unit="s").normalize()
    return pd.Series(r["c"], index=idx, name=sym)


def _load_feats_csv() -> pd.DataFrame:
    """
    Load the pre-built validated feature set (feats.csv) used in VIXY/SPY models.
    Daily data → resampled to monthly end-of-month values.
    Columns renamed feats_F1 … feats_F139 to avoid clashes with Finnhub features.
    """
    if not FEATS_CSV.exists():
        print(f"  feats.csv not found at {FEATS_CSV} — skipping")
        return pd.DataFrame()
    df = pd.read_csv(FEATS_CSV, index_col=0, parse_dates=True)
    monthly = df.resample("ME").last()
    monthly.columns = [f"feats_{c}" for c in monthly.columns]
    print(f"  feats.csv: {monthly.shape[0]} months x {monthly.shape[1]} features "
          f"({monthly.index[0].date()} to {monthly.index[-1].date()})")
    return monthly


def _load_vixy_parquets() -> pd.DataFrame:
    """
    Load the pre-built VIXY-model feature parquets: cross-asset, dispersion,
    VIX complex, yield curve, and economic calendar.
    All are daily → resampled to monthly end-of-month.
    The model is left to decide which features are useful via L1/LightGBM importance.
    """
    VIXY_DIR = Path("<local-path>")
    sources = {
        "xasset":  VIXY_DIR / "cross_asset_features.parquet",
        "disp":    VIXY_DIR / "stock_dispersion.parquet",
        "extras":  VIXY_DIR / "extras_data.parquet",
        "vix":     VIXY_DIR / "yahoo_extras.parquet",
        "xavol":   VIXY_DIR / "cross_asset_vol.parquet",
        "econ":    VIXY_DIR / "econ_features.parquet",
    }

    pieces = []
    for prefix, path in sources.items():
        if not path.exists():
            print(f"  {prefix}: not found — skipping")
            continue
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df.select_dtypes(include="number")
            monthly = df.resample("ME").last()
            monthly.columns = [f"{prefix}_{c}" for c in monthly.columns]
            pieces.append(monthly)
            print(f"  {prefix}: {monthly.shape[1]} features "
                  f"({monthly.index[0].date()} to {monthly.index[-1].date()})")
        except Exception as e:
            print(f"  {prefix}: error loading ({e}) — skipping")

    if not pieces:
        return pd.DataFrame()
    combined = pieces[0]
    for p in pieces[1:]:
        combined = combined.join(p, how="outer")
    return combined


def _download_prices(force: bool = False) -> pd.DataFrame:
    if MACRO_PARQUET.exists() and not force:
        df = pd.read_parquet(MACRO_PARQUET)
        if len(df) > 0:
            print("  Macro features loaded from cache.")
            return df

    print("  Downloading macro data from Finnhub ...")
    daily_prices = {}

    for sym in ALL_TICKERS:
        s = _finnhub_daily(sym)
        if s is not None:
            daily_prices[sym] = s
            print(f"  {sym}: {len(s)} rows  {s.index[0].date()} to {s.index[-1].date()}")
        time.sleep(1.2)

    if not daily_prices:
        raise RuntimeError("No macro data downloaded.")

    # ── common daily index ────────────────────────────────────────────────────
    all_idx = pd.DatetimeIndex(
        sorted(set().union(*[p.index for p in daily_prices.values()]))
    )
    px_daily = pd.DataFrame(
        {k: v.reindex(all_idx).ffill() for k, v in daily_prices.items()}
    )

    # ── monthly features ──────────────────────────────────────────────────────
    monthly_px  = px_daily.resample("ME").last()
    monthly_ret = monthly_px.pct_change()

    feats = pd.DataFrame(index=monthly_px.index)
    feats.index.name = "date"

    # ── core ETF features ─────────────────────────────────────────────────────
    for sym in CORE_TICKERS:
        if sym not in monthly_px.columns:
            continue
        px  = monthly_px[sym]
        ret = monthly_ret[sym]
        feats[f"{sym}_ret1"]     = ret
        feats[f"{sym}_ret3"]     = px.pct_change(3)
        feats[f"{sym}_ret12"]    = px.pct_change(12)
        feats[f"{sym}_vol3"]     = ret.rolling(3).std()  * np.sqrt(12)
        feats[f"{sym}_vol12"]    = ret.rolling(12).std() * np.sqrt(12)
        feats[f"{sym}_mom_z"]    = (ret - ret.rolling(12).mean()) / (ret.rolling(12).std() + 1e-8)
        feats[f"{sym}_drawdown"] = (px / px.rolling(24).max() - 1)

    # ── SPY REALIZED VOL from daily data (colleague's top-2 feature) ──────────
    if "SPY" in px_daily.columns:
        spy_d   = px_daily["SPY"].pct_change()
        spy_vol_21d = spy_d.rolling(21).std() * np.sqrt(252)
        spy_vol_63d = spy_d.rolling(63).std() * np.sqrt(252)
        feats["SPY_rvol21"] = spy_vol_21d.resample("ME").last()
        feats["SPY_rvol63"] = spy_vol_63d.resample("ME").last()
        # SPY vol change (regime shift signal)
        feats["SPY_rvol_chg1"] = feats["SPY_rvol21"].diff(1)
        feats["SPY_rvol_chg3"] = feats["SPY_rvol21"].diff(3)

    # ── SECTOR DISPERSION (colleague's top-5 feature) ────────────────────────
    avail_sectors = [s for s in SECTOR_TICKERS if s in monthly_px.columns]
    if len(avail_sectors) >= 4:
        sec_rets = monthly_ret[avail_sectors]
        # cross-sectional std of sector monthly returns (breadth of macro environment)
        feats["sec_disp"]      = sec_rets.std(axis=1)
        # best minus worst sector (momentum breadth)
        feats["sec_spread"]    = sec_rets.max(axis=1) - sec_rets.min(axis=1)
        # defensive vs cyclical spread: (XLP+XLV) - (XLE+XLI+XLY)
        def_cols  = [c for c in ["XLP", "XLV"] if c in sec_rets.columns]
        cyc_cols  = [c for c in ["XLE", "XLI", "XLY"] if c in sec_rets.columns]
        if def_cols and cyc_cols:
            feats["def_vs_cyc"] = (sec_rets[def_cols].mean(axis=1) -
                                   sec_rets[cyc_cols].mean(axis=1))
        # z-score of dispersion (high = stressed/dislocated regime)
        feats["sec_disp_z"]    = ((feats["sec_disp"] - feats["sec_disp"].rolling(24).mean()) /
                                  (feats["sec_disp"].rolling(24).std() + 1e-8))

    # ── derived cross-asset regime features ──────────────────────────────────
    if "HYG_ret3" in feats and "LQD_ret3" in feats:
        feats["cred_spread"]     = feats["HYG_ret3"] - feats["LQD_ret3"]
        feats["cred_spread_z"]   = ((feats["cred_spread"] - feats["cred_spread"].rolling(24).mean()) /
                                    (feats["cred_spread"].rolling(24).std() + 1e-8))
    if "TLT_ret3" in feats:
        feats["rates_vs_eq"]     = -feats["TLT_ret3"]
    if "XLF_ret3" in feats:
        feats["bank_stress"]     = -feats["XLF_ret3"]

    # ── merge validated feats.csv (F1-F139 proprietary feed) ─────────────────
    feats_ext = _load_feats_csv()
    if not feats_ext.empty:
        feats = feats.join(feats_ext, how="left")

    # ── merge VIXY-model pre-built feature parquets ───────────────────────────
    # cross-asset, dispersion, VIX complex, yield curve, econ calendar.
    # NOTE: these parquets start from 2016-2018 — only 2-4 years of training
    # data before the 2020 OOS window.  Adding them hurt OOS Sharpe (<real figure removed>)
    # because NaN-heavy columns caused silent model failures and spurious splits.
    # Disabled by default; re-enable once Client extends OOS past 2024.
    # vixy_feats = _load_vixy_parquets()
    # if not vixy_feats.empty:
    #     feats = feats.join(vixy_feats, how="left")

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats.to_parquet(MACRO_PARQUET)
    print(f"  Saved: {feats.shape[0]} months x {feats.shape[1]} macro features")
    return feats


def get_spy_daily(force: bool = False) -> pd.Series:
    """Return daily SPY prices (used by factor covariance model)."""
    cache = MACRO_PARQUET.parent / "spy_daily.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)["SPY"]
    s = _finnhub_daily("SPY")
    if s is None:
        raise RuntimeError("Could not download SPY daily data.")
    pd.DataFrame({"SPY": s}).to_parquet(cache)
    return s


def build_per_strategy_features(monthly_ret: pd.DataFrame) -> dict:
    """
    For each model, build technical features using only historical data.
    All features use shift(1) — available at month-end T, predict T+1.
    """
    features = {}
    for col in monthly_ret.columns:
        s = monthly_ret[col]
        f = pd.DataFrame(index=s.index)

        f["ret_lag1"]      = s.shift(1)
        f["ret_lag2"]      = s.shift(2)
        f["ret_lag3"]      = s.shift(3)
        f["ret_3m"]        = s.shift(1).rolling(3).mean()
        f["ret_6m"]        = s.shift(1).rolling(6).mean()
        f["ret_12m"]       = s.shift(1).rolling(12).mean()

        f["vol_3m"]        = s.shift(1).rolling(3).std()  * np.sqrt(12)
        f["vol_6m"]        = s.shift(1).rolling(6).std()  * np.sqrt(12)
        f["vol_12m"]       = s.shift(1).rolling(12).std() * np.sqrt(12)

        roll_mean = s.shift(1).rolling(12).mean()
        roll_std  = s.shift(1).rolling(12).std()
        f["mom_z12"]       = (s.shift(1) - roll_mean) / (roll_std + 1e-8)

        cum = (1 + s.shift(1).fillna(0)).cumprod()
        f["drawdown"]      = cum / cum.rolling(12).max() - 1

        f["win_rate_12m"]  = (s.shift(1) > 0).rolling(12).mean()
        f["sharpe_12m"]    = (s.shift(1).rolling(12).mean() /
                              (s.shift(1).rolling(12).std() + 1e-8)) * np.sqrt(12)

        features[col] = f
    return features


def build_cs_features(all_strat_feats: dict) -> dict:
    """
    Cross-sectional rank features: at each month, rank each strategy's
    key indicators relative to the full 62-strategy universe.
    Rank is expressed as a percentile (0=worst, 1=best).

    These capture relative momentum / quality — known strong signals for
    systematic futures portfolios (strategy is in top X% of universe).
    """
    rank_vars = ["ret_3m", "ret_6m", "ret_12m", "vol_12m", "sharpe_12m", "drawdown"]
    cols = list(all_strat_feats.keys())

    # Build cross-sectional panels: {feature: DataFrame(index=dates, columns=strategies)}
    panels = {}
    for var in rank_vars:
        panel = pd.DataFrame(
            {c: all_strat_feats[c][var] for c in cols if var in all_strat_feats[c].columns}
        )
        panels[var] = panel

    cs_feats = {}
    for col in cols:
        df = pd.DataFrame(index=next(iter(panels.values())).index)
        for var, panel in panels.items():
            if col in panel.columns:
                # pct=True gives 0..1 rank; for vol we invert (low vol = higher rank)
                ranked = panel.rank(axis=1, pct=True, na_option="keep")
                if var == "vol_12m":
                    df[f"cs_rank_{var}"] = 1.0 - ranked[col]   # low vol → high rank
                else:
                    df[f"cs_rank_{var}"] = ranked[col]
        cs_feats[col] = df

    return cs_feats


def build_feature_matrix(
    monthly_ret: pd.DataFrame,
    macro_feats: pd.DataFrame,
    model_col: str,
    all_strat_feats: dict | None = None,
    cs_feats: dict | None = None,
) -> pd.DataFrame:
    """
    Combine macro + per-strategy + cross-sectional features for one model.
    Target = 1 if next month's return > 0 else 0.

    Pass pre-computed all_strat_feats / cs_feats to avoid recomputing
    them for every strategy in the walk-forward loop.
    """
    if all_strat_feats is None:
        all_strat_feats = build_per_strategy_features(monthly_ret)
    if cs_feats is None:
        cs_feats = build_cs_features(all_strat_feats)

    strat_feats   = all_strat_feats[model_col]
    cs            = cs_feats.get(model_col, pd.DataFrame(index=monthly_ret.index))
    macro_aligned = macro_feats.reindex(monthly_ret.index, method="ffill")

    combined      = pd.concat([strat_feats, cs, macro_aligned], axis=1)
    combined["target"] = (monthly_ret[model_col].shift(-1) > 0).astype(float)
    combined      = combined[monthly_ret[model_col].notna()]
    return combined


if __name__ == "__main__":
    macro = _download_prices(force=False)
    print(f"\nFeature columns ({len(macro.columns)}):")
    for c in sorted(macro.columns):
        print(f"  {c}")
