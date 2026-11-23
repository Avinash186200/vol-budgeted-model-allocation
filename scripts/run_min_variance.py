"""Run the minimum-variance allocator on synthetic data, or on your own daily CSV.

Examples:
    python scripts/run_min_variance.py
    python scripts/run_min_variance.py --windows 3 --oos-days 252
    python scripts/run_min_variance.py --sweep
    python scripts/run_min_variance.py --daily data/returns.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volbudget import make_synthetic  # noqa: E402
from volbudget.data import load_daily_csv, make_independent_panel  # noqa: E402
from volbudget.min_variance import MVConfig, run_min_variance  # noqa: E402
from volbudget.min_variance.backtest import BASE, blend_name  # noqa: E402
from volbudget.min_variance.evaluation import (  # noqa: E402
    block_bootstrap_sharpe_diff,
    select_on_search,
    sign_test_p,
    summary,
    window_sharpes,
)
from volbudget.min_variance.sweep import construction_sweep, judge_specs, overlay_grid  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily", help="CSV: date + one column of daily returns per model")
    ap.add_argument("--models", type=int, default=24, help="synthetic: number of models")
    ap.add_argument("--months", type=int, default=156, help="synthetic: number of months")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--windows", type=int, default=5, help="number of back-to-back test windows")
    ap.add_argument("--oos-days", type=int, default=504, help="trading days per test window")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--independent", action="store_true",
                    help="synthetic control: independent models with equal Sharpe (nothing for min-var to find)")
    ap.add_argument("--sweep", action="store_true", help="also run the construction sweep and overlays (slow)")
    ap.add_argument("--out", default="results/min_variance")
    a = ap.parse_args(argv)

    market = None
    if a.daily:
        daily = load_daily_csv(a.daily)
    elif a.independent:
        daily = make_independent_panel(a.models, n_days=int(a.months * 21.75), seed=a.seed)
    else:
        p = make_synthetic(a.models, a.months, signal_strength=0.0, seed=a.seed)
        daily, market = p.daily, p.market_daily
    cfg = MVConfig(n_windows=a.windows, oos_days=a.oos_days, cost_bps=a.cost_bps,
                   search_windows=max(1, min(2, a.windows - 1)))
    res = run_min_variance(daily, cfg)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    tab = summary(res, BASE)
    ws = window_sharpes(res, "net")
    tab.to_csv(out / "summary.csv")
    ws.to_csv(out / "window_net_sharpe.csv")
    print(f"{daily.shape[1]} models, {len(res)} test windows of {cfg.oos_days} days, "
          f"net of {cfg.cost_bps} bps, T+{cfg.exec_lag}, refresh every {cfg.refresh} days")
    print("\nSummary (Sharpe = mean over windows; caps: 1.00 = at the cap, above 1 = breach)")
    print(tab)
    print("\nNet Sharpe per window (oldest first)")
    print(ws[BASE])

    wins = int((ws["blend"] > ws["vol_parity"]).sum())
    a_ = np.concatenate([w.net["blend"].to_numpy() for w in res])
    b_ = np.concatenate([w.net["vol_parity"].to_numpy() for w in res])
    bs = block_bootstrap_sharpe_diff(a_, b_, n_boot=a.boot)
    print(f"\nBlend ({cfg.alpha:.2f} min-var) vs vol parity, net: wins {wins}/{len(res)} windows "
          f"(sign test p = {sign_test_p(wins, len(res)):.3f}); pooled Sharpe diff {bs['diff']:+.3f}, "
          f"bootstrap P(diff<=0) = {bs['p_le_0']:.3f}, 95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}]")

    cands = [blend_name(x) for x in cfg.alpha_grid]
    sel = select_on_search(ws, cands, cfg.search_windows)
    vp_hold = ws["vol_parity"].iloc[cfg.search_windows:].mean()
    print(f"\nAlpha chosen on the first {cfg.search_windows} windows only: {sel['choice']} "
          f"(search {sel['search']:.3f}); hold-out {sel['holdout']:.3f} vs vol parity {vp_hold:.3f} "
          f"(delta {sel['holdout'] - vp_hold:+.3f})")

    wins_c = int((ws["blend_capped"] > ws["vol_parity_capped"]).sum())
    print(f"Capped blend vs capped vol parity, net: wins {wins_c}/{len(res)} windows; mean Sharpe "
          f"{ws['blend_capped'].mean():.3f} vs {ws['vol_parity_capped'].mean():.3f}")

    if a.sweep:
        streams = construction_sweep(daily, cfg)
        judged = judge_specs(streams, cfg.search_windows, n_boot=min(a.boot, 1000))
        judged.to_csv(out / "sweep.csv", index=False)
        fin = judged[judged["verdict"] != "not a finalist"]
        print(f"\nConstruction sweep: {len(judged)} specs; {len(fin)} beat vol parity on search; "
              f"{int((fin['holdout'] > 0).sum())} of those also on hold-out; "
              f"{int((judged['verdict'] == 'robust').sum())} clear the Bonferroni threshold")
        print(judged.head(10).to_string(index=False))
        ov = overlay_grid([w.net["vol_parity"] for w in res], market, cfg.search_windows)
        ov.to_csv(out / "overlays.csv", index=False)
        best = ov.iloc[0]
        print(f"\nOverlays on vol parity (Sharpe minus vol parity): best on search is {best['spec']} "
              f"({best['search']:+.3f}); its hold-out delta is {best['holdout']:+.3f}")
        print(ov.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
