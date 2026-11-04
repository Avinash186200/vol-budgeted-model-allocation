"""Run the walk-forward on synthetic data, or on your own CSVs.

Examples:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --signal 0.0
    python scripts/run_backtest.py --daily returns.csv --macro macro.csv --market market.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volbudget import Config, daily_to_monthly, make_synthetic, run_backtest  # noqa: E402
from volbudget.data import load_daily_csv  # noqa: E402
from volbudget.metrics import rank_ic, summary_table  # noqa: E402

NAMES = ["opt_raw", "opt_blend", "equal_weight", "vol_parity"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily", help="CSV: date + one column of daily returns per model")
    ap.add_argument("--macro", help="CSV: date + monthly macro features (dated by observation month)")
    ap.add_argument("--market", help="CSV: date,market monthly market returns")
    ap.add_argument("--models", type=int, default=24, help="synthetic: number of models")
    ap.add_argument("--months", type=int, default=156, help="synthetic: number of months")
    ap.add_argument("--signal", type=float, default=0.01, help="synthetic: planted macro signal")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--oos", type=int, default=36, help="out-of-sample months")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.daily:
        monthly = daily_to_monthly(load_daily_csv(a.daily))
        macro = pd.read_csv(a.macro, parse_dates=["date"]).set_index("date") if a.macro else None
        market = (pd.read_csv(a.market, parse_dates=["date"]).set_index("date")["market"]
                  if a.market else None)
    else:
        p = make_synthetic(a.models, a.months, a.signal, a.seed)
        monthly, macro, market = daily_to_monthly(p.daily), p.macro, p.market

    cfg = Config(oos_months=a.oos, cost_bps=a.cost_bps)
    ret, w, pred = run_backtest(monthly, macro, market, cfg)
    tab = summary_table(ret, NAMES)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ret.to_csv(out / "monthly_returns.csv")
    w.to_csv(out / "weights_opt_blend.csv")
    tab.to_csv(out / "summary.csv")

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(f"{monthly.shape[1]} models, {len(ret)} OOS months, net of {a.cost_bps} bps costs")
    print(tab)
    print(f"rank IC of P(up) vs next-month return: {rank_ic(pred):+.3f}")
    print(f"months over the portfolio vol cap (opt_blend): {(ret['pv_opt_blend'] > 0.06 + 1e-6).sum()}")


if __name__ == "__main__":
    main()
