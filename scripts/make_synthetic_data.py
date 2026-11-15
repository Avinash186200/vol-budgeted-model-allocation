"""Write the synthetic panel to CSV, in the formats run_backtest.py reads.

    python scripts/make_synthetic_data.py              # data/returns.csv, data/macro.csv, data/market.csv
    python scripts/make_synthetic_data.py --signal 0   # same files, no planted signal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volbudget import make_synthetic  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=int, default=24)
    ap.add_argument("--months", type=int, default=156)
    ap.add_argument("--signal", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    a = ap.parse_args(argv)

    p = make_synthetic(a.models, a.months, a.signal, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p.daily.to_csv(out / "returns.csv")
    p.macro.to_csv(out / "macro.csv")
    p.market.rename_axis("date").to_csv(out / "market.csv")
    print(f"wrote {p.daily.shape[0]} days x {p.daily.shape[1]} models, {len(p.macro)} months of macro "
          f"and market data to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
