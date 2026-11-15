"""
run_cpo.py — main orchestrator for the Client CPO pipeline.

Usage:
    python run_cpo.py                   # run full pipeline
    python run_cpo.py --force-features  # re-download macro features first

Pipeline:
    1. Load monthly returns          (data_prep.py)
    2. Build / load macro features   (features.py)
    3. Walk-forward backtest         (backtest.py)
    4. Performance report + charts   (report.py)
"""
import warnings; warnings.filterwarnings("ignore")
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import load_monthly_returns
from features  import _download_prices
from backtest  import run_backtest
from report    import compute_metrics, print_metrics, generate_charts, save_results
from config    import OOS_MONTHS, RESULTS_DIR


def main(force_features: bool = False) -> None:

    print("=" * 60)
    print("  Client CPO — Conditional Portfolio Optimisation")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data ...")
    monthly_ret = load_monthly_returns()
    macro_feats = _download_prices(force=force_features)
    model_cols  = list(monthly_ret.columns)

    print(f"  Strategies : {len(model_cols)}")
    print(f"  History    : {monthly_ret.index[0].date()} to {monthly_ret.index[-1].date()}")
    print(f"  OOS window : last {OOS_MONTHS} months")

    # ── 2. Walk-forward backtest ──────────────────────────────────────────────
    print("\n[2/4] Running walk-forward backtest ...")
    print("       (ensemble x5 per model x OOS months — takes ~20 min)")
    res = run_backtest(monthly_ret, macro_feats, verbose=True)

    # ── 3. Performance metrics ────────────────────────────────────────────────
    print("\n[3/4] Performance report")
    print("-" * 60)
    metrics = [
        compute_metrics(res["ret_cpo_shrunk"], "CPO (shrunk)"),
        compute_metrics(res["ret_cpo"],        "CPO (raw)"),
        compute_metrics(res["ret_ew"],         "Equal Weight"),
        compute_metrics(res["ret_vp"],         "Vol Parity"),
    ]
    print_metrics(metrics)

    print(f"\n  Vol discipline — CPO shrunk pred vol <= 6%: "
          f"{(res['pred_pv_shrunk'] <= 0.06).mean()*100:.0f}% of months")
    print(f"  Avg predicted port vol (CPO shrunk): "
          f"{res['pred_pv_shrunk'].mean()*100:.2f}%")

    # ── 4. Charts and outputs ─────────────────────────────────────────────────
    print("\n[4/4] Saving charts and results ...")
    generate_charts(res, metrics)
    save_results(res, metrics, model_cols)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Client CPO walk-forward backtest")
    ap.add_argument("--force-features", action="store_true",
                    help="Re-download macro features from Finnhub")
    args = ap.parse_args()
    main(force_features=args.force_features)
