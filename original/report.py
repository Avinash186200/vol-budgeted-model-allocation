"""
report.py — performance metrics, charts, and weight attribution.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import PORT_VOL_CAP, RESULTS_DIR


def compute_metrics(r: pd.Series, label: str) -> dict:
    ann_ret = r.mean() * 12
    ann_vol = r.std()  * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + r).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    hit     = (r > 0).mean()
    return dict(label=label, ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd, hit_rate=hit)


def print_metrics(metrics: list[dict]) -> None:
    header = f"{'Strategy':<22} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>8} {'HitRate':>8}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(f"  {m['label']:<20} {m['ann_ret']*100:>7.2f}%"
              f" {m['ann_vol']*100:>7.2f}%"
              f" {m['sharpe']:>8.2f}"
              f" {m['max_dd']*100:>7.2f}%"
              f" {m['hit_rate']*100:>7.1f}%")


def generate_charts(res: pd.DataFrame, metrics: list[dict]) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Cumulative returns ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for col, label, lw, ls in [
        ("ret_cpo_shrunk", "CPO (shrunk)",   2.5, "-"),
        ("ret_cpo",        "CPO (raw)",       1.5, "--"),
        ("ret_ew",         "Equal Weight",    1.5, "-."),
        ("ret_vp",         "Vol Parity",      1.0, ":"),
    ]:
        cum = (1 + res[col]).cumprod()
        ax1.plot(cum, label=label, linewidth=lw, linestyle=ls)
    ax1.set_title("Cumulative Return — OOS Walk-Forward", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # ── 2. Monthly returns bar ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ["green" if r > 0 else "red" for r in res["ret_cpo_shrunk"]]
    ax2.bar(range(len(res)), res["ret_cpo_shrunk"] * 100, color=colors, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("CPO Monthly Returns (%)", fontsize=10)
    ax2.set_ylabel("%")
    ax2.grid(alpha=0.3)

    # ── 3. Predicted portfolio vol ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(res["pred_pv_shrunk"] * 100, label="CPO (shrunk)", linewidth=2)
    ax3.plot(res["pred_pv_cpo"]    * 100, label="CPO (raw)", linewidth=1, linestyle="--")
    ax3.axhline(PORT_VOL_CAP * 100, color="red", linestyle="--", linewidth=1.5, label="6% cap")
    ax3.set_title("Predicted Portfolio Volatility", fontsize=10)
    ax3.set_ylabel("Ann. Vol %")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ── 4. Average ensemble probability ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(res["avg_prob"], linewidth=2, color="purple")
    ax4.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax4.fill_between(range(len(res)),
                     res["avg_prob"], 0.5,
                     where=res["avg_prob"] > 0.5, alpha=0.2, color="green")
    ax4.fill_between(range(len(res)),
                     res["avg_prob"], 0.5,
                     where=res["avg_prob"] < 0.5, alpha=0.2, color="red")
    ax4.set_title("Avg Ensemble Probability (active models)", fontsize=10)
    ax4.set_ylabel("P(positive return)")
    ax4.set_ylim(0.3, 0.7)
    ax4.grid(alpha=0.3)

    # ── 5. Sharpe comparison bar ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    labels  = [m["label"] for m in metrics]
    sharpes = [m["sharpe"] for m in metrics]
    colors5 = ["steelblue" if "CPO" in l else "gray" for l in labels]
    bars    = ax5.bar(labels, sharpes, color=colors5, alpha=0.85)
    ax5.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax5.set_title("Sharpe Ratio Comparison (OOS)", fontsize=10)
    ax5.set_ylabel("Sharpe")
    ax5.tick_params(axis="x", rotation=20, labelsize=8)
    ax5.grid(alpha=0.3, axis="y")

    chart_path = RESULTS_DIR / "cpo_backtest.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {chart_path}")


def save_results(res: pd.DataFrame, metrics: list[dict], model_cols: list[str]) -> None:
    # monthly returns
    res.drop(columns=["weights_cpo"]).to_csv(RESULTS_DIR / "monthly_returns.csv")

    # summary metrics
    pd.DataFrame(metrics).to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    # monthly weights
    weights_df = pd.DataFrame(
        res["weights_cpo"].tolist(),
        index=res.index,
        columns=model_cols,
    )
    weights_df.to_csv(RESULTS_DIR / "weights_cpo.csv")

    print(f"  Results saved to: {RESULTS_DIR}")
