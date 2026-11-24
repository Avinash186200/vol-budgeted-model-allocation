"""Scoring the selector against fixed methods, the oracle and each other."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..min_variance.evaluation import block_bootstrap_sharpe_diff, sharpe
from .methods import BookResult


def book_stats(bk: BookResult) -> dict[str, float]:
    """Net and gross Sharpe, return, vol, drawdown, turnover and the cost drag of one book.

    ``turnover_yr`` is the average one-way turnover per year; ``cost_drag_bp`` is
    the annual return lost to trading costs and borrow, in basis points.
    """
    net, gross = bk.net.to_numpy(), bk.gross.to_numpy()
    cum = np.cumprod(1 + net)
    years = len(net) / 252
    return {
        "net_sharpe": sharpe(net),
        "gross_sharpe": sharpe(gross),
        "ann_ret": float(net.mean() * 252),
        "ann_vol": float(net.std() * np.sqrt(252)),
        "max_dd": float((cum / np.maximum.accumulate(cum) - 1).min()),
        "turnover_yr": float(bk.turnover.sum() / years) if years > 0 else float("nan"),
        "cost_drag_bp": float((gross.mean() - net.mean()) * 252 * 1e4),
    }


def stats_table(books: dict[str, BookResult]) -> pd.DataFrame:
    return pd.DataFrame({k: book_stats(b) for k, b in books.items()}).T


def compare(a: BookResult, b: BookResult, n_boot: int, seed: int = 0, mean_block: int = 21) -> dict[str, float]:
    """Paired stationary block bootstrap of net Sharpe(a) - Sharpe(b) (mean block 21 days by default)."""
    x = a.net.align(b.net, join="inner")
    return block_bootstrap_sharpe_diff(x[0].to_numpy(), x[1].to_numpy(), n_boot=n_boot, mean_block=mean_block,
                                       seed=seed)


SELECTORS: tuple[str, ...] = ("logit_hard", "logit_soft", "forest_hard", "forest_soft")


def seed_record(res: dict) -> dict:
    """The compact per-seed record kept in the multi-seed file, from one ``results.json``."""
    h = res["holdout"]
    st = h["stats"]
    dev_best = h["best_fixed_dev_choice"]
    rec = {
        "seed": res["data"]["seed"],
        "best_fixed_dev": dev_best,
        "best_fixed_hindsight": h["best_fixed_hindsight"],
        "net_sharpe": {k: round(v["net_sharpe"], 4) for k, v in st.items()},
        "turnover_yr": {k: round(v["turnover_yr"], 3) for k, v in st.items()},
        "vs_dev_best": {},
    }
    for k in SELECTORS:
        b = h["bootstrap"][f"{k}_vs_{dev_best}"]
        b63 = h["bootstrap"][f"{k}_vs_{dev_best}_b63"]
        rec["vs_dev_best"][k] = {"diff": round(b["diff"], 4), "lo": round(b["lo"], 4), "hi": round(b["hi"], 4),
                                 "p_le_0": round(b["p_le_0"], 4), "p_le_0_b63": round(b63["p_le_0"], 4),
                                 "lo_b63": round(b63["lo"], 4), "hi_b63": round(b63["hi"], 4)}
    return rec


def summarise_seeds(records: list[dict], alpha: float = 0.05 / len(SELECTORS),
                    alpha_all: float = 0.05 / (len(SELECTORS) * 3)) -> dict:
    """Median, range and win shares of each selector against the dev-chosen best fixed method.

    ``share_beats`` counts seeds with a positive net Sharpe difference; ``share_significant`` those whose
    bootstrap P(diff <= 0) is below ``alpha`` (Bonferroni over the four selectors), with 21-day and 63-day
    mean blocks; the ``_12`` variants use ``alpha_all`` (Bonferroni over 4 selectors x 3 panels). P(diff <= 0)
    is the share of percentile-bootstrap draws at or below zero. ``share_beats_hindsight_best`` compares with the fixed method that was best on the held-out
    period itself; ``share_beats_regime_rule`` with the hard switch told the true state.
    """
    out = {"n_seeds": len(records), "alpha": alpha, "alpha_12": alpha_all, "selectors": {}}
    for k in SELECTORS:
        d = np.array([r["vs_dev_best"][k]["diff"] for r in records])
        p21 = np.array([r["vs_dev_best"][k]["p_le_0"] for r in records])
        p63 = np.array([r["vs_dev_best"][k]["p_le_0_b63"] for r in records])
        row = {"median_diff": round(float(np.median(d)), 4), "min_diff": round(float(d.min()), 4),
               "max_diff": round(float(d.max()), 4), "share_beats": round(float((d > 0).mean()), 4),
               "share_significant": round(float((p21 < alpha).mean()), 4),
               "share_significant_b63": round(float((p63 < alpha).mean()), 4),
               "share_significant_12": round(float((p21 < alpha_all).mean()), 4),
               "share_significant_12_b63": round(float((p63 < alpha_all).mean()), 4),
               "median_net_sharpe": round(float(np.median([r["net_sharpe"][k] for r in records])), 4)}
        hb = np.array([r["net_sharpe"][k] - r["net_sharpe"][r["best_fixed_hindsight"]] for r in records])
        row["share_beats_hindsight_best"] = round(float((hb > 0).mean()), 4)
        row["median_diff_hindsight_best"] = round(float(np.median(hb)), 4)
        if all("regime_rule" in r["net_sharpe"] for r in records):
            rr = np.array([r["net_sharpe"][k] - r["net_sharpe"]["regime_rule"] for r in records])
            row["share_beats_regime_rule"] = round(float((rr > 0).mean()), 4)
        out["selectors"][k] = row
    hs = np.array([r["net_sharpe"][k] for r in records for k in ("logit_hard", "forest_hard")])
    ss = np.array([r["net_sharpe"][k] for r in records for k in ("logit_soft", "forest_soft")])
    out["share_soft_beats_hard"] = round(float((ss > hs).mean()), 4)
    out["median_dev_best_net_sharpe"] = round(float(np.median([r["net_sharpe"][r["best_fixed_dev"]]
                                                               for r in records])), 4)
    out["dev_best_counts"] = {m: sum(r["best_fixed_dev"] == m for r in records)
                              for m in sorted({r["best_fixed_dev"] for r in records})}
    out["hindsight_best_counts"] = {m: sum(r["best_fixed_hindsight"] == m for r in records)
                                    for m in sorted({r["best_fixed_hindsight"] for r in records})}
    ls = [r["net_sharpe"]["long_short"] - r["net_sharpe"]["long_only_cvx"] for r in records
          if "long_short" in r["net_sharpe"]]
    if ls:
        out["share_long_short_beats_long_only"] = round(float((np.array(ls) > 0).mean()), 4)
    return out


def accuracy(probs: pd.DataFrame, y: pd.Series) -> float:
    """Share of months where the most probable method was the realised best one."""
    common = probs.index.intersection(y.index)
    if len(common) == 0:
        return float("nan")
    return float((probs.loc[common].to_numpy().argmax(axis=1) == y.loc[common].to_numpy()).mean())
