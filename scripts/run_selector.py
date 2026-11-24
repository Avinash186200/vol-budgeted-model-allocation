"""Conditional allocation selector: does picking the construction method each month beat a fixed method?

Runs the whole study on the synthetic panel and writes ``results.json`` plus CSV
tables to ``--out``. Every number in the README section comes from a field of
``results.json``.

Examples:
    python scripts/run_selector.py                         # planted regime effect, strength 2
    python scripts/run_selector.py --strength 1            # weaker planted effect
    python scripts/run_selector.py --mode control          # no regime effect
    python scripts/run_selector.py --no-long-short         # skip the cvxpy optimiser
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # before numpy: results depend on the BLAS thread count

try:  # cvxpy first: on some Windows installs, importing it after pandas crashes the interpreter
    import cvxpy  # noqa: F401
except ImportError:
    pass

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volbudget import make_synthetic  # noqa: E402
from volbudget.selector import METHODS, SelectorConfig, build_dataset, build_method_panel, make_market  # noqa: E402
from volbudget.selector import develop, run_book, run_holdout  # noqa: E402
from volbudget.selector.evaluation import accuracy, book_stats, compare, stats_table  # noqa: E402
from volbudget.selector.repro import code_hash  # noqa: E402
from volbudget.selector.walkforward import (  # noqa: E402
    GRID,
    TEMPERATURES,
    blend_weights,
    fixed_weights,
    oracle_weights,
)

SELECTORS = ["logit_hard", "logit_soft", "forest_hard", "forest_soft"]


def _clean(x):
    """Make numpy scalars JSON-friendly."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def _md(tab: pd.DataFrame) -> str:
    cols = ["net_sharpe", "gross_sharpe", "ann_ret", "ann_vol", "max_dd", "turnover_yr", "cost_drag_bp"]
    lines = ["| Strategy | Net Sharpe | Gross Sharpe | Ann. return | Ann. vol | Max DD | Turnover/yr | Cost drag (bp/yr) |",
             "|---|---|---|---|---|---|---|---|"]
    for k, r in tab[cols].iterrows():
        lines.append(f"| `{k}` | {r.net_sharpe:.2f} | {r.gross_sharpe:.2f} | {r.ann_ret:.1%} | {r.ann_vol:.1%} | "
                     f"{r.max_dd:.1%} | {r.turnover_yr:.1f} | {r.cost_drag_bp:.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["planted", "control"], default="planted")
    ap.add_argument("--strength", type=float, default=2.0, help="planted mode: size of the regime effect")
    ap.add_argument("--models", type=int, default=24)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout", type=int, default=60, help="held-out decision months (the last ones)")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--no-long-short", action="store_true", help="skip the cvxpy long-short optimiser")
    ap.add_argument("--out", default=None, help="output folder (default results/selector/<mode>[_s<strength>])")
    a = ap.parse_args(argv)
    if not a.no_long_short:
        try:
            import cvxpy  # noqa: F401
        except ImportError:
            print('cvxpy is not installed (pip install -e ".[longshort]"); skipping the long-short optimiser')
            a.no_long_short = True

    strength = a.strength if a.mode == "planted" else 0.0
    out = Path(a.out or f"results/selector/{a.mode}" + (f"_s{strength:g}" if a.mode == "planted" else ""))
    out.mkdir(parents=True, exist_ok=True)

    p = make_synthetic(a.models, a.months, signal_strength=0.0, seed=a.seed, regime_strength=strength)
    cfg = SelectorConfig(cost_bps=a.cost_bps, holdout_months=a.holdout)
    mk = make_market(p.daily)
    panel = build_method_panel(mk, cfg)
    ds = build_dataset(mk, panel, cfg)
    dev = develop(mk, panel, ds, cfg)
    held = run_holdout(mk, panel, ds, dev, cfg)

    def books_for(months, sel_books):
        b = {k: run_book(mk, fixed_weights(panel, months, k), cfg.cost_bps) for k in METHODS}
        b["equal_blend"] = run_book(mk, blend_weights(panel, months), cfg.cost_bps)
        b["oracle"] = run_book(mk, oracle_weights(panel, ds, months), cfg.cost_bps)
        b.update(sel_books)
        if not a.no_long_short:
            from volbudget.selector.longshort import LongShortConfig, long_short_weights
            ls = LongShortConfig()
            b["long_short"] = run_book(mk, long_short_weights(mk, months, cfg, ls), cfg.cost_bps, borrow=ls.borrow)
            lo = LongShortConfig(allow_short=False)
            b["long_only_cvx"] = run_book(mk, long_short_weights(mk, months, cfg, lo), cfg.cost_bps)
        return b

    extra = {}
    regime_rule = None
    if p.regime is not None:
        # A reference, not a ceiling: a hard switch told the current true state at t, using the method
        # that did best in that state over the development period. Not a tradable strategy.
        state = p.regime.to_numpy()
        regime_rule = {}
        for s_ in (0, 1):
            ms = [t for t in ds.dev if state[t] == s_]
            perf = {k: book_stats(run_book(mk, fixed_weights(panel, ms, k), cfg.cost_bps))["net_sharpe"]
                    for k in METHODS}
            regime_rule[s_] = max(perf, key=perf.get)
        extra["regime_rule"] = run_book(mk, {t: panel.weights[t][regime_rule[state[t]]] for t in ds.hold},
                                        cfg.cost_bps)
    hold_books = books_for(ds.hold, {**{k: held[k]["book"] for k in SELECTORS}, **extra})
    dev_books = books_for(ds.dev, {})
    hold_tab, dev_tab = stats_table(hold_books), stats_table(dev_books)

    fixed_hold = hold_tab.loc[list(METHODS), "net_sharpe"]
    best_hind = str(fixed_hold.idxmax())
    boots = {}
    for k in SELECTORS:
        boots[f"{k}_vs_{dev.best_fixed}"] = compare(hold_books[k], hold_books[dev.best_fixed], a.boot)
        # block-length sensitivity: 63-day mean blocks keep more of the serial dependence
        boots[f"{k}_vs_{dev.best_fixed}_b63"] = compare(hold_books[k], hold_books[dev.best_fixed], a.boot,
                                                        mean_block=63)
        if best_hind != dev.best_fixed:
            boots[f"{k}_vs_{best_hind}"] = compare(hold_books[k], hold_books[best_hind], a.boot)
    boots[f"equal_blend_vs_{dev.best_fixed}"] = compare(hold_books["equal_blend"], hold_books[dev.best_fixed], a.boot)
    if "long_short" in hold_books:
        boots["long_short_vs_long_only_cvx"] = compare(hold_books["long_short"], hold_books["long_only_cvx"], a.boot)
        boots[f"long_short_vs_{dev.best_fixed}"] = compare(hold_books["long_short"], hold_books[dev.best_fixed],
                                                          a.boot)

    y_hold = ds.y.loc[ds.y.index.intersection(ds.hold)]
    y_dev_train = ds.y.loc[ds.y.index.intersection(ds.dev)]
    n_configs = len(dev.table)
    res = {
        "code_hash": code_hash(),
        "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "data": {"mode": a.mode, "regime_strength": strength, "models": a.models, "months": a.months,
                 "seed": a.seed, "synthetic": True},
        "config": {k: v for k, v in vars(cfg).items()},
        "periods": {
            "dev_decision_months": len(ds.dev), "holdout_decision_months": len(ds.hold),
            "dates_note": "synthetic calendar labels, not real market dates",
            "dev_first_decision": str(mk.months[ds.dev[0]].date()),
            "holdout_first_decision": str(mk.months[ds.hold[0]].date()),
            "holdout_last_return_month": str(mk.months[ds.hold[-1] + 1].date()),
        },
        "features": ds.features,
        "target_counts_dev": {METHODS[int(k)]: int(v) for k, v in y_dev_train.value_counts().sort_index().items()},
        "development": {
            "configs_tried": n_configs,
            "grid": {k: list(v) for k, v in GRID.items()}, "temperatures": list(TEMPERATURES),
            "choices": dev.choices,
            "best_fixed": dev.best_fixed,
            "dev_stats": dev_tab.to_dict(orient="index"),
        },
        "holdout": {
            "stats": hold_tab.to_dict(orient="index"),
            "best_fixed_dev_choice": dev.best_fixed,
            "best_fixed_hindsight": best_hind,
            "accuracy": {k: accuracy(held[k]["probs"], y_hold) for k in SELECTORS},
            "base_rate_most_common_dev_class": float(y_dev_train.value_counts(normalize=True).max()),
            "holdout_share_of_dev_most_common_class": float(
                (y_hold == y_dev_train.value_counts().idxmax()).mean()),
            "bootstrap": boots,
            "multiple_testing": {
                "selectors_taken_to_holdout": len(SELECTORS),
                "bonferroni_alpha": 0.05 / len(SELECTORS),
                "configs_tried_on_dev": n_configs,
            },
        },
    }
    if p.regime is not None:
        res["data"]["stress_share"] = float(p.regime.mean())
        res["holdout"]["regime_rule_methods"] = {"calm": regime_rule[0], "stress": regime_rule[1]}
    (out / "results.json").write_text(json.dumps(_clean(res), indent=2))
    hold_tab.to_csv(out / "holdout_stats.csv")
    dev_tab.to_csv(out / "dev_stats.csv")
    dev.table.to_csv(out / "dev_configs.csv", index=False)

    print(f"Synthetic panel: mode={a.mode}, regime_strength={strength:g}, {a.models} models, {a.months} months, "
          f"seed {a.seed}, costs {cfg.cost_bps:g} bp per unit of turnover")
    print(f"Development: {len(ds.dev)} decision months from {res['periods']['dev_first_decision']}; held out: "
          f"{len(ds.hold)} from {res['periods']['holdout_first_decision']}")
    print(f"Configs tried on dev: {n_configs}; best fixed method on dev: {dev.best_fixed}")
    for k, c in dev.choices.items():
        print(f"  {k}: param={c['param']:g} temperature={c['temperature']:g} dev net Sharpe {c['dev_net_sharpe']:.2f}")
    print("\nHeld-out period (run once):")
    print(_md(hold_tab))
    print(f"\nBest fixed in hindsight on the held-out period: {best_hind}")
    print("\nHit rate of the most probable method (held out): "
          + ", ".join(f"{k} {v:.2f}" for k, v in res["holdout"]["accuracy"].items())
          + f"; always guessing the most common dev class: {res['holdout']['holdout_share_of_dev_most_common_class']:.2f}")
    print("\nPaired block bootstrap of the net Sharpe difference (held out):")
    for k, b in boots.items():
        print(f"  {k}: {b['diff']:+.2f}, 95% CI [{b['lo']:+.2f}, {b['hi']:+.2f}], P(diff <= 0) = {b['p_le_0']:.3f}")
    print(f"\nBonferroni threshold for {len(SELECTORS)} selectors: {0.05 / len(SELECTORS):.4f}")
    print(f"Wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
