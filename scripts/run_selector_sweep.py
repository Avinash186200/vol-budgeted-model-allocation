"""Run the selector study on many untouched seeds and write a compact multi-seed summary.

Seed 7 was used while designing the planted regime factor, so it is excluded by default.
Each run is a full ``run_selector.py`` study (about 1.5 minutes), single-threaded BLAS; the per-run
outputs go to ``--work`` (ignored by git) and the compact file to ``--out``. A cached run is reused only
if its ``code_hash`` matches the current code, and the summary refuses to mix runs from different code.

Examples:
    python scripts/run_selector_sweep.py                       # seeds 101-110, three modes, 2 workers
    python scripts/run_selector_sweep.py --seeds 101 102 --modes control
    python scripts/run_selector_sweep.py --summarise-only      # rebuild the summary from --work
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbudget.selector.evaluation import seed_record, summarise_seeds  # noqa: E402
from volbudget.selector.repro import code_hash, pin_threads  # noqa: E402

MODES = {"planted_s2": ["--strength", "2"], "planted_s1": ["--strength", "1"], "control": ["--mode", "control"]}


def _run(mode: str, seed: int, work: Path, boot: int) -> Path:
    """Run one study unless a result made by the current code (same ``code_hash``) is already cached."""
    out = work / f"{mode}_seed{seed}"
    res = out / "results.json"
    if res.exists() and json.loads(res.read_text()).get("code_hash") == code_hash():
        return res
    cmd = [sys.executable, str(ROOT / "scripts" / "run_selector.py"), *MODES[mode], "--seed", str(seed),
           "--boot", str(boot), "--out", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=pin_threads(dict(os.environ)))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 111)))
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--work", default="results/selector_runs")
    ap.add_argument("--out", default="results/selector/multiseed.json")
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args(argv)

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in a.modes for s in a.seeds]
    if a.summarise_only:
        paths = {(m, s): work / f"{m}_seed{s}" / "results.json" for m, s in jobs}
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {(m, s): ex.submit(_run, m, s, work, a.boot) for m, s in jobs}
            paths = {k: f.result() for k, f in futs.items()}

    out = {"seeds": a.seeds, "note": "seed 7 (used to design the planted factor) is not in this sweep",
           "boot": a.boot, "modes": {}}
    loaded = {k: json.loads(p.read_text()) for k, p in paths.items()}
    hashes = {r.get("code_hash") for r in loaded.values()}
    if hashes != {code_hash()}:
        raise SystemExit(f"cached runs come from different code ({sorted(map(str, hashes))}); delete {work} and rerun")
    out["code_hash"] = code_hash()
    for m in a.modes:
        recs = [seed_record(loaded[(m, s)]) for s in a.seeds]
        out["modes"][m] = {"records": recs, "summary": summarise_seeds(recs)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))

    for m in a.modes:
        sm = out["modes"][m]["summary"]
        print(f"\n{m}: {sm['n_seeds']} seeds; selector minus dev-chosen best fixed (net Sharpe)")
        print("| Selector | Median | Range | Beats | Signif. (21d) | Signif. (63d) |")
        print("|---|---|---|---|---|---|")
        for k, r in sm["selectors"].items():
            print(f"| `{k}` | {r['median_diff']:+.2f} | {r['min_diff']:+.2f} to {r['max_diff']:+.2f} | "
                  f"{r['share_beats']:.0%} | {r['share_significant']:.0%} | {r['share_significant_b63']:.0%} |")
    print(f"\nWrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
