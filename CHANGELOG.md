# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.3.0] - 2026-11-23

### Added
- `volbudget.min_variance`: the non-ML follow-up. Ledoit-Wolf minimum variance, ensembled over three look-backs
  and blended with vol parity, run daily with a T+2 lag over five back-to-back test windows. Includes a
  constrained min-variance solve, a final projection onto the per-model, group and portfolio vol caps, a paired
  block bootstrap, a sign test, search/hold-out selection of the blend weight, net-of-cost returns, and a sweep of
  the constructions and overlays the original research ruled out.
- `scripts/run_min_variance.py` (synthetic by default, `--independent` control panel, `--sweep`, `--daily` for
  your own CSV), run in CI.
- `tests/test_min_variance.py`: timing and look-ahead checks, cap compliance, cost accounting, bootstrap and
  selection checks, and a control panel on which min-variance must show no edge.
- `docs/min_variance.md`: the method, how it differs from the vol-budgeted optimiser, the audit of the original
  code and results on synthetic data.
- `original/min_variance/`: the original research code for this follow-up (read-only).
- `make_synthetic` now also returns the daily market series (`market_daily`); `make_independent_panel` builds the
  control panel. Existing outputs are unchanged.

## [1.2.0] - 2026-11-15

### Changed
- Docs: training-grade README and walkthrough. The README now has a plain-English problem statement, a pipeline
  diagram, key concepts, "What you will learn", a clearer synthetic label and an updated layout. New
  `docs/walkthrough.md`.
- README quick start: the `--daily returns.csv --macro macro.csv --market market.csv` placeholders are replaced by
  commands that run as written.

### Added
- `scripts/make_synthetic_data.py`, which writes the synthetic panel to `data/` in the CSV formats
  `run_backtest.py` reads, with a test.

## [1.1.0] - 2026-11-15

### Added
- `original/`: the original research code (read-only).
- `CHANGELOG.md`, `CONTRIBUTING.md` and a documentation index in `docs/README.md`.

## [1.0.0] - 2026-11-04

### Added
- First public release: the maintained code, documentation, tests and CI.
