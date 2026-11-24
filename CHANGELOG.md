# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.4.0] - 2026-11-24

### Added
- `volbudget.selector`: a conditional allocation selector. Each month it builds five candidate books (equal
  weight, vol parity, minimum variance, equal risk contribution and the existing vol-budgeted optimiser), computes
  point-in-time regime features (trailing vol, average correlation, first-PC share, dispersion, trend and each
  method's recent relative Sharpe), and trains a multinomial logistic regression and a random forest to predict
  which method will have the best risk-adjusted return next month. Hard (argmax) and soft (probability-blend)
  versions. Walk-forward with a purge and embargo, expanding-window refits, hyperparameters chosen on the
  development period only, and a final held-out period run once.
- `volbudget.selector.longshort`: a long-short optimiser written as one convex problem in cvxpy, with
  `w = w_long - w_short`, a borrow cost on the short leg, gross and net exposure limits, a turnover penalty and the
  same per-model, group and portfolio vol budget. cvxpy is an optional extra (`.[longshort]`).
- `make_synthetic(..., regime_strength=...)`: a regime-switching mode with a known planted effect (a hidden
  calm/stress state that changes which construction method wins). `regime_strength=0` (the default) leaves every
  existing output unchanged and serves as the control.
- `scripts/run_selector.py`, run in CI on the planted and control panels; results in `results/selector/`.
- `tests/test_selector.py` and `tests/test_longshort.py`: look-ahead checks on features, method books and targets,
  purge and embargo, a check that the development step never reads the held-out period, book and cost
  accounting, and the long-short constraints and long/short split.
- `scripts/run_selector_sweep.py` and `results/selector/multiseed.json`: the study on ten untouched seeds (101 to
  110) for the strong planted, weaker planted and control panels, with a test that recomputes every summary from
  the per-seed records. This is the main result; seed 7, used while designing the planted factor, is disclosed
  separately.
- A 63-day block-length check next to the 21-day paired bootstrap, and significance counts under a 12-test
  Bonferroni correction; the long-short solve is pinned to Clarabel.
- Reproducibility: the selector scripts pin BLAS/OpenMP to one thread, and every result is stamped with a
  `code_hash`; the sweep refuses to mix runs from different code.
- `docs/selector.md`: method, timeline, multi-seed results, the seed-7 case, disclosures and limitations.
- `pyproject.toml` with `dev` (pytest) and `longshort` (cvxpy) extras.

### Changed
- `requirements.txt` now lists runtime dependencies only; pytest moved to the `dev` extra. CI installs
  `.[dev,longshort]`.

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
