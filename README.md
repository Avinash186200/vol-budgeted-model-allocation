# Vol-Budgeted Model Allocation

A monthly allocator for a book of systematic trading models. For each model it predicts the probability that next month is positive, then sizes the book with a long-only optimiser under a three-tier volatility budget: per model, per group of models, and for the whole portfolio.

I built this for a book of systematic trading models. The return file is proprietary and is not in this repo. Everything here runs on a **synthetic daily return panel with the same shape** (a `date` column, one column per model, leading zeros before each model goes live), so you can clone it and reproduce every number below.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

## At a glance

| | |
|---|---|
| **Input** | Daily return panel, one column per trading model, staggered inception; optional monthly macro features and a market return series |
| **Method** | Per-model ensemble (logistic regression + gradient boosting) predicts P(next month > 0); factor covariance (market + PCA); SLSQP maximises `w'(P - 0.5)` under per-model, group and portfolio vol caps; 80/20 blend with equal weight |
| **Output** | Monthly weights, net-of-cost returns for the optimiser and for equal-weight and vol-parity baselines, predicted vol, and the rank IC of the forecasts |
| **Leakage controls** | Date-based train/predict split, macro features lagged one month, settings fixed in advance, costs on turnover, a future-perturbation test and a noise test in CI |
| **Status** | Research complete; no demonstrated edge over vol parity on the real book. Numbers below are synthetic only |

## How it works

```
daily returns ─► monthly compounding ─► features at t ─► per-model P(up) ─► factor Σ ─► vol-budgeted QP ─► blend with EW ─► returns in t+1
                 data.py                features.py      model.py          optimizer.py                     backtest.py
```

1. **Monthly returns**: compound the daily returns. Leading zeros are pre-inception and become NaN.
2. **Features** for the row dated month t use returns up to month t: trailing means, vol, Sharpe, win rate, drawdown, and cross-sectional percentile ranks across the book. Macro features are shifted by a one-month publication lag.
3. **Model**: for each live model, fit on rows dated before t (their labels are already realised) and predict the row dated t.
4. **Covariance**: `Σ = B Σ_F B' + D`, with the market as factor 1 and three PCs of the market residuals, over a 36-month window.
5. **Optimiser**: maximise `w'(P - 0.5)` with `w >= 0`, `sum(w) <= 1`, `w_i σ_i <= 1%`, group vol <= 3% and portfolio vol <= 6%. Then `w = 0.8 w_opt + 0.2 w_EW`.
6. **Costs**: 5 bps per unit of one-way turnover, with weights drifted between rebalances.

## Results on synthetic data

`python scripts/run_backtest.py` (24 models, 156 months, last 36 out of sample, 5 bps costs, seed 7):

| Portfolio | Ann. return | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| Optimiser (raw, `opt_raw`) | 9.3% | 3.5% | 2.68 | -1.7% |
| Optimiser (80/20 blend, `opt_blend`) | 8.7% | 3.2% | 2.74 | -1.5% |
| Equal weight | 6.3% | 2.6% | 2.41 | -1.1% |
| Vol parity | 6.1% | 2.5% | 2.40 | -0.9% |

Rank IC of the forecasts: +0.12. The synthetic models have a planted macro effect (`--signal 0.01`). With the signal switched off (`--signal 0`), the same pipeline gives a rank IC of +0.04, and the optimiser **loses** to both baselines (Sharpe 1.42 for the blend against 1.89 for vol parity). That is the honest behaviour: when the forecast is noise, maximising `w'μ` concentrates on noise, and a simple risk-based book wins. The high Sharpe ratios everywhere come from how the synthetic models are generated, not from the method. With 36 monthly observations, the standard error of each Sharpe is roughly 0.5 or more, so the 0.3 gap between the blend and vol parity on the signal run is suggestive, not significant.

> **About the real-data result.** The original research reported the blended optimiser well ahead of equal weight over a 24-month test window. That figure came out of a run with a positional indexing bug, macro features with no publication lag, no costs, and a feature set picked on the same test window (see the audit below). Even in that run, plain vol parity had the best Sharpe of the four. So even the inflated run showed no edge over a simple risk-based book. I don't publish real-data figures here, and I treat the forecast layer as unproven.

## Leakage and bias audit

Details are in [docs/methodology.md](docs/methodology.md).

| Issue | Effect | Fix |
|---|---|---|
| Train/predict rows picked by **position** in a feature frame that had dropped pre-inception rows | For late-starting models, the "row t" was a later month, so the model trained on labels from the future | Rows are selected by date: train on `date < t`, predict `date == t` |
| Macro features dated month t used at month t, described as lagged but not shifted | Look-ahead for any series published after month-end | Macro shifted by `macro_lag=1` month |
| Feature sets, ensemble size and blend rule compared on the same 24-month test window, best one reported | Selection bias on a tiny sample | One configuration, fixed in advance and listed in `Config` |
| Blend weight set by a confidence rule tuned during research | Another tuned knob | Fixed 80/20 blend |
| Market factor built as the **last daily** return of each month, not the monthly return | Wrong covariance factor | Market return compounded to monthly |
| Pre-inception gaps filled with 0 in the covariance window | Understated vol, so larger weights for new models | Gaps filled with the window mean |
| No transaction costs | Optimistic, especially for the optimiser, which turns over more | 5 bps on turnover, with drifted weights |
| Optimiser compared mainly with equal weight | The stronger simple baseline (vol parity) was ignored | Vol parity reported alongside |

## Repo layout

```
src/volbudget/   data.py · features.py · model.py · optimizer.py · backtest.py · metrics.py
scripts/         run_backtest.py
tests/           compounding, feature timing, macro lag, future-perturbation, budget, noise test
docs/            methodology.md
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/run_backtest.py                  # synthetic, with planted signal
python scripts/run_backtest.py --signal 0       # synthetic, pure noise
python scripts/run_backtest.py --daily returns.csv --macro macro.csv --market market.csv
```

## License

MIT
