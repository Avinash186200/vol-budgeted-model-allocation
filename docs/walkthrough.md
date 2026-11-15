# Walkthrough

This follows one run of `scripts/run_backtest.py` on the default synthetic panel (24 models, 156 months, planted
signal 0.01, seed 7), function by function. Every number here comes from running the code in this repo. To follow
along in a Python shell started at the repo root, first run `import sys; sys.path.insert(0, "src")`.

## 1. Daily panel to monthly

```python
from volbudget import make_synthetic, daily_to_monthly
p = make_synthetic(24, 156, 0.01, 7)
monthly = daily_to_monthly(p.daily)
```

`p.daily` has 3,393 business days and 24 model columns. Models start on different months, with zeros before they
go live. `daily_to_monthly` compounds each month and turns the pre-inception zeros into NaN. The result has 156
month-end rows, January 2008 to December 2020. Only one model is live in the first month, and the latest starter
still has 110 live months. `p.macro` holds two monthly macro series and `p.market` the monthly market return.
`scripts/make_synthetic_data.py` writes all three to `data/`.

## 2. Features

```python
from volbudget.features import build_panels, align_macro, feature_matrix
```

For each model and month-end t, `feature_matrix` builds 15 features from returns up to t: returns over 1, 3, 6 and
12 months, 6- and 12-month vol, 12-month Sharpe, win rate and drawdown, four cross-sectional percentile ranks
against the rest of the book, and the two macro series. `align_macro` shifts macro by one month, so the value used
at t is the one dated t-1. The target is whether month t+1 is positive.

## 3. Forecast

`model.predict_prob` fits a logistic regression and a gradient-boosting classifier on rows dated before t that
already have a realised target, and averages their probability for row t. Models with fewer than 36 training rows,
or fewer than 12 live months in the covariance window, keep a neutral 0.5.

## 4. Risk and weights

`factor_covariance` builds `B Σ_F B' + D` from the last 36 months: the market as one factor plus three principal
components of what is left. `optimize` maximises `w'(P - 0.5)` subject to `w >= 0`, `sum(w) <= 1`, a 1% vol cap per
model, 3% per group (4 groups) and 6% for the portfolio. The result is blended 80/20 with equal weight.

In this run the 36 rebalances run from January 2018 to December 2020 (the date is the month being earned). The
blended book holds all 24 models every month, with a largest single weight of 13.4%. Its average predicted vol is
3.1%, well inside the 6% portfolio cap.

## 5. Costs and comparison

Each month is charged 5 bp per unit of turnover, measured against the weights after they drifted with last month's
returns. The table printed by the script compares four books on the same 36 months. With the planted signal, the
blend has a Sharpe of 2.74 against 2.40 for vol parity, and the forecasts have a rank IC of +0.12. With
`--signal 0` the rank IC drops to +0.04 and the blend falls behind both baselines (1.42 against 1.89 for vol parity).

## Things to try

- Run `--signal 0` and `--signal 0.02` and watch the rank IC and the gap to vol parity.
- Change `--cost-bps` to 20. The optimiser turns over more than the baselines, so it pays more.
- Drop `--macro` from the CSV run. The planted signal lives in `macro_a`. What happens to the rank IC?
