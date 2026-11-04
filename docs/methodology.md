# Methodology and bias audit

## Problem

The book is a set of systematic trading models. The question was whether a forecast of each model's next-month return can improve on static allocation, while staying inside a fixed risk budget. The allocation is conditional: the expected return input to the optimiser changes with the current state of each model and of the macro environment.

## Data contract

| Field | Meaning |
|---|---|
| `date` | Trading day |
| `model 1` .. `model N` | Daily decimal return of each model. Zeros before a model's first non-zero day are pre-inception |
| macro CSV (optional) | `date` + monthly features, dated by the month they describe, not the month they are published |
| market CSV (optional) | `date,market` monthly market return, used as covariance factor 1 |

The real book was close to market-neutral (tiny average beta and R-squared against an equity index), and its covariance had an effective rank of about four. That is why the covariance is a low-rank factor model rather than a sample matrix.

## Timing

- Row dated month-end t: model features from returns up to and including month t; macro features from month t-1 (one-month lag).
- Target of row t: sign of the return in month t+1.
- Rebalance at t: fit on rows dated before t, whose targets (month <= t) are realised; predict row t; realise month t+1.
- Covariance: the 36 months ending at t.

## Risk budget

| Level | Constraint (annualised) |
|---|---|
| Per model | `w_i * σ_i <= 1%` (overrides per model supported) |
| Group | sub-portfolio vol <= group cap (default: 4 groups at 3%) |
| Portfolio | vol <= 6%, long-only, `sum(w) <= 1` (cash allowed) |

Real budgets usually have unequal group caps and tighter caps for some individual models. `Budget` takes any such structure.

## Audit of the original research code

1. **Positional indexing (look-ahead).** The per-model feature frame dropped months before the model was live, but the backtest then sliced it with `iloc[:t_pos]` and `iloc[[t_pos]]`, where `t_pos` was the position in the full calendar. For any model that started late, both the prediction row and the end of the training set pointed to later dates, so training included labels from after the rebalance date. Fix: all selection is by date. `test_prediction_unchanged_when_future_is_perturbed` changes every return after the rebalance and checks the forecasts do not move.
2. **Macro publication lag.** Docstrings said macro features were lagged one month, but they were joined unshifted. Daily market series are fine at month-end, but the external feature file contained economic series with unknown release dates. Fix: `macro_lag=1` by default; `test_macro_is_lagged`.
3. **Selection on the test window.** The research log compares four feature sets and several ensemble and blending variants on the same 24 out-of-sample months and reports the best. With 23 realised months, the standard error of a Sharpe estimate is around 0.7, so the spread between variants is mostly noise. Fix: one configuration, fixed in `Config` before running, and no real-data figures published.
4. **Tuned blending rule.** A confidence-based blend weight (0.5 to 0.9) was added after looking at results. Fix: a fixed 80/20 blend.
5. **Market factor bug.** `daily.pct_change().resample("ME").last()` gives the last day's return, not the month's. Fix: compound to monthly.
6. **Zero-filled gaps in covariance.** Fix: fill with the window mean, so a new model's variance is not diluted.
7. **No costs.** Fix: 5 bps per unit of one-way turnover, with weights drifted by realised returns between rebalances.
8. **Weak baseline.** The headline compared with equal weight. Vol parity had the highest Sharpe in the original run. Fix: report both.

## Checks in CI

- Pre-inception zeros become NaN; interior zeros are kept.
- Model features at t are unchanged when returns after t are changed.
- Macro features are shifted by the lag.
- Forecasts at a rebalance are unchanged when future returns are changed.
- The optimiser's weights satisfy every per-model, group and portfolio cap.
- Noise test: with no planted signal, the rank IC stays near zero, and with a planted signal it is clearly higher.

## Verdict

On synthetic data with a real planted signal, the pipeline finds it and the optimiser beats both baselines after costs. On noise, it loses to them. On the real book, even the original run, with the biases above still in it, had vol parity ahead of the optimiser on Sharpe. I see no evidence of an edge over vol parity: the macro and trailing-performance features carry very little information about next-month returns of these models, which matches an earlier diagnostic that put the median out-of-sample R-squared near zero. The useful output of the work was the risk-budget machinery and a clear answer that the forecast layer was not worth its turnover.

## Dropped from this repo

- The original 8-member ensemble (three gradient boosting configs, two logistic regressions, histogram boosting, extra trees, random forest) and a 4-member regression blend. Replaced with two fixed members to keep runs fast and remove a tuned knob.
- A vendor price downloader and an external pre-built feature file. Macro input is now a plain CSV.
- Charting and exploratory feature-set comparisons.

## Known limitations

- The test window on the real book was short (about two years). Any conclusion either way needs a longer hold-out.
- Costs are a flat rate; capacity and slippage per model are not modelled.
- Equal-weight in the blend is scaled to the portfolio cap only, so the blended book can exceed a group cap slightly.
