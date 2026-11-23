# Minimum variance: the non-ML follow-up

After the forecast layer in the main package showed no edge over vol parity, I tried the opposite approach: drop
return forecasts completely and build the book from the risk structure alone. This page describes that follow-up,
how it differs from the vol-budgeted optimiser, what I found when I audited my original code, and how the
maintained version behaves on synthetic data.

Code: [`src/volbudget/min_variance/`](../src/volbudget/min_variance/). Script:
[`scripts/run_min_variance.py`](../scripts/run_min_variance.py). Tests:
[`tests/test_min_variance.py`](../tests/test_min_variance.py). Original code:
[`original/min_variance/`](../original/min_variance/).

## The idea in one paragraph

A mean-variance optimiser needs two inputs: expected returns and a covariance. The main package spent its effort
on the first one and found that next-month returns of these trading models could not be forecast. The covariance
is a different matter: how the models move together is fairly stable and can be estimated from a few years of
daily data. Minimum variance uses only the covariance. It asks "which long-only mix of these models has the lowest
predicted volatility?" and nothing else. If every model has roughly the same Sharpe ratio, the lowest-risk mix is
also the best risk-adjusted one; if they don't, minimum variance can concentrate in the wrong place. That is the
whole bet.

## Method

At each rebalance (every 5 trading days, position `t` in the daily panel):

1. **Active models**: a model is live if it has at least 252 non-missing days in the last 504.
2. **Covariance**: for each look-back in {252, 504, 756} days, take the rows before `t` with no missing value
   among the active models and fit a **Ledoit-Wolf** shrunk covariance. Shrinkage pulls the noisy sample matrix
   towards a scaled identity, with the intensity estimated from the data, so there is nothing to tune.
3. **Min-variance ensemble**: solve long-only minimum variance (`min w'Σw`, `w >= 0`, `sum(w) = 1`) on each of the
   three covariances and average the three weight vectors. Averaging removes the "which window?" choice.
4. **Vol parity**: `w_i ∝ 1/σ_i`, the baseline. It ignores correlations.
5. **Blend**: `w = α·w_MV + (1-α)·w_VP`, with both legs first scaled to the same predicted vol. The original design
   used `α = 0.75`.
6. **Scale**: every book is scaled to 6% predicted annual vol under one shared covariance (the average of the
   three Ledoit-Wolf matrices). The same matrix is used for vol parity, the scaling and the cap check.
7. **Caps** (optional, `blend_capped`): re-solve min variance under the vol budget and project the final blend onto
   the caps (see below).
8. **Trade**: weights computed from rows `< t` are held from row `t + 2` (T+2) for 5 days. Costs of 5 bps per unit
   of one-way turnover are charged on the first day the new weights are held.

The history is cut into five back-to-back test windows of 504 trading days. Window 0 is the oldest.

## How this differs from the vol-budgeted optimiser

| | Vol-budgeted optimiser (main package) | Minimum variance (this page) |
|---|---|---|
| Signal | P(next month > 0) from logistic regression + boosting | None |
| Objective | maximise `w'(P - 0.5)` | minimise `w'Σw` |
| Covariance | monthly factor model (market + 3 PCs) | daily Ledoit-Wolf, three look-backs |
| Cadence | monthly | every 5 trading days, T+2 |
| Caps | inside the optimiser | min-variance re-solved under caps, then a final projection |
| Main risk | a noisy forecast concentrates the book on noise | an unequal Sharpe across models makes the lowest-risk mix the wrong mix |
| Gross exposure | `sum(w) <= 1` | whatever reaches 6% vol (above 1 when the models have low vol) |

## Why it is non-ML

There is no model fitted to returns, no label, no feature and no hyperparameter search inside the allocator. The
only estimated object is a covariance, with a shrinkage intensity that has a closed-form estimate. The design
choices that remain (the three look-backs, the blend weight, the refresh cadence) are a handful of numbers, and the
blend weight is the only one I let the data choose, on the search windows only. "Non-ML" does not mean "cannot be
overfit": choosing among many constructions on the same test windows is still a search, and the original research
did exactly that (see the audit).

## Constraints: the vol budget

The budget is the same three-tier one as the main package: per model `w_i σ_i <= 1%`, per group sub-portfolio vol
`<= 3%` (four groups by default), portfolio vol `<= 6%`.

- **Constrained min variance.** Long-only min variance has an equivalent form: the book with the largest gross
  exposure `sum(w)` for a given portfolio vol has the same KKT conditions as the min-variance direction. Written
  that way, the caps enter as plain absolute limits:
  `max sum(w)` s.t. `w >= 0`, `w_i σ_i <= cap_i`, group vol `<= group cap`, portfolio vol `<= 6%`. The start point
  (inverse variance scaled down by its worst cap ratio) is always feasible.
- **Final projection.** The blend of the capped min-variance leg with vol parity can still breach caps, because
  vol parity at 6% breaches them on its own. `project_to_caps` finds the closest weights (least squares) that meet
  every cap under the same covariance. All the cap constraints are positively homogeneous, so the input scaled down
  by its worst cap ratio is always feasible and is used as start point and fallback.

For a fair comparison the script also reports `vol_parity_capped`: vol parity passed through the same projection.

## Evidence: what each check can and cannot tell you

- **Per-window win count and sign test.** Five windows give a one-sided sign-test p-value of 1/32 ≈ 0.031 for
  5 of 5 wins, 0.19 for 4 of 5. The windows share most of their training history, so they are not fully
  independent either.
- **Paired stationary block bootstrap** of the Sharpe difference on the pooled daily returns (mean block 21
  days, same blocks for both series). It treats days as the unit of evidence, so it is optimistic when the real
  unit is a regime. I report it next to the win count, never alone.
- **Search/hold-out.** The blend weight `α` is chosen from {0, 0.25, 0.5, 0.75, 1} on the two oldest windows, then
  scored on the three newer ones. The test `test_selection_never_looks_at_holdout` checks the choice does not move
  when only the hold-out scores change.
- **Net of costs.** Costs are charged inside the daily stream on each rebalance. Turnover is measured on target
  weights (no drift between rebalances) and the first build of each window is not charged.

## Audit of the original research code

My original version (in [`original/min_variance/`](../original/min_variance/)) had no look-ahead in the weights:
the covariance at rebalance `t` used rows strictly before `t`, and returns were applied with a T+2 lag. Its own
audit checked this and I agree. The problems were elsewhere:

| Issue | Effect | Fix |
|---|---|---|
| The blend weight (0.25 / 0.5 / 0.75) and the pure-vs-blend choice were made by comparing all five test windows; the search/hold-out split was used for the overlays and the sweep, not for the headline design | The "5 of 5 windows" result and its bootstrap p-value are measured after selection on the same windows | `α` is chosen on the two oldest windows and checked on the other three |
| More than 180 construction specs were tried in the exhaustive sweep before settling on the design | Multiple-comparison risk; the best of many will beat the baseline by luck | The sweep is kept, but ranks on search windows, scores finalists on hold-out, and applies a Bonferroni threshold |
| The blend added a min-variance vector that sums to 1 to a vol-parity vector of `cap_i/σ_i` | The two legs were on different scales, so the real mix was not 75/25 and drifted with the book's vol level | Both legs scaled to the same predicted vol before blending |
| The blend, scaled to the 6% portfolio cap, breached per-model and group caps; the original's own audit flagged this as the open item before use | A headline result for a book that did not meet the mandate | Final projection onto all caps; `blend_capped` is compliant by construction and tested |
| Caps enforced with the shrunk covariance, but checked with a different (factor) covariance | Compliance in one matrix is not compliance in the other | One covariance for scaling, caps and checks |
| The constrained re-solve used a sum-to-1 problem with caps divided by a scale factor re-estimated in a loop; when SLSQP failed, the code silently used its starting point | Tight group caps can make that problem infeasible; in my checks on the synthetic panel the solver often failed to converge | Re-written as `max sum(w)` under absolute caps, which always has a feasible start |
| The claim that enforcing the caps improved Sharpe compared a capped blend with an uncapped blend built on different scales | Not a like-for-like comparison | Capped blend is compared with capped vol parity |
| Headline Sharpe ratios were gross; costs were applied afterwards as annual turnover times a rate | Fine as an approximation, but easy to skip; the headline table was gross | Costs charged in the daily stream; net Sharpe is the headline |
| Overlay scale normalised by the full-sample mean of the leverage series | Uses future values for a constant; it does not change a Sharpe ratio, but would change vol and return levels | Normalised by the expanding mean known the day before |
| The pooled daily bootstrap was presented with a very small p-value | Days are not independent draws of allocator skill; the effective sample is about five regimes | Reported next to the win count and the sign test |

**About the real-data result.** The original research reported a modest gain for the 75/25 blend over vol parity
on five test windows. Given the selection on those same windows, the scale mismatch in the blend and the cap
breaches, I don't treat that gain as established and I don't publish the figures. The honest status of the real
result is "not validated": it would need the corrected code, a capped comparison against capped vol parity, and a
forward hold-out.

## Results on synthetic data

All numbers below come from code in this repo on synthetic data. They show how the method behaves, not what it
earns. The synthetic models all have a positive drift, so the Sharpe ratios are high everywhere.

### Default panel (`python scripts/run_min_variance.py`)

24 models, 156 months of daily returns (seed 7, no planted signal), five test windows of 504 days, 5 bps costs.
The generator gives each model random-sign loadings on three hidden factors plus a drift unrelated to those
loadings. That is a structure minimum variance is built to exploit: it can hedge the hidden factors out and keep
the drift.

| Book | Net Sharpe (mean of 5) | Wins vs vol parity | Turnover / yr | Worst model cap | Worst group cap |
|---|---|---|---|---|---|
| Equal weight | 2.11 | 0 / 5 | 0.7 | – | – |
| Vol parity | 2.27 | – | 0.9 | 1.70× | 1.78× |
| Vol parity, capped | 2.32 | 4 / 5 | 0.6 | 1.00× | 1.00× |
| Min-variance ensemble | 3.63 | 5 / 5 | 5.3 | 7.10× | 3.92× |
| Blend (α = 0.75) | 3.51 | 5 / 5 | 4.3 | 6.04× | 3.54× |
| Blend, capped | 2.30 | 3 / 5 | 0.6 | 1.00× | 1.00× |

- Uncapped, the blend beats vol parity in 5 of 5 windows (sign test p = 0.031; pooled bootstrap 95% interval for
  the Sharpe difference +0.86 to +1.62).
- Choosing `α` on the two oldest windows picks `α = 1` (pure min variance); on the three hold-out windows it scores
  3.70 against 2.21 for vol parity.
- **The caps take the gain away.** The uncapped min-variance book reaches its low risk by putting up to seven times
  the per-model risk limit on a few models. Once the 1% model caps and 3% group caps are enforced, the capped blend
  (2.30) is no better than capped vol parity (2.32) and beats it in only 1 of 5 windows. On this panel, the
  advantage of minimum variance comes almost entirely from the concentration that the vol budget forbids.

### Control panel (`python scripts/run_min_variance.py --independent`)

24 independent models with different vols and the same Sharpe ratio. Here vol parity is already the best long-only
mix, so there is nothing for minimum variance to find.

| Book | Net Sharpe (mean of 5) | Wins vs vol parity | Turnover / yr |
|---|---|---|---|
| Vol parity | 4.84 | – | 0.7 |
| Min-variance ensemble | 4.73 | 2 / 5 | 3.5 |
| Blend (α = 0.75) | 4.81 | 2 / 5 | 2.7 |
| Blend, capped | 4.85 | identical to capped vol parity | 0.4 |

The blend loses slightly to vol parity (pooled Sharpe difference -0.03, 95% interval -0.21 to +0.15), with three
times the turnover. That is the expected behaviour: with no correlation structure to exploit, minimum variance only
adds estimation noise. `test_independent_equal_sharpe_panel_gives_min_variance_no_edge` runs a small version of
this in CI.

### Ruled-out approaches (`python scripts/run_min_variance.py --sweep`)

Same panel, all books uncapped and scaled to 6% vol, 5 bps costs. Specs are ranked on the two oldest windows and
only those that beat vol parity there are scored on the three hold-out windows.

- **Construction sweep**: 4 covariance estimators (sample, Ledoit-Wolf, OAS, EWMA with a 60-day half-life) x 2
  look-backs (252, 504 days) x 6 objectives = 48 specs. All 24 correlation-aware specs (min variance, max
  diversification, ERC) beat vol parity on search and on hold-out, and every one had hold-out bootstrap p < 0.001,
  the finest the 1,000-resample bootstrap can resolve, which is at the Bonferroni threshold of 0.05/48 (0.00104). The best on search was sample-covariance min variance on 504 days (+1.18 Sharpe on search,
  +1.52 on hold-out). Inverse variance and equal weight lose slightly to vol parity. Diagonal Kelly, the only
  objective that uses trailing mean returns, loses by 1.2 to 1.8 Sharpe: on this panel the trailing mean is noise
  and sizing on it hurts. The estimator matters much less than the objective here.
- **Overlays on vol parity** (time-varying leverage from the stream's own trailing vol, or from trailing market
  vol; no extra cost is charged for the daily leverage changes): the best on search was a vol-managed overlay with a 63-day look-back and power 2 (+0.06 Sharpe on search,
  +0.01 on hold-out). Every overlay is within about 0.1 of vol parity on hold-out, which is noise at this sample
  size, as it was in the original research.

These are uncapped results, so the warning from the default panel applies: the gain of the correlation-aware
constructions comes from concentration the vol budget would forbid. The sweep takes about four minutes.

## Kept, dropped and changed

- **Kept**: the min-variance ensemble, the blend, the constrained solve, the bootstrap, the net-of-cost view, the
  holdout discipline, the ERC / max-diversification / Kelly / estimator sweep, and the vol-managed and
  market-timing overlays.
- **Dropped**: the "ensemble of portfolios" experiment (it blended return streams from the ML pipeline, which this
  package does not produce), hierarchical risk parity and full Kelly from the sweep (to keep it fast), and the
  parallel launcher script.
- **Changed**: the vol-parity baseline uses the diagonal of the shrunk covariance rather than a market-factor
  covariance, and every book is scaled to the same predicted vol so the comparison and the blend are like for like.

## Known limitations

- Five test windows is a small sample, whatever the bootstrap says.
- The synthetic panel decides the answer: with hidden factors that can be hedged, minimum variance wins; with
  independent equal-Sharpe models, it does not. Real books sit somewhere in between, and the caps matter as much
  as the construction.
- Costs are a flat rate per unit of turnover; there is no capacity or slippage model, and gross exposure above 1 is
  assumed to be financeable at no cost.
