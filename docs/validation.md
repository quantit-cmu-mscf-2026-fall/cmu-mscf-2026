# Validation: deciding whether a candidate is real

Every candidate the project generates, whether hand-written, agent-proposed or
produced by a sweep, is judged by the same procedures, so "we found something"
means the same thing no matter who or what found it. This page is the contract
those procedures share. It grows one module at a time; the table at the bottom
says what exists today.

## The input: a performance matrix

Every method takes the same input: **one per-period performance series per
candidate**, side by side (dates x candidates, a pandas DataFrame, columns named
by candidate). Methods that work on p-values take a Series of p-values indexed by
candidate name.

The series can be whatever your pipeline scores candidates on, as long as it is
one number per period and higher is better:

- **Strategy returns**, the default. From a `SyntheticPanel`:
  `backtest.candidate_returns(panel)`. From your own signals: run each through
  `backtest.run_backtest` and put the series side by side, dropping the first
  (untraded) date.
- **Information coefficient**: a per-date rank IC per candidate, as the Alpha-GPT
  pipeline computes, is already this shape.
- **Anything else per-period**: long-short spread returns, excess returns over a
  benchmark.

The methods resample, split and test these series over time; they never see
signals, prices or summary rows. So the direction is one way: strategy and
generation code produce the matrix and hand it over, and validation never calls
back into them. A new pipeline plugs in by producing this matrix, without
changing anything here. Say in your results which kind of series you passed; a
Sharpe ratio of an IC series is an information ratio, not a strategy Sharpe.

Generated with known truth, for testing: `synth.make_return_matrix(...)`.

## The rule every method follows

A method ships with two kinds of test, run on data where the truth is known:

1. **Null calibration.** On a matrix with no real candidates it says "nothing
   here" at the rate it promises (for example, at most alpha of the time). A
   procedure that cannot return "nothing" is a ranking, not a validation.
2. **Power.** On a matrix with planted candidates it finds them.

Both run on the harder cases as well as the easy one, because real candidate
returns are not IID normal and not independent of each other:

| `make_return_matrix` option | What it breaks |
|---|---|
| `rho` (pairwise correlation) | "independent trials": BH's guarantee, Bonferroni's power, the trial count in the deflated Sharpe ratio |
| `ar1` (autocorrelation) | the IID standard error of the Sharpe ratio |
| `t_df` (fat tails) | normal-theory p-values |
| `garch` (volatility clustering) | any resampling that shuffles single days |

Each option keeps every column's mean and volatility where you set them, so a
null candidate stays exactly null whatever else is switched on.

## Trial counts

Every correction depends on how many candidates were tried. That number comes
from the run ledger (`capstone.runlog`), not from an argument someone typed:
log every trial before looking at its result.

## Graded evidence, not one yes/no

A candidate is not "real" because one procedure said so. Every correction here
also returns a graded score, the level at which it would first call the
candidate real, and `evaluate.evidence_profile` puts them side by side:

| Column | Level at which it is called real | Valid when |
|---|---|---|
| `holm` | family-wise error (any false call) | any dependence |
| `by` | false-discovery rate | any dependence |
| `bh` | false-discovery rate | positive dependence: one-sided p-values of positively correlated candidates |
| `q` (Storey) | estimated false-discovery rate | independent candidates only |

Lower is stronger in every column. Feed it one-sided p-values
(`sharpe_test(...).pvalue_greater`): only positive Sharpe ratios are
discoveries, and one-sided tests are the case BH's guarantee covers. Never
decide on `q` for correlated candidates; the share-of-nulls estimate behind it
collapses when candidates move together, and it calls false discoveries more
than twice as often as BH does (`tests/test_evaluate.py`).

## A staged funnel

Strictness is spent where a mistake costs money, not at the first look. Loose
early stages are safe only because later ones look at data the earlier ones
never touched.

| Stage | Question | Strictness |
|---|---|---|
| 1. Screen | Worth a closer look? | Loose and graded: `bh` or local FDR around 0.10-0.20 |
| 2. Robustness | Does it survive other periods, lags, costs and cross-validation? Is the search overfit? | Medium |
| 3. Incremental value | Does it add anything beyond known factors and signals already held? | Medium |
| 4. Final decision | Allocate capital? | Strict: deflated or haircut Sharpe with the full ledger trial count, on a holdout looked at once |
| 5. Incubation | Does it work live? | Small live or paper capital |

Every stage's score and threshold is fixed before the candidates are scored.
Choosing the procedure or threshold after seeing results is multiple testing
of its own. Thresholds are tuned on simulated candidate sets to a chosen ratio
of missed discoveries to false ones (Harvey & Liu 2020), not left at textbook
defaults.

## What exists

| Module | Methods | Status |
|---|---|---|
| `synth`, `backtest` | `make_return_matrix`, `candidate_returns` | done |
| `evaluate` | `sharpe_variance` / `sharpe_test` (normal, non-normal, autocorrelation-robust), PSR, MinTRL, implied independent trials | done |
| `evaluate` | `holm`, `benjamini_yekutieli`, `estimate_pi0`, `storey_qvalues`; adjusted p-values and `evidence_profile`; one-sided `pvalue_greater` | done |
| `evaluate` | local false-discovery rate and empirical null (Efron): a per-candidate probability of being real that allows for correlation | planned |
| `bootstrap` | stationary bootstrap, block-length selection | planned |
| `snooping` | White's Reality Check, Romano–Wolf step-down | planned |
| `cv` | purged k-fold with embargo (from the Alpha-GPT work), walk-forward | planned |
| `pbo` | CSCV / probability of backtest overfitting | planned |
| `spanning` | incremental value over known factors and accepted signals | planned |
| `evaluate` | `haircut_sharpe`: Harvey–Liu haircut (their reference code), ledger trial count | done |
| `gate` | staged, pre-registered scorecard; trial count from the ledger; holdout looked at once | planned |

Already in `evaluate`: `benjamini_hochberg`, `bonferroni`,
`deflated_sharpe_ratio`, `expected_max_sharpe`, and `false_discovery_rate` /
`power` for scoring a selection rule against known truth.

## Reading the haircut Sharpe

Notes from QUANTIT-51 and QUANTIT-52 (Pin-Hua).

**The haircut Sharpe is a graded score, not a gate** (stage 4 in
`validation/framework.md`; the gate there is the deflated Sharpe). The trial
count comes from `runlog.trial_count()`, and an empty ledger raises rather than
quietly applying no haircut.

`haircut_sharpe` follows **Harvey & Liu's own reference code**
([`Haircut_SR.m`, `sample_random_multests.m`](https://people.duke.edu/~charvey/backtesting/)),
not a reading of the paper, because the two differ in ways that change the
number. `scripts/haircut_reference.py` is a literal transcription of those two
files and is the oracle the tests compare against.

What the authors do, and what it costs us:

- **Two-sided p-values on a t distribution** with N−1 degrees of freedom for the
  reported Sharpe, and on a *normal* for the simulated trials. There is no
  one-sided option, so the haircut is the one place in this module that does not
  screen on `pvalue_greater`.
- **Everything is converted to months, and a daily year is 360 days.** This
  contradicts the `periods_per_year=252` convention every other function here
  uses. It is confined to `haircut_sharpe`, documented there, and a test pins
  `sharpe_pvalue` at 252 so nobody "tidies up" the inconsistency in the wrong
  direction. Pass `n_obs` in the units named by `frequency`.
- **The unreported trials are simulated, not assumed away**, from the Harvey,
  Liu & Zhu (2014) empirical p-value distribution selected by
  `avg_correlation`. Holm and BHY are computed on each simulated family and the
  **median** over `n_simulations` repetitions is taken, so the result is
  stochastic — fix `seed` to reproduce it. Measured spread across 30 seeds: sd
  ≤ 0.0003 (Holm), ≤ 0.0045 (BHY) in Sharpe units.
- **The trial count is split, and the split is not cosmetic.** `n_trials` is the
  ledger's total, *including* the strategy being haircut. The authors' `num_test`
  (their `M`) is the number of **other** trials, so `num_test = n_trials - 1`.
  Their own code is inconsistent about which to use: **Bonferroni multiplies by
  `num_test`**, while **Holm and BHY use a family of `num_test + 1 = n_trials`**,
  and BHY's harmonic constant runs over that same `n_trials`. Both are
  reproduced as written; `.attrs` records `n_trials` and `num_test` separately so
  the convention in force is never in doubt.

Reference agreement (ours vs. the transcription, seed 0, 2,000 repetitions):
closed-form quantities — the monthly conversion, the p-value and everything
Bonferroni — agree to floating-point precision; Holm and BHY agree to within
0.006 in Sharpe units, against a test tolerance of 0.02.

One result worth keeping in mind: **BHY is *less* strict than Bonferroni here**,
which is the opposite of what the same two procedures do when the unreported
trials are assumed away instead of simulated. An earlier draft of this function
padded the family with NaN p-values, which put the reported strategy at rank 1
and made BHY the *strictest* of the three — out by 12 percentage points of
haircut on the same inputs. Simulating the family is what fixes it.
