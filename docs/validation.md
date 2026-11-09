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

## White's Reality Check

`snooping.white_reality_check` is an early, search-level gate: one p-value for
the whole family of candidates, before any candidate is judged on its own.

```text
AI-generated candidate signals
  -> backtest / candidate return matrix
  -> White's Reality Check
  -> factor-adjusted alpha / robust Sharpe / FDR / stability / other candidate-level validation
```

**Hypothesis.** H0: `max_k E[f_k] <= 0`, where `f_k` is candidate `k`'s
performance over the benchmark: no candidate in the searched family truly
outperforms the benchmark. Rejecting H0 is evidence that at least one candidate
genuinely beats the benchmark after accounting for data snooping. The family
must be every candidate searched; one left out is a trial the test does not pay
for.

**Inputs and defaults.**

| Argument | Meaning | Default |
|---|---|---|
| `performance` | dates x candidates DataFrame, no NaN or infinite values | required |
| `benchmark` | a scalar, or a Series on exactly the index of `performance` | `0.0` |
| `mean_block` | expected stationary-bootstrap block length | `10` trading days |
| `n_boot` | bootstrap replicates | `1000` |
| `seed` | bootstrap RNG seed | `0` |

The significance level is 0.05. It is the convention for reading the result, not
an argument: the function returns the p-value and the caller compares it.

**Outputs.** A `RealityCheckResult` with `statistic` (the observed V), `pvalue`,
`best_candidate`, `mean_relative_performance` (per candidate) and
`bootstrap_statistics` (the empirical null distribution).

**Method.**

1. Relative performance is candidate return minus benchmark return:
   `f[k,t] = candidate_return[k,t] - benchmark[t]`.
2. The observed statistic is `V = max_k sqrt(n) * mean(f_k)`.
3. Recentre each candidate under the null:
   `f_centered[k,t] = f[k,t] - mean(f_k)`.
4. Resample the dates with the Politis–Romano stationary bootstrap, using the
   same resampled dates for every candidate column.
5. In each replicate, recompute all candidate means and take the maximum again,
   `V_boot = max_k sqrt(n) * bootstrap_mean_k`. These maxima are the empirical
   null distribution; no Normal approximation is assumed.
6. `p = (1 + count(V_boot >= V_obs)) / (n_boot + 1)`.

**Choosing `mean_block`.** `mean_block=10` is a practical default for the current
daily-return pipeline, not an automatically selected optimal block length.
Null-calibration tests show that autocorrelated data are more sensitive to block
length. Automatic block-length selection is still pending and is separate future
work.

**Evidence.** Null calibration is tested under IID returns, cross-candidate
correlation, AR(1), fat tails and GARCH; power is tested on the same cases with
planted positive signals (`tests/test_snooping.py`).

## What exists

| Module | Methods | Status |
|---|---|---|
| `synth`, `backtest` | `make_return_matrix`, `candidate_returns` | done |
| `evaluate` | `sharpe_variance` / `sharpe_test` (normal, non-normal, autocorrelation-robust), PSR, MinTRL, implied independent trials | done |
| `evaluate` | Holm, Benjamini–Yekutieli, Storey q-values | planned |
| `bootstrap` | stationary bootstrap index generation | done |
| `bootstrap` | automatic block-length selection | pending |
| `snooping` | White's Reality Check | done |
| `snooping` | Romano–Wolf step-down | planned |
| `cv` | purged k-fold with embargo (from the Alpha-GPT work), walk-forward | planned |
| `pbo` | CSCV / probability of backtest overfitting | planned |
| `evaluate` | Harvey–Liu haircut Sharpe | planned |
| `gate` | pre-registered acceptance decision, trial count from the ledger | planned |

Already in `evaluate`: `benjamini_hochberg`, `bonferroni`,
`deflated_sharpe_ratio`, `expected_max_sharpe`, and `false_discovery_rate` /
`power` for scoring a selection rule against known truth.
