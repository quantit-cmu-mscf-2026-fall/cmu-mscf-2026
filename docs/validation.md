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

**What counts as a trial:** every candidate whose performance anyone, or any
agent, has seen. That includes parameter variants, candidates that were
discarded, and every iteration inside an agent's search. An idea that never
produced a performance number is not a trial. Counts accumulate across rounds
of a search, because each round is chosen knowing the earlier results. The
deflated Sharpe ratio needs this full count and the spread of Sharpe ratios
across all of it (Bailey & López de Prado 2014).

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

Which method is the gate at each stage, and which only inform it:
[`validation/framework.md`](validation/framework.md) (proposed).

## Decisions the stages depend on

Proposed 2026-09-29. These are fixed before any agent generates candidates;
changing one later means re-running every candidate scored under the old one.

**Holdout period.** Language models can reproduce market history from before
their training cutoff, even when told not to (Glasserman & Lin 2023; Sarkar &
Vafa 2024). A holdout before the cutoff guards against search overfitting but
not against the model's memory, so there are two:

- *Pre-cutoff holdout:* the last 3 to 5 years before the agents' model cutoff,
  kept out of every agent run and every stage before the final one.
- *Post-cutoff holdout:* all data after the cutoff, which grows with each new
  month. This is the final test.

The exact dates depend on which model and version the agents run on; record
them here once that is fixed.

**Trading costs.** Candidates are scored net of costs from stage 1, with gross
returns reported alongside. Most published anomalies earn close to nothing
after bid-ask costs (Chen & Velikov 2023), so screening gross returns spends
the later stages on candidates that were never tradable. The cost model is an
estimated effective spread from CRSP times turnover, fixed before scoring.

**Stage 3 comparison set.**

- *Known factors:* Fama–French five factors plus momentum, from Ken French's
  library (`load_french`; the momentum file still needs adding).
- *Known signals:* the Chen & Zimmermann open-source anomaly portfolios (over
  200 published signals), plus any signal we already hold. An agent that has
  read the literature tends to rediscover published anomalies, and those earn
  about half as much after publication (McLean & Pontiff 2016). Adding value
  beyond this set is the test of novelty.

The anomaly portfolios are monthly, so stage 3 runs on monthly returns.

**Calibration range for the share of real candidates.** Start at 0 to 5% for
agent-generated candidates. Harvey & Liu (2020) use up to 20% for a curated
database of strategies already known to have worked, which is not our setting.
Update the range as planted signals and the ledger's history give base rates.

**Data.** Index membership as of each date, not today's members, and returns
that include delisting returns (Shumway 1997). Either mistake biases every
backtest upward, and no method in this document can correct it.

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
| `evaluate` | Harvey–Liu haircut Sharpe | planned |
| `search_fdr` | search-adjusted p-value for a hypothesis's best of K variants, independent or equicorrelated (`rho=`); search-adjusted FDR (López de Prado & Fabozzi 2026); max-of-mixture fit over assumed K, a sensitivity table, not a gate | done |
| `gate` | staged, pre-registered scorecard; trial count from the ledger; holdout looked at once | planned |

Already in `evaluate`: `benjamini_hochberg`, `bonferroni`,
`deflated_sharpe_ratio`, `expected_max_sharpe`, and `false_discovery_rate` /
`power` for scoring a selection rule against known truth.
