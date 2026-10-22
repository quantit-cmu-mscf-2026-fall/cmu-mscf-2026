# Validation handoff: start here

The validation framework is being built by five people in parallel. This page
is what every one of them, and every agent working for them, needs before
starting a task. Read it, then [`../validation.md`](../validation.md) (the
contract), then your workstream brief below.

Tasks are tracked on the team Jira board under the **Validation** epic
(QUANTIT-6). Each task's description has a checklist and a "Done when" line;
this page and the briefs hold the context the tickets don't.

## Who owns what

| Workstream | Owner | Jira | Brief |
|---|---|---|---|
| Merge the #28–#32 stack; local FDR; acceptance gate | Cal Vinson | QUANTIT-23 to 36 | (this page) |
| Stationary bootstrap, Reality Check, Romano–Wolf, SPA | Carl Cui | QUANTIT-37 to 42 | [bootstrap-and-snooping.md](bootstrap-and-snooping.md) |
| Purged k-fold, walk-forward | Lauren Liao | QUANTIT-43 to 45 | [cross-validation.md](cross-validation.md) |
| PBO (CSCV); Harvey–Liu haircut Sharpe | Pin-Hua Chen | QUANTIT-46, 12, 47, 51, 52 | [pbo-and-haircut.md](pbo-and-haircut.md) |
| Incremental value over factors and accepted signals | Eunice Yang | QUANTIT-48 to 50 | [incremental-value.md](incremental-value.md) |

The workstreams are independent of each other. Each builds its own module and
plugs into the same input; the only shared dependency is the stack below.

## Where the code is today

The validation code lands in two ways. For what has merged since this page
was written, check the
[pull request list](https://github.com/quantit-cmu-mscf-2026-fall/cmu-mscf-2026/pulls?q=is%3Apr).

**A stack of PRs.** Each one's base is the branch of the PR below it, and a
reviewed follow-up merges into the branch it builds on, so the top branch,
`vincal848/graded-evidence`, has all of it. The stack reaches `main` from the
bottom up:

| PR | Branch | Adds |
|---|---|---|
| #28 | `vincal848/validation-harness` | `backtest.candidate_returns`, `synth.make_return_matrix`, `docs/validation.md`; with #29 merged in, `sharpe_variance` / `sharpe_test` (HAC), PSR, MinTRL |
| #31 | `vincal848/fdr-dependence` | `holm`, `benjamini_yekutieli`, `storey_qvalues` |
| #32 | `vincal848/graded-evidence` | adjusted p-values (`holm_adjusted`, `by_adjusted`, `bh_adjusted`), `evidence_profile`, the stages of the calibrated validation funnel; with #59 merged in, the Sharpe follow-ups from #29's review |

**Separate PRs against `main`.** Each one's reviewer is the person who builds
on it. Already on `main`: Lauren's purged k-fold CV, `cv.py` (#34); the
shared-data loader, `shared_data.py` (#26); and the backtest's first-period
fix (#30). Still open when this was written:

| PR | Adds | Reviewer |
|---|---|---|
| #33 | `runlog.read_entries()`, `runlog.trial_count()` | Pin-Hua |
| #35 | CSCV / probability of backtest overfitting (`pbo.py`) | Pin-Hua |

So until the stack reaches `main`, neither branch has everything: the stack
branch has `make_return_matrix` and the Sharpe and multiple-testing code but
not `cv.py` or `shared_data.py`, and `main` has those but not the stack.

**Which branch to start from:**

- Your work uses the stack (`make_return_matrix`, `sharpe_test`, the adjusted
  p-values): branch from `origin/vincal848/graded-evidence` and set your PR's
  base to that branch. Once #32 is in `main`, change the base to `main` and
  rebase.
- Your work needs only what's on `main`: branch from `origin/main`.
- It needs both (for example `cv.py` with `make_return_matrix`): branch from
  `origin/vincal848/graded-evidence` and run `git merge origin/main`. If git
  reports a conflict in `tests/test_synth.py`, both sides added tests at the
  end of the file; keep both. Until the stack is on `main`, your PR's diff
  against the stack branch also shows `main`'s changes, so say so in the PR
  description.

Name branches `<name>/<topic>`, and put the Jira key in the PR title (for
example `QUANTIT-37: stationary bootstrap resampler`).

## The contract, in code

Everything below is on `vincal848/graded-evidence`.

**Input.** Every method takes a performance matrix: a `pd.DataFrame`, dates x
candidates, one per-period series per candidate, higher is better. Methods
that work on p-values take a `pd.Series` indexed by candidate name. Validation
never calls strategy code; strategies hand it the matrix.

**Output.** A method that scores candidates returns a `pd.Series` indexed by
candidate name (an adjusted p-value, a probability of being real, an alpha
p-value), so it can sit next to the columns of `evaluate.evidence_profile`.

**Test data with known truth:**

```python
from capstone.synth import make_return_matrix

m = make_return_matrix(
    n_obs=2520, n_candidates=200, n_real=20,   # n_real=0 -> pure null
    sharpe_real=1.0, vol=0.01,
    rho=0.5,            # pairwise correlation via one common factor
    ar1=0.3,            # autocorrelation
    t_df=5,             # fat tails
    garch=(0.05, 0.9),  # volatility clustering
    seed=0,
)
m.returns  # DataFrame, dates x candidates
m.truth    # bool Series: True = planted
```

Every option keeps each column's mean and volatility fixed, so null candidates
stay exactly null with every option on.

**Reuse these; don't write your own:**

| Need | Use (`capstone.evaluate`) |
|---|---|
| Sharpe test / p-value | `sharpe_test(returns, method="hac")` returns `.sharpe`, `.se`, `.pvalue`, `.pvalue_greater` (one-sided), `.psr` |
| Sharpe standard error, Newey–West lag | `sharpe_variance(..., method="hac")`, `newey_west_lags(n_obs)` |
| Multiple testing, yes/no | `holm`, `benjamini_yekutieli`, `benjamini_hochberg`, `bonferroni` |
| Multiple testing, graded | `holm_adjusted`, `by_adjusted`, `bh_adjusted`, `evidence_profile` |
| Scoring a rule against known truth | `false_discovery_rate(rejected, truth)`, `power(rejected, truth)` |
| Correlation between candidates | `average_correlation`, `implied_independent_trials` |
| Deflated Sharpe | `deflated_sharpe_ratio`, `expected_max_sharpe` (already on `main`) |

## The rule every method follows

From [`../validation.md`](../validation.md): every method ships with

1. a **null-calibration test**: on data with no real candidates, it says
   "nothing here" at the rate it promises; and
2. a **power test**: on data with planted candidates, it finds them.

Both run on correlated (`rho`) and autocorrelated (`ar1`) nulls, not only IID.
`tests/test_evaluate.py` has the pattern to copy (`TestSharpeVarianceCalibration`
and the multiple-testing tests):

- Measure a rate over many seeds or many candidates, not one draw.
- Set the tolerance from the sampling error, and **check it across 50–200
  seeds before committing**, not only the committed seed. In #28 one test
  passed at seed 0 by luck and failed on others.
- Write down in the test's docstring or comment what range you measured.
- Keep the suite fast. Put long sweeps in a script and report the numbers in
  the PR description.

## The data split

Real-data runs use the fixed CRSP data, 1990–2025 (the shared-data loader on
`main`). Search, including every fit, screen and choice of parameters, uses
**1990–2020** only. **2021–2025 is the holdout**: never used in search, and
looked at once, at the final decision.

## What we already know (don't re-learn it)

- **Storey q-values are unsafe for correlated candidates.** On all-null sets
  with pairwise correlation 0.5, they make a discovery in 10–14% of sets at a
  nominal 5%, because the estimate of the share of nulls collapses when
  candidates move together (#32). Use BY or Holm when candidates are correlated.
- **The IID Sharpe standard error is wrong under autocorrelation.** On AR(1)
  nulls with coefficient 0.3, the "nonnormal" variance rejects about 15% of the
  time at a nominal 5%. Use `method="hac"`; it is much closer but still a little
  liberal in finite samples (#29).
- **Screen on one-sided p-values** (`pvalue_greater`). Only positive Sharpe
  ratios are discoveries, and BH's guarantee under positive correlation covers
  the one-sided case (#29, #32).
- **PBO near 0.5 is the no-skill baseline, not a clean bill of health.** With
  independent no-skill candidates the in-sample winner's out-of-sample rank is
  uniform. PBO climbs toward 1 only when candidates are coupled, such as variants
  of one idea. Measured on `make_return_matrix` nulls: about 0.47 on average,
  but anywhere from 0.06 to 0.78 for a single set
  ([pbo-and-haircut.md](pbo-and-haircut.md)).
- **Correct p-values with the ledger's registered trial count as it is.**
  Holm, BY, the haircut Sharpe and the search adjustment take the full count.
  Don't shrink it to an "effective" number of independent trials to allow for
  correlation: #51's review found that anti-conservative for the search
  adjustment (size 0.058–0.071 at a nominal 0.05). Allow for correlation in
  the method instead (BY, Holm, or a bootstrap that keeps the correlation
  between candidates).
- **The deflated Sharpe is the one exception, as the code says.**
  `expected_max_sharpe` and `deflated_sharpe_ratio` count independent trials
  and accept a fractional count from `implied_independent_trials` (Bailey &
  López de Prado's rough correction). If you pass one, report the ledger count
  next to it, and use the full ledger count for the final decision
  (`docs/validation.md`, stage 4).

## Trial counts and the ledger

Every correction depends on how many candidates were tried. That number comes
from the run ledger, `capstone.runlog`, never from a number typed into a call.
Log every trial with `runlog.log_run(...)` **before** looking at its result,
and read the count with:

```python
from capstone.runlog import trial_count
m = trial_count()               # every logged trial
m = trial_count("mom-sweep")    # only trials logged under one experiment name
```

It raises `LookupError` instead of returning 0 when nothing matches, because
a correction against zero trials applies no correction. It arrives in its own
small PR, #33, against `main` and separate from the stack. If it isn't on
your branch yet, the command line gives the same number:
`python -m capstone.runlog stats`. Tests must set `CAPSTONE_LEDGER_DIR` to a
temporary directory and never touch `experiments/runs.jsonl`.

## Avoiding merge conflicts

- Put new methods in **your own module** (`bootstrap.py`, `snooping.py`,
  `cv.py`, `pbo.py`, `spanning.py`) with its own test file.
- `evaluate.py` is shared by local FDR (Cal) and the haircut Sharpe (Pin-Hua).
  Add new functions at the end of the file, in a section of their own.
- In `docs/validation.md`, change only your own row of the "What exists"
  table (from "planned" to "done") and add your notes under a heading of your
  own. Rows conflict trivially; keep both sides.

## Before opening a PR

From `CLAUDE.md` and `CONTRIBUTING.md`:

- `pytest -q -m "not network"` green; `ruff check . && ruff format --check .` clean.
- New behavior ships with a test that fails without it.
- Small PRs, one logical change (about 400 changed lines is the guideline).
- Fill in the PR template, including seeds and trial counts for any experiment.
- A teammate reviews; squash-merge.
- This repository is public: no credentials, no data files, no ledger.

## Working with an agent

Point it at this page, your brief, `docs/validation.md` and the Jira task's
checklist, then give it one task at a time. `CLAUDE.md` already tells agents
to use the shared validation methods and not to write their own corrections.
