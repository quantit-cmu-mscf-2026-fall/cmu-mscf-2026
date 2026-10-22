# Brief: cross-validation (purged k-fold and walk-forward)

**Owner:** Lauren Liao. Read [handoff.md](handoff.md) first.

## Why this workstream matters

When a strategy fits a model or picks parameters, its reported performance
only counts if it was measured on dates it wasn't fitted on. With labels that
span several days, and features that are correlated over time, ordinary k-fold
leaks the test period into training. This workstream provides the splitters
every strategy should use. Their out-of-sample results become the performance
matrix that the rest of validation judges.

Note the direction of the contract: CV is a tool strategies call to *produce*
the matrix. Validation methods never call strategy code or CV splitters.

## Tasks

| Jira | Task | Blocked by |
|---|---|---|
| QUANTIT-43 | Review PR #34 (your `cv.py`, brought to `main`) | done: #34 is on `main` |
| QUANTIT-44 | Purged k-fold: leakage tests on the performance matrix | nothing, but needs `cv.py` (`main`) and `make_return_matrix` (#28, on the stack) together; see below |
| QUANTIT-45 | Walk-forward splitter and CV docs | nothing (`cv.py` is on `main`) |

## Most of this already exists: `cv.py` on `main`

Your `capstone/cv.py` and `tests/test_cv.py` (16 tests), from commit
`f94b6d9` on `laurenli/alpha-gpt-replication`, merged to `main` in **#34**,
unchanged and with you as the commit author. It has:

- `label_end_times(dates, horizon)`
- `PurgedKFold(n_splits=4, *, horizon=1, embargo_pct=0.01)` with
  `.split(dates)`, which yields `(train_dates, test_dates)` with purging and an
  embargo never shorter than `horizon`
- `average_uniqueness(t1, dates)` sample weights (López de Prado 2018, ch. 4)

**Checked 2026-09-27, and again in the review of this brief.** The full suite
passes on `main` with #34, and also on the validation stack (#32).
`PurgedKFold.split` works directly on the index of a `make_return_matrix`
matrix (a `DatetimeIndex`). With 5 folds on 1,000 dates and horizons of 1, 5
and 21, no training label's return window, (t, t1], overlaps any test label's
(0 overlaps), and the embargo band after every test fold is excluded from
training (0 violations).

So:

- **QUANTIT-43 is done.** No code changes were needed to fit the input format.
- **QUANTIT-44:** the splitter itself is done. What's left is a leakage test
  written against the performance matrix, over several horizons and fold
  counts, like the check above.
- **QUANTIT-45 (walk-forward) is the real new work.** Nothing for it exists yet.
  Build it in `cv.py` on a branch off `main`.

## Getting `cv.py` and `make_return_matrix` together

Until the validation stack reaches `main`, they live on different branches:
`cv.py` is on `main`, and `make_return_matrix` is on the stack (#28, and so on
`vincal848/graded-evidence`). QUANTIT-44 needs both, and so does the optional
null test below. Branch from `origin/vincal848/graded-evidence` and merge
`main` into it:

```
git switch -c laurenli/<topic> origin/vincal848/graded-evidence
git merge origin/main
```

If git reports a conflict in `tests/test_synth.py`, both sides added tests at
the end of the file; keep both. Set the PR's base to
`vincal848/graded-evidence`, and say in the description that its diff also
shows `main`'s changes until the stack is merged. See
[handoff.md](handoff.md) for the general rule.

When checking leakage yourself, use the same convention as `cv.py`: a label at
date t covers the returns in (t, t1], so a training label ending exactly on the
first test date does **not** overlap. Treating the ends as closed reports one
false overlap per fold boundary.

If `laurenli/alpha-gpt-replication` merges later, its `cv.py` is identical
to `main`'s and won't conflict. #28 also made `backtest.backtest_components`
public under the same name your branch uses.

## Design notes

- **Split on the matrix's dates.** The splitters take a `DatetimeIndex`, which
  is the index of the performance matrix, so any strategy that produces the
  matrix can use them.
- **Walk-forward** (QUANTIT-45) is new: rolling and expanding training windows,
  an optional gap between train and test, and every training date strictly
  before every test date.
- **Keep CV separate from PBO.** Pin-Hua's CSCV in `pbo.py` does its own
  combinatorial splitting for a different purpose (whether choosing the best
  in-sample candidate is overfit). Don't merge the two.
- The docs task should say when to use which: purged k-fold when there isn't
  much data and the model has no memory of time; walk-forward when the strategy
  must only ever train on the past, or to simulate how it would be refitted
  live.

## Tests to write

- Leakage property tests: no training date's label window overlaps any test
  label window, across several `horizon` and `n_splits` values.
- Embargo: the dates right after each test fold are excluded from training.
- Walk-forward: every training date is before every test date, gaps are
  respected, windows move forward.
- Optional null test: choosing the best of many null candidates in-sample with
  the splitter, then scoring it out of sample, gives an out-of-sample Sharpe
  near zero on `make_return_matrix(n_real=0)` data.

## Who uses your output

Every strategy that fits anything, including what the discovery pipeline's
formula screening and strategy combination graph produce. Their out-of-sample
series feed the performance matrix.
