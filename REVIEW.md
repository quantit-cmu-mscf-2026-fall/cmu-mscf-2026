# What a review must catch here

Read by reviewers, human or agent (the `review-pr` skill). `CLAUDE.md` has the
conventions; `docs/code_review.md` teaches the method. This page says what
blocks a merge, what doesn't, and when a review stops.

## What breaks here

Ordered by cost. Any of these is **Important**.

1. **It sees the future.** A position not lagged by `shift(1)`, returns
   computed by hand instead of through `backtest`, fundamentals used before
   `sd.available_from`, CV without purging and an embargo, or search code that
   can read 2021-2025.
2. **It misstates the search.** A run not logged with `log_run` before its
   result is used, a trial count typed in rather than read from the ledger,
   the best of N reported as one test, or a hypothesis family that undercounts
   the variants searched.
3. **It can't say "nothing here".** A validation method shipped without a
   null-calibration test and a power test on `synth.make_return_matrix` with
   `rho` and `ar1`, or a test that passes for any implementation.
4. **It invents or hides data.** `fillna(0)` on returns, idle days counted as
   observations, a bare `except`, rows or columns dropped silently, a NaN or
   inf turned into a p-value.
5. **It hand-rolls validation.** A correction or significance test inside
   strategy code instead of the shared methods in `docs/validation.md`, or
   validation calling strategy code.
6. **It leaks.** Credentials, account identifiers, CRSP- or Compustat-derived
   files, the ledger, `factors.db` or session logs in the diff. The repo is
   public.
7. **It contradicts its docs.** A docstring, README or `docs/validation*` claim
   the code doesn't meet, or a doc change that overwrites a shared page.
8. **It breaks on CI.** CI installs the latest pandas (3.x), which differs from
   2.x on all-NaN groups, copy-on-write and string dtypes.

Past catches, for scale: #55 decided idle days from the weights rather than
the signal; #64 dropped winners that went flat out of sample, lowering PBO;
#54 crashed on a p-value of 0; #67 gave an infinite statistic p = 1/(B+1);
#68 failed on pandas 3; #75 replaced `docs/validation.md`; a merge into #50
duplicated a section without a conflict.

## Severity

- **Important:** an item above, or a bug with a concrete failing input. Blocks.
- **Nit:** worth one line, never blocks. At most three per review. No
  formatting comments; ruff settles those.
- **Pre-existing:** real but not introduced by this diff. Mention it once,
  don't block, and open an issue.

Name a defect class when there is one ("every `idxmax` over a group that can
be all-NaN"), so the author fixes every instance.

## Evidence

No finding without `file:line` and either a repro (input, observed, expected)
or a quoted line it contradicts. A guess from a function's name is not a
finding. Say what you ran and on which commit.

## Risk

Every PR needs one approving review. When a PR touches the paths below, that
review can't be a read-through: check out the branch, run the tests on it
merged with its base, and try a repro. Tag the method's owner
(`docs/validation/handoff.md`) so they see it.

- `capstone/`: `evaluate`, `search_fdr`, `calibration`, `cv`, `backtest`,
  `runlog`, `shared_data`, `factors/evaluation`
- `.github/`, `.claude/`, `scripts/capture/`

An approval covers one commit. After new commits, re-review the new diff.
After merging `main` into a stacked branch, diff it against `main` and check
for duplicated sections before merging.

## When a review stops

- Re-review only what changed since the commit you last reviewed. Don't
  repeat a finding already raised.
- Every Important finding gets an answer: a fix, with its commit, or a
  reasoned dispute.
- After three rounds on one PR, stop reviewing lines and raise the approach
  in the thread or at the weekly sync.

## Testing the reviewer

`review-practice/` is the reviewer's test set. When a bug reaches `main` or a
review misses one, add it there as a case, or as a line under "What breaks
here". Before changing the `review-pr` skill or the `pr-reviewer` agent, run
it on every case; it must still catch each one.
