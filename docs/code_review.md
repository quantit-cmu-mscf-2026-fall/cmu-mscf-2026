# Reviewing a Teammate's Pull Request

**Audience**: capstone students, Group 008
**Status**: student handout, September 2026

Everything below is about one job: reading code somebody else wrote and saying something
useful about it. It takes fifteen minutes and it is the only part of this project that you
cannot do alone.

---

## 1 · What a review is, and what it is not

A review is **not** permission. Nobody is checking whether you were allowed to write the
code. A review is a second pair of eyes on a claim.

Here is the asymmetry that makes it work. When you write code you know what you *meant*,
so you read the intention instead of the text. Your teammate does not know what you meant,
so they can only read what is actually there. That is why they find things you cannot, and
why you find things in their work that they stared past for an hour.

Three consequences worth internalising:

- **The reviewer is not a gatekeeper.** You are not defending the repository against your
  teammate. You are the first reader of a result that the team is about to rely on.
- **"Looks good to me" on code you did not read is worse than nothing.** It creates the
  record of a review that never happened, and the next person trusts that record.
- **Reviewing is how you learn what to distrust.** Especially now, when most of the code is
  written by an agent. Reading five agent-written diffs a week teaches you the failure
  patterns faster than writing twenty yourself.

## 2 · Who reviews, and when

Our `CONTRIBUTING.md` already answers this: *a teammate reviews, and the role rotates.*
Mentors read pull requests as participants, not as gatekeepers. What it never said is whose
turn it is, so for four weeks it has been nobody's. From this week there is an order.

**The ring**: you review the next person's pull request. It shifts by one each week, so you
never review the same author twice running.

```
Vinson  ->  Liao  ->  Chen  ->  Yang  ->  Cui  ->  Vinson
```

**Target**: a first response within one working day. A blocked pull request blocks a
teammate, and the whole point of small pull requests is that they move.

If your reviewer has gone quiet for more than a day, bump the pull request in the channel and
tag them. That is not nagging. That is the process working exactly as designed.

## 3 · The loop, step by step

This is the whole mechanic. Run it once and it becomes muscle memory.

**Step 1 — Get the code onto your machine.** Reading a diff in the browser is fine for a
two-line change. For anything real, check it out:

```
gh pr list
gh pr checkout 12
```

**Step 2 — Read the description before the diff.** What does the author claim this does? If
the description is still the template text, that is your first comment, and it is a real one:
a pull request body nobody wrote is a pull request nobody re-read, including its author.

**Step 3 — Run the tests yourself.** Do not trust the green check alone; you want to see it.

```
pytest -m "not network"
ruff check .
```

**Step 4 — Now read the diff, with the three checks in section 4.** Read the test file first.
Tests tell you what the author believed the code should do, which is usually clearer than the
code.

**Step 5 — Leave comments on the lines they belong to.** Not one summary comment at the
bottom. A line comment is anchored to the thing you are talking about, so the author does not
have to guess.

**Step 6 — Give a verdict.** Approve, request changes, or comment. Section 6 says which.

**Step 7 — The author merges.** Squash-merge, imperative title, delete the branch. The
reviewer does not merge someone else's work.

## 4 · The three checks, in this order

Order matters. The first one catches the expensive mistakes; the last one catches the cheap
ones. Reviewers who start with style never get to the top of the list.

**Check 1 — Does the claim match the evidence?**

If the description says "improves Sharpe to 1.9", ask how many things were tried before this
one. A Sharpe of 1.9 from a single pre-registered test and a Sharpe of 1.9 from the best of
forty sweeps are different claims, and only one of them survives a correction. Our repository
has the tools that settle it: `expected_max_sharpe(n_trials, n_obs)` tells you what the best
of `n_trials` nulls would have produced anyway, and `log_run(...)` is where the trial count
comes from. If the run was never logged, the trial count is unknown, and "unknown" is not the
same as "one".

**Check 2 — Is there lookahead?**

This is the single highest-value bug you can catch in this codebase, because it is invisible
in the output. It does not crash. It just makes everything look wonderful.

The guard lives in `_backtest_components`, which `run_backtest` calls:

```
weights   = to_weights(signal, demean=demean, gross=gross)
positions = weights.shift(1)
```

That `.shift(1)` is what makes today's position come from *yesterday's* signal. Any code path
that multiplies weights by same-day returns without it has imported the future. Watch for a
diff that computes returns by hand instead of calling `run_backtest` — that is how the guard
gets bypassed, and it usually looks like a harmless refactor.

**Check 3 — Would this be readable to you in six weeks?**

Not "is it clever". Could a teammate change it in November without archaeology? Names,
one obvious path through the function, a docstring that says why rather than what.

Style nits come after all three, and the formatter settles most of them anyway.

## 5 · Writing a comment somebody can act on

A useful comment does three things: it points at a specific line, it says what is wrong or
unclear, and it leaves the author a move. Compare:

| Weak | Useful |
|---|---|
| "This looks wrong" | "Line 34 multiplies weights by same-day returns. Should this go through `run_backtest` so it picks up the `shift(1)`?" |
| "Add tests" | "Is there a case where `signal` is all-NaN for a date? `to_weights` returns all-zero there, and I could not tell from the test whether that is intended." |
| "LGTM" | "Ran `pytest -m 'not network'`, green. Read the diff. The claim in the description is a single test, not a sweep, so the number stands. Approving." |

Two habits that pay for themselves:

- **Ask rather than assert when you are not sure.** "Should this...?" costs you nothing if you
  are wrong and does not put the author on the defensive if you are right.
- **Say which ones are blocking.** A reviewer who marks nothing as blocking, and a reviewer who
  marks everything, are equally useless. Two or three real ones is a good review.

## 6 · The three verdicts

| Verdict | Use it when | In practice |
|---|---|---|
| **Approve** | You would be comfortable building on this | Green tests, claim supported, nothing that will bite in six weeks |
| **Request changes** | Something must change before this lands | Lookahead, a claim the diff does not support, a test that cannot fail |
| **Comment** | You have questions but no blockers | You read it, you are not the right person to judge one part, say so |

Approving is a statement about your own confidence, not a favour. Request changes is not an
insult. Disagreement is content: argue in the thread, decide, merge. If it stalls, bring it to
the weekly sync.

---

# Practice: five cases

Do these **on your own** before the sync, then compare. That is the point: five people
reviewing the same diff produce five different lists, and the gap between your list and
someone else's is the actual lesson.

For each case, write down two things: (a) what you would comment, and (b) the check that would
prove you right. Then read the answer key at the end.

## Case 1

A pull request titled *"Speed up the momentum backtest"*. The diff:

```
-    rets = run_backtest(signal, returns, cost_bps=5.0)
-    sharpe = rets.mean() / rets.std() * np.sqrt(252)
+    w = to_weights(signal)
+    rets = (w * returns).sum(axis=1)
+    sharpe = rets.mean() / rets.std() * np.sqrt(252)
```

The description says: *"Same result, avoids the overhead of the full backtest path. Sharpe went
from 0.9 to 2.4, so the old path was also dropping data somewhere."*

## Case 2

A pull request titled *"20-day momentum: Sharpe 1.87 on the sample panel"*. The diff adds one
file, `research/momentum_sweep.py`, which loops over lookbacks 5, 10, 15 … 120, runs a backtest
for each, and prints the best one. There is no call to `log_run`. The description reports the
1.87 and nothing else.

## Case 3

A pull request titled *"Add tests for the weighting function"*. The diff:

```
def test_to_weights_gross():
    w = to_weights(SIGNAL, gross=1.0)
    assert w.abs().sum(axis=1).max() >= 0
```

CI is green.

## Case 4

A pull request titled *"Handle missing prices"*. The diff:

```
-    returns = prices.pct_change()
+    try:
+        returns = prices.pct_change().fillna(0.0)
+    except Exception:
+        returns = pd.DataFrame(index=prices.index, columns=prices.columns)
```

The description says: *"Some tickers have gaps, this stops the backtest from crashing."*

## Case 5

Two open pull requests from the same author, three days apart. Both add
`research/momentum_baseline.py`; the second version is six lines longer. Both also add five
`.DS_Store` files. The first one's description is still the template text with the headings
unfilled. CI is green on both.

---

## Answer key

Read this only after you have written your own two lines for each case.

**Case 1 — lookahead, and the Sharpe is the evidence.** The new code multiplies same-day
weights by same-day returns; the `.shift(1)` in `_backtest_components` is gone. The jump from
0.9 to 2.4 is not a bug fix, it is the size of the look-ahead. Comment: *request changes,* route
it back through `run_backtest`. **The check**: run both versions on a signal that cannot
predict anything — `make_panel` with zero true signals, or shuffle the signal's dates. The
guarded path gives roughly zero. The new one will not.

Worth saying out loud: a large unexplained improvement is a defect signal, not a result. This
is the case where "the number got better" is exactly what should worry you.

**Case 2 — the claim outruns the evidence.** 1.87 is the maximum of about 24 trials, and the
description reports it as if it were one. `expected_max_sharpe(24, n_obs)` tells you what the
best of 24 pure nulls would have produced; if 1.87 is not comfortably above that, there is no
finding here. And because nothing called `log_run`, the trial count is not recorded anywhere,
so the next person cannot even reconstruct it. Comment: *request changes,* log the runs and
report the number against the multiple-testing correction. **The check**: run the same sweep on
`make_panel(n_true=0)` and see what the best lookback scores. That is your null.

**Case 3 — a test that cannot fail.** An absolute sum is never negative, so the assertion is
true no matter what `to_weights` does. Green CI here means nothing. Comment: *request changes,*
assert the actual contract — that each row's absolute weights sum to `gross`, and that an
all-NaN row comes back all-zero. **The check**: the discrimination test. Break the function on
purpose (return `signal` unchanged) and re-run. A test that stays green while the code is
broken is decoration.

**Case 4 — two silent failures in four lines.** `fillna(0.0)` on returns does not handle missing
prices, it invents a flat day, and a flat day is a tradeable observation the backtest will
happily use. The bare `except Exception` then converts any real error into an empty frame that
propagates quietly. Comment: *request changes,* drop missing observations rather than filling
them, and let the exception raise. **The check**: count how many filled zeros the change creates
on the sample panel. If it is thousands, you are no longer backtesting the data you think.

**Case 5 — process, not code.** Nothing here is a numerical bug, and all of it costs the team.
Two pull requests for one change means the reviewer does not know which is real. `.DS_Store` is
your operating system's clutter, not the project's. A description left as template text means
nobody re-read the change, including the author. Comment: keep the later one, close the other,
add `.DS_Store` to `.gitignore`, fill in the description. **The check**: `git log --stat` on both
branches, and open the two files side by side.

---

## What to bring to the sync

One line per case: what you would have said. Where your answer and someone else's differ, that
gap is the fifteen minutes worth having.
