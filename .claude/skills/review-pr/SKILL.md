---
name: review-pr
description: Review a pull request, or the current branch before opening one, the way REVIEW.md says. Parallel finders, then every finding verified by a fresh agent; only confirmed issues are reported. Use when asked to review a PR, or as the self-review step before opening one.
---

# Review a PR

Input: a PR number, or nothing to review the current branch against `main`.

1. **Gather.** Get the title, description, base, head commit, linked issue and
   earlier reviews and replies (`gh pr view <n> --comments`), and the diff
   against the base. A stacked PR's base is the PR below it. On a re-review,
   take only the diff since the commit last reviewed, plus the findings
   already raised. Read `CLAUDE.md` and `REVIEW.md`, and note whether any
   path under REVIEW.md's "Risk" changed.
2. **Run.** In a scratch worktree at the head, merged with the current base,
   run `pytest -q -m "not network"` and `ruff check . && ruff format --check .`.
   If pandas-sensitive code changed, also run its tests under pandas 3.
   Record the results and the commit.
3. **Find.** Launch three `pr-reviewer` agents at once on the same diff and
   worktree, one lens each:
   - future and search: REVIEW.md items 1, 2 and 5;
   - data and tests: items 3, 4 and 8, and whether the new code does what its
     docstring says;
   - claims and leaks: items 6 and 7, and whether the description's claims
     hold (its numbers, "fails without the change").
4. **Verify.** Merge duplicates. For each remaining finding, launch a fresh
   `pr-reviewer` to refute it. Keep only `CONFIRMED` findings; check
   Pre-existing ones against `main`.
5. **Write.** Start with the verdict (Approve, Request changes or Comment).
   Then give what was run and on which commit, and the Important findings,
   most severe first, each with `file:line`, a one-line repro and a fix. Add at
   most three Nits and one line for Pre-existing issues. Don't summarise the
   diff or list what is fine.
6. **Post only when the user says so** (`gh pr review <n> --approve`,
   `--request-changes` or `--comment`, with `--body-file`). For the current
   branch, fix the Important findings before opening the PR instead.

If this would be the fourth round on the PR, say so and suggest discussing
the approach rather than reviewing lines again (REVIEW.md, "When a review
stops"). A round is one submitted review, from anyone, that requested changes
or commented: count the `CHANGES_REQUESTED` and `COMMENTED` entries in
`gh pr view <n> --json reviews`.
