# Git exercise — week 3

**Do this before Thursday, Sep 10.** Six drills, about 45 minutes total. Every one has a check you
can run yourself, so you never have to wonder whether you did it right.

The rulebook is `CONTRIBUTING.md`; the map is the wiki page *GitHub, A to Z*. This sheet is the
drill. It exists because the first two weeks produced nineteen commits from four of you, one merged pull request
and one review — not from lack of work, but because the loop had not been run end to end by each
of you yet.

---

## Drill 0 · Prove your commits are yours (5 min)

A commit carries whatever email your machine was configured with. If that address is not on your
GitHub account, the commit still lands, GitHub shows a plain name instead of your profile, and
your contribution history reads empty. This has already happened in this repository.

```bash
git config user.name
git config user.email
```

Set it to the address attached to your GitHub account:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@andrew.cmu.edu"
```

**Check**: `git log -1 --format='%an <%ae>'` on your last commit shows that address, and on
GitHub the commit shows your avatar rather than a grey placeholder.

If earlier commits used the wrong address, add that address to your GitHub account under
*Settings → Emails* — GitHub then links the old commits retroactively. Do not rewrite published
history to fix this.

---

## Drill 1 · Stop committing machine noise (5 min)

`.DS_Store` files are macOS Finder metadata. Five of them are currently tracked in this
repository. They are noise in every future diff and they make a two-line change look like a
six-file change.

```bash
git rm --cached .DS_Store '**/.DS_Store'          # untrack, keep on disk
printf '.DS_Store\n' >> .gitignore
```

**Check**: `git status` shows the deletions plus the `.gitignore` edit, and
`git ls-files | grep -c DS_Store` returns `0`. Ship this as part of Drill 3's pull request.

While you are there, know what else is deliberately ignored here and why:
`experiments/runs.jsonl` (the ledger — synced to the private archive, never to a public repo),
`.sessions/` (transcripts), `data_cache/`, `.venv/`.

---

## Drill 2 · Branch, and know where you are (5 min)

```bash
git switch main && git pull
git switch -c yourname/short-topic
```

Naming is `<your-name>/<topic>`, and branches are short-lived — days, not weeks.

**Check**: `git status -sb` first line reads `## yourname/short-topic...` and
`git log --oneline main..HEAD` is empty (you have not committed yet).

**The rule this drill is really about**: *work on a branch nobody can see is work the team does
not have.* There is currently a branch in this repository carrying a validation core — CSCV/PBO,
a Benjamini-Hochberg comparison, a two-phase ledger — that has never been in a pull request.
Nobody can review it, reuse it, or build on it. If you have local work older than three days,
that is your Drill 3.

---

## Drill 3 · One change, one pull request (15 min)

Make **one logical change**. If you need the word "and" to describe it, it is two pull requests.
Keep it under roughly 400 changed lines: a big PR gets skimmed, and a skimmed review is no review.

```bash
ruff check . && ruff format --check . && pytest -q -m "not network"   # what CI runs
git add -p                                                            # stage deliberately
git commit -m "Add a purged k-fold splitter"                          # imperative mood
git push -u origin yourname/short-topic
gh pr create --fill
```

Then **fill the template in**. Every heading in `.github/pull_request_template.md` is there
because someone needed the answer later:

- **What** — one sentence, the change.
- **Why** — the problem or question. Link the issue.
- **How it was tested** — the commands you ran. "CI is green" is not a test plan.
- **Results (research PRs)** — the numbers, the seed, and **how many configurations you tried**.

That last field is the one people delete. Do not. A Sharpe of 1.4 found on the first attempt and
a Sharpe of 1.4 found on the two-hundredth are different numbers, and the trial count is the only
thing that tells them apart. It is a direct input to `deflated_sharpe_ratio` in
`capstone/evaluate.py`.

**Check**: `gh pr view --json body -q .body` contains no `<!--` template comments, and
`gh pr checks` shows the CI job green.

---

## Drill 4 · Fold two pull requests into one (10 min, whoever owns duplicates)

Two open PRs currently add the same file. That is a merge conflict waiting to be handed to
whoever merges second. Fixing it is the most common piece of real git work you will do.

```bash
git switch main && git pull
git switch -c yourname/momentum-baseline-v2
git checkout <better-branch> -- research/momentum_baseline.py   # take the better version
# remove the .DS_Store files (Drill 1), run the checks, commit, push, open one PR
gh pr close <the-other-number> --comment "Superseded by #<new>, same change without the noise"
```

**Check**: one open PR for that file, and the closed one links to its replacement.

Closing your own pull request is not a failure. Leaving two open is.

---

## Drill 5 · Review somebody else's pull request (10 min)

You will review as often as you are reviewed, and the role rotates. There are open PRs right now
with zero reviews on them.

```bash
gh pr list
gh pr checkout <number>       # actually run it — do not review from the diff alone
gh pr review <number> --comment --body "..."
```

Three questions, in this order:

1. **Does the claim match the evidence?** If the description says "improves Sharpe", is the trial
   count consistent with that claim surviving a correction?
2. **Is there lookahead?** In this codebase the guard is the `.shift(1)` in `run_backtest`:
   positions are formed from *yesterday's* signal. Any path around that shift is the
   highest-value bug you can catch all semester.
3. **Would you understand this in six weeks?**

**Check**: your review appears under `gh pr view <number> --json reviews`.

Approve when you would be comfortable building on it. "Looks good to me" on a PR you did not read
is worse than silence, because it creates the record of a review that never happened.

---

## Drill 6 · Survive a conflict (5 min, optional but do it once)

Conflicts are routine; the first one is only alarming because it is the first.

```bash
git switch main && git pull
git switch yourname/short-topic
git rebase main
# conflict: open the file, delete the <<<<<<< ======= >>>>>>> markers, keep the right code
git add <file> && git rebase --continue
git push --force-with-lease        # your own branch only, never main
```

**Check**: `git log --oneline main..HEAD` shows your commits sitting on top of current `main`.

`--force-with-lease` rather than `--force`: it refuses if someone else pushed to your branch in
the meantime. On a shared branch, force-pushing without it destroys other people's work.

---

## Done when

- [ ] `git log -1 --format='%ae'` shows your GitHub address
- [ ] `git ls-files | grep -c DS_Store` returns 0
- [ ] One open PR of yours, template filled, trial count present, CI green
- [ ] One review written on somebody else's PR
- [ ] Any local branch older than three days is either in a PR or deleted on purpose

Bring the number of the PR you opened and the number of the PR you reviewed to Thursday.
