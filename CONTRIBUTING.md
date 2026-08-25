# How we work

This repository runs the way a production research team runs. None of this is
ceremony — every rule below exists because skipping it costs more than following
it. All of it is standard, public software practice.

## Ground rules

- **`main` is protected.** No direct pushes, no force pushes. Code lands
  through a pull request with green CI and at least one approving review.
- **`main` is always deployable.** If it's on `main`, it should be safe to run.
- **You own what you merge.** The agents write most of the code; a human is
  accountable for every line that lands. Review accordingly.

## What needs review, and what doesn't

Review exists to protect the thing that breaks when it's wrong — the code that
runs. It is not a gate on thinking out loud, and putting one there would only
teach everyone to think somewhere we can't see.

**Goes through a PR**: anything in `main` — code, tests, `deploy/strategies.yml`,
and the docs that describe how the code works.

**Post it yourself, no review**:

- **Paper rounds** — open an issue with the *Paper round* template. It's yours;
  edit it after the discussion and fill in the report-back the week after.
- **Topics and work items** — the *Topic* and *Work item* templates. The team's
  map of what this project is about is written by the team. Open topics when you
  have a direction, edit anyone's when the thread has moved past what it says,
  close the ones that went nowhere (with a line saying so).
- **The wiki** — notes, references, derivations, write-ups. Write freely.
- **Discussions** — questions, half-formed ideas, anything not yet a topic.

You do not need permission for any of the second group, and you don't need to
wait for us. We read all of it as participants.

## Branches and pull requests

- Branch from `main`, name it `<your-name>/<short-topic>`
  (e.g. `cal/purged-cv`, `lauren/ib-order-shim`).
- Keep branches short-lived — days, not weeks. Long-lived branches rot.
- One logical change per PR. Small PRs (roughly under ~400 changed lines) get
  reviewed fast and reverted cleanly; big ones get skimmed, and skimmed review
  is no review.
- Fill in the PR template. For research PRs, the **trial count** field is not
  optional: the number of configurations you tried is an input to every
  multiple-testing correction this project runs. Losing track of it in the repo
  means reconstructing it in week 12.
- Squash-merge. The PR title becomes the commit message — write it as one
  clear sentence in the imperative ("Add purged k-fold splitter").
- Delete the branch after merge (the repo is configured to do this for you).

## Review

- **A teammate reviews, and the role rotates.** Don't route every PR to the
  same person, and don't wait for the mentors — we read PRs as participants,
  not as gatekeepers.
- Reviewer's job, in order: (1) is the claim in the description actually
  supported by the diff and the results, (2) would a stranger be able to rerun
  this, (3) is the code clear enough to change later. Style nits come last;
  the formatter settles most of them.
- Target: first review within one working day. A blocked PR blocks a teammate.
- Disagreement is content, not friction — argue in the PR thread, decide,
  merge. If the thread stalls, bring it to the weekly sync.

## CI

- Every PR runs lint (`ruff check`, `ruff format --check`) and the test suite
  (`pytest -m "not network"`). Keep it green and keep it fast — if CI creeps
  past ~5 minutes, fixing that is a PR someone should open.
- A flaky test is a bug in the test. Fix or quarantine it the day it flakes;
  a suite people re-run until it passes teaches everyone to ignore red.
- New behavior ships with a test that fails without the change. If you can't
  demonstrate the test discriminates, it's decoration.

## Research hygiene

- Experiments are code: fixed seeds, parameters in files (not in your shell
  history), runnable by a stranger with one command.
- Results live in the PR description (numbers, small plots) — not as large
  binaries in git history. Anything over a few hundred KB belongs in a release
  asset or external storage, referenced by path.
- Prefer plain `.py` scripts over notebooks for anything meant to be re-run.
  If you commit a notebook, strip its outputs first.

## Secrets

- **Nothing secret enters this repository, ever.** It is public: a leaked
  token is compromised the moment it is pushed, and rewriting git history
  doesn't un-leak it.
- Paper-trading credentials live outside the repo (environment variables on
  the machine that runs execution; GitHub Actions secrets if a workflow needs
  them). The repo's secret-scanning push protection is enabled — treat a block
  as a save, not an obstacle.
- The paper account holds no real money, but its credentials are still
  credentials. Same discipline as production.

## Deploying to the paper account

The end-to-end leg follows the same promote-by-PR pattern as everything else:

1. **The deploy registry is a file in this repo** (`deploy/strategies.yml`).
   What runs on the Interactive Brokers paper account is exactly what that
   file says on `main` — nothing else.
2. **A strategy enters the registry by PR**, and that PR must link the
   validation artifact (the gate report for that strategy). No gate report,
   no deploy — the reviewer's job is to hold that line.
3. **Releases are tags.** Promoting `main` to the paper runner happens by
   tagging (`deploy-YYYY.MM.DD`); the runner only ever runs a tag, never a
   branch tip. That makes "what was live last Tuesday?" a lookup, not a
   reconstruction.
4. **Rollback is a revert.** Reverting the registry commit and re-tagging is
   the whole procedure. If rolling back feels scary, the deploy unit was too
   big.

## The short version

Small PRs, reviewed by a rotating teammate, squash-merged onto a protected
`main` that is always deployable; releases are tags; secrets never touch the
repo; and every research claim carries its seed and its trial count.
