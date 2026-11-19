---
name: research-pr
description: How to prepare, check and describe a pull request to the public cmu-mscf-2026 repo - branch naming, one logical change, tests and ruff, the PR template with seeds and ledger trial count, review rotation and squash merge, and what must never be in the diff. Use when opening, writing up or reviewing a PR, or finishing an experiment that will be reported.
---

# Research PR

Full workflow: `CONTRIBUTING.md`. Template: `.github/PULL_REQUEST_TEMPLATE.md`.

## Before opening

1. Branch `<name>/<topic>`, short-lived, one logical change. If the description needs
   "and", split it.
2. New behavior has a test that fails without it.
3. `pytest -q -m "not network"` green; `ruff check . && ruff format --check .` clean.
4. Diff hygiene (the repo is public):
   - `git diff --stat origin/main...` shows no `runs.jsonl`, `*.parquet`, `data_cache/`,
     `.sessions/`, notebook outputs, images over a few hundred KB, or course material.
   - `git diff origin/main... | grep -iE "token|secret|password|api[_-]?key|wrds"` finds
     nothing that is a value.
5. Experiments: seeds fixed in a parameter file; every trial in the ledger; the count from
   `runlog.trial_count(name=... or include_tags=[...], exclude_tags=["synthetic"])`.

## The description

Fill every section of the template:

```markdown
## What
<one change, one sentence>

## Why
<question answered or problem fixed; link issue / Jira QUANTIT-nn>

## How it was tested
<commands run, tests added; "CI is green" is not a plan>

## Results (research PRs)
- Series: <returns net of quoted_spread | rank IC>, period <start>-<end> (search only)
- Data version: <sd.data_version()>
- Seeds: <list>
- Trial count: <n> from `trial_count(<name or tags>, exclude_tags=["synthetic"])`, <m> total in ledger
- Gate: <stage gate>; graded scores: <evidence profile>
- <key numbers or one small plot>

## Checklist
<template boxes, ticked only if true>
```

Results live in the description, not in committed files.

## Review and merge

- Request the rotating reviewer, not the same person each time; validation PRs go to the
  person who builds on them (`docs/validation/handoff.md`).
- Squash merge after green CI and one approval. Never `git push --force` to a shared branch;
  `--force-with-lease` on your own PR branch is fine.
- Deploy changes (`deploy/strategies.yml`) follow `CONTRIBUTING.md`, "Deploying to the paper
  account", and are tagged `deploy-YYYY.MM.DD`.

## Reviewing someone else's PR

Run `/review-pr` (it launches the `pr-reviewer` agents, see `REVIEW.md`) for a first pass,
then read the diff yourself. Check:
ledger before results, trial count sourced from the ledger, `shared_data` rules, holdout
untouched, one gate per stage with graded scores reported, null and power tests for any new method.
