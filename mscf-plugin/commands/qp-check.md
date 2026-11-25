---
description: Pre-PR check - the offline test suite, ruff, and the trial count from the run ledger
argument-hint: "[nothing]"
allowed-tools: Bash(python -m pytest:*), Bash(python -m ruff:*), Bash(ruff:*), Bash(python -m capstone.runlog:*), Read
---

Run the repository's acceptance checks before a PR (`CLAUDE.md`, "Verify before any PR").
Run from the repository root with the project's virtual environment active.

## Commands

```bash
python -m pytest -q -m "not network"
python -m ruff check .
python -m ruff format --check .
python -m capstone.runlog stats
```

`ruff format --check` only reports; run `ruff format .` yourself if it lists files.

## Check

- Report the passed, failed and skipped counts and each command's exit code; all must be 0.
- Ruff must report no findings and no files to reformat.
- Report the trial count from `runlog stats`. Never type one in, and never edit
  `experiments/runs.jsonl`: the ledger is local and gitignored, and is never committed.

## Refuse

- Reporting "green" when any exit code is nonzero, a command was skipped, or the suite was run
  with a different marker expression.
- Running an experiment or a script that reads data as part of the check.
