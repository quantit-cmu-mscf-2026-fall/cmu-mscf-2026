---
name: run-ledger
description: How to record every experiment in the cmu-mscf-2026 run ledger with capstone.runlog.log_run before looking at results, what counts as a trial, the stage tags the live viewer reads, and how to get the trial count for corrections. Use when writing or running an experiment, sweep or agent search, or when a correction needs n_trials.
---

# Run ledger

The ledger (`experiments/runs.jsonl`, or `$CAPSTONE_LEDGER_DIR/runs.jsonl`) is the project's
trial count. Every multiple-testing correction depends on it, so an unlogged run silently
lowers the bar for every candidate.

## Logging a run

Call `log_run` once per trial, at the point the result is produced and **before** printing,
plotting or returning it:

```python
from capstone import runlog, shared_data as sd

entry = runlog.log_run(
    "momentum-12-1",                       # reuse the same name across a sweep
    params={"lookback": 252, "skip": 21, "cost": "quoted_spread",
            "end": "2020-12-31", "data_version": sd.data_version()},
    metrics={"sharpe": sr, "pvalue_greater": p},
    seed=SEED,
    tags=["stage:screen", "family:momentum", "series:returns"],
    notes="net of costs; gross sharpe in params",
)
```

- `params`: everything that defines the trial, including `data_version` and the date range.
- `metrics`: the tested quantity. Put the number here, then look at it.
- `seed`: a fixed integer from the script's parameters, never `None` for random work.
- Scripts, not notebooks: a notebook re-run logs nothing or logs twice.

## What counts as a trial

Every candidate whose performance anyone or any agent has seen: parameter variants,
discarded candidates, each iteration inside an agent's search. An idea that never produced
a performance number is not a trial. Counts accumulate across rounds of a search.

## Stage and verdict tags

Use these tag prefixes so the funnel can be read straight off the ledger:

| Tag | Meaning |
|---|---|
| `stage:screen` / `stage:robustness` / `stage:incremental` / `stage:final` / `stage:incubation` | Funnel stage 1-5 the run belongs to |
| `family:<id>` | Hypothesis family (variants of one idea); stage 1 corrects across families |
| `series:returns` / `series:ic` | Which per-period series was scored |
| `verdict:pass` / `verdict:fail` | Outcome against the stage's pre-registered gate |
| `holdout` | The run read 2021-2025 data. Only stage 4 may carry it, once |

## Reading the count

```bash
python -m capstone.runlog stats          # total runs, runs per name, distinct users
python -m capstone.runlog list --last 20
```

Take `n_trials` for `deflated_sharpe_ratio` and friends from this output (or from
`runlog.trial_count()` once PR #33 lands), never from a typed number. Report the count and
seeds in the PR description.

## Tests and scratch work

Point throwaway runs and tests elsewhere so they do not inflate the team count:
`CAPSTONE_LEDGER_DIR=.ledger-scratch` (gitignored). Never commit `runs.jsonl`; it syncs to the
team's private archive.
