---
name: run-ledger
description: How to record every experiment in the cmu-mscf-2026 run ledger with capstone.runlog.log_run before looking at results, what counts as a trial, the stage tags the pipeline viewer reads, and how to get the trial count for corrections. Use when writing or running an experiment, sweep or agent search, or when a correction needs n_trials.
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
    tags=["stage:screen"],
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

Use these tag prefixes so the funnel can be read straight off the ledger. `stage:`,
`verdict:` and `holdout` are read by the pipeline viewer (#192, still open). Nothing reads
or defines other prefixes, so do not invent ones that look official. Group a family's runs
by reusing one experiment `name`, which `trial_count(name=...)` reads.

| Tag | Meaning |
|---|---|
| `stage:screen` / `stage:robustness` / `stage:incremental` / `stage:final` / `stage:incubation` | Funnel stage 1-5 the run belongs to |
| `verdict:pass` / `verdict:fail` | Outcome against the stage's pre-registered gate |
| `holdout` | The run read 2021-2025 data. Only stage 4 may carry it, once |
| `synthetic` | Methodology run on simulated data; kept out of a strategy's count with `exclude_tags` |

## Reading the count

The count for a correction is scoped to the family being corrected, not the whole ledger:

```python
from capstone import runlog

n = runlog.trial_count(name="momentum-12-1", exclude_tags=["synthetic"])
# or, for related runs that share a tag:
n = runlog.trial_count(include_tags=["stage:screen"], exclude_tags=["synthetic"])
```

Bare `trial_count()` and `python -m capstone.runlog stats` count every entry (every
contributor, every methodology run), which inflates `m` and throws away power. Use `stats`
and `list --last 20` to look at the ledger, not to source `n_trials`. Never type the
number. Report the count, the filter used and the seeds in the PR description.

## Tests and synthetic runs

Runs on synthetic data (null calibrations, simulation studies) belong in the real ledger,
tagged `synthetic`: `CLAUDE.md` requires every experiment to be logged, the `trial_count`
docstring sets the tag convention, and `scripts/calibrate_validation.py` follows it.
Exclude them from a strategy's count with `exclude_tags=["synthetic"]`. Unit tests are
different: point `CAPSTONE_LEDGER_DIR` at pytest's `tmp_path` so test runs never touch the
team ledger. Never commit `runs.jsonl`; it syncs to the team's private archive.
