# Experiment ledger

`runs.jsonl` in this directory is the append-only record of every experiment the
team runs. Its length is the trial count — the denominator of every
multiple-testing correction in `capstone.evaluate`. A run that is not logged is
a hypothesis test the corrections never hear about, which silently inflates
every "significant" result that survives them.

Log a run at the point where the result is produced:

```python
from capstone.runlog import log_run

log_run("momentum_sweep", params={"lookback": 60}, metrics={"sharpe": 0.9}, seed=0)
```

Inspect the ledger:

```
python -m capstone.runlog stats
python -m capstone.runlog list --last 20
```

The rule: **if it ran, it goes in the ledger.**

Note: `runs.jsonl` is local and **gitignored** — it never enters the public
repo. `python scripts/capture/sync_sessions.py` copies it (redacted) to the
team's private `session-archive/ledger/`, where per-user ledgers combine into
the team-wide trial count.
