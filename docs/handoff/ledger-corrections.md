# Testing handoff: ledger corrections

## 1. What this PR adds
- QUANTIT-112. `runlog.trial_count` and `python -m capstone.runlog stats` apply
  ledger corrections: a later entry named `ledger-correction` lists runs that
  are not trials. Nothing is deleted from the ledger.
- Code: `capstone/runlog.py` (`CORRECTION_NAME`, `_corrections`, `_is_trial`,
  `trial_count`, `_cmd_stats`). Tests: `tests/test_runlog.py`. Docs: one
  paragraph in `docs/validation/handoff.md`.
- Not included: a command to write corrections, and removing the factor
  evaluation's own counter (#78's refresh does that).

## 2. Contract
- An item `{"ts_utc": ..., **params}` excludes the run logged under
  `ledger_name` at that `ts_utc` whose params hold every other key with the
  same value. The correction entry is never a trial.
- Corrections apply before the `name` and tag filters, so they compose.
- A malformed correction (no `ledger_name`, `excludes` not a list, an item
  without `ts_utc`) raises `ValueError`. Zero trials after corrections still
  raises `LookupError`.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | 7 new tests (8 cases), `tmp_path` ledger | pass | `pytest -q tests/test_runlog.py -k correction` |
| Fails without the change | stash `capstone/runlog.py`, rerun | 8 of 8 fail | see the PR description |
| Full suite and lint | CI commands | green | `pytest -q -m "not network"`; `ruff check . && ruff format --check .` |
| Edge cases | wrong param, other name, no such timestamp, excludes everything, malformed (3 shapes), tag filters | pass | test names in `tests/test_runlog.py` |

## 4. Provenance
- Ported from Cal's development sandbox, runlog part only; sandbox-only
  provenance-id code and comments were cut. Builds on #33 (with #92).

## 5. Known limits and areas to improve
- Matching is exact on `ts_utc` and on each named param's JSON value.
- A correction for an experiment name that never ran excludes nothing, silently.
- Reviewer checklist: the matching rule in `_is_trial`; that `stats` and
  `trial_count` agree; the error message for a malformed correction.

## 6. Public-repo safety
No ledger, data, credentials, account ids or machine paths. Tests use `tmp_path`.
