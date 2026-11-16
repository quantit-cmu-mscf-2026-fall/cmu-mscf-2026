# Testing handoff: ledger corrections

## 1. What this PR adds
- QUANTIT-112. `runlog.trial_count` and `python -m capstone.runlog stats` apply
  ledger corrections: a later entry named `ledger-correction` lists runs that
  are not trials. Nothing is deleted from the ledger.
- Code: `capstone/runlog.py` (`CORRECTION_NAME`, `_Corrections`,
  `trial_count`, `_cmd_stats`). Tests: `tests/test_runlog.py`. Docs: one
  paragraph in `docs/validation/handoff.md`.
- Not included: a command to write corrections, and removing the factor
  evaluation's own counter (#78's refresh does that).

## 2. Contract
- An item `{"ts_utc": ..., **params}` excludes the run logged under
  `ledger_name` at that `ts_utc`, *before the correction*, whose params hold
  every other key with the same JSON value (`1` does not match `true`). The
  correction entry is never a trial.
- `params.reason` is required: a non-empty string.
- A correction is undone by a later entry `{"retracts": [<ts_utc of earlier
  corrections>], "reason": ...}`. A retraction can be retracted in turn. The
  ledger is never rewritten.
- Corrections apply before the `name` and tag filters, so they compose.
- A malformed correction (no reason, `ledger_name` not a non-empty string,
  `excludes` not a list, an item without a string `ts_utc`, a bad `retracts`,
  both kinds in one entry) raises `ValueError` naming the file and line. Zero
  trials after corrections still raises `LookupError`.
- A run with recorded `metrics` may have had its result seen. `trial_count`
  warns when a correction drops one; `count_seen_exclusions=True` counts it.
  `stats` prints `excluded by corrections`, how many were seen, the trial
  count if seen exclusions were counted, and exclusions per name.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | 17 test functions (31 cases) added after review, on top of the first 6; `tmp_path` ledger | pass | `pytest -q tests/test_runlog.py` |
| Fails without the change | the PR's previous `capstone/runlog.py`, new tests | the 31 added cases fail | see the PR description |
| Full suite and lint | CI commands | green | `pytest -q -m "not network"`; `ruff check . && ruff format --check .` |

## 4. Provenance
- Ported from Cal's development sandbox, runlog part only; sandbox-only
  provenance-id code and comments were cut. Builds on #33 (with #92).

## 5. Known limits and areas to improve
- Matching is exact on `ts_utc` and on each named param's JSON value.
- "Seen" is a heuristic: the run logged non-empty `metrics`.
- A correction that matches no run is not an error; `stats` counts such items.
- Whether a run whose result was seen may be excluded at all is an open team
  policy question. The code does not decide it: exclusions are honoured but
  warned about, and the stricter count is available.
- Reviewer checklist: the matching rule in `_is_trial`; that `stats` and
  `trial_count` agree; the error message for a malformed correction.

## 6. Public-repo safety
No ledger, data, credentials, account ids or machine paths. Tests use `tmp_path`.
