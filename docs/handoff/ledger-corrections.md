# Testing handoff: ledger corrections

## 1. What this PR adds
- QUANTIT-112. `runlog.trial_count` and `python -m capstone.runlog stats` apply
  ledger corrections: a later entry named `ledger-correction` can exclude
  unseen runs from the count. Seen results always count. Nothing is deleted.
- Code: `capstone/runlog.py` (`CORRECTION_NAME`, `_Corrections`,
  `trial_count`, `_cmd_stats`). Tests: `tests/test_runlog.py`. Docs: one
  paragraph in `docs/validation/handoff.md`.
- Not included: a command to write corrections, and removing the factor
  evaluation's own counter (#78's refresh does that).

## 2. Contract
- An item `{"ts_utc": ..., **params}` targets one run logged under
  `ledger_name` at that `ts_utc`, *before the correction*, whose params hold
  every other key with the same JSON value (`1` does not match `true`). The
  correction entry is never a trial. Each item consumes one earlier matching
  entry; two identical ledger lines require two items to target both.
- `params.reason` is required: a non-empty string.
- A correction is undone by a later entry `{"retracts": [<ts_utc of earlier
  corrections>], "reason": ...}`. A retraction can be retracted in turn. The
  ledger is never rewritten.
- Corrections apply before the `name` and tag filters, so they compose.
- A malformed correction (no reason, `ledger_name` not a non-empty string,
  `excludes` not a list, an item without a string `ts_utc`, a bad `retracts`,
  both kinds in one entry) raises `ValueError` naming the file and line. Zero
  trials after corrections still raises `LookupError`.
- A run with non-empty `metrics` may have had its result seen and always
  remains a trial, even when targeted by a correction. `trial_count` defaults
  to this conservative count; the existing `count_seen_exclusions=True`
  argument is accepted, while False raises `ValueError`. `stats` reports the
  same count and per-name totals, correction targets, excluded unseen runs,
  and seen targets retained as trials. Family and tag filters still apply.

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
- No attempt is made to infer whether a result without recorded metrics was
  seen elsewhere; recording results truthfully remains the caller's duty.
- Reviewer checklist: the matching rule in `_Corrections`; that `stats` and
  `trial_count` agree; the error message for a malformed correction.

## 6. Public-repo safety
No ledger, data, credentials, account ids or machine paths. Tests use `tmp_path`.
