# Testing handoff: the acceptance gate

The acceptance gate (QUANTIT-23..36, Cal's workstream) arrives as a stack of
small PRs. Each one adds its rows here. Long vetting detail is in each PR's
description.

## 1. What this adds
- **Dated funnel statistics** (QUANTIT-34): `capstone/funnel.py` with
  `family_statistics` (periods cut by date), `holdout_statistics` (stage 4's
  input alone) and `best_by_family`. `calibration.family_statistics` now maps
  its simulated periods to dates and calls it. Tests: `tests/test_funnel.py`.
- **Stages 1-3 and the thresholds** (QUANTIT-32): `Thresholds` (checked when
  built, `stage1` a field, default "rho" as on `main`), `thresholds_dict`
  (plain data to register), `screen` (stages 1-3, no holdout),
  `stage1_pvalues`, `run_funnel`. `calibration` re-exports them.
- **The scorecard** (QUANTIT-35): `capstone/scorecard.py`, the row and
  header shape (`SCORECARD_COLUMNS`, `GateHeader`, `TrialScope`,
  `GateResult`) and the report (`write_report`/`read_report`, JSON + CSV).
- **Registration** (QUANTIT-33): `capstone/gate.py`, `GateConfig` (hashed
  from its content), `register` (logs it to the ledger once, tag "gate"),
  `required_n_boot` and the input checks a run makes first.
- **Graded scores** (QUANTIT-35): `capstone/grading.py`, reported and never
  read back into a decision.
- **The run** (QUANTIT-34): `gate.run_gate`. Refuses an unregistered or
  changed config (`GateNotRegistered`) and a config whose holdout was already
  looked at (`HoldoutAlreadyOpened`); needs a named `authorizer`, recorded in
  the `gate-holdout-look` ledger entry written before the holdout provider is
  called. Stages 1-3 get the search period only; stage 3 is not measured yet
  (`s3_measured` False). Tests: `tests/test_gate_run.py`.
- **Holdout inputs** (QUANTIT-34): the provider's answer is checked on
  arrival (exactly the requested variants, registered holdout dates only,
  finite); a survivor with no holdout returns fails stage 4. Tests:
  `tests/test_gate_holdout_inputs.py`; the decision matrix (null and planted
  markets on iid, `rho` and `ar1` data), scorecard, report and expected-mean
  tests are in `tests/test_gate_decisions.py`.
- Not yet: the stage-3 spanning test and the power header.

## 2. Contract
- Input: a performance matrix (dates x variants, `docs/validation.md`) and a
  hypothesis label per variant.
- Stages 1-3 read dates up to `search_end` only. A frame with later dates is
  refused unless `holdout_start` (> `search_end`) is passed explicitly.
- `holdout_statistics` reads only the columns it is asked for.
- Errors: `TypeError` without a date index; `ValueError` for unordered
  dates, an empty period, a holdout that does not follow the search, or a
  variant with no hypothesis label.
- `screen` never reads `ho_*`. Stage 1 refuses a missing family p-value
  (BH would drop it); stage 4 refuses a live candidate with no holdout
  statistics. Stage 3 passes through (spanning test not built yet).
- `run_gate` takes no default authorizer and ships no holdout provider: the
  caller supplies both. A blank authorizer, a failed input check, too few
  ledger trials or a failed stage 1-3 raise before the look, so none of them
  spends it; once the look is logged the config cannot be run again.
- A config registered with `factors_digest` is refused until stage 3 exists.
- Real data cannot search the holdout: unless `data_version` is "synthetic",
  `GateConfig` refuses a `search_end` on or after `gate.HOLDOUT_OPENS`
  (2021-01-01, where `docs/validation.md` starts the holdout), when it is
  built, so such a config can't be registered or run. A test keeps this
  constant and the docs' date equal.
- `GateConfig`'s default thresholds use stage 1 on the bootstrap family
  maximum (`Thresholds()` itself still defaults to "rho", as on `main`).
- A config's hash is SHA-256 of its canonical JSON, equal in every process.
  The gate's ledger entries are tagged "gate" and never counted as trials.
- The report is standard JSON (NaN as null) and reads back to the same
  result; `read_report` refuses an accepted list the scorecard disagrees with.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | `tests/test_funnel.py`, 73 tests | pass | `pytest -q tests/test_funnel.py` |
| Holdout blindness | shock holdout dates, compare stage-1/2 numbers | unchanged | `pytest -q tests/test_funnel.py -k only_search` |
| Calibration unchanged | old vs new `calibration.family_statistics`, 4 sets x 2 `n_boot` x 2 designs | exactly equal | see the PR description |
| Scorecard and report | `tests/test_scorecard.py`, 9 tests (14 cases), hand-built result | pass, pandas 3.1 and 2.2 | `pytest -q tests/test_scorecard.py` |
| Registration | `tests/test_gate.py`, 16 tests (27 cases), `tmp_path` ledger | pass | `pytest -q tests/test_gate.py` |
| Calibration runs unchanged | old vs new `run_funnel`/`run_naive_stack`/`gates_alone`, 96 settings | exactly equal | see the PR description |
| pandas 3 | all-NaN family, nothing passes the naive stack | no warning | `pytest -q tests/test_funnel.py -k Pandas3` |

## 4. Provenance
Ported from Cal's development sandbox. Cut there-only pieces: the
`validation/` package layout (see the split-`evaluate.py` Topic).

## 5. Known limits
- A family with one variant still fails in `average_correlation` (as on
  `main`); a follow-up gives it rho 0.
- Reviewer: check that nothing before stage 4 can read a holdout date.

## 6. Public-repo safety
Synthetic data only; no data, credentials, paths or ledger.
