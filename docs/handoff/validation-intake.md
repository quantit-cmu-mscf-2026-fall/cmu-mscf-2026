# Testing handoff: validation intake

## 1. What this PR adds
- QUANTIT-81. `capstone/intake.py`: `FunnelInput` (returns, family, generator,
  source), `from_returns` and `stage_inputs`, the one hand-off any generator
  (factor trees, Alpha-GPT, a strategy grid) uses to give candidates to the
  funnel. Tests: `tests/test_intake.py`.
- Not included: `from_evaluation` (factor evaluation's adapter, the next PR),
  thresholds, the gate or the scorecard.

## 2. Contract
- Input: dates x candidate net returns (NaN = no position) and one hypothesis
  label per candidate, the performance matrix of `docs/validation.md`.
- Checked once at the boundary: increasing unique dates; unique candidates;
  every candidate labelled and every label a candidate; no +/-inf; no
  candidate without returns; a non-blank generator name. Rows where no
  candidate traded are dropped and counted in `source["n_dropped_rows"]`.
- `stage_inputs` cuts periods by date: stages 1-3 see `<= search_end` only;
  returns past `search_end` without `holdout_start` raise. It takes the
  registered stage-1 BH level `q` and raises if `n_boot` can't resolve it.
- Imports no generator, backtest or data code.
- Failures: `TypeError` (not a dated frame / Series), `ValueError` (the rest).

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | `tests/test_intake.py`, 18 tests | pass | `pytest -q tests/test_intake.py` |
| Holdout never seen by stages 1-3 | holdout rows poisoned with +1e6; bootstrap spied | stage 1-3 columns identical | `pytest -q tests/test_intake.py -k holdout` |
| Null calibration | 20 null family sets each, iid, `rho=0.3`, `ar1=0.2`; BH at 0.10 on `p1_bootstrap` | mean share accepted <= 0.10 | `pytest -q tests/test_intake.py -k null` |
| Power | 3 of 10 planted families, Sharpe 2 | all planted pass BH at 0.10 | `pytest -q tests/test_intake.py -k planted` |
| No generator imports | fresh interpreter, `sys.modules` | none loaded | `pytest -q tests/test_intake.py -k imports` |
| Full suite and lint | CI commands | green | `pytest -q -m "not network"`; `ruff check . && ruff format --check .` |

## 4. Provenance
- Ported from Cal's development sandbox; the factor-evaluation adapter was cut
  (next PR). Builds on `funnel.family_statistics` (QUANTIT-34, the base PR).

## 5. Known limits and areas to improve
- `source` is a free-form dict; the gate decides what it must hold.
- Reviewer checklist: the input checks in `_check`; that dropping all-missing
  rows is the right shared rule for every generator; the date cut.

## 6. Public-repo safety
No data, ledger, credentials, account ids or machine paths. Synthetic data only.
