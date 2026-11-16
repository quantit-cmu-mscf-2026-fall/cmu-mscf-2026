# Testing handoff: the acceptance gate

The acceptance gate (QUANTIT-23..36, Cal's workstream) arrives as a stack of
small PRs. Each one adds its rows here. Long vetting detail is in each PR's
description.

## 1. What this adds
- **Dated funnel statistics** (QUANTIT-34): `capstone/funnel.py` with
  `family_statistics` (periods cut by date), `holdout_statistics` (stage 4's
  input alone) and `best_by_family`. `calibration.family_statistics` now maps
  its simulated periods to dates and calls it. Tests: `tests/test_funnel.py`.
- Not yet: the stages and thresholds, the scorecard, registration and the
  single-look run (later PRs in this stack).

## 2. Contract
- Input: a performance matrix (dates x variants, `docs/validation.md`) and a
  hypothesis label per variant.
- Stages 1-3 read dates up to `search_end` only. A frame with later dates is
  refused unless `holdout_start` (> `search_end`) is passed explicitly.
- `holdout_statistics` reads only the columns it is asked for.
- Errors: `TypeError` without a date index; `ValueError` for unordered
  dates, an empty period, a holdout that does not follow the search, or a
  variant with no hypothesis label.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | `tests/test_funnel.py` | pass | `pytest -q tests/test_funnel.py` |
| Holdout blindness | shock holdout dates, compare stage-1/2 numbers | unchanged | `pytest -q tests/test_funnel.py -k only_search` |
| Calibration unchanged | old vs new `calibration.family_statistics`, 4 sets x 2 `n_boot` x 2 designs | exactly equal | see the PR description |
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
