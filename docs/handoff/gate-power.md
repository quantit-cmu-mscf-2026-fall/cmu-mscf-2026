# Testing handoff: the gate's power analysis

Details: the PR descriptions. Each PR in this track adds its rows here.

## 1. What this adds
- QUANTIT-NEW-POWER. `capstone/power.py` builds the simulated worlds a power analysis of the acceptance gate runs on: a null batch of families (`null_world`, `null_side`), a target family planted at a true Sharpe (`plant`, `planted_side`). `PowerSpec` is the configuration measured.
- `calibration.make_family_set` takes one variant count per family; a scalar `k` gives exactly the data it did.
- Tests: `tests/test_power.py`, `TestVariantsPerFamily` in `tests/test_calibration.py`.
- Not here: one world's stage-by-stage pass flags, the batched Sharpe grid, the power curves and minimum detectable Sharpe (next PR), the gate header and script (the one after), stage 3's spanning p-value (the stage-3 PR).

## 2. Contract
- Adds no statistics: every number comes from `funnel.family_statistics` on `make_family_set` data.
- Deterministic for a seed. Simulated data only: no real data, no ledger writes.
- `PowerSpec` raises `ValueError` on a configuration the funnel can't run: under 10 bootstrap blocks of search data, `n_boot + 1 < 20 x families` with the bootstrap stage 1, `within_corr < across_corr`, fewer trials than variants.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | `tests/test_power.py`, 17 tests | pass | `pytest -q tests/test_power.py` |
| Same code paths | a family alone = its row in the full batch; `plant` = the generator's planted family | equal (1e-12 / 1e-15) | `pytest -q tests/test_power.py -k SameCodePaths` |
| Correlations | `within_corr`/`across_corr` measured on a 58-year null world | within 0.02 / 0.03 | `pytest -q tests/test_power.py -k correlations` |
| Determinism | same seed twice, `null_side` and `planted_side` | identical | `pytest -q tests/test_power.py -k Determinism` |
| Generator | listed equal counts = scalar `k`, bit for bit | pass | `pytest -q tests/test_calibration.py -k VariantsPerFamily` |
| Null calibration and power | the curves are the next PR's | n/a here | n/a |

## 4. Provenance
- Ported from Cal's development sandbox (`validation/power.py`). Cut here: the stage-3 spanning p-value (stage-3 PR); the world loop, curves and report types (next PRs); the gate hooks.

## 5. Known limits and reviewer checklist
- Each family's statistics must depend only on its own columns and the seed; `planted_side` relies on it, and the first same-code-path test pins it.
- Worth a look: `within_rho`'s mapping from the measured correlations; `PowerSpec`'s refusals against the funnel's own floors.

## 6. Public-repo safety
No data, credentials, account ids, machine paths or private notes.
