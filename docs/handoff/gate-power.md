# Testing handoff: the gate's power analysis

Details: the PR descriptions. Each PR in this track adds its rows here.

## 1. What this adds
- QUANTIT-124. `capstone/power.py` builds the simulated worlds a power analysis of the acceptance gate runs on: a null batch of families (`null_world`, `null_side`), a target family planted at a true Sharpe (`plant`, `planted_side`). `PowerSpec` is the configuration measured, including the synthetic defects the worlds can carry: `ar1`, `t_df`, `garch` (`synth.make_return_matrix`'s ranges, refused outside them; none by default, which is the cross-correlated Gaussian world, unchanged). A plant target listed twice, or naming no family, raises `ValueError`.
- `calibration.make_family_set` takes one variant count per family; a scalar `k` gives exactly the data it did.
- Stacked on it, `power_analysis(spec, budget)` simulates `PowerBudget.n_sims` worlds, each scored at every Sharpe of the grid (`planted_sides`, a few funnel calls per world), and returns a `PowerReport`: per stage a `StagePower` (nominal level, `null_rate` at Sharpe 0, power curve, minimum detectable Sharpe at 50% and 80%) and the end-to-end curve; `min_detectable_sharpe()` lists `MinDetectableSharpe`s. The minimum is the first grid Sharpe where the curve's running maximum reaches the power, interpolated; None when never reached or the stage is no gate (`alpha2 = 1`).
- Tests: `tests/test_power.py`, `tests/test_power_analysis.py`, `TestVariantsPerFamily` in `tests/test_calibration.py`.
- **Gate report** (QUANTIT-124): `power.for_gate(config, search, family, n_trials, budget)` turns a registered `GateConfig` and its search period into the header's `min_detectable_sharpe` (per stage and end to end) through `spec_for_gate`: the config's thresholds and bootstrap, the family sizes, the search length, the holdout's business days, and the within- and across-family correlations measured on the search period. `GateHeader` carries `min_detectable_sharpe` and `power_sims` (None and 0 when no budget was run; reports written before the fields existed still read back). `scripts/power_analysis.py` simulates the registered funnel, taking the thresholds and the bootstrap draws from the `gate_config` in `scripts/gate_crsp_v1.json` (crsp-v1: stage 1 at q₁ = 0.10 on the bootstrap, q₄ = 0.50, 1,999 draws; `--config` points at another registered file), and the run's shape from `scripts/power_analysis.json` (which holds no thresholds). It logs the config's hash with the params to the ledger as `validation-power` (tags `synthetic`, `power`) before printing, and prints the table that goes in a PR description. It simulates the funnel only: the registered decision rule (the baseline, BH at 0.05) is not simulated, so these are the funnel's detection limits, not the decision's. Tests: `tests/test_power_report.py`.
- Not here: stage 3 (spanning is not in the simulated worlds: reported as no gate, passing everything) and calling `for_gate` from the gate run itself; both follow once the gate run and the spanning test are on `main`.

## 2. Contract
- Adds no statistics: every number comes from `funnel.family_statistics` on `make_family_set` data.
- Deterministic for a seed. Simulated data only: no real data, no ledger writes.
- `PowerSpec` raises `ValueError` on a configuration the funnel can't run: a family with fewer than 2 variants, under 10 bootstrap blocks of search data, `n_boot + 1 < families / q` whenever the bootstrap runs (`n_boot > 0`, or a bootstrap stage 1), with q = min(0.05, q1) for a bootstrap stage 1 and 0.05 otherwise (`null_side` checks the bootstrap at that level), `within_corr < across_corr`, fewer trials than variants.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Unit tests | `tests/test_power.py`, 24 tests | pass | `pytest -q tests/test_power.py` |
| Same code paths | a family alone = its row in the full batch; `planted_side` = the full planted batch recomputed; `plant` = the generator's planted family | equal (1e-12 / 1e-15) | `pytest -q tests/test_power.py -k SameCodePaths` |
| Correlations | `within_corr`/`across_corr` measured on a 58-year null world | within 0.02 / 0.03 | `pytest -q tests/test_power.py -k correlations` |
| Determinism | same seed twice, `null_side` and `planted_side` | identical | `pytest -q tests/test_power.py -k Determinism` |
| Generator | listed equal counts = scalar `k`, bit for bit | pass | `pytest -q tests/test_calibration.py -k VariantsPerFamily` |
| Defects | AR(1) / t / GARCH worlds measured: lag-1 autocorrelation, excess kurtosis, squared-return autocorrelation; null stays null with unit volatility; planting equals the generator and the full-batch recompute under defects | pass | `pytest -q tests/test_power.py -k Defects` |
| Null calibration | 150 null worlds each on iid, cross-correlated (`rho`), AR(1) and t/GARCH data: stage 1 any-family rate ~q1, stage 4 lone-survivor rate ~q4, end-to-end acceptance <= 0.09 | pass | `pytest -q tests/test_power_analysis.py -k NullLevels` |
| Power | 40 worlds each on iid, `rho` and `ar1` data: curves rise with Sharpe, end to end >= 0.8 at 2.5; one family's holdout power and minimum detectable Sharpe equal the one-sided closed form (250 worlds) | pass | `pytest -q tests/test_power_analysis.py -k TestPower` |
| Batched grid | each Sharpe of `planted_sides` = `planted_side` at that Sharpe, under a forced small chunk | equal (1e-12) | `pytest -q tests/test_power_analysis.py -k batched` |
| Determinism | same seed twice | identical | `pytest -q tests/test_power_analysis.py -k Determinism` |

## 4. Provenance
- Ported from Cal's development sandbox (`validation/power.py`). Cut here: the stage-3 spanning p-value (stage-3 PR) and the call from the gate run (the gate-run PRs).

## 5. Known limits and reviewer checklist
- The AR(1) null's stage-1 rate measured 0.13 over 150 worlds (standard error 0.027): within noise of q1 = 0.10, but HAC-free formula p-values can run liberal on autocorrelated data. The test bounds it (0.03 to 0.20); it does not claim exactness.
- The grid is simulated worlds, not an analytic curve: its Monte Carlo error is ~0.03-0.07 per point at the default 200 worlds, and the minimum detectable Sharpe is interpolated between grid points.
- The stage columns (`best`, `p1_*`, `p2`, `ho_*`) must depend only on the family's own columns and the seed; `planted_side` relies on it, and the first same-code-path test pins it.
- The naive stack's columns (`naive_pass`, `nv_*`) depend on the whole batch (Holm, BY, BH, the DSR's trial count), so `planted_side` drops them. A caller that needs them runs `funnel.family_statistics` on the whole planted batch.
- Worth a look: `within_rho`'s mapping from the measured correlations; `PowerSpec`'s refusals against the funnel's own floors.

## 6. Public-repo safety
No data, credentials, account ids, machine paths or private notes.
