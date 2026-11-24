# Validation methods in `capstone` (cmu-mscf-2026)

Status as of `docs/validation.md` on `main` (late Oct 2026). Check the PR list for what has
merged since; a function listed here as planned may exist on a branch only.

## Inputs and synthetic truth

| Function | Purpose |
|---|---|
| `synth.make_return_matrix(n_obs, n_candidates, n_real, *, sharpe_real, vol, rho, ar1, t_df, garch, seed)` | Known-truth return matrix; `.returns`, `.truth`, `.params` |
| `synth.make_panel` / `backtest.candidate_returns(panel)` | Same shape the long way, through signals and a backtest |
| `calibration.make_family_set(n_families, k, share_real, sharpe_real, *, within_rho, ...)` | Families of K near-copy variants with a search and a holdout split |

## Sharpe inference (`evaluate`)

| Function | Notes |
|---|---|
| `sharpe_test(returns, method="hac")` | `.pvalue_greater` (one-sided), `.pvalue`, `.psr`, `.se`; methods `normal`, `nonnormal`, `hac` |
| `sharpe_variance(returns, method, lags)` | `hac` uses Newey-West lags (`newey_west_lags`) |
| `probabilistic_sharpe_ratio`, `min_track_record_length` | PSR and MinTRL |
| `deflated_sharpe_ratio(sharpe, n_trials, n_obs, ...)` | Against `expected_max_sharpe` of n_trials nulls; n_trials from the ledger |
| `average_correlation`, `implied_independent_trials` | Effective trial count under correlation |

## Multiple testing (`evaluate`)

| Function | Controls | Valid under |
|---|---|---|
| `holm`, `holm_adjusted` | FWER | any dependence; graded score (too strict for a screen) |
| `benjamini_yekutieli`, `by_adjusted` | FDR | any dependence; graded score |
| `benjamini_hochberg`, `bh_adjusted` | FDR | independence or positive dependence (one-sided); the **gate** at stages 1 and 4 |
| `estimate_pi0`, `storey_qvalues` | pFDR | independence only; graded score |
| `evidence_profile(p)` | all four side by side, as graded scores (one correction per stage is the gate) | - |
| `false_discovery_rate(rejected, truth)`, `power(rejected, truth)` | score a rule on synthetic truth | - |

## Search (`search_fdr`)

| Function | Notes |
|---|---|
| `bootstrap_family_pvalues` | **Stage-1 gate input**: block bootstrap of the family maximum; `block_length_rule(n_obs)` |
| `search_adjusted_pvalue` | Fast cross-check only, not the gate: exact for independent or equicorrelated variants (`rho=`); with it the baseline ran at an FDR of 0.13-0.16 |
| `search_adjusted_fdr`, `fdr_by_search_intensity` | Search-adjusted FDR (Lopez de Prado & Fabozzi 2026) |
| `fit_max_of_mixture` | Sensitivity table over assumed K; not a gate |

## Cross-validation (`cv`)

`PurgedKFold(n_splits, *, horizon, embargo_pct)` (the last two are keyword-only:
`PurgedKFold(5, horizon=5, embargo_pct=0.01)`), `WalkForward`, `label_end_times`,
`average_uniqueness`.

## Funnel calibration (`calibration`)

`family_statistics`, `Thresholds`, `run_funnel`, `run_baseline`, `run_naive_stack`, `score`,
`gate_report`, `gates_alone`; driver `scripts/calibrate_validation.py` with JSON configs.

The defaults of `Thresholds()` (`q1=0.1`, `q4=0.3`, `stage1='rho'`) and of
`run_baseline(stats, *, q=0.10, stage1="rho")` are a synthetic starting point, not
registered thresholds (see the `Thresholds` docstring in `funnel.py`). On real candidates
pass `stage1="bootstrap"`; registered values live in `docs/validation/framework.md`.

## Planned (check open PRs before building)

Stationary bootstrap and block-length selection (QUANTIT-37/38); White's Reality Check,
Romano-Wolf, SPA (#91); spanning over signals already held
(QUANTIT-50). Built: `pbo` CSCV, `evaluate.haircut_sharpe`, `evaluate.local_fdr`,
`spanning.factor_alpha` and the `gate` pre-registered scorecard.
