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
| `holm`, `holm_adjusted` | FWER | any dependence |
| `benjamini_yekutieli`, `by_adjusted` | FDR | any dependence |
| `benjamini_hochberg`, `bh_adjusted` | FDR | independence or positive dependence (one-sided) |
| `estimate_pi0`, `storey_qvalues` | pFDR | independence only |
| `evidence_profile(p)` | all four side by side | - |
| `false_discovery_rate(rejected, truth)`, `power(rejected, truth)` | score a rule on synthetic truth | - |

## Search (`search_fdr`)

| Function | Notes |
|---|---|
| `search_adjusted_pvalue` | Best-of-K p-value; exact for independent or equicorrelated variants (`rho=`) |
| `bootstrap_family_pvalues` | Block bootstrap of the family maximum; `block_length_rule(n_obs)` |
| `search_adjusted_fdr`, `fdr_by_search_intensity` | Search-adjusted FDR (Lopez de Prado & Fabozzi 2026) |
| `fit_max_of_mixture` | Sensitivity table over assumed K; not a gate |

## Cross-validation (`cv`)

`PurgedKFold(n_splits, horizon, embargo_pct)`, `WalkForward`, `label_end_times`,
`average_uniqueness`.

## Funnel calibration (`calibration`)

`family_statistics`, `Thresholds`, `run_funnel`, `run_baseline`, `run_naive_stack`, `score`,
`gate_report`, `gates_alone`; driver `scripts/calibrate_validation.py` with JSON configs.

## Planned (check open PRs before building)

Stationary bootstrap and block-length selection; White's Reality Check, Romano-Wolf, SPA
(Carl); `pbo` CSCV (#35, Pin-Hua); Harvey-Liu haircut Sharpe (#65); local FDR / empirical
null (#54); spanning alpha (Eunice); `gate` pre-registered scorecard; `runlog.trial_count()`
(#33).
