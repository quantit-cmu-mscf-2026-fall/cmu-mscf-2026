# Brief: incremental value over known factors and accepted signals

**Owner:** Eunice Yang. Read [handoff.md](handoff.md) first.

## Why this workstream matters

A candidate can pass every significance test and still be a known factor in
disguise, or a near-copy of something we already hold. Stage 3 of the
calibrated validation funnel in `docs/validation.md` asks "does it add
anything beyond known factors and signals already held?" This workstream
answers that with a time-series regression and a test on its intercept (the
alpha).

## Tasks

| Jira | Task | Blocked by |
|---|---|---|
| QUANTIT-48 | Decide on factor data | daily FF5 + Momentum, in #75 |
| QUANTIT-49 | Alpha test against known factors | QUANTIT-48 / #75; #28 (QUANTIT-23) is on `main` |
| QUANTIT-50 | Incremental value over accepted signals | QUANTIT-49; Cal: accepted signals as regressors alone or with the factors |

New module: `capstone/spanning.py`, with `tests/test_spanning.py`. The first
`spanning` PR implements QUANTIT-49 only; QUANTIT-50 is not implemented yet
(see "Accepted signals" below).

## Factor data (QUANTIT-48)

`capstone/data.py` already loads Ken French's library with no credentials:

```python
from capstone.data import load_french
ff5 = load_french("factors5_daily")   # daily Fama–French 5, decimal returns
```

Available datasets: `factors_daily`, `factors_monthly`, `factors5_daily`,
`industry49_daily`, `industry12_daily`. #75 adds `momentum_daily` to
`FRENCH_DATASETS` and `load_ff5_momentum()`, and records the choice in the
Stage 3 comparison set of `docs/validation.md`: daily FF5 + Momentum from the
public Ken French files first, with Chen & Zimmermann's monthly anomaly
portfolios considered later. The
other option was WRDS, through the shared-data loader,
`capstone/shared_data.py` (#26). That loader and `make_return_matrix` are both
on `main`, so a branch from `origin/main` has them together.

## Design notes (QUANTIT-49)

- **Regression:** for each candidate, regress its per-period returns on the
  factor returns over the same dates. The test is on the intercept.
- **Standard errors must allow for autocorrelation:** Newey–West. `statsmodels`
  is a dependency: `sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})`.
  Use `evaluate.newey_west_lags(n_obs)` for L, so the lag rule matches the
  Sharpe tests.
- **Excess returns:** with its default `demean=True`, the backtest builds
  dollar-neutral long-short weights, so those candidate returns are already
  excess returns; don't subtract the risk-free rate from them. A long-only
  series does need `RF` subtracted. Either way, drop `RF` from the regressors.
- **Output:** a Series of one-sided alpha p-values (H1: alpha > 0) indexed by
  candidate, so they go straight into `holm_adjusted` / `by_adjusted` /
  `evidence_profile`. Returning a DataFrame with alpha, t-stat and p-value as
  well is fine.
- **Date alignment:** inner-join on dates and say how many were dropped.

## Tests to write

- A planted candidate built as a mix of factor returns plus noise is rejected
  (no alpha).
- A planted candidate with its own alpha on top of factor exposure is kept.
- Null calibration: on candidates with factor exposure but no alpha, the
  rejection rate is close to α, including with `ar1` noise (the reason for
  Newey–West).

`make_return_matrix` doesn't produce factor exposure on its own. Build test
cases as `factors @ betas + make_return_matrix(...).returns`, with synthetic
factors generated in the test, so the tests don't download anything.

## QUANTIT-49 implementation choices

The design notes and tests above are the team contract. The choices below
fill gaps the contract leaves open; they were agreed for the first
`spanning` PR and bind only `spanning`. No threshold is set here: α₃ is
registered when stage 3 is calibrated (`framework.md`), and stage 3 stays
unmeasured in the funnel and the gate until that is done.

```python
factor_alpha(returns, factors, *, returns_are_excess: bool, lags: int | None = None) -> pd.DataFrame
```

- **Excess returns.** `returns_are_excess` has no default, so the caller
  always says which kind of series it passes and `RF` is never subtracted
  twice. `True`: the returns are used as given (the dollar-neutral
  long-short backtest default). `False`: each date's `RF` is subtracted
  once. Either way the regressors are exactly `FF5_MOMENTUM_REGRESSORS`;
  `RF` is never one.
- **Estimator.** `sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC",
  cov_kwds={"maxlags": L}, use_t=False)`: the call above, with the normal
  reference that `evaluate.sharpe_test` uses made explicit. Otherwise
  statsmodels' defaults: Bartlett kernel, no small-sample correction.
  `L = evaluate.newey_west_lags(n_obs)` per candidate, unless `lags` is
  passed, as in `sharpe_test`.
- **p-value.** `pvalue_greater = norm.sf(alpha / se)`, testing α = 0
  against α > 0.
- **Alignment, per candidate.** A candidate's own missing values are
  removed first; what remains are its *available observations*. They are
  matched to the factor dates that have every regressor (and `RF` when it
  is subtracted). `n_dropped` counts available observations with no
  complete factor row, so `n_obs + n_dropped` is the number of available
  observations. Each candidate gets its own `n_obs` and `L`.
- **Invalid input raises.** `ValueError` for: a frame without a unique,
  increasing, timezone-naive `DatetimeIndex`; `returns` and `factors`
  sharing no dates; non-numeric or infinite values in `returns` or in the
  factor columns used (`RF` only when it is subtracted; other factor
  columns are ignored); duplicate candidate names; a missing or duplicated
  factor column (`RF` included when it is subtracted); a candidate date on
  or after the holdout; input that is not daily; a negative or
  non-integer `lags`. `TypeError` when `returns_are_excess` is not a bool.
- **A candidate that cannot be estimated gets NaN, not an error.** That
  covers: `n_obs` at most 7 (the intercept plus six loadings), `L` not
  below `n_obs`, a rank-deficient design, or residuals with no variance
  (residual RMS at most √ε times the largest return, which catches a
  candidate that is an exact mix of the factors). Its statistics are NaN
  and the rest of the batch still runs; `n_obs`, `n_dropped` and `lags`
  are still reported. `holm_adjusted` and `by_adjusted` count a NaN toward
  the family size and `bh_adjusted` does not (`evaluate.py`).
- **Search period only.** If the candidate returns have any date on or
  after `gate.HOLDOUT_OPENS` (2021-01-01), the call raises, for synthetic
  data too. That is stricter than the gate's exemption for
  `data_version="synthetic"`. Factor rows after that date are not an
  error, because alignment never uses them. `HOLDOUT_OPENS` is imported
  inside the function, not at module import, so `gate` can later import
  `spanning` without a cycle, and the date is not copied. An earlier
  registered `search_end` is still enforced by the funnel and the gate,
  not here.
- **Daily only.** In `returns` and in `factors`, consecutive dates must be
  at least one day apart, and their median gap at most 4 calendar days
  (`MAX_MEDIAN_GAP_DAYS`). Holidays and missing sessions leave the median
  at 1; weekly or monthly series are refused.
- **Output.** A DataFrame indexed by candidate, in `returns` column order:
  `alpha` (per period, decimal), `se`, `tstat`, `pvalue_greater`, `n_obs`,
  `n_dropped`, `lags` (nullable int), and one `beta_<factor>` column per
  regressor. `pvalue_greater` is the Series for `holm_adjusted` /
  `by_adjusted` / `evidence_profile`.
- **Tests run offline.** Factors are generated in the test; candidates are
  `factors @ betas + make_return_matrix(...).returns`, placed on dates
  before 2021 (`make_return_matrix` dates start in 2015, so a long matrix
  would reach the holdout). They cover: null calibration with IID, `ar1`
  and `rho` noise; power with planted alpha; `RF` subtracted only when
  asked; missing dates and `n_dropped`; degenerate designs returning NaN;
  holdout and non-daily input refused.

## Accepted signals (QUANTIT-50): an open dependency

Testing against "signals already held" needs a list of accepted signals and
their return series. **That doesn't exist yet.** The ledger records runs, not
which candidates were accepted, and the gate that will decide acceptance
(Cal, QUANTIT-32 to 36) isn't built. So:

- Write the function to take the accepted signals' return matrix as an
  **argument** (the same shape as the performance matrix). Don't read it from
  anywhere yet.
- Test it with synthetic "accepted" series: a near-copy of one of them should
  be rejected.
- Where the accepted set comes from will be decided with the gate. Raise it
  with Cal when you get there.

**Deferred from the first PR.** `spanning` does not implement QUANTIT-50
yet. The docs do not settle whether accepted signals enter the regression
alone or together with FF5 + Momentum, and that is waiting on Cal. Whichever
is chosen, accepted signals' returns will be passed in as a matrix argument,
never loaded inside `spanning`.

## Who uses your output

The acceptance gate's stage 3 (Cal).
