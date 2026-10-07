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
| QUANTIT-48 | Decide on factor data | nothing (a decision; can start now) |
| QUANTIT-49 | Alpha test against known factors | QUANTIT-48, #28 (QUANTIT-23) |
| QUANTIT-50 | Incremental value over accepted signals | QUANTIT-49 |

New module: `capstone/spanning.py`, with `tests/test_spanning.py`.

## Factor data (QUANTIT-48)

`capstone/data.py` already loads Ken French's library with no credentials:

```python
from capstone.data import load_french
ff5 = load_french("factors5_daily")   # daily Fama–French 5, decimal returns
```

Available datasets: `factors_daily`, `factors_monthly`, `factors5_daily`,
`industry49_daily`, `industry12_daily`. **Momentum is not on `main` yet:**
#75 adds `momentum_daily` to `FRENCH_DATASETS` and `load_ff5_momentum()`, and
records the choice of the public Ken French files in `docs/validation.md`. The
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

## Who uses your output

The acceptance gate's stage 3 (Cal).
