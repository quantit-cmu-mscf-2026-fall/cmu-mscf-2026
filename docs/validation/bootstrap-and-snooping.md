# Brief: stationary bootstrap and data-snooping tests

**Owner:** Carl Cui. Read [handoff.md](handoff.md) first.

## Why this workstream matters

Almost every test we have so far computes a p-value for one candidate at a time
from a formula. When there are many correlated candidates, the question that
matters is about the best of them: "is the best candidate better than the best
of this many no-skill candidates, given how they move together?" A formula
can't answer that. Resampling the data can, as long as the resampling keeps the
autocorrelation and the correlation between candidates. That's what this
workstream provides, and the acceptance gate depends on it.

## Tasks

| Jira | Task | Blocked by |
|---|---|---|
| QUANTIT-37 | Stationary bootstrap resampler | #28 (QUANTIT-23) |
| QUANTIT-38 | Block-length selection | QUANTIT-37 |
| QUANTIT-39 | Bootstrap Sharpe confidence intervals | QUANTIT-37, 38 |
| QUANTIT-40 | White's Reality Check | QUANTIT-37 |
| QUANTIT-41 | Romano–Wolf step-down | QUANTIT-37 |
| QUANTIT-42 | Hansen SPA and snooping docs | QUANTIT-40 |

New modules: `capstone/bootstrap.py` and `capstone/snooping.py`, with
`tests/test_bootstrap.py` and `tests/test_snooping.py`.

## Design notes

**Resample dates, not cells.** Draw one set of date indices per bootstrap
replicate and take those rows of the whole matrix, so every candidate is
resampled on the same dates. That keeps the correlation between candidates,
which is the thing the snooping tests need. A small, reusable interface keeps
both modules simple:

```python
def stationary_bootstrap_indices(
    n_obs: int, mean_block: float, n_boot: int, *, seed: int | None = 0
) -> np.ndarray:  # shape (n_boot, n_obs), row indices into the matrix
```

This is a suggestion, not a requirement. The point is that `snooping.py`
should never need to know how the indices were drawn.

**Stationary bootstrap** (Politis & Romano 1994): blocks start at random dates
and have geometric lengths with mean `mean_block`, wrapping around the end of
the sample. Why not an IID bootstrap: `docs/validation.md` lists volatility
clustering (`garch`) as exactly what a bootstrap of single days destroys.

**Block length** (QUANTIT-38): Politis & White (2004), with the correction in
Patton, Politis & White (2009). It is computed per series, so decide on one
rule for a whole matrix (for example the median across candidates) and write it
down. `statsmodels` is already a dependency; the `arch` package is not. Don't
add a dependency just for this. If you want to check against `arch` locally,
keep that out of the committed tests.

**Memory and speed.** 200 candidates x 2,520 dates x 1,000 replicates is
about 4 GB as float64 if you build it all at once. Loop over replicates, or
compute only the statistic you need per replicate (column means or Sharpes of
`values[idx]`), and keep the full resampled matrix out of memory.

**Reality Check** (White 2000): the statistic is the maximum across candidates
of the (scaled) mean performance over the benchmark (zero by default). Its null
distribution comes from the bootstrap maxima of the *recentred* series.

**Romano–Wolf** (Romano & Wolf 2005): a step-down version that says *which*
candidates beat the benchmark, keeping the chance of any false call at α. Use
studentized statistics. Return **adjusted p-values as a Series indexed by
candidate**, so they can sit next to `holm` and `by` in `evidence_profile`.
Under correlation it should be at least as powerful as Holm from #31; that
comparison is a useful power test.

**SPA** (Hansen 2005): studentized, and recentres only candidates that look
poor, so adding many useless candidates hurts it less than it hurts the
Reality Check. That is the property to test.

## Tests to write

- Bootstrap: lag-1 autocorrelation of AR(1) data (`ar1=0.3`) survives
  resampling. So does correlation across candidates (`rho=0.5`).
- Block length: stronger `ar1` or `garch` gives longer blocks; IID data gives
  short ones.
- Sharpe intervals: coverage close to nominal across many seeds on `ar1`,
  `t_df` and `garch` data. Compare with the analytic intervals from
  `sharpe_test`.
- Reality Check, Romano–Wolf, SPA: on correlated all-null matrices, the share
  of sets with any rejection is at most α (null calibration). On matrices with
  planted candidates, they are found (power).

## Who uses your output

- **Acceptance gate (Cal, QUANTIT-31 to 36).** Romano–Wolf is one of the main
  options for how the gate approves a *set* of candidates (QUANTIT-31 is linked
  to QUANTIT-41). A bootstrap of the maximum Sharpe across candidates is also a
  candidate reference for the gate's final stage, because it keeps the
  correlation that formula-based hurdles ignore.
- **Anyone who needs a resampling scheme later.** Keep the bootstrap interface
  free of snooping-specific details.
