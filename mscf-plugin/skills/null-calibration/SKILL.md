---
name: null-calibration
description: The test recipe every new validation method in cmu-mscf-2026 must ship with - a null-calibration test and a power test on synth.make_return_matrix, including correlated (rho), autocorrelated (ar1), fat-tailed (t_df) and GARCH nulls, with seeds and tolerance bands. Use when adding or changing a validation method, correction, bootstrap, CV scheme or gate, or reviewing such a PR.
---

# Null calibration and power

A procedure that cannot return "nothing here" is a ranking, not a validation. Every method
ships with both tests, in `tests/test_<module>.py`, marked fast enough for
`pytest -q -m "not network"`.

## 1. Null calibration

On a matrix with **no** real candidates (`n_real=0`), the method says "nothing" at the rate
it promises.

```python
import numpy as np
import pytest
from capstone.synth import make_return_matrix

NULL_CASES = {
    "iid": {},
    "correlated": {"rho": 0.4},
    "autocorrelated": {"ar1": 0.3},
    "fat_tails": {"t_df": 5},
    "garch": {"garch": (0.1, 0.85)},
}

@pytest.mark.parametrize("case", NULL_CASES)
def test_my_method_is_calibrated_under_the_null(case):
    alpha, reps = 0.05, 200
    rejected = 0
    for seed in range(reps):
        m = make_return_matrix(n_obs=1260, n_candidates=50, n_real=0, seed=seed, **NULL_CASES[case])
        rejected += my_method(m.returns, alpha=alpha).any()     # family-wise: any false call
    rate = rejected / reps
    assert rate <= alpha + 2.5 * np.sqrt(alpha * (1 - alpha) / reps)
```

- Test the error rate the method claims: FWER (any rejection), FDR (mean false-discovery
  proportion, needs some real candidates), or per-test size (rejection share of nulls).
- Set the band from the binomial standard error of the replication count, and say so in a
  comment. Fixed seeds (`range(reps)`), so the test is deterministic.
- If a case is outside the method's stated validity (Storey's q under `rho`), write the
  test anyway and assert it **fails** in the expected direction. That documents the limit,
  as `tests/test_evaluate.py` does for q and for IID Sharpe variance under `ar1`.

## 2. Power

On a matrix with planted candidates, the method finds them.

```python
def test_my_method_finds_planted_candidates():
    m = make_return_matrix(n_obs=2520, n_candidates=200, n_real=20, sharpe_real=1.0, rho=0.2, seed=0)
    calls = my_method(m.returns, alpha=0.10)
    from capstone.evaluate import power, false_discovery_rate
    assert power(calls, m.truth) >= 0.5
    assert false_discovery_rate(calls, m.truth) <= 0.10 + 0.05
```

- Use a Sharpe the method should plausibly detect at that sample length (about 1.0 over 10
  years); also show it is weak where it should be (`sharpe_real=0.25`, `n_obs=756`).
- Include at least the `rho` and `ar1` cases: real candidates from one search are both
  correlated and autocorrelated.

## 3. Ship

- Each test must fail without the change (`CLAUDE.md`: new behavior ships with a failing
  test). Show it failing once, by breaking the method locally.
- Keep runtime under a few seconds per case; lower `reps` and widen the band rather than
  mark it slow.
- Log simulation runs to the real ledger tagged `synthetic` (exclude them from a
  strategy's count with `exclude_tags`); in pytest, point `CAPSTONE_LEDGER_DIR` at `tmp_path`.
- In the PR, report each case's realised rate and band, and add the method to the "What
  exists" table in `docs/validation.md`.
