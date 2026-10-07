# Brief: probability of backtest overfitting and the haircut Sharpe

**Owner:** Pin-Hua Chen. Read [handoff.md](handoff.md) first.

## Why this workstream matters

Both methods answer "how much of the best result is just the result of
searching?" PBO asks whether choosing the best candidate on one half of the
data tells you anything about the other half. The haircut Sharpe shrinks a
Sharpe ratio to allow for how many trials were run. Both are used at the
robustness and final-decision stages of the calibrated validation funnel in
`docs/validation.md`.

## Tasks

| Jira | Task | Blocked by |
|---|---|---|
| QUANTIT-46 | Review PR #35 (CSCV / PBO) | nothing |
| QUANTIT-12 | Check the PBO statistic against the paper (in #35) | nothing |
| QUANTIT-47 | PBO calibration tests on `make_return_matrix`, and docs | #35, #28 |
| QUANTIT-51 | Harvey–Liu haircut Sharpe | #32 (`holm_adjusted`, `by_adjusted`) |
| QUANTIT-52 | Use the ledger's trial count in the haircut | QUANTIT-51, #33 |

You're the requested reviewer on **#33** (the ledger reader your haircut
uses) and **#35** (PBO).

## PBO: already in PR #35

`capstone/pbo.py` and `tests/test_pbo.py` are in **PR #35** against `main`,
unchanged from an earlier unmerged implementation. It already matches the
contract:

```python
cscv(returns_matrix: pd.DataFrame, n_blocks: int = 16, *, periods_per_year: int = 252) -> dict
# keys: pbo, degradation_slope, degradation_intercept, oos_ranks,
#       winner_is_sharpe, winner_oos_sharpe, winner_columns
```

It follows Bailey, Borwein, López de Prado & Zhu (2016/2017): split the sample
into `n_blocks` (even) contiguous blocks, use every combination of half the
blocks as in-sample, find the in-sample winner, and record its relative rank
out of sample. PBO is the share of splits where the winner lands below the
out-of-sample median. Rows with any missing value are dropped so every
candidate is scored on the same dates.

**Checked 2026-09-27.** The full suite passes on plain `main` with #35, and
also on the validation stack (#32). `cscv` works directly on
`make_return_matrix` data (1,000 dates x 50 candidates, `n_blocks=10`):

| Matrix | Mean PBO | Range across seeds |
|---|---|---|
| IID null, 100 seeds | 0.478 (se 0.017) | |
| Fat-tailed null (`t_df=5`), 100 seeds | 0.471 (se 0.020) | |
| Correlated null (`rho=0.5`), 20 seeds | 0.46 | 0.10–0.75 |
| Autocorrelated null (`ar1=0.3`), 20 seeds | 0.46 | 0.06–0.71 |
| Volatility-clustering null (`garch=(0.05, 0.9)`), 20 seeds | 0.47 | 0.10–0.78 |
| 1 planted candidate, Sharpe 3, among 49 nulls | 0.02 | 0.00–0.25 |
| 5 planted, Sharpe 2, `rho=0.5` | 0.00 | 0.00–0.00 |

So:

- **QUANTIT-46 and QUANTIT-12 are reviewing #35.** `cscv` already generates
  the splits and computes PBO and the performance-drop fit.
- **QUANTIT-47** is the remaining work: calibration tests on
  `make_return_matrix` (the tests in #35 use `make_panel`), covering the cases
  in the table, plus docs. Also update the module docstring: it says to build
  the input with `run_backtest` per candidate, and `backtest.candidate_returns`
  (#28) now does that.
- **PBO from one candidate set is noisy.** On no-skill sets it ranged from
  about 0.06 to 0.78 depending on the seed. A calibration test has to average
  over many seeds, and the gate shouldn't treat a single PBO near a threshold
  as decisive.

**Reading the number:**

- **PBO around 0.5 is the no-skill baseline, not a clean bill of health.**
  With independent no-skill candidates, the winner's out-of-sample rank is
  uniform, so about half the splits land below the median by symmetry.
  The tests in #35 calibrate this to a band around 0.5 across seeds; a threshold like
  "> 0.7 means overfit" is wrong for independent nulls.
- PBO climbs toward 1 when candidates are **coupled**, for example many
  variants of one idea, where the in-sample winner is systematically the one
  that fit noise. The tests in #35 have a coupled case; keep it.
- A strong planted candidate gives low PBO.
- Watch the number of combinations: C(16, 8) = 12,870 splits. Keep
  `n_blocks` modest in tests.

## Haircut Sharpe

Harvey & Liu (2015), "Backtesting", *Journal of Portfolio Management*. Turn
each Sharpe into a p-value, adjust it for the number of trials, and convert
the adjusted p-value back into a Sharpe. The haircut is the difference.

- **Reuse the adjustments from #32**: `holm_adjusted` and `by_adjusted` (BY is
  the paper's BHY). Bonferroni is `min(1, p * M)`. Don't reimplement them.
- Check the p-value convention against the paper before you code; the paper's
  worked examples are your test cases. Our screening convention elsewhere is
  one-sided (`pvalue_greater`), so say which one the function uses.
- The trial count M must come from the ledger (QUANTIT-52). With correlated
  trials, don't substitute `implied_independent_trials` for M in the decision;
  report it alongside if useful.

## Trial count (QUANTIT-52)

Use `capstone.runlog.trial_count()`; don't write another reader. It comes in
its own PR, #33 (see handoff.md). It counts every
logged trial, or only those under one experiment name, and raises
`LookupError` instead of returning 0. Let that error propagate: a haircut
computed against zero trials is no haircut. Tests set `CAPSTONE_LEDGER_DIR` to
a temporary directory and log a known number of trials.

## Where the code goes

- `capstone/pbo.py`, `tests/test_pbo.py` (your own module).
- Haircut in `capstone/evaluate.py`: add it **at the end of the file, in its
  own section**. Cal's local-FDR work also lands in `evaluate.py`.
