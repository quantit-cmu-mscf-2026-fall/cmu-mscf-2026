---
name: validate-candidates
description: How to judge whether candidate signals or strategies in cmu-mscf-2026 are real - build the dates x candidates performance matrix, test Sharpe ratios one-sided with autocorrelation-robust variance, pick the right multiple-testing correction for the dependence, deflate by the ledger's trial count, and place the result in the five-stage funnel. Use when scoring candidates, reporting a discovery, or choosing a correction.
---

# Validate candidates

The contract is `docs/validation.md`; the stage design is `docs/validation/framework.md`.
Before a validation *task* (building a method), read `docs/validation/handoff.md` and your
workstream brief first. Method details and module status: `references/methods.md`.

## 1. Build the performance matrix

One per-period series per candidate, dates x candidates, columns named by candidate,
higher is better, search period only (<= 2020-12-31).

- Synthetic panel: `backtest.candidate_returns(panel, cost_bps=...)`.
- Own signals: `backtest.run_backtest` each, side by side, drop the first (untraded) date.
- Per-date rank IC is also valid. Say which series you passed: a Sharpe of an IC series is
  an information ratio, not a strategy Sharpe.
- Score net of costs (`sd.quoted_spread`) from stage 1; report gross alongside.

Validation code never imports strategy code. Hand the matrix over; do not pass signals.

## 2. Log, then test

Log each candidate with `runlog.log_run` (see `run-ledger`) before reading its numbers. Then:

```python
from capstone import evaluate as ev

tests = {c: ev.sharpe_test(matrix[c], method="hac") for c in matrix}   # HAC: autocorrelation-robust
p = pd.Series({c: t.pvalue_greater for c, t in tests.items()})        # one-sided: only SR > 0 counts
profile = ev.evidence_profile(p)                                       # holm, by, bh, q side by side
```

## 3. Choose the correction by dependence, not by result

| Situation | Use | Do not use |
|---|---|---|
| Any dependence, want no false call at all | `holm` | `bonferroni` (dominated by Holm) |
| Any dependence, FDR | `benjamini_yekutieli` / `by` | |
| One-sided p-values of positively correlated candidates (one search) | `benjamini_hochberg` / `bh` | two-sided p-values |
| Independent candidates only | `storey_qvalues` / `q` | `q` on correlated candidates: it over-calls >2x |
| Best of K variants of one idea | `search_fdr.search_adjusted_pvalue` / `bootstrap_family_pvalues`, then BH across families | the raw best p-value |
| Headline Sharpe of the winner | `deflated_sharpe_ratio(..., n_trials=<ledger count>)` | a typed `n_trials` |

Pick before looking. Report the full `evidence_profile` row, not only the column that says
yes; a candidate strong only under `q` is not a discovery.

## 4. Place it in the funnel

| Stage | Question | Gate |
|---|---|---|
| 1 Screen | Worth a closer look? | BH across families on each family's search-adjusted p-value |
| 2 Robustness | Survives other lags, costs, sub-periods? Search overfit? | graded checks (CV, PBO) |
| 3 Incremental | Adds beyond FF5 + momentum and known anomalies? | spanning alpha (planned) |
| 4 Final | Allocate? | BH across survivors' holdout p-values, holdout looked at once |
| 5 Incubation | Works live? | small paper capital |

Thresholds come from `capstone.calibration` simulations, fixed before scoring. Tag the
ledger entry with the stage (`stage:screen`, ...).

## 5. Report

State: series type, period, cost model, seed(s), ledger trial count, the correction and why,
the full evidence profile, and for anything that passed the holdout, that the model may
remember 2021-2025.
