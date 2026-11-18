---
name: validate-candidates
description: How to judge whether candidate signals or strategies in cmu-mscf-2026 are real - build the dates x candidates performance matrix, test Sharpe ratios one-sided with autocorrelation-robust variance, apply the stage's gate (one luck correction per stage) plus graded scores, deflate by the ledger's trial count, and place the result in the five-stage funnel. Use when scoring candidates, reporting a discovery, or placing a result in the funnel.
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

The profile is a set of graded scores, not a menu of decisions (section 3).

## 3. One luck correction per stage is the gate; every other column is a graded score

`docs/validation/framework.md` ("One correction for luck per stage"): each stage uses one
correction as its gate (section 4 says which). The others may be reported as graded
scores. Two gates at one stage charge for the same search twice (charging the deflated
Sharpe the full count cut power from 0.47 to 0.08). Do not choose the gate by which column
says yes.

| Column | Role | Notes |
|---|---|---|
| `benjamini_hochberg` / `bh` over each family's bootstrap p-value | **Gate**, stage 1 | One-sided p-values, positive dependence; two-sided p-values void the guarantee |
| `benjamini_hochberg` / `bh` over survivors' holdout p-values | **Gate**, stage 4 | The deflated Sharpe lost to it in every calibration run |
| `benjamini_yekutieli` / `by` | Graded score | Valid under any dependence; costs power |
| `holm` | Graded score | Strictest guarantee; far too strict for a screen |
| `storey_qvalues` / `q` | Graded score, independent candidates only | Over-calls more than 2x on correlated candidates |
| `deflated_sharpe_ratio(..., n_trials=...)` | Graded score | Headline Sharpe of the winner; `n_trials` from the ledger (`run-ledger`), never typed |
| `search_fdr.search_adjusted_pvalue` | Cross-check only | Fast formula for a family's best-of-K p-value; not the gate |

Pick before looking. Report the full `evidence_profile` row, not only the column that says
yes; a candidate strong only under `q` is not a discovery.

## 4. Place it in the funnel

| Stage | Question | Gate |
|---|---|---|
| 1 Screen | Worth a closer look? | BH across families on each family's `bootstrap_family_pvalues` (block bootstrap of the family maximum); `search_adjusted_pvalue` is a cross-check, not the gate |
| 2 Robustness | Survives other lags, costs, sub-periods? Search overfit? | graded checks (CV, PBO) |
| 3 Incremental | Adds beyond FF5 + momentum and known anomalies? | spanning alpha (planned) |
| 4 Final | Allocate? | BH across survivors' holdout p-values, holdout looked at once |
| 5 Incubation | Works live? | small paper capital |

Until the funnel is calibrated and shown to beat it, **the baseline decides** (Q5 in
`framework.md`): BH at the registered level on each family's bootstrap p-value (10% in
`framework.md` at the time of writing; re-read it, since values are registered there). The
funnel replaces the baseline only once it beats it.

Threshold values come from `capstone.calibration` simulations of the real generator and are
registered in `docs/validation/framework.md` before scoring; none were registered when this
was written. The defaults in code (`Thresholds()`, `run_baseline`) are not registered, and
both default to `stage1="rho"`, the formula p-value. For real candidates stage 1 must pass
`stage1="bootstrap"`. Tag the ledger entry with the stage (`stage:screen`, ...).

## 5. Report

State: series type, period, cost model, seed(s), ledger trial count, the stage's gate and the graded scores,
the full evidence profile, and for anything that passed the holdout, that the model may
remember 2021-2025.
