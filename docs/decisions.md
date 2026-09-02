# Decision Log

## 2026-08-25

Decision: Use Benjamini-Hochberg (FDR control) as the correction at the candidate-screening stage, not Bonferroni.

Why: On the effect-size sweep, BH recovers more real signal in the faint regime. At effect size 0.005, BH power was 0.25 versus Bonferroni power 0.083, while both methods had 0.00 FDR in this run. At 0.0075 and 0.010, BH power was 0.75 and 1.00 versus Bonferroni power 0.667 and 0.917, with BH FDR at 0.10 and 0.077 and Bonferroni FDR at 0.00. Faint signals are the project's actual regime. BH's false positives are caught downstream by the deflated-Sharpe acceptance gate, which Bonferroni-level strictness at screening would forfeit real signal to reach.

Pre-registered: yes — decided from the sweep BEFORE running real data.

Evidence: exp_candidate_corrections, effect-size sweep, logged in ledger.

## 2026-09-02

Decision: Removed 45 rows from the run ledger (`experiments/runs.jsonl`) that were
written by the test suite, not by experiments. Ledger went 91 -> 46 runs.

Why: `tests/test_ml_candidates.py` called `screen_ml_candidates`, which calls
`log_run`/`log_outcome`, but unlike `tests/test_runlog.py` it never redirected
`CAPSTONE_LEDGER_DIR` to a tmp path. Every `pytest` run by anyone, on any branch,
appended 9 rows to the real ledger: 3 `ml_candidate` trials, 3
`ml_candidate_screen` trials, and 3 outcomes. Five such runs had accumulated (two
on 2026-08-26 by the original author, three on 2026-09-02 during agent work).

The ledger's length is the denominator of every multiple-testing correction in
`capstone.evaluate`. Test residue inflates the trial count with hypotheses nobody
ever posed, which makes every downstream correction wrongly conservative and the
project's own trial accounting untrue. The ledger is append-only by convention,
so removing rows is itself notable -- it is recorded here rather than done
silently, and the pre-purge file was backed up before the edit.

Identification: the residue is exactly the rows carrying `kind: trial`/`outcome`
with name `ml_candidate` or `ml_candidate_screen`, plus the outcome rows
referencing those trials. They are distinguishable from real runs by
`params.n_candidates: 3`, `metrics.signal_shape: [600, 25]` (the test's panel
dimensions), `user: "unknown"`, and timestamps landing exactly on pytest
invocations. No orphaned outcome rows were left behind.

Kept deliberately: 12 older rows named `ml_candidate` (9) and
`ml_candidate_screen` (3) from 2026-08-26 03:47-03:52 predate the two-phase
ledger format and came from real manual runs of the module, not from tests.
Deleting the module does not un-try those trials, so they stay in the count.

Root cause fixed by deletion: `capstone/ml_candidates.py`,
`tests/test_ml_candidates.py`, and `experiments/exp_candidates_sample.py` were
removed as search-side extras (commit 03eed06 said it dropped them; it did not).
Verified afterwards that a full `pytest -q -m "not network"` run leaves the
ledger byte-for-byte unchanged.
