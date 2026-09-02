# Decision Log

## 2026-08-25

Decision: Use Benjamini-Hochberg (FDR control) as the correction at the candidate-screening stage, not Bonferroni.

Why: On the effect-size sweep, BH recovers more real signal in the faint regime. At effect size 0.005, BH power was 0.25 versus Bonferroni power 0.083, while both methods had 0.00 FDR in this run. At 0.0075 and 0.010, BH power was 0.75 and 1.00 versus Bonferroni power 0.667 and 0.917, with BH FDR at 0.10 and 0.077 and Bonferroni FDR at 0.00. Faint signals are the project's actual regime. BH's false positives are caught downstream by the deflated-Sharpe acceptance gate, which Bonferroni-level strictness at screening would forfeit real signal to reach.

Pre-registered: yes — decided from the sweep BEFORE running real data.

Evidence: exp_candidate_corrections, effect-size sweep, logged in ledger.
