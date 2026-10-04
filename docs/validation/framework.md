# Validation framework: one standard, calibrated to the generator

**Proposed 2026-10-04 for the team to confirm in review.** This page says how
the methods in [`../validation.md`](../validation.md) and the workstream briefs
fit together into one decision. It adds no method of its own.

## The problem this solves

Each method we have, or have planned, answers a real question. Applied
together as a checklist, though, they reject almost everything, including real
signals of the strength an agent can plausibly find. That is not a bug in any
one method. A candidate that must pass every gate faces a rule that is never
more powerful than its strictest gate, and usually much less, because each
gate spends part of the same evidence. Several of our methods also apply the same
penalty: Holm, BY, the deflated Sharpe, the haircut Sharpe and the Reality
Check all correct for how many candidates were tried, so applying three of them
charges that penalty three times.

So the framework does two things:

1. **Each method gets exactly one role**, and each stage has exactly one gate.
2. **Gate thresholds are calibrated** to how good our generator actually is
   (its base rate of real candidates, their strength and how correlated its
   output is), measured on simulated candidate sets with known truth. They are not left at
   textbook defaults.

## Four roles

| Role | What it does to a candidate | Costs power? |
|---|---|---|
| **Construction requirement** | How every candidate's performance series is *built*. Not a test: a series built any other way isn't admissible. | No |
| **Gate** | Pass or fail at a threshold fixed before scoring. Exactly one per stage. | Yes, and this is the only role that does |
| **Graded score** | Reported next to the gate on the scorecard, never used to reject. Lets a reader see *how* strong the evidence is. | No |
| **Batch diagnostic** | Judges the search that produced a set of candidates, not any one candidate. It tells us how strict the gates need to be. | No, not directly |

The point of separating them is that only gates remove candidates. Everything
else either guarantees the series is honest (construction) or informs a person
(scores, diagnostics), without stacking another rejection rule on top.

## One correction for luck per stage

These all answer the same question: *is this better than the best of what
searching through that many no-skill candidates would produce?*

- Bonferroni, Holm, Romano–Wolf: chance of *any* false call
- BH, BY, Storey q, local FDR: share of false calls among those made
- Deflated Sharpe, haircut Sharpe, Reality Check, SPA: the best result
  compared with the best of many null candidates

**Each stage uses one of them as its gate.** The others may appear as graded
scores. Using two as gates at one stage charges for the same search twice.

## The catalog

Known weaknesses are measured ones, from the PRs cited.

| Method | Failure mode it catches | Unit | Strength | Known weakness | Role | Stage |
|---|---|---|---|---|---|---|
| Purged k-fold with embargo (`cv`, #34) | Look-ahead through overlapping labels | candidate | Removes leakage when a model is fitted | Not needed for a series with no fitted parameters | Construction | all |
| Walk-forward (`cv`, planned) | Using the future to refit | candidate | Mimics live refitting | Fewer test dates than k-fold | Construction | all |
| Net-of-cost returns (#43) | Untradable signals | candidate | Most anomalies vanish after costs | Cost model is an estimate | Construction | all |
| Point-in-time data, delisting returns (#43) | Survivorship and look-ahead in data | data | No method below can fix this | Needs the right data source | Construction | all |
| Holdouts looked at once (#43) | Reusing the final test; model memory of history | data | Post-cutoff holdout guards against memorisation | Short holdouts have little power | Construction | 4 |
| HAC Sharpe test, one-sided (`sharpe_test`, #29) | One series being luck | candidate | Allows for autocorrelation and fat tails | Slightly liberal in small samples (5.5–8% at a nominal 5% on AR(1) nulls) | Input to gates | 1, 2 |
| BH adjusted p (`bh_adjusted`, #32) | Luck across many candidates | candidate | Most power of the FDR controls | Guarantee needs positive dependence, i.e. one-sided tests | **Gate** | 1 |
| BY adjusted p (`by_adjusted`, #32) | Same | candidate | Valid under any dependence | Costs power; never failed in our simulations where BH held | Graded score | 1 |
| Holm adjusted p (`holm_adjusted`, #32) | Any false call | candidate | Strictest guarantee | Far too strict for a screen | Graded score | 1, 4 |
| Storey q (#31, #32) | Same as BH | candidate | More power when candidates are independent | **Unsafe when correlated**: 10–14% any-discovery rate at a nominal 5% | Graded score, independent candidates only | — |
| Local FDR / empirical null (planned) | Same as BH, under correlation | candidate | A probability of being real per candidate | Not built yet | Alternative stage-1 gate | 1 |
| Family-level bootstrap max + BH/BY (Carl, `carl/validation1-hypothesis-fdr`) | Search within one hypothesis | hypothesis | Uses the correlation between a hypothesis's variants instead of counting each one | Needs the family structure recorded | **Gate**, when families exist | 1 |
| PSR, MinTRL (#29) | Too short a track record | candidate | Readable: "how many years would we need?" | Single-candidate; no multiple testing | Graded score | 2, 4 |
| Robustness to lags, costs, sub-periods | Fragile signals | candidate | Catches results that hang on one choice | Each check is another test if used as a gate | Graded score | 2 |
| PBO / CSCV (`pbo`, #35) | Overfitting in the *selection* of the best | batch | Measures the search, not one candidate | 0.5 is the no-skill baseline; a single set ranges 0.06–0.78 | Batch diagnostic | 2 |
| Average correlation, implied independent trials (#29) | Overcounting correlated trials | batch | Explains why a correction is strict | Must not replace the ledger count in a decision | Batch diagnostic | all |
| Search-adjusted FDR estimate (below) | Hidden specification search | batch | Uses what we can see: the whole search | Depends on the search being fully logged | Batch diagnostic | 1 |
| Spanning alpha (`spanning`, planned) | Known factor in disguise; copy of a held signal | candidate | The novelty test | Needs the comparison set (#43) | **Gate** | 3 |
| Deflated Sharpe (#29) | Best of N trials being luck | candidate | Uses the ledger count and the spread of trial Sharpes | Needs the full trial count | **Gate** | 4 |
| Haircut Sharpe (planned) | Same | candidate | Readable as "the Sharpe after the search" | Same question as the deflated Sharpe | Graded score | 4 |
| Reality Check, Romano–Wolf, SPA (planned) | Best of many correlated candidates | set | Keeps correlation by resampling | Not built yet; same question as the deflated Sharpe | Alternative stage-4 gate | 4 |

## The funnel, with roles assigned

The stages are the ones in [`../validation.md`](../validation.md). Each stage
looks at dates the earlier stages never used; that is what makes a loose screen
safe.

| Stage | Data | Gate | Graded scores reported | Batch diagnostics |
|---|---|---|---|---|
| 1. Screen | Development period | BH-adjusted one-sided HAC p ≤ q₁ across hypothesis families (search-adjusted within each family) | BY, Holm, raw p | Search-adjusted FDR, average correlation |
| 2. Robustness | A later period unused in stage 1 | One-sided HAC p ≤ α₂ on that period, for stage-1 survivors only | PSR, lag / cost / sub-period sensitivity | PBO of the search |
| 3. Incremental value | Same period as stage 2, monthly returns | Spanning alpha p ≤ α₃ | Factor loadings | — |
| 4. Final decision | Holdout, looked at once | Deflated Sharpe ≥ d₄, with the ledger's full trial count | Haircut Sharpe, Holm, MinTRL | — |
| 5. Incubation | Live or paper | — | Realised vs. expected Sharpe | — |

Stage 2 tests only the few stage-1 survivors on new data, so its family is small
and its threshold can be modest. The multiple-testing penalty for the full search is
charged twice on purpose: loosely at stage 1 (over the families screened) and
strictly at stage 4 (over the ledger's full count). It is not charged at every stage.

## Search-adjusted testing

López de Prado & Fabozzi (2026, SSRN 6450418; issue #41) argue that the
false-discovery rate **cannot be identified** from reported statistics when
those statistics are the winners of a search over many specifications. Very
different mixes of real candidates, effect sizes and search intensity produce
nearly the same profile of reported significance and imply very different FDRs.
In their illustration (Section 4.4), a world where each report is the best of
10 variants and 95% of variants are null, and a world with no search and 15%
nulls, report almost the same upper tail; their FDRs are 0.575 and 0.083.
Fitting their best-of-K model to 212 published predictors gives an FDR of 0.83
at the best-fitting K = 5, which the authors stress is conditional on the
assumed search model, not a measurement of it.

Our position differs from the published literature in one useful way: **we can
see the whole search.** The ledger records every trial, and the factor store
(#46–#49) records every proposal with its lineage. So:

- **The stage-1 unit is the hypothesis family**: one economic idea and all the
  expressions and parameters tried for it. Its p-value is adjusted for its own
  search, with Carl's bootstrap max when the variants are correlated, or
  1 − (1 − p)^K for K independent variants. BH then runs across families. This
  *replaces* BH across every individual variant; it is not added on top.
- **The deflated Sharpe's trial count is everything searched**, not only what
  was reported.
- **Methods that estimate the share of nulls from the candidates' own
  statistics** (local FDR, empirical Bayes, Storey's π₀) have the same problem
  if they are fed only winners; the paper's Theorem 1 names them. Feed them
  every trial, or the family-adjusted p-values.
- **The search-adjusted FDR estimate is a batch diagnostic.** A high estimate
  for a batch says the generator is searching hard for little, and stage 1
  should be tightened for that generator, not loosened.
- **The ledger has to record the search.** Each trial carries
  `hypothesis_id`, `parent_id` (the trial it was revised from) and whether its
  result was shown to the agent, as `log_run` params. Without these, K is
  unknown and the identification problem comes back.

## Calibration

The thresholds q₁, α₂, α₃ and d₄ are set by simulation, before any real
candidate is scored, and written into this page.

**Assumptions, from our generator:**

- Share of real candidates π₁: 0 to 5% ([`../validation.md`](../validation.md),
  "Decisions the stages depend on").
- Strength of a real candidate: Sharpe 0.5 to 1.5 net of costs.
- Search intensity K per hypothesis, correlation between candidates, lag-1
  autocorrelation and kurtosis: measured from the generator's own output
  (`evaluate.average_correlation` and the series themselves), then set in
  `synth.make_return_matrix`.

**Objective:** the largest end-to-end power (share of planted candidates that
reach stage 5) subject to a target false-discovery rate among candidates that
pass stage 4. The target, and how a missed signal is weighed against a false
one (Harvey & Liu 2020), is a team decision; record it here.

**Procedure:** run the whole funnel, not each method alone, on simulated
candidate sets across the assumption grid, for each candidate threshold set.
Report, by stage, how many real and null candidates survive. Pick the
threshold set that meets the objective at the central assumptions and degrades
gracefully at the edges. Also report the naive alternative (every method as a
gate on all the data) so the cost of stacking is visible.

**Recalibrate when the generator changes:** a new agent, model, prompt set or
search budget changes π₁ and K. The thresholds belong to a generator, not to
the project.

**The null check still applies:** on sets with no real candidates, the
calibrated funnel must pass almost nothing (`CLAUDE.md`, "Research hygiene").

## The scorecard

One row per candidate that reached stage 1, so a reader can see why each one
stopped:

- identity: candidate, hypothesis family, K, generator and version
- per stage: the gate's statistic, its threshold, pass or fail
- graded scores: BY, Holm, PSR, MinTRL, haircut Sharpe, sensitivity checks
- header for the batch: ledger trial count, search-adjusted FDR estimate, PBO,
  average correlation, the threshold set used and when it was calibrated

A candidate is accepted when it passes every gate. Graded scores explain the
decision; they never overturn it.

## Questions for the team

1. **#37 and #31 overlap.** Both add BH and BY. Proposal: keep #31's functions,
   since the rest of the stack builds on them, and fold anything #37 adds that
   #31 lacks into it.
2. **Stage-1 gate when families exist:** Carl's bootstrap max plus BH (proposed
   here), or local FDR once it is built?
3. **Stage-4 gate:** the deflated Sharpe (proposed here, since it exists),
   Romano–Wolf or SPA once built? Only one is the gate.
4. **Trial registry fields:** agree on `hypothesis_id`, `parent_id` and
   `shown_to_agent` as standard `log_run` params, and whether the factor store
   or the ledger is their source of truth.
5. **The calibration target:** the end-to-end FDR target and the
   miss-to-false-discovery ratio.

## What this page doesn't change

Every method keeps its owner, module and tests. The contract in
[`../validation.md`](../validation.md) is unchanged: a performance matrix in,
graded scores out, trial counts from the ledger, a null-calibration test and a
power test for every method. This page only says which output decides.
