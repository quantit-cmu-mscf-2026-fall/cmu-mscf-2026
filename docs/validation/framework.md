# Validation framework: one standard, calibrated to the generator

**Proposed 2026-10-04; team decisions recorded 2026-10-05** (see "Decisions"
at the end, one still open). This page says how
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
| Holdout 2021–2025, looked at once (#43) | Reusing the final test | data | Five years can confirm an annual Sharpe of about 0.75 | Predates the agents' model cutoff, so a pass is not established against memorisation | Construction | 4 |
| HAC Sharpe test, one-sided (`sharpe_test`, #29) | One series being luck | candidate | Allows for autocorrelation and fat tails | Slightly liberal in small samples (5.5–8% at a nominal 5% on AR(1) nulls) | Input to gates | 1, 2 |
| BH adjusted p (`bh_adjusted`, #32) | Luck across many candidates | candidate | Most power of the FDR controls | Guarantee needs positive dependence, i.e. one-sided tests | **Gate** | 1 |
| BY adjusted p (`by_adjusted`, #32) | Same | candidate | Valid under any dependence | Costs power; never failed in our simulations where BH held | Graded score | 1 |
| Holm adjusted p (`holm_adjusted`, #32) | Any false call | candidate | Strictest guarantee | Far too strict for a screen | Graded score | 1, 4 |
| Storey q (#31, #32) | Same as BH | candidate | More power when candidates are independent | **Unsafe when correlated**: 10–14% any-discovery rate at a nominal 5% | Graded score, independent candidates only | — |
| Local FDR / empirical null (`local_fdr`, #54) | Same as BH, under correlation | candidate | A probability of being null per candidate, against a null that moves with a shared factor | Needs 200+ p-values; "real" means stands out from the other candidates, not Sharpe above zero | Graded score (`lfdr` column); replaces the stage-1 gate only if the harness shows a power gain | 1 |
| Family-level bootstrap max + BH (Carl, `carl/validation1-hypothesis-fdr`) | Search within one hypothesis | hypothesis | Keeps any correlation pattern between a hypothesis's variants | Needs the family structure recorded; not yet a module | **Gate** | 1 |
| Search-adjusted family p-value (`search_adjusted_pvalue`, #51) | Search within one hypothesis | hypothesis | Exact for independent or evenly correlated variants (`rho=`); used for the baseline | The independent formula over-corrects near-copies: power 0.64 vs 0.81 at K = 20, ρ = 0.9 | **Gate** input until the bootstrap is a module; the baseline's input | 1 |
| PSR, MinTRL (#29) | Too short a track record | candidate | Readable: "how many years would we need?" | Single-candidate; no multiple testing | Graded score | 2, 4 |
| Robustness to lags, costs, sub-periods | Fragile signals | candidate | Catches results that hang on one choice | Each check is another test if used as a gate | Graded score | 2 |
| PBO / CSCV (`pbo`, #35) | Overfitting in the *selection* of the best | batch | Measures the search, not one candidate | 0.5 is the no-skill baseline; a single set ranges 0.06–0.78 | Batch diagnostic | 2 |
| Average correlation, implied independent trials (#29) | Overcounting correlated trials | batch | Explains why a correction is strict | Must not replace the ledger count in a decision | Batch diagnostic | all |
| Search-adjusted FDR estimate (below) | Hidden specification search | batch | Uses what we can see: the whole search | Depends on the search being fully logged | Batch diagnostic | 1 |
| Spanning alpha (`spanning`, planned) | Known factor in disguise; copy of a held signal | candidate | The novelty test | Needs the comparison set (#43) | **Gate** | 3 |
| Deflated Sharpe (#29) | Best of N trials being luck | candidate | Uses the ledger count and the spread of trial Sharpes | As a per-candidate gate over 1,000 hypotheses with the searched trial count it finds 2–11% of real signals (#51), so only for the few that reach stage 4 | **Gate** (proposed; open) | 4 |
| Haircut Sharpe (`haircut_sharpe`, #65) | Same | candidate | Readable as "the Sharpe after the search"; follows the authors' code | Same question as the deflated Sharpe | Graded score | 4 |
| Reality Check, Romano–Wolf, SPA (planned) | Best of many correlated candidates | set | Keeps correlation by resampling | Not built yet; same question as the deflated Sharpe | Alternative stage-4 gate | 4 |

## The funnel, with roles assigned

The stages are the ones in [`../validation.md`](../validation.md). Each stage
looks at dates the earlier stages never used; that is what makes a loose screen
safe.

| Stage | Data | Gate | Graded scores reported | Batch diagnostics |
|---|---|---|---|---|
| 1. Screen | Search period, 1990–2020 (#43) | BH ≤ q₁ across hypothesis families, on each family's search-adjusted p-value (bootstrap max) | BY, Holm, `lfdr`, raw p | Search-adjusted FDR, average correlation |
| 2. Robustness | Not yet decided: see "Stage 2's fresh data" below | One-sided HAC p ≤ α₂ on that data, for stage-1 survivors only | PSR, lag / cost / sub-period sensitivity | PBO of the search |
| 3. Incremental value | Same period as stage 2, monthly returns | Spanning alpha p ≤ α₃ | Factor loadings | — |
| 4. Final decision | Holdout, 2021–2025, looked at once | Deflated Sharpe ≥ d₄ with the full searched trial count (proposed; open) | Haircut Sharpe, Holm, MinTRL | — |
| 5. Incubation | Live or paper | — | Realised vs. expected Sharpe | — |

Stage 2 tests only the few stage-1 survivors on new data, so its family is small
and its threshold can be modest. The multiple-testing penalty for the full search is
charged twice on purpose: loosely at stage 1 (over the families screened) and
strictly at stage 4 (over the ledger's full count). It is not charged at every stage.

### Stage 2's fresh data

#43 gives agents and stages 1–3 the whole search period, 1990–2020. Stage 2
is only meaningful on dates the search never saw, and there are none inside
that period once the agents have used all of it. Two ways out, for the team to
choose:

- **Split the search period:** agents and stage 1 see 1990–2013; stage 2
  confirms on 2014–2020, untouched until then. Simple, but the agents search
  on seven fewer years.
- **Walk-forward inside it:** stage 2 re-fits and scores each survivor on
  rolling out-of-sample windows (`cv`, walk-forward, QUANTIT-45). Keeps all
  the data for search, but only works for candidates with fitted parameters;
  a fixed formula has nothing to re-fit, so its windows were still seen.

Until this is decided, the calibration harness treats stage 2 as a separate
period of configurable length.

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
  search: Carl's bootstrap max (decided, Q2), or `search_adjusted_pvalue`
  (#51) with the family's measured correlation `rho` until the bootstrap is a
  module. BH then runs across families. This *replaces* BH across every
  individual variant; it is not added on top. Use the correlation: an agent's
  variants of one idea are near-copies (ρ ≈ 0.9), and the independent formula
  then over-corrects.
- **Use the real K.** An "effective" count such as
  `implied_independent_trials` makes the adjustment anti-conservative (#51
  review); allow for correlation through `rho` or the bootstrap instead.
- **The deflated Sharpe's trial count is everything searched**, not only what
  was reported.
- **Methods that estimate the share of nulls from the candidates' own
  statistics** (local FDR, empirical Bayes, Storey's π₀) have the same problem
  if they are fed only winners; the paper's Theorem 1 names them. Feed them
  every trial, or the family-adjusted p-values.
- **The search-adjusted FDR estimate is a batch diagnostic,** and a
  sensitivity table rather than a gate: on pure noise it can read "mostly
  real" (#51). A high estimate for a batch says the generator is searching
  hard for little, and stage 1 should be tightened for that generator, not
  loosened.
- **The search has to be recorded** (decided, Q4). The factor store is the
  source of truth for lineage: its proposals table already records
  `hypothesis_id` and `parent_factor_id` for every proposal, rejected ones
  included, so K counts what was searched. A trial is logged to the ledger,
  with `factor_id` and `hypothesis_id` as `log_run` params, when its
  performance is computed. `shown_to_agent` will be added to the store in the
  run loop (#49), where results go back to the agent. Without these, K is
  unknown and the identification problem comes back.

## Calibration

The thresholds q₁, α₂, α₃ and d₄ are set by simulation, before any real
candidate is scored, and written into this page.

**Assumptions, from our generator:**

- Share of real candidates π₁: 0 to 5% ([`../validation.md`](../validation.md),
  "Decisions the stages depend on").
- Strength of a real candidate: Sharpe 0.5 to 1.5 net of costs.
- Search intensity K per hypothesis, correlation between candidates and
  within a family (near-copies, ρ ≈ 0.9), lag-1 autocorrelation and kurtosis:
  measured from the generator's own output (`evaluate.average_correlation`
  and the series themselves), then set in the simulation.
- Periods: 31 years of search and a 5-year holdout (#43).

**Target (decided, Q5):** one flat **end-to-end false-discovery rate of 10%**
among candidates that pass the final stage. Incubation is a further check, not
part of the target. No separate miss-to-false-discovery ratio: fixing the FDR
already implies it.

**Baseline first (decided, Q5):** until the funnel is calibrated and shown to
do better, the decision rule is the baseline: **BH at 10% on each family's
search-adjusted p-value** (#51). The funnel's stage thresholds are then
calibrated on simulated candidates to the same 10% across the whole 0–5%
base-rate range, registered here before any real data is scored, and the
funnel replaces the baseline only once it beats it.

**The harness reports**, for the baseline, the calibrated funnel and the
naive stack (every method as a gate on all the data): end-to-end FDR, power,
and expected true discoveries per incubation slot, by stage where there are
stages. Running the whole funnel, not each method alone, is the point: it is
how the cost of stacking becomes visible.

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
- graded scores: BY, Holm, `lfdr`, PSR, MinTRL, haircut Sharpe, sensitivity checks
- header for the batch: ledger trial count, search-adjusted FDR estimate, PBO,
  average correlation, the threshold set used and when it was calibrated

A candidate is accepted when it passes every gate. Graded scores explain the
decision; they never overturn it.

## Decisions

Recorded 2026-10-05 from the review of #50.

| | Question | Decision |
|---|---|---|
| Q1 | #37 and #31 both add BH and BY | Keep #31's functions. #37 is closed; it returns later as a thin wrapper over `evaluate`, bringing its Newey–West p-values from returns |
| Q2 | Stage-1 gate when families exist | Carl's bootstrap max + BH. Local FDR (#54) is a graded score, and replaces the gate only if the harness shows a real power gain at our base rate |
| Q3 | Stage-4 gate: deflated Sharpe, Romano–Wolf or SPA | **Open.** Proposed: the deflated Sharpe, since it exists; only one decides |
| Q4 | Where the search is recorded | The factor store, for lineage; the ledger, for trials with computed performance; `shown_to_agent` added in #49 |
| Q5 | Calibration target | A flat 10% end-to-end FDR; the baseline (BH on #51's family p-values) decides until the funnel beats it |
| — | Where stage 2's fresh data comes from | **Open.** See "Stage 2's fresh data" |
| — | Stage 3's gate is the spanning alpha test | **To confirm** with its owner |

## What this page doesn't change

Every method keeps its owner, module and tests. The contract in
[`../validation.md`](../validation.md) is unchanged: a performance matrix in,
graded scores out, trial counts from the ledger, a null-calibration test and a
power test for every method. This page only says which output decides.
