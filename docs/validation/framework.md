# Validation framework: one standard, calibrated to the generator

**Proposed 2026-10-04; team decisions recorded 2026-10-05 and 2026-10-06;
method for calibration settled 2026-10-21; thresholds to be registered from
the real generator** (see "Decisions" at the end; three items still open). This page says how
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
| Holdout 2021–2025, looked at once (#43) | Reusing the final test; search overfitting of any kind | data | Independent of the search, so it guards the end-to-end FDR however hard the agents searched | The pipeline's real bottleneck: in the method's test runs (q₄ = 0.50) about 11% of real Sharpe-1.0 survivors and 40% of Sharpe-0.5 survivors failed it. Predates the agents' model cutoff, so a pass is not established against memorisation | Construction | 4 |
| HAC Sharpe test, one-sided (`sharpe_test`, #29) | One series being luck | candidate | Allows for autocorrelation and fat tails | Slightly liberal in small samples (5.5–8% at a nominal 5% on AR(1) nulls), more so in the far tail under fat tails and GARCH | Input to the stage-4 gate (holdout p-values); graded score at stage 1 | 1, 4 |
| BH adjusted p (`bh_adjusted`, #31) | Luck across many candidates | candidate | Most power of the FDR controls | Guarantee needs positive dependence, i.e. one-sided tests | **Gate** | 1 |
| BY adjusted p (`by_adjusted`, #31) | Same | candidate | Valid under any dependence | Costs power; never failed in our simulations where BH held | Graded score | 1 |
| Holm adjusted p (`holm_adjusted`, #31) | Any false call | candidate | Strictest guarantee | Far too strict for a screen | Graded score | 1, 4 |
| Storey q (#31) | Same as BH | candidate | More power when candidates are independent | **Unsafe when correlated**: 10–14% any-discovery rate at a nominal 5% | Graded score, independent candidates only | — |
| Local FDR / empirical null (`local_fdr`, #54) | Same as BH, under correlation | candidate | A probability of being null per candidate, against a null that moves with a shared factor | Needs 200+ p-values; "real" means stands out from the other candidates, not Sharpe above zero | Graded score (`lfdr` column); replaces the stage-1 gate only if the harness shows a power gain | 1 |
| Family-level bootstrap max + BH (`bootstrap_family_pvalues`, #67, adapted from Carl's `carl/validation1-hypothesis-fdr`) | Search within one hypothesis | hypothesis | Keeps any correlation pattern between a hypothesis's variants; each family's null rate 0.047–0.060 at a nominal 0.05 | Needs the family structure recorded and at least 10 blocks of data per variant; 1.3–1.7× liberal at the 1% level with fat tails over 4 years (finite-sample) | **Gate** (stage 1) and the baseline's input | 1 |
| Search-adjusted family p-value (`search_adjusted_pvalue`, #51) | Search within one hypothesis | hypothesis | Exact for independent or evenly correlated variants (`rho=`); no resampling | The independent formula over-corrects near-copies (power 0.64 vs 0.81 at K = 20, ρ = 0.9); built on HAC p-values, too optimistic in the tail under fat tails and GARCH | Fast cross-check on the bootstrap | 1 |
| PSR, MinTRL (#29) | Too short a track record | candidate | Readable: "how many years would we need?" | Single-candidate; no multiple testing | Graded score | 2, 4 |
| Robustness to lags, costs, sub-periods | Fragile signals | candidate | Catches results that hang on one choice | Each check is another test if used as a gate | Graded score | 2 |
| PBO / CSCV (`pbo`, #35) | Overfitting in the *selection* of the best | batch | Measures the search, not one candidate | 0.5 is the no-skill baseline; a single set ranges 0.06–0.78 | Batch diagnostic | 2 |
| Average correlation, implied independent trials (#29) | Overcounting correlated trials | batch | Explains why a correction is strict | Must not replace the ledger count in a decision | Batch diagnostic | all |
| Search-adjusted FDR estimate (below) | Hidden specification search | batch | Uses what we can see: the whole search | Depends on the search being fully logged | Batch diagnostic | 1 |
| Spanning alpha (`spanning`, planned) | Known factor in disguise; copy of a held signal | candidate | The novelty test | Needs the comparison set (#43) | **Gate** | 3 |
| Deflated Sharpe (#29) | Best of N trials being luck | candidate | Uses the ledger count and the spread of trial Sharpes | Answers "is the best of N real?", not "is each survivor real?". As the stage-4 gate it lost to BH across survivors in every calibration run, and charging the searched count cut power to 0.08 | Graded score | 4 |
| Haircut Sharpe (`haircut_sharpe`, #65) | Same | candidate | Readable as "the Sharpe after the search"; follows the authors' code | Same question as the deflated Sharpe | Graded score | 4 |
| Reality Check, Romano–Wolf, SPA (planned) | Best of many correlated candidates | set | Keeps correlation by resampling | Not built yet; same question as the deflated Sharpe | Alternative stage-4 gate | 4 |

## The funnel, with roles assigned

The stages are the ones in [`../validation.md`](../validation.md). Stages 1–3
work on the search period and stage 4 on a holdout that nothing before it has
seen; that untouched holdout is what makes a loose screen safe.

| Stage | Data | Gate | Graded scores reported | Batch diagnostics |
|---|---|---|---|---|
| 1. Screen | Search period, 1990–2020 (#43) | BH ≤ q₁ across hypothesis families, on each family's search-adjusted p-value (bootstrap max) | BY, Holm, `lfdr`, raw p | Search-adjusted FDR, average correlation |
| 2. Robustness | Search period | — (no gate; see "Stage 2" below) | PSR, lag / cost / sub-period sensitivity | PBO of the search |
| 3. Incremental value | Search period; frequency to decide (monthly anomaly portfolios, or daily factors as #75 proposes) | Spanning alpha p ≤ α₃ | Factor loadings | — |
| 4. Final decision | Holdout, 2021–2025, looked at once | BH ≤ q₄ across the survivors' one-sided holdout p-values | Deflated Sharpe, haircut Sharpe, Holm, MinTRL | — |
| 5. Incubation | Live or paper | — | Realised vs. expected Sharpe | — |

The search is paid for once, at stage 1, where each family's p-value is
adjusted for everything it tried and BH runs across families. Stage 4 charges
only for the candidates taken to the holdout: it is independent data, so the
search no longer competes there (López de Prado & Fabozzi 2026 make the same
point about independent validation data). Charging the full searched count
again at stage 4 is the double penalty this page warns against; in calibration
it cut power from about 0.47 to 0.08 (#68). Stage 4's level can be loose
because the stages multiply: in the method's test runs stage 1 removed over 99% of null families (99.4% in the confirmation run),
and the target is the end-to-end FDR.

### Stage 2

#43 gives agents and stages 1–3 the whole search period, 1990–2020, so stage 2
has no unseen dates to confirm on. Calibration compared splitting off the
last seven years for a stage-2 confirmation against giving stage 1 all 31
years and letting the holdout confirm. Not splitting found more real signals
in all three runs (mean power 0.54 vs 0.47, 0.66 vs 0.59 and 0.69 vs 0.63) at
the same FDR target. The holdout already guards the end-to-end FDR, however hard the agents
searched, and splitting costs stage 1 seven years.

So stage 2 is not a gate. It reports graded robustness checks on the search
period (sensitivity to costs, lags and sub-periods; PSR; the search's PBO) for
a reader to weigh, and nothing is rejected there. Walk-forward (QUANTIT-45)
belongs here too, for candidates with fitted parameters.

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
  search, by a bootstrap of the family maximum (`bootstrap_family_pvalues`,
  #67, adapted from Carl's branch; decided, Q2). `search_adjusted_pvalue`
  (#51) with the family's measured correlation `rho` is a fast cross-check,
  not the gate: under fat tails and GARCH its HAC inputs are too optimistic in
  the tail (see "What the method's test runs showed"). BH then runs across families. This *replaces* BH
  across every individual variant; it is not added on top. Every evaluated
  variant needs a family: the gate raises on one without.
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
  source of truth for lineage: its proposals table records `hypothesis_id`
  and `parent_factor_id` for every proposal, rejected ones included (#47).
  Since #77 the unit of trial is a *variant* (a factor under one trading
  spec). #78 logs each evaluated variant to the ledger with `factor_id` and
  `variant_id`, and its family resolves through the store (`hypothesis_of`).
  **K for a family is its evaluated variants**: one ledger variant entry
  each, and the columns the stage-1 gate receives for that family.
  Results already go back to the agent: the deepen move scores each edit on
  its search-period p-value (#79). So the search adapts to search-period
  scores, which is what the sealed holdout guards against. Whether a result
  was shown to the agent is not yet recorded per trial. Without this record,
  K is unknown and the identification problem comes back.

## Calibration

This page fixes the **method**; the threshold **values** q₁, α₃ and q₄ come
from running that method for our real generator, and are written here, with
the date and the parameter file, before any real candidate is scored. None
is registered yet.

**What "calibrated on real data" means.** The thresholds are tuned on
*simulated* candidate sets whose structure is measured from the real
generator's output, never on which real candidates pass. Picking a cut-off
by looking at real candidates' results is multiple testing of its own, and
the holdout is never used for it. What we measure from the generator is
structure only, none of which reveals whether a strategy works:

- search intensity K per hypothesis (evaluated variants per family, from the
  ledger, #78);
- correlation within a family and across families
  (`evaluate.average_correlation` on the search-period returns);
- lag-1 autocorrelation, tails and volatility clustering of the variants'
  returns;
- the periods: 31 years of search and a 5-year holdout (#43).

What we cannot measure is how many ideas are real or how strong they are, so
those are a grid: a share of real ideas π₁ from 0 to 5%
([`../validation.md`](../validation.md), "Decisions the stages depend on")
and an annual Sharpe of 0.5 to 1.5 net of costs.

**Target (decided, Q5):** one flat **end-to-end false-discovery rate of 10%**
among candidates that pass the final stage. Incubation is a further check, not
part of the target. No separate miss-to-false-discovery ratio: fixing the FDR
already implies it.

**The procedure** (`scripts/calibrate_validation.py`, #68):

1. Simulate candidate sets with the measured structure, planting real ideas
   at each share and strength in the grid, with known truth.
2. Run the whole pipeline on each set, not each method alone, with the
   stage-1 gate as it will run on real data: the bootstrap of the family
   maximum (#67; `"stage1": "bootstrap"`, as in
   `calibrate_validation_confirm_bootstrap.json`).
3. For each candidate threshold set, measure the end-to-end FDR and power in
   every setting, and how bad the bad runs get (the chance of any false
   discovery, the 90th-percentile and worst count per run).
4. Choose the set with the most mean power whose **worst-setting** FDR is at
   most 10%, with margin, because false discoveries cluster (below). A set
   that only just meets the target in simulation is not safe to register.
5. Register it here, before scoring. Recalibrate whenever the generator
   changes: a new agent, model, prompt set or search budget changes K and
   the correlations, and the thresholds belong to a generator, not to the
   project.

Stage 3 is not simulated yet (`run_funnel` passes everything there). α₃ is
registered only once it is, and a real stage-3 gate will lower power, and
possibly the FDR, below what the funnel alone shows.

**Baseline first (decided, Q5):** until the funnel is calibrated and shown to
do better, the decision rule is the baseline: **BH at 10% on each family's
bootstrap p-value** (#67). The baseline has no holdout behind it, so its FDR
rests entirely on those p-values. The same procedure measures it, the funnel
and the naive stack side by side, and the funnel replaces the baseline only
once it beats it.

**The null check still applies:** on sets with no real candidates, the
calibrated funnel must pass almost nothing (`CLAUDE.md`, "Research hygiene").

### What the method's test runs showed (2026-10-06, synthetic; #68)

Before any real generator exists, the procedure was run on a stand-in:
200 hypotheses × 10 near-copy variants (within-family correlation 0.9),
cross-candidate correlation 0.1, AR(1) 0.3, t₅ tails, GARCH (0.05, 0.9).
Three runs (seeds 0–29, 100–119, 200–259), logged to the ledger as
`validation-calibration`, tag `synthetic`. **These runs used the HAC-based
formula at stage 1, not the bootstrap, so their threshold values are not
registered.** They settled design choices, which the real calibration
inherits:

- **Stage 4: BH across survivors.** It beat the deflated Sharpe in every run;
  charging the deflated Sharpe the full searched count cut power from 0.47 to
  0.08 (run 1).
- **Stage 2: no gate, no separate period.** See "Stage 2".
- **Stacking costs real signals.** The naive stack's FDR was 0.00–0.01, but
  it found 0.63–0.73 of real Sharpe-1.0 ideas against the funnel's 0.85–0.93
  (confirmation run, q₄ = 0.50), and 0.03 against 0.18–0.20 at Sharpe 0.5.
- **The baseline needs the bootstrap.** With the formula p-values it ran at
  an FDR of 0.13–0.16 in every setting (60 seeds), up to 12 false discoveries
  in one run; with the bootstrap, 0.067 when nothing is real and 0.089 at 2%
  real (15 seeds).
- **False discoveries cluster.** A shared market factor makes many null
  candidates look good together: most runs had none, a few had 5–10. That is
  why step 4 asks for margin.
- **The holdout is the bottleneck.** At q₄ = 0.50, about 11% of real
  Sharpe-1.0 survivors and about 40% of Sharpe-0.5 survivors failed there.
  Only a longer holdout or live incubation changes that.
- **Funnel vs baseline (Q5) is not settled by simulation.** The bootstrap
  baseline matched the funnel's FDR with more power in the one setting
  checked (0.95 vs 0.85). The funnel's case is its holdout, which guards
  against what the simulation doesn't model: agents adapting to
  search-period scores, which the deepen move already does (#79).

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

Recorded 2026-10-05 and 2026-10-06 from the review of #50; updated 2026-10-21.

| | Question | Decision |
|---|---|---|
| Q1 | #37 and #31 both add BH and BY | Keep #31's functions. #37 is closed; it returns later as a thin wrapper over `evaluate`, bringing its Newey–West p-values from returns |
| Q2 | Stage-1 gate when families exist | Carl's bootstrap max + BH. Local FDR (#54) is a graded score, and replaces the gate only if the harness shows a real power gain at our base rate |
| Q3 | Stage-4 gate: deflated Sharpe, Romano–Wolf or SPA | **Proposed from calibration:** BH across the survivors' holdout p-values; the deflated Sharpe becomes a graded score (#68) |
| Q4 | Where the search is recorded | The factor store, for lineage; the ledger, for trials with computed performance. **Not implemented yet:** a per-trial record of whether a result was shown to the agent. Results reach the agent through the deepen scorer (#79), which records no such exposure; that audit is a follow-up |
| Q5 | Calibration target | A flat 10% end-to-end FDR; the baseline (BH at 10% on the bootstrap family p-values, #67) decides until the funnel beats it |
| — | Where stage 2's fresh data comes from | **Proposed from calibration:** no separate period; stage 2 reports graded robustness checks and the holdout confirms (#68) |
| — | Stage 3's gate is the spanning alpha test | **To confirm** with its owner |
| — | Stage 3's frequency: monthly anomaly portfolios or daily factors (#75) | **Open**, with its owner; needed before α₃ means anything |
| — | Funnel or bootstrap baseline as the deciding rule | **Open**; see "What the method's test runs showed" |

## What this page changes, and what it doesn't

Every method keeps its owner, module and tests. The contract in
[`../validation.md`](../validation.md) is unchanged: a performance matrix in,
graded scores out, trial counts from the ledger, a null-calibration test and a
power test for every method. What changes is which output decides: the
funnel table in `../validation.md` now names this page's gates instead of its
earlier strictness column (BH at 0.10–0.20 at stage 1; the deflated or haircut
Sharpe with the full trial count at stage 4; thresholds tuned to a
miss-to-false-discovery ratio), which calibration replaced.
