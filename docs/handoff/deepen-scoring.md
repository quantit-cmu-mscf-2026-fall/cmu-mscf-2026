# Testing handoff: score the deepen move with factor evaluation

## 1. What this PR adds
- QUANTIT-114. `capstone/factors/scoring.py`: `EvaluationScorer`, which scores
  a factor tree as one evaluated variant (one `TradingSpec`) on the discovery
  period, net of half the quoted spread, and `crsp_settings`, which reads
  `config/deepen_crsp.toml`.
- Quality is the funnel's stage-1 statistic as a z-score,
  `Phi^-1(1 - p)` with `p = evaluation.screen_pvalue(net returns)`. The values
  the pool correlates are the variant's daily net returns.
- `deepen.Scored` gains `params` and `metrics`; the deepen move writes them
  into each score's ledger entry (parents and children). A scorer never logs.
- `FactorConfig` gains `memory_quality_edges` and `memory_signed_quality`
  (defaults unchanged: |ICIR| bins, sign ignored); `memory.parent_context`
  takes them. Tests: `tests/test_scoring.py`, `tests/test_deepen.py`,
  `tests/test_factor_memory.py`.

## 2. Contract
- The scorer refuses a panel holding a date after `DISCOVERY_END`
  (`require_discovery`) at construction; it returns `None` for a factor that
  cannot be computed or whose p-value is not finite.
- `config/deepen_crsp.toml` holds proposals for the z-scale thresholds (to be
  calibrated on simulated candidate sets) and the one trading spec. No paths.
- No model or network calls; stages 2-4 of the funnel never reach the scorer.

## 3. How it was vetted

| Check | How | Result | Reproduce with |
|---|---|---|---|
| Quality and values | planted-signal synthetic panel; z equals `norm.isf(screen_pvalue)`, sign kept | pass | `pytest -q tests/test_scoring.py` |
| Scorer never logs; deepen does | empty ledger after scoring; deepen entries carry scorer params/metrics | pass | `pytest -q tests/test_deepen.py -k scorer` |
| Memory bins | signed z edges bin a loser as `q0` | pass | `pytest -q tests/test_factor_memory.py` |
| Holdout and period | holdout panel refused; params log the panel's own dates | pass | `pytest -q tests/test_scoring.py` |

## 4. Provenance
Ported from Cal's development sandbox. Not included: a holdout loader for the
funnel's stage 4.

## 5. Known limits and areas to improve
- The z thresholds in `config/deepen_crsp.toml` are proposals, not calibrated.
- One trading spec per run; choosing it per hypothesis is a follow-up.
