## Review chain (how plans get vetted before they get built)

Plans go through critics before code gets written. The orchestrator runs this;
you do not invoke each critic by hand.

1. The research agent (or a human) proposes a PLAN — approach, smallest trial,
   failure condition, the one file to create.
2. The orchestrator fans the plan out to the RELEVANT critics (not always all):
   - critic-stats          — trial counting, multiple testing, p-hacking
   - critic-timeseries     — leakage, purge/embargo, autocorrelation
   - critic-ml             — overfitting, complexity justification, data hygiene
   - critic-quant          — survivorship / PIT universe, reinventing the wheel
   - critic-swe            — reuse, placement, interfaces, test quality
   - critic-reproducibility— seeds, logged-before-looking, honest N, stranger-run
   Pick the critics whose lens the plan actually touches. A pure-refactor PR does
   not need critic-timeseries; a new forecaster needs most of them.
3. Critics return objections from their lens only. They critique the PLAN; they
   never write code and never decide whether a result is real.
4. The orchestrator collects objections, folds in the non-conflicting fixes, and
   SURFACES DISAGREEMENTS between critics to the human to decide. It does not
   silently pick a side on a real design choice (e.g. BH vs Bonferroni).
5. Human approves the vetted plan. Only then does the build agent implement it.
6. After build: validation agent attacks the result; ops checks process. Neither
   is in the pass/fail path — the deterministic harness (evaluate.py,
   acceptance.py) decides outcomes.

Hard boundary: critics and agents review PLANS and CODE. They never decide
whether a strategy is real. That verdict is deterministic and agent-blind. An
agent anywhere between "candidate" and "pass/fail" turns validation into ranking.