---
name: paper-round
description: Runs the weekly paper round for cmu-mscf-2026 - turns a finance or ML paper into the paper-round issue one-pager (claim, what we would do differently, smallest experiment), designs that experiment against the shared data and validation methods, and fills in the Report back section after it runs. Use when someone says paper round, one-pager, presents a paper, or asks what a paper means for the pipeline.
---

# Paper round

Template: `.github/ISSUE_TEMPLATE/paper-round.md`, an issue (no PR), title
`[paper] <paper title> - <name>`, label `paper-round`. Five minutes to present.

## Write the one-pager

Write in the person's own words and argue a position. Do not paste the abstract or more than
a short phrase of the paper.

1. **The claim** - one or two sentences someone could push back on. Not the abstract.
2. **If it's right, what would we do differently?** - concretely, for this pipeline: which
   stage, method, data rule or search space changes. Map it to a module (`evaluate`,
   `search_fdr`, `cv`, `calibration`, `shared_data`, the agents' search). "Nothing" is a
   valid finding; say why.
3. **The smallest experiment** - hours, not days. Name:
   - data: `synth.make_return_matrix` / `calibration.make_family_set` with which defects, or
     which `shared_data` file, columns and dates (search period only);
   - metric: realised FDR, power, rejection rate, Sharpe, IC;
   - the result that would change our mind, written before running it.

Leave **Report back** empty until the experiment has run.

## Run the experiment

- A `.py` script under `experiments/` or `scripts/` with seeds and parameters in a file.
- Every trial through `runlog.log_run` with tags `paper:<short-id>` plus the stage tag if
  it touches real candidates; synthetic-only checks are logged too, tagged `synthetic`.
- If it needs a new method, the method ships with `null-calibration` tests.

## Report back (following week)

Expected vs. observed, the numbers with seeds and trial count, and the next step: adopt
(open a PR, `research-pr`), drop, or a sharper experiment. A null result is a result.

## Carrying a knowledge skill into the team

Per `docs/exercise_knowledge_to_skill.md` Part 8: a skill that changes an experiment
(e.g. purged CV) is ported by PR as the exercise says, into `.claude/skills/<name>/` (the team's project
skills; this plugin's `skills/` only if the team agrees, see the plugin README), an experiment is rerun with it
active, and the before/after goes in the paper-round issue's Report back.
