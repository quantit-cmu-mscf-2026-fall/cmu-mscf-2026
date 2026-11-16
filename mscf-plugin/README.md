# mscf-pipeline

Claude Code plugin for the Quantit x CMU MSCF 2026 alpha-discovery and validation pipeline
(`quantit-cmu-mscf-2026-fall/cmu-mscf-2026`). It targets this repository's `capstone` API
(`shared_data`, `runlog`, `evaluate`, `search_fdr`, `cv`, `calibration`, `synth`, `backtest`).

## Install

```text
/plugin marketplace add <path to this checkout>
/plugin install mscf-pipeline@cmu-mscf-2026
```

For development: `claude --plugin-dir ./mscf-plugin`.

## Contents

| Component | Name | What it does |
|---|---|---|
| Skill | `pipeline-boundaries` | Sealed holdout, ledger before results, one-way validation, pre-registered thresholds, public-repo hygiene |
| Skill | `shared-data` | Loading CRSP/Compustat through `capstone.shared_data` and the returns/costs/fundamentals rules |
| Skill | `run-ledger` | `log_run` before looking, what counts as a trial, stage tags, trial counts |
| Skill | `validate-candidates` | Performance matrix, one-sided HAC Sharpe tests, choosing a correction, the five-stage funnel |
| Skill | `null-calibration` | Null and power tests every new validation method ships with |
| Skill | `research-pr` | Branch, checks, PR template with seeds and trial count, review and merge |
| Skill | `paper-round` | Weekly paper one-pager, smallest experiment, report back |

The skills only describe rules that are written down in `CLAUDE.md`, `CONTRIBUTING.md`,
`docs/validation.md` and `docs/shared_data.md`; they add no policy of their own. For PR review
use the repository's own `/review-pr` skill and `pr-reviewer` agent (`.claude/`, see
`REVIEW.md`); this plugin does not duplicate them.

## Adding a team member's skill

Knowledge skills from the `kb-<name>` repositories (exercise Part 8) go in `skills/<name>/`
by PR, with a description that says when to use it. `tests/test_mscf_plugin_layout.py` checks
that every skill has frontmatter and that every `capstone.<module>.<function>` it names exists.

## Tests

`pytest -q tests/test_mscf_plugin_layout.py`: manifest and marketplace entry, frontmatter, and
that every `capstone` function a skill names still exists.
