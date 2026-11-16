# mscf-pipeline

Claude Code plugin for the Quantit x CMU MSCF 2026 alpha-discovery and validation pipeline
(`quantit-cmu-mscf-2026-fall/cmu-mscf-2026`). It targets this repository's `capstone` API
(`shared_data`, `runlog`, `evaluate`, `search_fdr`, `cv`, `calibration`, `synth`, `backtest`).

## Install

```text
/plugin marketplace add <path to this checkout>
/plugin install mscf-pipeline@cmu-mscf-2026
```

For development: `claude --plugin-dir ./mscf-plugin`. The hook runs `python`, so it must be on
`PATH`.

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
| Skill | `pipeline-view` | The live pipeline lab over the run ledger (local page or one-file snapshot), and how an experiment reports its progress (`mscf_progress.track`, `pipeline_run.py`) |
| Command | `qp-check` | The pre-PR acceptance check: offline tests, ruff, and the trial count from `runlog stats` |
| Agent | `ship-checker` | Read-only pre-push check of a branch against the PR rules (title, size, template, failing-without-change test, safety scan) |
| Hook | `hooks/guard.py` | Denies holdout dates in commands, force pushes, printing secrets, committing working data, editing the ledger; warns on holdout dates in code; reports the ledger count after experiment scripts |

The skills only describe rules that are written down in `CLAUDE.md`, `CONTRIBUTING.md`,
`docs/validation.md` and `docs/shared_data.md`; they add no policy of their own. For PR review
use the repository's own `/review-pr` skill and `pr-reviewer` agent (`.claude/`, see
`REVIEW.md`); this plugin does not duplicate them, and `ship-checker` only covers the author's
side before a push.

## Guard settings

The guard never grants anything; it denies or warns. Two rules are off until you set them
(environment variables, e.g. in your shell profile or `.claude/settings.local.json`):

| Variable | Effect |
|---|---|
| `MSCF_GUARD_DENY_PATHS` | Path prefixes (separated by `;` on Windows, `:` elsewhere) that commands and edits may not touch, e.g. a second checkout you want left alone. Unset: nothing is denied, and this repository is never denied by name. |
| `MSCF_GUARD_NOREPLY_REMOTE` | A substring of a remote URL. A `git push` there is denied if a commit it sends has an author or committer email that is not a `users.noreply.github.com` address. Unset: not checked. |

## Adding a team member's skill

Knowledge skills from the `kb-<name>` repositories (exercise Part 8) go in `skills/<name>/`
by PR, with a description that says when to use it. `tests/test_mscf_plugin_layout.py` checks
that every skill has frontmatter and that every `capstone.<module>.<function>` it names exists.

## Tests

`pytest -q tests/test_mscf_plugin_layout.py tests/test_mscf_plugin_guard.py`: manifest,
marketplace entry and hook wiring, frontmatter, that every `capstone` function a skill names
still exists, and the guard's deny/allow cases.
