---
name: pipeline-boundaries
description: The rules no change to the cmu-mscf-2026 pipeline may break - sealed 2021-2025 holdout, ledger before results, validation never calls strategy code, pre-registered thresholds, public-repo hygiene. Use before running an experiment, editing capstone/ code, or reviewing a PR in the team repo.
---

# Pipeline boundaries

These are the project's invariants. Each one names where it is written down; the plugin's
guard hook (`hooks/guard.py`) is a coarse second line, never a replacement. When a request
would cross one, stop and say which rule it crosses instead of finding a way around it.

## 1. The holdout is sealed

- Search period: 1990-01-02 to 2020-12-31. Holdout: 2021-01-01 to 2025-12-31
  (`docs/validation.md`, "Decisions the stages depend on").
- Agents and stages 1-3 see only the search period. Pass `end="2020-12-31"` (or earlier)
  to `sd.load` for any discovery or validation work; `sd.load` does not filter the holdout
  for you.
- The holdout is looked at **once**, by stage 4, on survivors whose thresholds were fixed
  beforehand. A second look is a new trial and invalidates the first.
- A pass on the holdout is not proof against memorisation: the agents' model was trained
  after 2025, so results must say so.
- Changing the agents' model means re-deciding the dates. Do not change `config/factors.toml`
  model settings as a side effect of other work.

## 2. Log before you look

- Every experiment goes through `capstone.runlog.log_run` at the moment the result exists,
  before anyone (human or agent) inspects it. See the `run-ledger` skill.
- A trial is every candidate whose performance anyone has seen, including discarded
  variants and every iteration inside an agent's search.

## 3. Validation is one-way

- Strategies produce a dates x candidates performance matrix and call validation.
  Validation never imports or calls strategy or generation code (`docs/validation.md`,
  "The input").
- Never hand-roll a correction or significance test inside a strategy; use the shared
  methods (`validate-candidates` skill).

## 4. Thresholds are pre-registered

- Every stage's score and threshold is fixed before candidates are scored. Choosing a
  procedure or a threshold after seeing results is multiple testing of its own.
- Thresholds come from simulating the whole funnel on known-truth candidate sets
  (`capstone.calibration`), to a 10% end-to-end FDR, not textbook defaults.

## 5. The repository is public

- No credentials, tokens or account identifiers in files, commit messages or echoed
  commands. Never print `ANTHROPIC_*`, `WRDS_*` or broker variables.
- No working data: never commit `experiments/runs.jsonl`, `data_cache/`, `*.parquet`,
  `.sessions/`, transcripts, notebook outputs or large result binaries. Results go in the
  PR description.
- Do not commit lecture slides, problem sets or verbatim course text.

## 6. Data comes from shared_data

- CRSP and Compustat load only through `capstone.shared_data` (`shared-data` skill). Never
  read the parquet files directly or query WRDS, unless you are the data maintainer
  refreshing the shared files.

## 7. `main` is always deployable

- Changes reach `main` only by PR with green CI and a rotating teammate's review, squash
  merged. Paper deploys go through `deploy/strategies.yml` and a tag; never edit its
  `enabled`/state keys outside a reviewed deploy PR.

## Before you finish

Check each changed file against rules 1-6, and name in your reply any rule the work came
close to.
