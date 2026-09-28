# Project conventions (read by the AI agents working in this repo)

Agentic alpha discovery and statistical validation — Quantit × CMU MSCF
capstone. The humans steer; you build. These rules are not optional.

## Verify before any PR

- Tests: `pytest -q -m "not network"` — must be green.
- Lint: `ruff check . && ruff format .` — must be clean.
- New behavior ships with a test that fails without the change.

## The run ledger is sacred

- EVERY experiment goes through `capstone.runlog.log_run` — log **before**
  looking at the result. The ledger is the project's trial count; an unlogged
  run corrupts every downstream multiple-testing correction.
- The ledger file (`experiments/runs.jsonl`) is local and gitignored — it is
  synced to the team's private archive, never committed to this public repo.

## This repository is public

- Never write credentials, tokens, or account identifiers into any file,
  commit message, or command that echoes them.
- Working data stays out: no session logs, no transcripts, no large result
  binaries. Results go in the PR description.

## Pull requests

- Small PRs, one logical change; branch naming `<name>/<topic>`.
- Fill the PR template — including seeds and the trial count for research PRs.
- See CONTRIBUTING.md for the full workflow (review rotation, squash merge,
  deploy registry).

## Research hygiene

- Prefer `.py` scripts over notebooks; fixed seeds; parameters in files.
- A procedure that cannot return "nothing here" is not a validation procedure
  — run methods on signal-free data before trusting them.

## Validation

- Judge candidates with the shared methods listed in `docs/validation.md`;
  don't hand-roll a correction or a significance test inside a strategy.
- Hand them a performance matrix: dates x candidates, one per-period series
  per candidate (returns via `backtest.candidate_returns`, or per-date IC).
  Validation never calls strategy code; strategies call validation.
- Take the trial count from the ledger (`python -m capstone.runlog stats`),
  never from a number typed into the call.
- A new validation method ships with a null-calibration test and a power test
  on `synth.make_return_matrix` data, including correlated (`rho`) and
  autocorrelated (`ar1`) nulls.

## Session capture (by design)

Sessions run inside this repository are captured to `.sessions/` and synced,
credential-redacted, to the team's **private** archive — for the team's own
learning. Nothing outside this directory is touched. See `docs/telemetry.md`.
Work normally.
