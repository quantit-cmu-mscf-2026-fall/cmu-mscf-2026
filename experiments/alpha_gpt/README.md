# Alpha-GPT baseline: Seed + Analyst, cross-validated

Methodological replication of Alpha-GPT (arXiv 2308.00016v2); the experiment
definition is in `docs/experiments/alpha_gpt.md`. Code: `capstone/alpha_gpt/`.

## Run

```bash
python -m capstone.alpha_gpt.run --config experiments/alpha_gpt/baseline_synth.toml
python -m capstone.alpha_gpt.run --config experiments/alpha_gpt/baseline_null.toml
python -m capstone.runlog stats          # trial counts
```

`--skip-test` stops before TEST. `--reuse-test` evaluates a TEST that an earlier
run already looked at, and tags that in the ledger. With `n_splits = 4` and 3
rounds a run makes about `(n_splits + 1) * (2 * n_rounds - 1)` = 25 `claude -p`
calls (more on retries).

## Splits

| | dates | used for |
|---|---|---|
| **TRAIN** | purged, embargoed folds of DEVELOPMENT | the procedure: Quant Developer proposals, trial scores, Analyst feedback, selection |
| **VALIDATION** | each held-out DEVELOPMENT fold | scoring the procedure's picks out of fold; the procedure never sees it |
| **TEST** | the last `test_frac`, after an embargo | the final selection, once |

## What happens

1. **The procedure** (`run_procedure`): the Quant Developer proposes `n_alphas`
   expressions; each valid, new one is scored on TRAIN and **logged to the ledger
   at that moment**; the Analyst reads the TRAIN table and revises the idea; after
   `n_rounds` the best alpha by TRAIN IC t-stat is selected.
2. **Cross-validation** (`cross_validate_procedure`, López de Prado AFML ch. 7):
   DEVELOPMENT is split by `capstone.cv.PurgedKFold`. For each fold the whole
   procedure runs on a panel where the fold — and every return its labels are
   built from — is masked to NaN, on TRAIN dates purged of label overlap and
   embargoed after the fold. Its pick is then scored on the fold.
3. **Final**: the procedure runs once on all of DEVELOPMENT.
4. **TEST**: that selection is scored once (`final_test.json`).

The report prints *in-sample t → cross-validated held-out t → TEST t*, the family
size (every search trial in every fold plus the final run) and the deflated
Sharpe ratio for that family. The gap between in-sample and held-out is the
procedure's overfitting, measured.

## Run directory (`experiments/alpha_gpt/out/<run_id>/`, gitignored)

| path | contents |
|---|---|
| `config.json` | config, panel descriptor, split dates |
| `fold<j>/`, `final/` | per procedure: `llm_calls.jsonl` (every prompt and raw reply), `trials.jsonl`, `reviews.jsonl`, `events.jsonl` |
| `cv_folds.jsonl` | per fold: TRAIN/held-out dates, picks with TRAIN and held-out metrics |
| `final_test.json` | TEST metrics of the final selection |
| `summary.json` | in-sample vs cross-validated vs TEST, family size, DSR |

Keep run outputs out of the repo; put results in the PR description.

## Guarantees and their limits

- **The LLM sees only its prompt.** Role calls run `claude -p` in an empty temp
  directory with `--tools ""`, one turn and no session persistence. Those nested
  calls are not captured by this repo's session hook; `llm_calls.jsonl` is their
  record.
- **No arbitrary code.** Alphas are parsed with `ast` against a whitelist; lags
  cannot be negative; every operator is tested to be causal and to never turn a
  masked row into a finite value that depends on it.
- **Held-out data is absent, not skipped.** The procedure holds a panel with TEST
  removed and its CV fold masked; the evaluator refuses a panel containing TEST
  dates; prompts render TRAIN metrics only. Tests scramble TEST data and a CV
  fold's data and require the corresponding prompts to stay byte-identical.
- **Sample weights.** Label-uniqueness weights (AFML ch. 4) are used for TRAIN,
  held-out and TEST scores alike; with `label_horizon = 1` they are all 1.
- **TEST is locked** per run directory and, through the ledger, per dataset and
  split. Reuse is possible only with `--reuse-test` and is tagged.
- **Trial counting.** Rejected (unparseable) expressions are not trials; a
  canonical duplicate within one procedure is not re-logged. Each CV fold's
  procedure is a separate search, so its trials count toward the family.
- **Not yet:** a significance gate (Benjamini–Hochberg on held-out p-values) and
  Newey–West standard errors; both are later phases.
