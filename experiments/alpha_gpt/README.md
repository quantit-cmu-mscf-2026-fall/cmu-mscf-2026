# Alpha-GPT baseline: Seed + Analyst loop

Methodological replication of Alpha-GPT (arXiv 2308.00016v2); the experiment
definition is in `docs/experiments/alpha_gpt.md`. Code: `capstone/alpha_gpt/`.

## Run

```bash
python -m capstone.alpha_gpt.run --config experiments/alpha_gpt/baseline_synth.toml
python -m capstone.alpha_gpt.run --config experiments/alpha_gpt/baseline_null.toml
python -m capstone.runlog stats          # trial counts
```

`--skip-test` stops after selection without touching TEST. Each run calls
`claude -p` roughly `2 * n_rounds - 1` times (more on retries).

## What happens

1. **Quant Developer** (`claude -p`) turns the idea into `n_alphas` expressions in
   the operator DSL (`capstone/alpha_gpt/prompts/quant_developer.md`).
2. Each valid, new expression is evaluated on TRAIN and VALIDATION and **logged
   to the ledger at that moment** — before the loop, the Analyst or you see it.
3. **Analyst** (`claude -p`) reads the results table (TRAIN/VALIDATION only) and
   writes a revised idea for the next round.
4. After the last round, the best alpha by VALIDATION IC t-stat is evaluated
   **once** on TEST (`final_test.json`; a second attempt refuses).

## Run directory (`experiments/alpha_gpt/out/<run_id>/`, gitignored)

| file | contents |
|---|---|
| `config.json` | config, panel descriptor, split dates |
| `llm_calls.jsonl` | every prompt and raw reply, with rejection reasons |
| `trials.jsonl` | canonical expression + TRAIN/VALIDATION metrics per trial |
| `reviews.jsonl` | Analyst summaries, diagnoses, revised ideas |
| `events.jsonl` | rounds, DSL rejections, duplicates, selection |
| `final_test.json` | TEST metrics of the selected alpha(s) |

Keep run outputs out of the repo; put results in the PR description.

## Guarantees and their limits

- **The LLM sees only its prompt.** Role calls run `claude -p` in an empty temp
  directory with `--tools ""`, one turn and no session persistence. Those
  nested calls are therefore not captured by this repo's session hook; the run
  directory's `llm_calls.jsonl` is their record.
- **No arbitrary code.** Alphas are parsed with `ast` against a whitelist; lags
  cannot be negative; every operator is tested to be causal.
- **TEST is structurally hidden** from search: the evaluator holds a panel with
  the TEST period removed, and no prompt renderer accepts TEST data.
- **Trial counting.** Rejected (unparseable) expressions are not trials. A
  canonical duplicate within a run is not re-logged. How repeated runs on the
  same data count toward a family size is a team decision; the ledger keeps
  expression, data descriptor and split dates for every trial so it can be
  computed later.
- **VALIDATION is not clean.** The Analyst's feedback is built from VALIDATION
  results and steers later rounds — inherent to the paper's design. Only TEST
  is an unbiased estimate, which is why it is evaluated once.
- **No significance gate yet.** Selection is "best VALIDATION t". The
  Benjamini–Hochberg FDR gate is a later phase.
