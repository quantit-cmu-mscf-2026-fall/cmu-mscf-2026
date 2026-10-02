# Factors from papers: `capstone.factors`

One generation framework (Cal's). Papers become hypotheses, hypotheses
become factor formulas, and formulas are stored as expression trees with
their lineage. It stops before evaluation: no market data reaches the model
and no performance is computed, so nothing here is a ledger trial. Whatever
evaluates these factors later must log every evaluated factor to the ledger
and hand validation the usual dates x candidates matrix
([validation.md](validation.md)).

## Pipeline

```
papers (file or paperlog) -> hypotheses.extract -> store.hypotheses
                          -> proposer.propose   -> parse / repair -> originality -> store
```

- **Hypotheses** use AlphaAgent's five fields (Tang et al. 2025): observation,
  knowledge, justification, specification, falsification condition.
- **Kinds**: a hypothesis is `market` (a claim about returns, which becomes
  factors) or `method` (about how to search or validate, such as "simpler
  factors decay less"). Method hypotheses are stored and stop there.
- **Complexity**: a formula with more than `max_nodes` tree nodes is rejected
  (AlphaAgent's complexity control). Every Alpha101 member has at most 26.
- **Usage**: every paper's model calls and tokens go in the `usage` table,
  including for a paper that failed partway; `stats` totals them.
- **Factors** are formulas in a closed grammar over daily open, high, low,
  close, volume, `returns` (CRSP total return), `shares` (outstanding) and
  `cap` (market cap), so turnover is `volume / shares` (`capstone/factors/tree.py`).
  Evaluating them later needs split-adjusted prices and shares. A formula that does not parse goes back to
  the model with the error, up to `max_repairs` times.
- **Originality** (AlphaAgent Eq. 6) is the share of a formula inside its
  largest subtree shared with a reference, compared by shape, so a retuned
  window does not make a copy original. Rejected at or above `max_zoo_share`
  against Alpha101 and `max_store_share` against what is already stored.
- **Identity**: `factor_id` keeps windows and constants and ignores operand
  order under `+` and `*`. Each factor is stored once; every proposal of it,
  including duplicates, rejections and parse failures, is kept in
  `proposals`, which is the record of what the search actually produced.

## Running it

```bash
pip install -e ".[agents]"                # anthropic; the key comes from ANTHROPIC_API_KEY
python -m capstone.factors run --source file config/papers.example.toml
python -m capstone.factors run --source paperlog --min-relevance 3 --limit 10
python -m capstone.factors stats          # papers, hypotheses, factors, proposals by status
python -m capstone.factors list
python -m capstone.factors lineage <factor_id>
```

`--source paperlog` needs the paperlog package importable and uses its
`PAPERLOG_DB`. A paper is marked ingested in paperlog only after all its
hypotheses and factors are stored, so a failed run offers it again.

Parameters are in `config/factors.toml`; bump `prompt_version` whenever a
prompt or threshold changes, since every stored row records it.

## Where the store lives

`CAPSTONE_FACTOR_DB`, else `experiments/factors.db`. It is gitignored: this
repository is public, and the store holds the team's ideas. Sync it
privately, like the run ledger.
