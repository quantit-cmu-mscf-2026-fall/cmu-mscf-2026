# paperlog

A weekly, automated log of academic papers for this project: LLM and agentic
generation of signals, factors and hypotheses, and the statistical methods that
decide which candidates are real (multiple testing, backtest overfitting, Sharpe
inference, purged cross-validation). English and Chinese-language sources feed one
combined set. The queries, anchor papers, rubric and tags in `config.yaml` are set
for this project; the paper-round issues on GitHub are the natural anchors to add.

Retrieval, deduplication and storage are plain Python. The only LLM call is the
screening step, which scores each paper, tags it, extracts key fields, and
translates Chinese-language papers into English.

## How a run works

1. **Collect to a target, not a window.** A run aims for `target_new_papers` new
   papers. It starts with the normal date window and escalates only if it falls short:
   widen the window, then go deeper per query, then add exploration queries and loosen
   the prefilter. Everything is deduplicated against the log first, so widening never
   re-fetches value you already have, and a quiet week automatically searches wider
   instead of returning five papers. Escalation stops early when a level stops yielding.
2. **Sources.** arXiv (submissions and revisions), OpenAlex (English and Chinese queries,
   by publication date and by index date, cursor-paginated so nothing is truncated),
   and Semantic Scholar (papers citing your anchors, plus the references of newly
   adopted papers).
3. **Deduplicate.** By arXiv ID, DOI, and fuzzy title, within the batch and against the
   log. Records whose identifiers disagree are never merged.
4. **Prefilter.** A keyword gate that *routes* rather than drops: rejected papers keep
   their abstracts and are revisited whenever the prefilter version or mode changes.
5. **Screen.** Claude Haiku scores 0-5 against your rubric, tags, extracts methods,
   datasets, markets and code links, flags look-ahead bias, and translates. Every scoring
   event is recorded, so a score is a dated opinion under a known rubric version, not a
   permanent fact. Queue order: new papers, then unscreened backlog, then papers scored
   under an older rubric.
6. **Output.** `papers.db`, `digests/YYYY-MM-DD.md`, `papers.csv`.

## Growing the search

The search widens on its own as the loop runs:

- **Every adopted paper becomes a citation anchor**, so future runs pull what cites it.
- **References of newly adopted papers** are pulled once, which surfaces foundational
  work that keyword queries miss because the terminology has changed.
- **`paperlog suggest-queries --write`** reads your adopted papers and proposes new
  queries (English and Chinese) into `learned_queries.yaml`, kept separate from
  `config.yaml` so you can see what was added and prune it.
- **`paperlog add`** puts anything you find yourself straight into the log.

## Setup

From the repo root:

```bash
pip install -e "research/paperlog[dev]"   # add ",embeddings" or ",dataframe" for extras
export PAPERLOG_DB=/path/outside/the/repo/papers.db   # see "Where things live"
export ANTHROPIC_API_KEY=...     # only needed for screening
export OPENALEX_API_KEY=...      # free, instant: openalex.org/settings/api
export S2_API_KEY=...            # optional, free on request from Semantic Scholar
paperlog run --no-screen         # retrieval only: costs nothing, needs no Anthropic key
paperlog run                     # first run backfills 60 days, then screens
```

Without `S2_API_KEY`, Semantic Scholar's shared pool is often rate-limited (HTTP 429).
`paperlog add <arXiv id>` then falls back to arXiv itself for the metadata, but
citation expansion from the anchors will be patchy until you add a key.

Tests run offline, with mocked sources and a fake LLM: `cd research/paperlog && pytest -q`.
CI runs them on every PR.

## Commands

| Command | What it does |
|---|---|
| `run` | Collect to target, screen, write digest and CSV. `--target N`, `--days N`, `--since`, `--no-screen` |
| `add <id>` | Add a paper you found: arXiv id/URL, DOI, Semantic Scholar URL, or a title. `--adopt`, `--anchor`, `--rating`, `--notes` |
| `rescreen` | Rescore under the current rubric. Default: papers on an older rubric version. `--all`, `--filtered`, `--limit` |
| `pending` | Papers not yet ingested downstream, best first. `--json`, `--claim`, `--min-relevance` |
| `suggest-queries` | Propose new search queries from adopted papers. `--write` to save |
| `review` | Adopt / read / reject with your own 0-5 rating |
| `stats` | Counts, rubric versions, and screener-vs-you calibration including the biggest misses |
| `export` | CSV for Google Sheets or Excel |

`stats` compares the screener's scores with your ratings, so you can see when the rubric needs tightening.

## Using inside a larger project

It lives at `research/paperlog/`. Install it in editable mode from the repo root, so every
script and notebook in the project can import it:

```bash
pip install -e research/paperlog
```

**Where things live.** Code and `config.yaml` are in the repo like any other code. The
database, digests and CSV are working data, and this repository is public (`CLAUDE.md`:
working data stays out), so they don't go in it: `papers.db` holds the team's ratings,
notes and what it adopted. Keep the log somewhere shared and private, such as the shared
Drive folder, and point the tools at it with:

```bash
export PAPERLOG_DB=/path/to/papers.db      # default: ./papers.db
export PAPERLOG_CONFIG=/path/to/config.yaml # default: the config.yaml in this folder
```

**Reading the log from other code:**

```python
from paperlog import api

api.papers(min_relevance=4, tags=["alpha_factor_mining"], since="2026-06-01")
api.pending(min_relevance=3)                   # not yet consumed downstream
api.mark_ingested([p["key"] for p in batch])   # idempotent handoff
api.adopted()                                  # your research backlog
api.mark("arxiv:2412.20138", status="adopted", notes="experiment exp-014")
api.history("arxiv:2412.20138")                # every score it has ever had
api.add("https://arxiv.org/abs/2412.20138", adopt=True)   # agent-callable
api.dataframe(min_relevance=3)                 # pandas, with the [dataframe] extra
```

**Weekly automation (not enabled).** `workflows/paperlog.yml` is a template, not an active
workflow. It commits the log to a `paperlog-data` branch, which in this public repo would
publish it, so enabling it needs a private destination for the data first, plus the API
keys below as repository secrets. The original steps, for reference:

1. Create the data branch once:
   ```bash
   git switch --orphan paperlog-data
   git commit --allow-empty -m "init paperlog data"
   git push -u origin paperlog-data
   git switch main
   ```
2. Copy `workflows/paperlog.yml` to `.github/workflows/paperlog.yml` at the repo root,
   and set `PAPERLOG_DIR` in it if the folder isn't `research/paperlog`.
3. Add `ANTHROPIC_API_KEY`, `OPENALEX_API_KEY` (and optionally `S2_API_KEY`) under
   Settings → Secrets and variables → Actions.
4. If you already ran it locally, seed the history: copy your `papers.db` onto the
   `paperlog-data` branch and push, so the first automated run continues your log.

To get the latest log locally: `git fetch origin paperlog-data` then
`git worktree add ../paperlog-data paperlog-data` (or `git -C ../paperlog-data pull` after the first time),
and set `PAPERLOG_DB=../paperlog-data/papers.db`.

## Costs

arXiv, Semantic Scholar and OpenAlex are free at this volume (OpenAlex's free key has a
daily allowance far above what a weekly run uses). The only cost is screening: with Haiku,
each paper is one short call, and `max_papers_per_run` caps the total. Check the token
counts printed at the end of each run against current pricing at claude.com/pricing.

## Tuning

Everything lives in `config.yaml`:

- **Anchors** (`semantic_scholar.anchors`): replace the examples with the papers you consider central.
  Citation expansion is often the highest-yield source.
- **Rubric and taxonomy** (`screening`): bump `rubric_version` whenever you edit the
  rubric. Papers scored under the old version are rescored automatically when there's
  budget left, or immediately with `paperlog rescreen`. `stats` shows which papers you
  rated much higher than the screener did: those are the misses worth rewriting for.
- **Queries and labs** (`openalex.queries`, `openalex.institutions`): add or remove
  English or Chinese queries and institutions freely; everything lands in the same set.
- **Prefilter** terms: if you see relevant papers marked `filtered` in the DB, widen the terms or set `mode: either`.

## Known limits

- **Chinese-language search quality in OpenAlex is uneven.** Many Chinese journal records
  lack abstracts, and keyword matching on Chinese text is less reliable than on English.
  Check the first few runs' results for the Chinese queries and adjust them.
- **Cross-language duplicates** (a Chinese journal version of an English preprint) aren't
  caught by ID or title matching. Enable `embeddings` (`pip install -e ".[embeddings]"`)
  to link them. The model is a ~2 GB download, so it's off by default.
- **Not included:** CNKI/Wanfang (no free API), and grey literature such as brokerage reports.
  The `tier` column is there so these can be added later without mixing them into academic results.

## Tests

`python -m pytest -q` runs an offline end-to-end test with mocked sources and a fake LLM.
