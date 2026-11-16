---
name: paperlog
description: Search, summarise or add to the team's paper log (papers.db in the shared Drive folder, built by research/paperlog), or run the weekly paper collection. Use when asked to find papers on a topic, show what's new or most relevant, list papers the factor pipeline hasn't used yet, add a paper, or update the log.
---

# The team's paper log

`research/paperlog` collects papers from arXiv, OpenAlex and Semantic Scholar,
and Claude Haiku screens each one: relevance 0-5 against the team's rubric,
topic tags, a look-ahead check and an English summary
(`research/paperlog/README.md`). The log is private team data and lives only
in the shared Drive:

    ~/Library/CloudStorage/GoogleDrive-*/My Drive/mscf-capstone-data/papers/papers.db

## Rules

- Never copy `papers.db`, digests or CSVs into the repository (it is public),
  and never commit them. Never enable `research/paperlog/workflows/paperlog.yml`:
  as written it would publish the log.
- Never write API keys to a file or print them. Use the user's environment.
- Read from a **temporary copy**, never the Drive file, so a read can't change
  or lock the shared log.
- Change the shared log (run, add, review) only when the user asks, and only
  after step 2 of "Updating the log".

## Reading (default)

Copy the log, then query the copy read-only with Python's `sqlite3`; nothing
needs installing. The `papers` table holds each paper's latest screening:
`title`, `authors`, `published`, `url`, `abstract`, `relevance` (0-5),
`tags` (JSON list), `summary_en`, `methods`, `lookahead_bias`, `status`
(`screened`, `read`, `adopted`, `rejected`, ...), `human_rating`,
`ingested_at` (taken by the factor pipeline) and `duplicate_of`.

```bash
DB=$(ls ~/Library/CloudStorage/GoogleDrive-*/My\ Drive/mscf-capstone-data/papers/papers.db)
TMP=$(mktemp -d) && cp "$DB" "$TMP/papers.db"
python3 - "$TMP/papers.db" <<'EOF'
import sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
rows = con.execute("""
    select relevance, published, title, tags, url, summary_en from papers
    where duplicate_of is null and relevance >= 4
      and (title like '%hypothesis%' or abstract like '%hypothesis%')
    order by relevance desc, published desc limit 15""").fetchall()
for r in rows: print(r)
EOF
rm -rf "$TMP"
```

Adapt the `where` clause to the request: a topic (title or abstract), a tag
(`tags like '%alpha_factor_mining%'`), recent papers (`first_seen >= ...`),
papers the pipeline hasn't taken (`ingested_at is null and status = 'screened'`),
or adopted ones (`status = 'adopted'`). Answer with a short list: score, year,
title, one line from `summary_en`, and the link. Say how many matched.

## Updating the log

1. **Install the tool** once, from the repository root:
   `pip install -e research/paperlog`.
2. **Check before writing.** Only one person runs the weekly collection, the
   one the team agreed on; two runs on the same file through Drive sync can
   corrupt it. Ask the user to confirm they are that person. Then check that
   the Drive file isn't mid-sync or in use: no `papers.db-journal` or
   `papers.db-wal` beside it, and not modified in the last 10 minutes.
3. **Run** with `PAPERLOG_DB` set to the Drive path:
   - `paperlog add <arXiv id | DOI | title>`: add one paper the user found.
   - `paperlog run --no-screen`: collect only; free.
   - `paperlog run`: collect and screen. It calls the Anthropic API (capped by
     `budget_per_run` in `research/paperlog/config.yaml`); confirm the cost
     with the user first. It needs `ANTHROPIC_API_KEY`, and preferably
     `OPENALEX_API_KEY` and `S2_API_KEY` (without it Semantic Scholar
     rate-limits).
4. **Report** what the run printed: papers found and screened, any
   "truncated at N pages" lines, and where the digest was written. Put
   digests beside the log in the Drive (`--digest-dir`), never in the repo.
