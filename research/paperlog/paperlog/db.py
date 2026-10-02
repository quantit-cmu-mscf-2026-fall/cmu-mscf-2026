import json
import sqlite3
from datetime import UTC, datetime

from .models import Paper, normalize_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  key TEXT PRIMARY KEY,
  title TEXT NOT NULL, title_norm TEXT, abstract TEXT, authors TEXT, published TEXT,
  url TEXT, sources TEXT, doi TEXT, arxiv_id TEXT, openalex_id TEXT, s2_id TEXT,
  institutions TEXT, countries TEXT, language TEXT, venue TEXT, found_via TEXT,
  tier TEXT DEFAULT 'academic',
  first_seen TEXT, run_id INTEGER,
  -- screening (LLM)
  -- status: new | filtered | screened | read | adopted | rejected | duplicate
  status TEXT DEFAULT 'new',
  screened_at TEXT, screen_model TEXT, relevance INTEGER, tags TEXT,
  title_en TEXT, summary_en TEXT, methods TEXT, datasets TEXT, markets TEXT,
  eval_method TEXT, code_url TEXT, lookahead_bias TEXT, screen_notes TEXT,
  -- human feedback
  human_rating INTEGER, human_notes TEXT, reviewed_at TEXT,
  rubric_version INTEGER, prefilter_version INTEGER, screen_count INTEGER DEFAULT 0,
  ingested_at TEXT,                  -- claimed by a downstream consumer
  duplicate_of TEXT, embedding BLOB
);
CREATE INDEX IF NOT EXISTS ix_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS ix_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS ix_title ON papers(title_norm);
CREATE INDEX IF NOT EXISTS ix_status ON papers(status);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, started TEXT, finished TEXT, since TEXT,
  fetched INTEGER, new INTEGER, screened INTEGER, notes TEXT
);
CREATE TABLE IF NOT EXISTS anchors (
  title TEXT PRIMARY KEY, s2_id TEXT, source TEXT DEFAULT 'config', refs_done TEXT
);
-- one row per scoring event: scores are dated opinions, not permanent facts
CREATE TABLE IF NOT EXISTS screenings (
  id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, screened_at TEXT, model TEXT,
  rubric_version INTEGER, relevance INTEGER, tags TEXT, summary_en TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_screenings_key ON screenings(key);
"""

LIST_FIELDS = (
    "authors",
    "sources",
    "institutions",
    "countries",
    "found_via",
    "tags",
    "methods",
    "datasets",
    "markets",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# columns added after v0.2; existing databases are upgraded in place
MIGRATIONS = {
    "rubric_version": "INTEGER",
    "prefilter_version": "INTEGER",
    "screen_count": "INTEGER DEFAULT 0",
    "ingested_at": "TEXT",
}


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    have = {r["name"] for r in con.execute("PRAGMA table_info(papers)")}
    for col, decl in MIGRATIONS.items():
        if col not in have:
            con.execute(f"ALTER TABLE papers ADD COLUMN {col} {decl}")
    have_anchor = {r["name"] for r in con.execute("PRAGMA table_info(anchors)")}
    for col, decl in {"source": "TEXT DEFAULT 'config'", "refs_done": "TEXT"}.items():
        if col not in have_anchor:
            con.execute(f"ALTER TABLE anchors ADD COLUMN {col} {decl}")
    con.commit()
    return con


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for f in LIST_FIELDS:
        if d.get(f):
            try:
                d[f] = json.loads(d[f])
            except (TypeError, json.JSONDecodeError):
                pass
    d.pop("embedding", None)
    return d


def find_existing(con, p: Paper) -> str | None:
    if p.arxiv_id:
        r = con.execute("SELECT key FROM papers WHERE arxiv_id=?", (p.arxiv_id,)).fetchone()
        if r:
            return r["key"]
    if p.doi:
        r = con.execute("SELECT key FROM papers WHERE doi=?", (p.doi,)).fetchone()
        if r:
            return r["key"]
    r = con.execute("SELECT key FROM papers WHERE title_norm=?", (p.title_norm,)).fetchone()
    return r["key"] if r else None


def insert(con, p: Paper, run_id: int, key: str | None = None) -> str:
    key = key or p.key
    con.execute(
        """INSERT INTO papers (key,title,title_norm,abstract,authors,published,url,sources,doi,
           arxiv_id,openalex_id,s2_id,institutions,countries,language,venue,found_via,
           first_seen,run_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            key,
            p.title,
            p.title_norm,
            p.abstract,
            json.dumps(p.authors, ensure_ascii=False),
            p.published,
            p.url,
            json.dumps(p.sources),
            p.doi,
            p.arxiv_id,
            p.openalex_id,
            p.s2_id,
            json.dumps(p.institutions, ensure_ascii=False),
            json.dumps(p.countries),
            p.language,
            p.venue,
            json.dumps(p.found_via, ensure_ascii=False),
            now(),
            run_id,
        ),
    )
    return key


def add_sighting(con, key: str, p: Paper) -> None:
    """A known paper turned up again: record extra sources/affiliations, fill gaps."""
    row = con.execute("SELECT * FROM papers WHERE key=?", (key,)).fetchone()
    old = row_to_dict(row)
    merged = Paper(
        title=old["title"],
        abstract=old["abstract"] or "",
        authors=old["authors"] or [],
        published=old["published"] or "",
        url=old["url"] or "",
        sources=old["sources"] or [],
        doi=old["doi"],
        arxiv_id=old["arxiv_id"],
        openalex_id=old["openalex_id"],
        s2_id=old["s2_id"],
        institutions=old["institutions"] or [],
        countries=old["countries"] or [],
        language=old["language"] or "",
        venue=old["venue"] or "",
        found_via=old["found_via"] or [],
    )
    merged.merge(p)
    con.execute(
        """UPDATE papers SET abstract=?, sources=?, doi=COALESCE(doi,?),
           arxiv_id=COALESCE(arxiv_id,?),
           openalex_id=COALESCE(openalex_id,?), s2_id=COALESCE(s2_id,?), institutions=?,
           countries=?, found_via=?, language=COALESCE(NULLIF(language,''),?) WHERE key=?""",
        (
            merged.abstract,
            json.dumps(merged.sources),
            merged.doi,
            merged.arxiv_id,
            merged.openalex_id,
            merged.s2_id,
            json.dumps(merged.institutions, ensure_ascii=False),
            json.dumps(merged.countries),
            json.dumps(merged.found_via, ensure_ascii=False),
            merged.language,
            key,
        ),
    )


def all_titles(con) -> list[tuple]:
    return [
        (r["key"], r["title_norm"], r["arxiv_id"], r["doi"])
        for r in con.execute("SELECT key, title_norm, arxiv_id, doi FROM papers")
    ]


def normalize(t):  # re-export for convenience
    return normalize_title(t)
