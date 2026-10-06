"""The factor store: papers, hypotheses, factor trees and every proposal.

A local SQLite file, append-only like the run ledger. It sits at
`CAPSTONE_FACTOR_DB` if that is set, else `experiments/factors.db`; it is
gitignored and never committed, because this repository is public.

Each factor is stored once, under its `factor_id`. Every time an agent
proposes it, from any hypothesis, a `proposals` row records that, as well as
proposals that were rejected or failed to parse. Proposals are the lineage
edges (paper -> hypothesis -> factor, and parent factor -> child), and their
count is what the search actually produced: N_searched, not N_reported.

A `variants` row is one way of trading a stored factor (a trading spec:
holding period, sector neutrality, ...). Evaluation trades variants, not
bare formulas, so each variant is a candidate with lineage to its factor.

Nothing here computes performance, so nothing here is a ledger trial.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from capstone.factors.tree import Node, factor_id, node_count, param_count, parse
from capstone.factors.tree import structural_key as _structural_key
from capstone.factors.tree import unparse as _unparse

DB_FILENAME = "factors.db"
ALIGNMENT_COLUMNS = (
    ("alignment", "REAL"),
    ("alignment_reason", "TEXT"),
    ("alignment_model", "TEXT"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  key TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  url TEXT,
  citation TEXT,
  source TEXT,
  added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypotheses (
  id TEXT PRIMARY KEY,
  paper_key TEXT NOT NULL REFERENCES papers(key),
  observation TEXT NOT NULL,
  knowledge TEXT NOT NULL,
  justification TEXT NOT NULL,
  specification TEXT NOT NULL,
  falsification_condition TEXT NOT NULL,
  model TEXT,
  prompt_version TEXT,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'market'
);
CREATE TABLE IF NOT EXISTS factors (
  id TEXT PRIMARY KEY,
  expression TEXT NOT NULL,
  structural_key TEXT NOT NULL,
  node_count INTEGER NOT NULL,
  param_count INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hypothesis_id TEXT NOT NULL REFERENCES hypotheses(id),
  expression TEXT NOT NULL,
  factor_id TEXT REFERENCES factors(id),
  parent_factor_id TEXT REFERENCES factors(id),
  edge_type TEXT NOT NULL,
  status TEXT NOT NULL,
  reason TEXT,
  zoo_similarity REAL,
  nearest_zoo TEXT,
  store_similarity REAL,
  nearest_factor TEXT,
  rationale TEXT,
  model TEXT,
  prompt_version TEXT,
  alignment REAL,
  alignment_reason TEXT,
  alignment_model TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_key TEXT NOT NULL REFERENCES papers(key),
  model TEXT NOT NULL,
  calls INTEGER NOT NULL,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bonds (
  id TEXT PRIMARY KEY,
  rule TEXT NOT NULL,
  leaves TEXT NOT NULL,
  leaf_nodes INTEGER NOT NULL,
  member_share REAL NOT NULL,
  nearest_members TEXT,
  bond_overlap REAL NOT NULL,
  nearest_bond TEXT,
  features TEXT NOT NULL,
  rationale TEXT,
  model TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bond_members (
  bond_id TEXT NOT NULL REFERENCES bonds(id),
  member_id TEXT NOT NULL,
  member_kind TEXT NOT NULL,
  PRIMARY KEY (bond_id, member_id)
);
CREATE TABLE IF NOT EXISTS moves (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  action TEXT NOT NULL,
  from_id TEXT,
  result_id TEXT,
  hypothesis_id TEXT,
  status TEXT NOT NULL,
  reason TEXT,
  policy TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS variants (
  id TEXT PRIMARY KEY,
  factor_id TEXT NOT NULL REFERENCES factors(id),
  spec TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_variants_factor ON variants(factor_id);
CREATE INDEX IF NOT EXISTS ix_proposals_factor ON proposals(factor_id);
CREATE INDEX IF NOT EXISTS ix_proposals_hypothesis ON proposals(hypothesis_id);
CREATE INDEX IF NOT EXISTS ix_hypotheses_paper ON hypotheses(paper_key);
"""

EDGE_TYPES = ("proposed", "refined")

# A market hypothesis claims something about returns and becomes factors; a
# method hypothesis is about how to search or validate, and is kept for the
# record but never turned into a factor.
KINDS = ("market", "method")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def default_path() -> Path:
    """`CAPSTONE_FACTOR_DB` if set, else `<repo>/experiments/factors.db`."""
    override = os.environ.get("CAPSTONE_FACTOR_DB")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "experiments" / DB_FILENAME


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the store, creating the file and tables if needed."""
    path = Path(path) if path is not None else default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    columns = {row[1] for row in con.execute("PRAGMA table_info(hypotheses)")}
    if "kind" not in columns:  # a store created before hypotheses had a kind
        con.execute("ALTER TABLE hypotheses ADD COLUMN kind TEXT NOT NULL DEFAULT 'market'")
    columns = {row[1] for row in con.execute("PRAGMA table_info(proposals)")}
    for name, kind in ALIGNMENT_COLUMNS:  # a store created before the alignment check
        if name not in columns:
            con.execute(f"ALTER TABLE proposals ADD COLUMN {name} {kind}")
    return con


# ---------------------------------------------------------------------------
# Papers and hypotheses


def add_paper(
    con: sqlite3.Connection,
    key: str,
    title: str,
    *,
    url: str = "",
    citation: str = "",
    source: str = "",
) -> None:
    """Record a source paper. A key already present is left as it is."""
    with con:
        con.execute(
            "INSERT OR IGNORE INTO papers (key, title, url, citation, source, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, title, url, citation, source, _now()),
        )


@dataclass(frozen=True)
class Hypothesis:
    """An economic claim stated before any performance is seen (AlphaAgent's format)."""

    paper_key: str
    observation: str
    knowledge: str
    justification: str
    specification: str
    falsification_condition: str
    kind: str = "market"

    def __post_init__(self) -> None:
        empty = [name for name, value in asdict(self).items() if not str(value).strip()]
        if empty:
            raise ValueError(f"hypothesis fields must be non-empty: {', '.join(empty)}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def add_hypothesis(
    con: sqlite3.Connection, hypothesis: Hypothesis, *, model: str = "", prompt_version: str = ""
) -> str:
    """Record a hypothesis under its paper and return its id. Repeats are ignored."""
    row = asdict(hypothesis)
    with con:
        con.execute(
            "INSERT OR IGNORE INTO hypotheses (id, paper_key, observation, knowledge, "
            "justification, specification, falsification_condition, model, prompt_version, "
            "created_at, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                hypothesis.id,
                row["paper_key"],
                row["observation"],
                row["knowledge"],
                row["justification"],
                row["specification"],
                row["falsification_condition"],
                model,
                prompt_version,
                _now(),
                row["kind"],
            ),
        )
    return hypothesis.id


def record_usage(
    con: sqlite3.Connection,
    paper_key: str,
    model: str,
    *,
    calls: int,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Record what processing one paper cost in model calls and tokens."""
    with con:
        con.execute(
            "INSERT INTO usage (paper_key, model, calls, input_tokens, output_tokens, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (paper_key, model, calls, input_tokens, output_tokens, _now()),
        )


# ---------------------------------------------------------------------------
# Proposals and factors


def _insert_proposal(con: sqlite3.Connection, **row) -> int:
    if row["edge_type"] not in EDGE_TYPES:
        raise ValueError(f"edge_type must be one of {EDGE_TYPES}, got {row['edge_type']!r}")
    row.setdefault("created_at", _now())
    columns = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    cursor = con.execute(f"INSERT INTO proposals ({columns}) VALUES ({marks})", tuple(row.values()))
    return int(cursor.lastrowid)


def add_factor(
    con: sqlite3.Connection,
    node: Node,
    hypothesis_id: str,
    *,
    expression: str | None = None,
    parent_factor_id: str | None = None,
    edge_type: str = "proposed",
    rationale: str = "",
    model: str = "",
    prompt_version: str = "",
    zoo_similarity: float | None = None,
    nearest_zoo: str = "",
    store_similarity: float | None = None,
    nearest_factor: str = "",
    alignment: float | None = None,
    alignment_reason: str = "",
    alignment_model: str = "",
) -> tuple[str, bool]:
    """Store a factor and the proposal that produced it.

    Returns (factor_id, is_new). A factor already in the store is not stored
    again, but the proposal is still recorded, with status "duplicate".
    """
    fid = factor_id(node)
    with con:
        is_new = (
            con.execute(
                "INSERT OR IGNORE INTO factors (id, expression, structural_key, node_count, "
                "param_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fid,
                    _unparse(node),
                    _structural_key(node),
                    node_count(node),
                    param_count(node),
                    _now(),
                ),
            ).rowcount
            == 1
        )
        _insert_proposal(
            con,
            hypothesis_id=hypothesis_id,
            expression=expression if expression is not None else _unparse(node),
            factor_id=fid,
            parent_factor_id=parent_factor_id,
            edge_type=edge_type,
            status="stored" if is_new else "duplicate",
            reason="" if is_new else "already in the store",
            zoo_similarity=zoo_similarity,
            nearest_zoo=nearest_zoo,
            store_similarity=store_similarity,
            nearest_factor=nearest_factor,
            rationale=rationale,
            model=model,
            prompt_version=prompt_version,
            alignment=alignment,
            alignment_reason=alignment_reason,
            alignment_model=alignment_model,
        )
    return fid, is_new


def reject(
    con: sqlite3.Connection,
    hypothesis_id: str,
    expression: str,
    reason: str,
    *,
    node: Node | None = None,
    parent_factor_id: str | None = None,
    edge_type: str = "proposed",
    rationale: str = "",
    model: str = "",
    prompt_version: str = "",
    zoo_similarity: float | None = None,
    nearest_zoo: str = "",
    store_similarity: float | None = None,
    nearest_factor: str = "",
    alignment: float | None = None,
    alignment_reason: str = "",
    alignment_model: str = "",
) -> int:
    """Record a proposal that was not stored: it failed to parse, was not original, or
    did not implement its hypothesis.

    The factor itself is not added to `factors`. Pass `node` when the
    expression parsed, so the proposal's row still names which factor it was.
    """
    with con:
        return _insert_proposal(
            con,
            hypothesis_id=hypothesis_id,
            expression=expression,
            factor_id=None,
            parent_factor_id=parent_factor_id,
            edge_type=edge_type,
            status="rejected",
            reason=reason if node is None else f"{reason} [factor {factor_id(node)}]",
            zoo_similarity=zoo_similarity,
            nearest_zoo=nearest_zoo,
            store_similarity=store_similarity,
            nearest_factor=nearest_factor,
            rationale=rationale,
            model=model,
            prompt_version=prompt_version,
            alignment=alignment,
            alignment_reason=alignment_reason,
            alignment_model=alignment_model,
        )


# ---------------------------------------------------------------------------
# Variants: how a factor is traded


def variant_id(fid: str, spec: dict) -> str:
    """Stable id of a (factor, trading spec): the factor id, then a short hash of the spec."""
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return f"{fid}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]}"


def add_variant(con: sqlite3.Connection, fid: str, spec: dict) -> str:
    """Record a way of trading a stored factor and return its id. Repeats are ignored."""
    vid = variant_id(fid, spec)
    with con:
        con.execute(
            "INSERT OR IGNORE INTO variants (id, factor_id, spec, created_at) VALUES (?, ?, ?, ?)",
            (vid, fid, json.dumps(spec, sort_keys=True), _now()),
        )
    return vid


def variants(con: sqlite3.Connection, fid: str | None = None) -> list[dict]:
    """Stored variants (spec decoded), optionally of one factor."""
    if fid is None:
        rows = con.execute("SELECT * FROM variants ORDER BY created_at, id")
    else:
        rows = con.execute(
            "SELECT * FROM variants WHERE factor_id = ? ORDER BY created_at, id", (fid,)
        )
    return [{**dict(r), "spec": json.loads(r["spec"])} for r in rows]


# ---------------------------------------------------------------------------
# Reading


def factors(con: sqlite3.Connection) -> list[dict]:
    """Every stored factor, oldest first."""
    return [dict(r) for r in con.execute("SELECT * FROM factors ORDER BY created_at, id")]


def factor_trees(con: sqlite3.Connection) -> list[tuple[str, Node]]:
    """Stored factors as (factor_id, tree), the shape `zoo_similarity` takes."""
    return [(row["id"], parse(row["expression"])) for row in factors(con)]


def hypotheses(con: sqlite3.Connection, paper_key: str | None = None) -> list[dict]:
    """Stored hypotheses, optionally for one paper."""
    if paper_key is None:
        rows = con.execute("SELECT * FROM hypotheses ORDER BY created_at, id")
    else:
        rows = con.execute(
            "SELECT * FROM hypotheses WHERE paper_key = ? ORDER BY created_at, id", (paper_key,)
        )
    return [dict(r) for r in rows]


def proposals(con: sqlite3.Connection, status: str | None = None) -> list[dict]:
    """Every proposal, optionally only one status (stored, duplicate, rejected)."""
    if status is None:
        rows = con.execute("SELECT * FROM proposals ORDER BY id")
    else:
        rows = con.execute("SELECT * FROM proposals WHERE status = ? ORDER BY id", (status,))
    return [dict(r) for r in rows]


def lineage(con: sqlite3.Connection, fid: str) -> list[dict]:
    """Every proposal of one factor, each with its hypothesis and paper."""
    rows = con.execute(
        "SELECT p.id AS proposal_id, p.status, p.edge_type, p.parent_factor_id, p.expression, "
        "p.rationale, p.model, p.created_at, h.id AS hypothesis_id, h.observation, "
        "h.specification, pa.key AS paper_key, pa.title AS paper_title, pa.url AS paper_url "
        "FROM proposals p JOIN hypotheses h ON h.id = p.hypothesis_id "
        "JOIN papers pa ON pa.key = h.paper_key WHERE p.factor_id = ? ORDER BY p.id",
        (fid,),
    )
    return [dict(r) for r in rows]
