"""Read API for the rest of your research code.

    from paperlog import api
    for p in api.papers(min_relevance=4, tags=["valuation"]):
        print(p["key"], p["title_en"])

Set PAPERLOG_DB (or pass db_path=) to point at the log.
"""

from __future__ import annotations

import sqlite3

from . import db as dbm
from .paths import db_path as default_db

_SKIP = ("filtered", "duplicate")


def _connect(db_path: str | None, readonly: bool = True) -> sqlite3.Connection:
    path = db_path or default_db()
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con
    return dbm.connect(path)


def papers(
    min_relevance: int = 0,
    tags: list[str] | None = None,
    status: str | list[str] | None = None,
    since: str | None = None,
    search: str | None = None,
    db_path: str | None = None,
) -> list[dict]:
    """Query the log. Filters combine with AND; `tags` matches any of the given tags."""
    sql = "SELECT * FROM papers WHERE status NOT IN (?, ?) AND COALESCE(relevance, 0) >= ?"
    args: list = [*_SKIP, min_relevance]
    if status:
        statuses = [status] if isinstance(status, str) else list(status)
        sql += f" AND status IN ({','.join('?' * len(statuses))})"
        args += statuses
    if since:
        sql += " AND published >= ?"
        args.append(since)
    if search:
        sql += " AND (title LIKE ? OR title_en LIKE ? OR abstract LIKE ? OR summary_en LIKE ?)"
        args += [f"%{search}%"] * 4
    sql += " ORDER BY COALESCE(relevance, -1) DESC, published DESC"
    with _connect(db_path) as con:
        rows = [dbm.row_to_dict(r) for r in con.execute(sql, args)]
    if tags:
        want = set(tags)
        rows = [r for r in rows if want & set(r.get("tags") or [])]
    return rows


def get(key: str, db_path: str | None = None) -> dict | None:
    with _connect(db_path) as con:
        r = con.execute("SELECT * FROM papers WHERE key=?", (key,)).fetchone()
    return dbm.row_to_dict(r) if r else None


def pending(min_relevance: int = 0, limit: int = 100, db_path: str | None = None) -> list[dict]:
    """Screened papers your pipeline hasn't taken yet, best first. Call
    mark_ingested() once you've consumed them so reruns are idempotent."""
    with _connect(db_path) as con:
        rows = [
            dbm.row_to_dict(r)
            for r in con.execute(
                "SELECT * FROM papers WHERE ingested_at IS NULL AND status NOT IN (?, ?) "
                "AND COALESCE(relevance,0) >= ? "
                "ORDER BY COALESCE(relevance,-1) DESC, published DESC "
                "LIMIT ?",
                (*_SKIP, min_relevance, limit),
            )
        ]
    return rows


def mark_ingested(keys: list[str], db_path: str | None = None) -> int:
    con = _connect(db_path, readonly=False)
    try:
        con.executemany(
            "UPDATE papers SET ingested_at=? WHERE key=?", [(dbm.now(), k) for k in keys]
        )
        con.commit()
        return len(keys)
    finally:
        con.close()


def history(key: str, db_path: str | None = None) -> list[dict]:
    """Every scoring event for a paper, oldest first: score, rubric version, model."""
    with _connect(db_path) as con:
        return [
            dict(r)
            for r in con.execute(
                "SELECT screened_at, model, rubric_version, relevance, tags, summary_en "
                "FROM screenings WHERE key=? ORDER BY id",
                (key,),
            )
        ]


def add(
    identifier: str,
    adopt: bool = False,
    anchor: bool = False,
    screen: bool = True,
    notes: str | None = None,
    config_path: str | None = None,
    db_path: str | None = None,
) -> str:
    """Add a paper you found. Returns its key. Usable by an agent:
    api.add("https://arxiv.org/abs/2412.20138", adopt=True)"""
    from argparse import Namespace

    from .cli import cmd_add, load_config
    from .paths import config_path as default_config

    cfg = load_config(config_path or default_config())
    args = Namespace(
        identifier=identifier,
        adopt=adopt,
        anchor=anchor,
        rating=None,
        notes=notes,
        no_screen=not screen,
        db=db_path or default_db(),
    )
    return cmd_add(args, cfg)


def adopted(db_path: str | None = None) -> list[dict]:
    """Papers you've marked as adopted: the natural input to a research backlog."""
    return papers(status="adopted", db_path=db_path)


def mark(
    key: str,
    status: str | None = None,
    rating: int | None = None,
    notes: str | None = None,
    db_path: str | None = None,
) -> None:
    """Write feedback from other tools, e.g. mark a paper 'adopted' when an
    experiment based on it is created."""
    if status and status not in ("read", "adopted", "rejected"):
        raise ValueError("status must be read, adopted or rejected")
    con = _connect(db_path, readonly=False)
    try:
        if not con.execute("SELECT 1 FROM papers WHERE key=?", (key,)).fetchone():
            raise KeyError(key)
        con.execute(
            "UPDATE papers SET status=COALESCE(?, status), human_rating=COALESCE(?, human_rating), "
            "human_notes=COALESCE(?, human_notes), reviewed_at=? WHERE key=?",
            (status, rating, notes, dbm.now(), key),
        )
        con.commit()
    finally:
        con.close()


def dataframe(**filters):
    """Same filters as papers(), as a pandas DataFrame (needs the [dataframe] extra)."""
    import pandas as pd

    return pd.DataFrame(papers(**filters))
