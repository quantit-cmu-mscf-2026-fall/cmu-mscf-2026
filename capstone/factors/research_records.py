"""Content-addressed typed artifacts in the existing research/factor SQLite store."""

from __future__ import annotations

import json
import sqlite3

from capstone.factors.research_contracts import canonical, digest


def put_artifact(connection, kind, payload):
    """Insert idempotently inside the caller's transaction; never replace an artifact.

    The legacy method-spec identity rule remains unchanged. Callers validate typed
    payloads before saving; storing a record grants no qualification or authority.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("artifact kind is required")
    encoded = canonical(payload)
    key = digest(payload) if kind == "method_spec" else digest({"kind": kind, "payload": payload})
    connection.execute("""CREATE TABLE IF NOT EXISTS research_artifacts
        (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)""")
    connection.execute(
        "INSERT OR IGNORE INTO research_artifacts VALUES (?,?,?)", (key, kind, encoded)
    )
    row = connection.execute(
        "SELECT kind,payload FROM research_artifacts WHERE id=?", (key,)
    ).fetchone()
    if tuple(row) != (kind, encoded):
        raise ValueError("existing research artifact differs from its identity")
    return key


def load_artifact(connection, key, *, kind):
    """Verify complete saved payload identity before any typed loader consumes it."""
    try:
        row = connection.execute(
            "SELECT kind,payload FROM research_artifacts WHERE id=?", (key,)
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            raise
        raise ValueError("research artifact is missing") from exc
    if row is None or row[0] != kind:
        raise ValueError("research artifact is missing or has a different kind")
    payload = json.loads(row[1])
    expected = (
        digest(payload) if kind == "method_spec" else digest({"kind": kind, "payload": payload})
    )
    if key != expected or canonical(payload) != row[1]:
        raise ValueError("research artifact contents changed")
    return payload
