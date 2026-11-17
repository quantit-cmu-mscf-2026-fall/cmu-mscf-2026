"""The `research` loop-guard file and the request journal's open pauses.

`<run>/autopilot/research-guard.json` is the stage's own state. It is written ahead of
every action it records, atomically, so a kill at any instant leaves a decision that the
next tick finishes instead of counting twice:

```json
{"schema_version": 1, "consecutive_pauses": 1, "last_pause": ["<request id>"],
 "clears_seen": 0,
 "decisions": [{"request_id": "...", "packet": "idea-0", "bucket": "...",
                "decision": "retry", "pause": 1, "at": "..."}]}
```

A pause is identified by the sorted ids of the requests awaiting resolution, so the same
unresolved pause seen again (after a kill, or a STOP a person cleared) is not a new one.
A `clear` of the stage records `cleared_pause` (the pause the STOP was about: autopilot
never answers it, the person does in the request journal) and `restart_count` (the next
new pause counts as the first in a row).
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from capstone.factors.research_contracts import digest
from capstone.factors.research_records import load_artifact

GUARD = ("autopilot", "research-guard.json")
AWAITING = ("inflight", "uncertain", "paused", "failed")
# a generation request's task is the attempt id; an advisory one appends this suffix
TASKS = ("", ":source-rule-alignment")
MAX_ORDINAL = 4


def blank():
    return {
        "schema_version": 1,
        "consecutive_pauses": 0,
        "last_pause": None,
        "clears_seen": 0,
        "cleared_pause": None,
        "restart_count": False,
        "decisions": [],
    }


def load(root):
    path = Path(root).joinpath(*GUARD)
    if not path.is_file():
        return blank()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    return {**blank(), **saved}


def save(root, guard):
    path = Path(root).joinpath(*GUARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(guard, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def retries(guard, packet):
    """How often this packet's requests were re-registered (explicit `retry`)."""
    return sum(row["packet"] == packet and row["decision"] == "retry" for row in guard["decisions"])


def _connect(path):
    return closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True))


def awaiting(source_run):
    """Requests awaiting an explicit decision: `[{id, logical, bucket, state, error}]`."""
    path = Path(source_run) / "requests.db"
    if not path.is_file():
        return []
    with _connect(path) as db:
        rows = db.execute(
            f"""SELECT id, logical, bucket, state, error FROM model_requests r
            WHERE state IN ({",".join("?" * len(AWAITING))})
            AND NOT EXISTS(SELECT 1 FROM request_resolutions s WHERE s.request_id = r.id)
            ORDER BY rowid""",
            AWAITING,
        ).fetchall()
    keys = ("id", "logical", "bucket", "state", "error")
    return [dict(zip(keys, row, strict=True)) for row in rows]


def resolution(source_run, request_id):
    """The recorded decision for one request, or None."""
    path = Path(source_run) / "requests.db"
    if not path.is_file():
        return None
    with _connect(path) as db:
        row = db.execute(
            "SELECT decision FROM request_resolutions WHERE request_id = ?", (request_id,)
        ).fetchone()
    return None if row is None else row[0]


def packets(source_run, requests, buckets):
    """`{logical: packet_id}` for the given requests, from the controller's attempt records.

    A generation request's logical id is `digest(bucket, attempt_id, ordinal)`; an
    advisory one uses `attempt_id + ":source-rule-alignment"`. A request no attempt
    explains keys as `request:<logical>` (its own packet for the re-registration count).
    """
    wanted = {row["logical"] for row in requests}
    found = {}
    path = Path(source_run) / "controller.db"
    if wanted and path.is_file():
        with _connect(path) as db:
            ids = db.execute(
                "SELECT id FROM research_artifacts WHERE kind = 'idea_attempt_v1' ORDER BY id"
            ).fetchall()
            for (key,) in ids:
                attempt = load_artifact(db, key, kind="idea_attempt_v1")
                for bucket in buckets:
                    for suffix in TASKS:
                        for ordinal in range(MAX_ORDINAL):
                            logical = digest(
                                {
                                    "bucket": bucket,
                                    "task": attempt["attempt_id"] + suffix,
                                    "ordinal": ordinal,
                                }
                            )
                            if logical in wanted:
                                found[logical] = attempt["packet_id"]
    return {logical: found.get(logical, f"request:{logical}") for logical in wanted}
