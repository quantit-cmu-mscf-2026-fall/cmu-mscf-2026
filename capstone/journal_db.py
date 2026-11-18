"""A process-owned SQLite journal connection; callers create their own tables."""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from capstone.locking import exclusive_file


@contextmanager
def locked_journal(path, *, message="another process owns this journal"):
    """Open `path` under an exclusive `<path>.lock` (WAL, synchronous=FULL); fail on contention."""
    path = Path(path).resolve()  # one lock per database, however it is reached
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file(path.with_name(path.name + ".lock"), message=message):
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            yield db
