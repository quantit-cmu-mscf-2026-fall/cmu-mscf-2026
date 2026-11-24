"""AUTOPILOT.json (per-stage state, atomic) and AUTOPILOT.events.jsonl (append-only).

The state file is replaced atomically (temp file, fsync, rename), so a kill leaves the
previous or the next state, never a torn one. Events are appended one fsynced line each
and chained: every event carries the sha256 of the line before it, so an edited or
dropped line is detected by `verify_events`. A stage found `running` at the start of a
tick was interrupted; the runner reruns it (stages are idempotent).

Single writer: one process at a time may write a run root. Two writers break the hash
chain, and a second `Journal` instance's `update` overwrites the first one's stage records.
The runner must hold `LOCK` in the run root for the whole tick; the runner is not in this
module, so nothing here takes it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
STATE, EVENTS, LOCK = "AUTOPILOT.json", "AUTOPILOT.events.jsonl", "AUTOPILOT.lock"
GENESIS = "0" * 64


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class JournalError(ValueError):
    pass


class Journal:
    def __init__(self, run_root, *, clock=None):
        self.root = Path(run_root)
        self.clock = clock or (lambda: datetime.now(UTC).isoformat(timespec="seconds"))
        path = self.root / STATE
        if path.exists():
            self.doc = json.loads(path.read_text(encoding="utf-8"))
            if self.doc.get("schema_version") != SCHEMA_VERSION:
                raise JournalError(f"{path}: unsupported schema_version")
        else:
            self.doc = {"schema_version": SCHEMA_VERSION, "policy_id": None, "stages": {}}

    def stage(self, name) -> dict:
        """One stage's record (a copy); a stage never run is `pending`."""
        blank = {"state": "pending", "attempts": 0, "crashes": 0, "artifacts": {}, "data": {}}
        return copy.deepcopy(self.doc["stages"].get(name, blank))

    def update(self, name, **fields):
        """Merge `fields` into one stage record and save atomically."""
        self.doc["stages"][name] = {**self.stage(name), **fields}
        self.save()

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / STATE
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(self.doc, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _lines(self):
        """(complete event lines, raw text). A line cut by a kill is not an event."""
        path = self.root / EVENTS
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        complete = []
        for line in text.split("\n"):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            complete.append(line)
        return complete, text

    def events(self):
        return [json.loads(line) for line in self._lines()[0]]

    def event(self, kind, **fields):
        complete, text = self._lines()
        previous = hashlib.sha256(complete[-1].encode()).hexdigest() if complete else GENESIS
        row = {"seq": len(complete), "at": self.clock(), "kind": kind, "prev": previous, **fields}
        line = json.dumps(row, sort_keys=True, separators=(",", ":"))
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / EVENTS).open("a", encoding="utf-8", newline="\n") as stream:
            # a torn tail is closed off as its own ignored line, never overwritten
            stream.write(("\n" if text and not text.endswith("\n") else "") + line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return row

    def verify_events(self):
        """The event count; raises unless the log is the unbroken chain that was appended."""
        previous = GENESIS
        complete = self._lines()[0]
        for count, line in enumerate(complete):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise JournalError(f"{self.root / EVENTS}: line {count} is not a JSON object")
            if row.get("seq") != count or row.get("prev") != previous:
                raise JournalError(f"{self.root / EVENTS}: event {count} breaks the chain")
            previous = hashlib.sha256(line.encode()).hexdigest()
        return len(complete)
