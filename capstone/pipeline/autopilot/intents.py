"""Write-ahead intents and exactly-once audit events.

One pattern for every effect a stage must not repeat or lose across a kill:

1. **Intent first.** Before an effect that is not idempotent by itself (a model call, a
   write to a store outside the run, a counter), `IntentFile.open(key, ...)` records what
   is about to happen under a deterministic `key`, atomically. A record already there is
   returned unchanged, so a resumed attempt reuses the first attempt's values (`at`, the
   counts it saw) instead of minting new ones.
2. **Effect, reconciled.** The effect is an idempotent call, or the resumed attempt finds
   what the killed one did (by the intent's key or its recorded values) before acting.
3. **Event keyed by the intent.** `event_once(ctx, key, kind, ...)` appends the audit
   event unless one carrying `once == key` is already in the log. It is called on *every*
   pass that finds the effect done, not only on the pass that did it, so a kill between
   the effect and the event loses nothing and a kill after the event repeats nothing.
4. **Close.** `IntentFile.close(key, ...)` marks the intent done with its result.

`fault(point)` is the named kill point the fresh-process kill tests use
(`capstone.faults.maybe_fault`; a no-op unless a test arms it).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from capstone.faults import maybe_fault

FOLDER = ("autopilot", "intents")


def fault(point):
    """A named kill point (`autopilot:<stage>:<where>`); a no-op in production."""
    maybe_fault(point)


def intent_id(*parts):
    """A deterministic id for an intent: sha256 of the canonical JSON of `parts`, 16 hex."""
    text = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def event_once(ctx, key, kind, **fields):
    """Append audit event `kind` with `once = key` unless the log already has one."""
    for row in ctx.journal.events():
        if row.get("once") == key:
            return row
    return ctx.event(kind, once=key, **fields)


class IntentFile:
    """`{key: {"status": "open" | "closed", ...}}` in one JSON file, replaced atomically."""

    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def of(cls, ctx):
        """The current stage's intent file, `<run>/autopilot/intents/<stage>.json`."""
        return cls(ctx.path(*FOLDER, f"{ctx.stage}.json"))

    def read(self):
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key):
        return self.read().get(key)

    def _write(self, document):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def open(self, key, **payload):
        """The intent `key`: the recorded one if any, else `payload` recorded now (open)."""
        document = self.read()
        if key not in document:
            document[key] = {"status": "open", **payload}
            self._write(document)
        return document[key]

    def put(self, key, **payload):
        """Record intent `key` as `payload` (open), replacing any earlier record."""
        document = self.read()
        document[key] = {"status": "open", **payload}
        self._write(document)
        return document[key]

    def close(self, key, **result):
        """Mark intent `key` closed with `result`; returns the record."""
        document = self.read()
        document[key] = {**document.get(key, {}), **result, "status": "closed"}
        self._write(document)
        return document[key]
