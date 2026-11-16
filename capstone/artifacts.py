"""Portable artifact identity: canonical JSON, its digest, and an atomic JSON write."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def canonical(value) -> str:
    """Portable identity, independent of mapping order; reject nonfinite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    """Write `value` as canonical JSON so a reader sees the old file or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
