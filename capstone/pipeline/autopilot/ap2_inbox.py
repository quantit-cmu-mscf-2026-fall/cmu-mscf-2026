"""The inbox folder as a source list, and the input archive's `data_version`.

Inbox grammar: a flat folder; names starting with `.` are
ignored. `*.pdf` is a `local_pdf`, `*.txt|*.md|*.markdown|*.html|*.htm` a `local_text`.
`<file>.source.json` beside `<file>` is that file's sidecar (`key`, `title`, `author`,
`published_date`, `role`); a `*.source.json` with no such file is a source of its own
(`text` for an inline note, or `url`). Anything else is refused (`InboxError`), never
skipped, so the frozen list is exactly what was dropped.

A local file's identity is its inbox-relative name and its content: unless its
sidecar gives a `key`, the row carries `key = "inbox:<name>@sha256:<hex>"`, so its
`source_id` does not depend on where the inbox or the run root lives (the absolute
`path` is still how the batch reads it).
"""

from __future__ import annotations

import json
from pathlib import Path

from capstone.factors.research_contracts import digest
from capstone.factors.source_intake import SourceRequest
from capstone.pipeline.autopilot.journal import sha256_file

SIDECAR = ".source.json"
PDF = {".pdf"}
TEXT = {".txt", ".md", ".markdown", ".html", ".htm"}
SIDECAR_FIELDS = {"key", "title", "author", "published_date", "role"}
STANDALONE_FIELDS = SIDECAR_FIELDS | {"text", "url"}


class InboxError(ValueError):
    """The inbox holds something the grammar does not accept."""


def fingerprint(inbox):
    """{file name: sha256} of every inbox entry the grammar reads (sorted by name)."""
    root = Path(inbox)
    if not root.is_dir():
        return {}
    out = {}
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if path.name.startswith("."):
            continue
        if not path.is_file():
            raise InboxError(f"{path.name}: the inbox is flat; no folders")
        out[path.name] = sha256_file(path)
    return out


def _json(path, allowed):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InboxError(f"{path.name}: {error}") from error
    if not isinstance(value, dict) or value.keys() - allowed:
        extra = sorted(value.keys() - allowed) if isinstance(value, dict) else "not an object"
        raise InboxError(f"{path.name}: unsupported fields {extra}")
    return value


def source_rows(inbox, default_role):
    """[(file name, source_intake row)] for the inbox, sorted by file name.

    Local rows carry absolute paths, so the list resolves the same from anywhere, and a
    content + relative-name `key`, so their identity does not depend on that path."""
    root = Path(inbox).resolve()
    hashes = fingerprint(root)
    names = list(hashes)
    rows = []
    for name in names:
        path = root / name
        if name.endswith(SIDECAR):
            target = name[: -len(SIDECAR)]
            if target in names:
                continue  # read with its file
            fields = _json(path, STANDALONE_FIELDS)
            if ("text" in fields) == ("url" in fields):
                raise InboxError(f"{name}: a standalone source has exactly one of text, url")
            row = {"kind": "inline_text" if "text" in fields else "url", **fields}
            row.setdefault("role", default_role)
        else:
            suffix = path.suffix.lower()
            if suffix not in PDF | TEXT:
                raise InboxError(f"{name}: unsupported file type {suffix or '(none)'}")
            sidecar = root / (name + SIDECAR)
            meta = _json(sidecar, SIDECAR_FIELDS) if sidecar.is_file() else {}
            kind = "local_pdf" if suffix in PDF else "local_text"
            row = {"kind": kind, "path": str(path), "role": default_role, **meta}
            row.setdefault("key", local_key(name, hashes[name]))
        rows.append((name, row))
    return rows


def local_key(name, sha256):
    """The work key of a local inbox file: its inbox-relative name and its content."""
    return f"inbox:{name}@sha256:{sha256}"


def requests(rows):
    """{file name: SourceRequest}; refuses a row the batch would refuse, and two files
    that name one source (`source_id`), before anything is frozen."""
    out, seen = {}, {}
    for name, row in rows:
        fields = dict(row)
        kind = fields.pop("kind")
        locator = fields.pop("path", fields.pop("url", ""))
        try:
            request = SourceRequest(kind=kind, locator=locator, **fields)
        except (TypeError, ValueError) as error:
            raise InboxError(f"{name}: {error}") from error
        if request.source_id in seen:
            raise InboxError(f"{name} and {seen[request.source_id]} are one source (same key)")
        seen[request.source_id] = name
        out[name] = request
    return out


def archive_data_version(archive):
    """The `data_version` of a qualified input archive, after the descriptor checks that
    need no frame read: schema, cutoff, `input_hash`, and each frame file's sha256.
    The freeze (`--input-archive`) verifies the frames themselves."""
    root = Path(archive)
    path = root / "historical-inputs.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("cutoff") != "2020-12-31":
        raise ValueError(f"{path}: unsupported historical input archive")
    if digest({k: v for k, v in metadata.items() if k != "input_hash"}) != metadata.get(
        "input_hash"
    ):
        raise ValueError(f"{path}: descriptor changed (input_hash)")
    for label, item in metadata["frames"].items():
        frame = root / item["file"]
        if not frame.is_file() or sha256_file(frame) != item["sha256"]:
            raise ValueError(f"{path}: frame {label} is missing or changed")
    version = metadata.get("data_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{path}: no data_version")
    return version
