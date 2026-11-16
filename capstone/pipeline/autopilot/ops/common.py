"""Shared pieces of the operations hook: the ops folder, atomic writes, typed ops reasons.

Everything the hook writes lives in `<run>/autopilot/ops/` except `<run>/ALERT.json` / `.md`
(the alert files). Every file is replaced atomically (temp file, fsync, rename), so a kill
leaves the previous or the next version.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from capstone.pipeline.autopilot.reasons import ReasonSpec, record

OPS_DIR = ("autopilot", "ops")

# Reason codes the hook reports in `report.problems` (`"stage": null`); each alerting one is
# written to ALERT.json and passed to the notifier.
REASONS = {
    "alert_notifier_failed": ReasonSpec(
        "Operator", "fix `[alert] notifier` (module:function); the next alert calls it again"
    ),
    "ops_status_failed": ReasonSpec("Operator", "read `detail`; the next tick writes it again"),
}


def problem(code, detail, status="blocked"):
    """One `report.problems` row for an ops reason."""
    return {"stage": None, "status": status, "reason": record(code, detail, REASONS)}


def ops_dir(root) -> Path:
    return Path(root).joinpath(*OPS_DIR)


def write_text(path, text):
    """Replace `path` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, document):
    write_text(path, json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path, default=None):
    """The document, or `default` when the file is absent or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_line(path, row):
    """Append one fsynced JSON line; a line torn by a kill is closed off, never extended."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    line = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(("\n" if text and not text.endswith("\n") else "") + line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_lines(path):
    """The complete JSON lines of `path` (a torn line is skipped)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.split("\n")[:-1]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows
