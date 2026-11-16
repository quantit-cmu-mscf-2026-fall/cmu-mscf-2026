"""Status aggregation: the autopilot journal, the newest alert and its history.

`write(config, names)` writes `<run>/autopilot/ops/status.json` and `status.md` once the run
root exists (it never creates the run root); the operations hook writes the same files at
the end of every tick. `ok` is false when any stage is blocked or stopped. `alert` is the
newest `<run>/ALERT.json` (alerts are history: a clean tick keeps it); `warnings` are the
run's current typed early warnings (`early_warnings`). A broken journal
makes `aggregate` raise; the hook turns that into the typed problem `ops_status_failed`.
"""

from __future__ import annotations

from collections import Counter

from capstone.pipeline.autopilot.journal import Journal
from capstone.pipeline.autopilot.ops import alert as alerts
from capstone.pipeline.autopilot.ops import early_warnings
from capstone.pipeline.autopilot.ops.common import (
    ops_dir,
    read_json,
    read_lines,
    write_json,
    write_text,
)

SCHEMA = "autopilot-status-v1"
JSON_FILE, MD_FILE = "status.json", "status.md"


def aggregate(config, names, *, journal=None):
    """The status document for stage `names` (in DAG order)."""
    root = config.run_root
    journal = journal or Journal(root)
    rows = []
    for name in names:
        rec = journal.stage(name)
        rows.append(
            {
                "stage": name,
                "state": rec["state"],
                "attempts": rec["attempts"],
                "reason": rec.get("reason"),
            }
        )
    attention = [r for r in rows if r["state"] in ("blocked", "stopped")]
    return {
        "schema": SCHEMA,
        "run_root": str(root),
        "policy_id": journal.doc["policy_id"],
        "events": journal.verify_events(),
        "ok": not attention,
        "counts": dict(sorted(Counter(r["state"] for r in rows).items())),
        "attention": [r["stage"] for r in attention],
        "stages": rows,
        "alert": read_json(root / alerts.JSON_FILE),
        "alerts_total": len(read_lines(ops_dir(root) / alerts.HISTORY)),
        "warnings": (read_json(ops_dir(root) / early_warnings.FILE) or {}).get("warnings", []),
    }


def markdown(doc):
    lines = [
        "# Autopilot status",
        "",
        f"- Run: `{doc['run_root']}`",
        f"- Policy: `{doc['policy_id']}`",
        f"- OK: {'yes' if doc['ok'] else 'no'}",
        f"- Events: {doc['events']}; alerts so far: {doc['alerts_total']}",
        "",
        "| Stage | State | Attempts | Reason | Cleared by |",
        "|---|---|---|---|---|",
    ]
    for row in doc["stages"]:
        reason = row["reason"] or {}
        cells = [
            row["stage"],
            row["state"],
            row["attempts"],
            reason.get("code", ""),
            reason.get("cleared_by", ""),
        ]
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    if doc["alert"]:
        last = doc["alert"]
        lines += ["", f"Last alert ({last['severity']}, {last['at']}): {last['line']}"]
    for row in doc.get("warnings", []):
        reason = row["reason"]
        lines += ["", f"Warning ({reason['code']}): {reason['detail']}"]
    return "\n".join(lines) + "\n"


def write(config, names, *, journal=None):
    doc = aggregate(config, names, journal=journal)
    if not config.run_root.is_dir():
        return doc  # nothing has run: status does not create the run root
    folder = ops_dir(config.run_root)
    write_json(folder / JSON_FILE, doc)
    write_text(folder / MD_FILE, markdown(doc))
    return doc
