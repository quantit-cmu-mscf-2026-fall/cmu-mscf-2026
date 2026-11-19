"""`<run>/ALERT.json` / `ALERT.md` for every STOP (and a new blocker, `[alert] on_blocked`).

A tick whose `report.problems` holds a *new* STOP (or, with `on_blocked`, a new
blocker) writes both files and appends the same document to
`<run>/autopilot/ops/alerts.jsonl`; a tick with nothing new removes neither (alerts are
history). A new `warning` row of the report alerts
too, with `severity = "warning"` when nothing is stopped or blocked. Then the notifier
runs:
`services["notifier"]` when injected (tests), else `[alert] notifier` =
`module:function`, called with a copy of the alert. ALERT.json is written before the
call with `notifier.status = "pending"` and rewritten with the outcome, so a kill during
the call leaves a readable alert and the next tick (the STOP is still there) calls the
notifier again. A notifier that cannot be resolved or raises is the typed problem
`alert_notifier_failed`, never a traceback.

Every alerting problem is deduplicated by its *problem key*:
`<run>/autopilot/ops/alert-seen.json` holds the keys already alerted and delivered, and a
tick alerts only when it has a key not in that set (one alert per new problem). The key
is `status|stage|code|<sha256 of detail>`, so the same code for another session (another
`detail`) is a new problem; a code whose detail reports progress rather than which
problem it is (`COARSE`, e.g. a resume count) is keyed
without it. A stage's STOP also carries the stage's attempt count: a STOP that lasts
alerts once, not on every tick, and a STOP that comes back after a person's `clear` (the
stage ran again) is new. A problem that goes away is forgotten, so it alerts again if it
comes back. The set is saved only after the alert is in the history and the notifier did
not fail, so a kill or a failed notifier alerts again on the next tick.
"""

from __future__ import annotations

import copy
import hashlib
import importlib

from capstone.pipeline.autopilot.ops.common import (
    append_line,
    ops_dir,
    problem,
    read_json,
    write_json,
    write_text,
)

SCHEMA = "autopilot-alert-v1"
JSON_FILE, MD_FILE, HISTORY = "ALERT.json", "ALERT.md", "alerts.jsonl"
SEEN = "alert-seen.json"
SEEN_SCHEMA = 2  # problem keys; an older file (blocker labels) reads as empty
# Codes whose `detail` reports progress (counts, timings), not which problem it is.
COARSE = frozenset({"spend_lock_busy", "autopilot_locked"})


def blocker_key(row):
    """The readable label of one problem: `<stage or "">|<reason code>`."""
    return f"{row['stage'] or ''}|{row['reason']['code']}"


def problem_key(row, journal=None):
    """The dedupe identity of one alerting problem : status, stage, code, a digest
    of `detail` (`*` for a `COARSE` code) and, for a stage's STOP, its attempt count."""
    reason = row["reason"]
    detail = str(reason["detail"]).encode("utf-8")
    digest = "*" if reason["code"] in COARSE else hashlib.sha256(detail).hexdigest()[:16]
    key = f"{row['status']}|{row['stage'] or ''}|{reason['code']}|{digest}"
    if row["status"] == "stopped" and row["stage"] and journal is not None:
        key += f"|a{journal.stage(row['stage'])['attempts']}"
    return key


def seen_keys(path):
    """The problem keys already alerted and delivered."""
    document = read_json(path, None) or {}
    if document.get("schema") != SEEN_SCHEMA:
        return set()
    return set(document.get("seen", []))


def save_seen(path, keys):
    write_json(path, {"schema": SEEN_SCHEMA, "seen": sorted(keys)})


def alerting(problems, on_blocked):
    """The problems an alert reports: every STOP, plus blockers when `on_blocked`, plus
    every `warning` row of the report."""
    keep = ("stopped", "blocked", "warning") if on_blocked else ("stopped", "warning")
    return [p for p in problems if p["status"] in keep]


def resolve(target):
    """`module:function` -> the callable; ValueError names what is wrong."""
    module_name, _, function_name = (target or "").partition(":")
    if not module_name or not function_name:
        raise ValueError(f"notifier {target!r} is not module:function")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"notifier module {module_name!r}: {error}") from error
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(f"notifier {target!r} is not a callable")
    return function


def markdown(alert):
    lines = [
        f"# Autopilot alert: {alert['severity']}",
        "",
        f"- Run: `{alert['run_root']}`",
        f"- Policy: `{alert['policy_id']}`",
        f"- At: {alert['at']}",
        f"- Line: {alert['line']}",
        f"- Notifier: {alert['notifier']['status']} {alert['notifier']['detail']}".rstrip(),
        "",
        "| Stage | Status | Code | Class | Detail | Cleared by |",
        "|---|---|---|---|---|---|",
    ]
    for row in alert["problems"]:
        reason = row["reason"]
        cells = [
            row["stage"] or "(run)",
            row["status"],
            reason["code"],
            reason["class"],
            reason["detail"],
            reason["cleared_by"],
        ]
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def save(root, alert):
    write_json(root / JSON_FILE, alert)
    write_text(root / MD_FILE, markdown(alert))


def notify(ctx, report, alert):
    """Call the notifier; returns its outcome row (and adds a problem when it fails)."""
    injected = ctx.services.get("notifier")
    target = ctx.config.sections["alert"]["notifier"]
    if injected is None and not target:
        return {"target": None, "status": "none", "detail": ""}
    name = getattr(injected, "__name__", "service") if injected is not None else target
    try:
        function = injected if injected is not None else resolve(target)
        function(copy.deepcopy(alert))
    except (
        Exception,
        SystemExit,
    ) as error:  # a notifier is operator code: typed, never a traceback
        detail = f"{name}: {type(error).__name__}: {error}"
        report.problems.append(problem("alert_notifier_failed", detail))
        return {"target": name, "status": "failed", "detail": detail}
    return {"target": name, "status": "sent", "detail": ""}


def run(ctx, report):
    """Write and send the alert for this tick; returns the alert, or None."""
    section = ctx.config.sections["alert"]
    found = alerting(report.problems, section["on_blocked"])
    root = ctx.run_root
    seen_path = ops_dir(root) / SEEN
    seen = seen_keys(seen_path)
    keyed = [(problem_key(p, ctx.journal), p) for p in found]
    present = {key for key, _ in keyed}
    new = sorted(present - seen)
    stopped = any(p["status"] == "stopped" for p in found)
    if not new:  # one alert per new problem, a lasting STOP included
        if seen - present:  # a problem that went away alerts again when it comes back
            save_seen(seen_path, seen & present)
        return None
    fresh = [p for key, p in keyed if key in new]
    alert = {
        "schema": SCHEMA,
        "at": ctx.now(),
        "run_root": str(root),
        "policy_id": ctx.config.core["policy_id"],
        "severity": (
            "stop"
            if stopped
            else "blocked"
            if any(p["status"] == "blocked" for p in found)
            else "warning"
        ),
        "line": report.line(),
        "problems": copy.deepcopy(found),
        "new_blockers": sorted({blocker_key(p) for p in fresh if p["status"] == "blocked"}),
        "new_problems": new,
        "states": dict(report.states),
        "notifier": {"target": None, "status": "pending", "detail": ""},
    }
    save(root, alert)
    outcome = notify(ctx, report, alert)
    alert["notifier"] = outcome
    if outcome["status"] == "failed":
        alert["problems"] = copy.deepcopy(alerting(report.problems, section["on_blocked"]))
        alert["line"] = report.line()
    save(root, alert)
    append_line(ops_dir(root) / HISTORY, alert)
    if outcome["status"] != "failed":
        save_seen(seen_path, present)
    return alert
