"""A paper tick's refusals and failures reach an ALERT.

The paper tick runs outside the autopilot, so a refusal would otherwise be a traceback
and an exit code only. Here every refusal the tick raises (a coded one such as
`paper_reconcile_failed` or `paper_reconcile_divergence` included), every paper-guard
refusal and any other failure writes `<paper folder>/ALERT.json` and `ALERT.md` in the
autopilot's alert schema (`alert.SCHEMA`, one `report.problems`-shaped row with
`"stage": "paper_tick"`), appends it to `<paper folder>/alerts.jsonl`, and calls the
same notifier the autopilot calls (`[alert] notifier`, `module:function`, read from the
autopilot config with `notifier_from_config`, or given directly).

The notifier is called once per new problem (the autopilot's problem key: status, code
and a digest of the detail, kept in `<paper folder>/alert-seen.json`); a repeat rewrites
the ALERT files with `notifier.status = "repeat"`. A tick that succeeds forgets the seen
keys (`cleared`), so a problem that comes back alerts again. A notifier that cannot be
resolved or raises is recorded in the alert (`status = "failed"`) and never hides the
tick's own error, which is re-raised unchanged. The original exception is marked, so an
outer wrapper does not alert twice.

A paper-guard refusal (`PaperGuardError`: not a paper account, `CAPSTONE_PAPER_TRADING`
unset, not a paper port) happens before anything is read or written, so it is sent to
the notifier on every tick but writes no file under the paper folder. A simulated kill
(`InjectedFault`, tests only) is a dead process, not a refusal: no alert.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path

from capstone.pipeline.autopilot.ops import alert as autopilot_alert
from capstone.pipeline.autopilot.ops.common import (
    append_line,
    read_json,
    write_json,
    write_text,
)
from capstone.pipeline.autopilot.reasons import ReasonSpec, record

STAGE = "paper_tick"
SEEN = autopilot_alert.SEEN
MARK = "_paper_alerted"
# Codes that need a person before the next session can trade; the rest are blockers the
# next tick may clear by itself (a drop-folder session arriving, a registry fixed).
STOPPED = frozenset(
    {
        "kill_switch",
        "paper_guard",
        "mixed_roles",
        "gate_report_mismatch",
        "paper_tick_failed",
        "deploy_refused",
    }
)
REASONS = {
    "kill_switch": ReasonSpec("Operator", "remove the kill-switch file when trading may resume"),
    "paper_guard": ReasonSpec(
        "Broker", "a DU account, CAPSTONE_PAPER_TRADING=1 and a paper port; fix the environment"
    ),
    "holdout_sealed": ReasonSpec("Operator", "the session is inside the sealed holdout"),
    "nothing_enabled": ReasonSpec("Admission", "enable a strategy and push a new deploy-* tag"),
    "mixed_roles": ReasonSpec("Admission", "one deploy-* tag per role; re-tag"),
    "gate_report_mismatch": ReasonSpec(
        "Admission", "pass the run's own gate report, byte for byte the one at the tag"
    ),
    "current_registry_unreadable": ReasonSpec("Operator", "fix the registry checkout"),
    "plan_not_transmittable": ReasonSpec("Planning", "read the blockers in `detail`"),
    "paper_reconcile_failed": ReasonSpec(
        "Data", "read the tick's monitor_error; the send stands, the next tick reconciles again"
    ),
    "paper_reconcile_divergence": ReasonSpec(
        "Data",
        "compare the paper account with the plans for the session in `detail`; accept it in "
        "<paper folder>/reconcile-accepted.json (reviewer, reason) or fix the account; the "
        "next tick re-attributes from the last good session",
    ),
    "no_deploy_tag": ReasonSpec("Admission", "a person merges the deploy PR and pushes the tag"),
    "no_session": ReasonSpec("Data", "the export has not filled the drop folder yet"),
    "capital_not_configured": ReasonSpec(
        "Operator", "set --capital, the context's capital or CAPSTONE_PAPER_CAPITAL"
    ),
    "deploy_refused": ReasonSpec("Admission", "read `detail`; fix the registry at the tag"),
    "paper_tick_failed": ReasonSpec("Operator", "read `detail` (an unexpected error)"),
}
DEFAULT = ReasonSpec("Data")  # a drop-folder or contract code the tick passes through


def classify(error):
    """(code, status) of an exception the paper tick raised."""
    from capstone.execution.broker import PaperGuardError
    from capstone.execution.deploy import Refused

    code = getattr(error, "code", None)
    if isinstance(error, PaperGuardError):
        code = "paper_guard"
    elif code is None:
        code = "deploy_refused" if isinstance(error, Refused) else "paper_tick_failed"
    return code, "stopped" if code in STOPPED else "blocked"


def problem_row(error):
    code, status = classify(error)
    detail = str(error) if code != "paper_tick_failed" else f"{type(error).__name__}: {error}"
    return {"stage": STAGE, "status": status, "reason": record(code, detail, _Reasons())}


class _Reasons(dict):
    def __missing__(self, code):
        return REASONS.get(code, DEFAULT)


def notifier_from_config(path):
    """`[alert] notifier` of an autopilot config (`module:function`), or None."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return (raw.get("alert") or {}).get("notifier") or None


def _key(row):
    reason = row["reason"]
    digest = hashlib.sha256(str(reason["detail"]).encode("utf-8")).hexdigest()[:16]
    return f"{row['status']}|{row['stage']}|{reason['code']}|{digest}"


def _seen(folder):
    document = read_json(Path(folder) / SEEN, None) or {}
    if document.get("schema") != autopilot_alert.SEEN_SCHEMA:
        return set()
    return set(document.get("seen", []))


def _now():
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def alert(folder, error, *, notifier=None, context=None, clock=None):
    """Write the ALERT for `error` under the paper folder and call the notifier once per
    new problem; returns the alert document."""
    folder = Path(folder)
    row = problem_row(error)
    key = _key(row)
    # A paper-guard refusal happens before anything is read or written: the
    # notifier hears of it, but nothing is written under the paper folder.
    quiet = row["reason"]["code"] == "paper_guard"
    seen = set() if quiet else _seen(folder)
    context = dict(context or {})
    line = f"paper tick {row['status']}: {row['reason']['code']}: {row['reason']['detail']}"
    document = {
        "schema": autopilot_alert.SCHEMA,
        "at": (clock or _now)(),
        "run_root": str(folder),
        "policy_id": context.get("tag"),
        "severity": "stop" if row["status"] == "stopped" else "blocked",
        "line": line,
        "problems": [row],
        "new_blockers": [autopilot_alert.blocker_key(row)] if row["status"] == "blocked" else [],
        "new_problems": [key] if key not in seen else [],
        "states": {STAGE: row["status"]},
        "paper": context,
        "notifier": {"target": None, "status": "pending", "detail": ""},
    }
    name = notifier if isinstance(notifier, str) else getattr(notifier, "__name__", None)
    if key in seen:
        document["notifier"] = {"target": name, "status": "repeat", "detail": key}
        _save(folder, document)
        return document
    if not quiet:
        _save(folder, document)
    if notifier is None:
        outcome = {"target": None, "status": "none", "detail": ""}
    else:
        try:
            function = autopilot_alert.resolve(notifier) if isinstance(notifier, str) else notifier
            function(copy.deepcopy(document))
        except Exception as failure:  # operator code: recorded, never hides the tick's error
            detail = f"{name}: {type(failure).__name__}: {failure}"
            outcome = {"target": name, "status": "failed", "detail": detail}
        else:
            outcome = {"target": name, "status": "sent", "detail": ""}
    document["notifier"] = outcome
    if quiet:
        return document
    _save(folder, document)
    append_line(folder / autopilot_alert.HISTORY, document)
    if outcome["status"] != "failed":
        write_json(
            folder / SEEN, {"schema": autopilot_alert.SEEN_SCHEMA, "seen": sorted(seen | {key})}
        )
    return document


def _save(folder, document):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / autopilot_alert.JSON_FILE, document)
    write_text(folder / autopilot_alert.MD_FILE, autopilot_alert.markdown(document))


def cleared(folder):
    """A tick that succeeded: forget the seen keys (the ALERT files are history)."""
    path = Path(folder) / SEEN
    if path.is_file():
        write_json(path, {"schema": autopilot_alert.SEEN_SCHEMA, "seen": []})


@contextmanager
def alerting(folder, *, notifier=None, context=None, clock=None):
    """Alert on any exception raised inside (once), then re-raise it unchanged."""
    from capstone.faults import InjectedFault

    try:
        yield
    except Exception as error:
        # A simulated kill (InjectedFault) is a dead process, not a refusal: no alert.
        if not getattr(error, MARK, False) and not isinstance(error, InjectedFault):
            try:
                alert(folder, error, notifier=notifier, context=context, clock=clock)
            except Exception as failure:  # the tick's own error is what matters
                print(f"paper alert not written: {failure}", file=sys.stderr)
            setattr(error, MARK, True)
        raise
