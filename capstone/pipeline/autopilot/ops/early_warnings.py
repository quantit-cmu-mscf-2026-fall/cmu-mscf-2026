"""Typed early warnings, before the run commits to a long wait or blocks.

A warning is not a problem: it never changes a tick's exit code or a stage's state. It
names something the run will run into later, while a person can still act on it:

- `chronological_cash_period`: the portfolio `daily` would trade is chronological, so it
  plans cash until its members have `AllocationPolicy.minimum` prefix receipts (1,260
  sessions, about five years, for the registered policy), and an admitted one waits the
  same number of live sessions after 2025. Checked as soon as `portfolios` has chosen
  the comparison (until `admission` admits no portfolio), and by `daily` itself before
  it plans the first session of an admitted portfolio, so it comes before any cash plan.
- `final_protocol_manifest_missing`: `[final_protocol] manifest` is unset or not a file,
  so `final_protocol` falls back to the shared-data cache's `SHA256SUMS` and blocks after
  a gate pass without one (checked from the first tick until the stage is done).
- `spend_runs_dir_unset`: `[spend] run_ceiling` allows calls but `[spend] runs_dir` is
  unset, so the global ceiling counts only this run and `accounting_roots`, the
  cross-run spend lock is off, and `real_model_calls = true` is refused at load.

Each warning row has the shape of a `report.problems` row with `status = "warning"`.
`deliver` keeps the current warnings in `<run>/autopilot/ops/warnings.json` (no clock:
the same state gives the same bytes) and sends each warning once per run, keyed like an
alert (`alert.problem_key`), to the `[alert]` notifier (`services["notifier"]` in
tests) as a document of `severity = "warning"` whose `problems` are the new warnings.
Warnings never write `<run>/ALERT.json`. A notifier that fails is the ops problem
`alert_notifier_failed` at the end of the tick; the warning is sent again next tick.
"""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

from capstone.pipeline.autopilot.ops import alert
from capstone.pipeline.autopilot.ops.common import ops_dir, problem, read_json, write_json
from capstone.pipeline.autopilot.reasons import ReasonSpec, record

SCHEMA = "autopilot-warnings-v1"
NOTICE = "autopilot-warning-v1"
FILE = "warnings.json"
SESSIONS_PER_YEAR = 252  # trading sessions, for the "about N years" in a detail

WARNINGS = {
    "chronological_cash_period": ReasonSpec(
        "Planning",
        "nothing, if the wait is intended; otherwise a person chooses a non-chronological "
        "comparison or another `[daily] saved_portfolio` before `daily` trades it",
    ),
    "final_protocol_manifest_missing": ReasonSpec(
        "Operator", "set `[final_protocol] manifest` to the shared-data SHA256SUMS file"
    ),
    "spend_runs_dir_unset": ReasonSpec(
        "Spend", "set `[spend] runs_dir` to the folder holding every run root"
    ),
}


def warning(stage, code, detail):
    """One warning row (the shape of a `report.problems` row)."""
    return {"stage": stage, "status": "warning", "reason": record(code, detail, WARNINGS)}


def _section(ctx, name):
    return ctx.config.sections.get(name) or {}


def _chronological_minimum(spec):
    """`AllocationPolicy.minimum` of a chronological saved portfolio spec, else None."""
    try:
        document = tomllib.loads(Path(spec).read_text(encoding="utf-8"))
        definition = json.loads(document["payload"])["definition"]
        if definition.get("membership_clock") is None:
            return None
        return int(definition["policy"]["minimum"])
    except (OSError, KeyError, TypeError, ValueError):  # decode / TOML / JSON errors too
        return None


def chronological(ctx):
    """The cash-period warning for the portfolio `daily` would trade, if chronological."""
    if "daily" not in ctx.config.section_of:
        return []
    admission = ctx.record("admission")
    if admission["state"] == "done" and not admission["data"].get("portfolios"):
        return []  # nothing admitted that plans on a portfolio's clock
    spec = _section(ctx, "daily").get("saved_portfolio")
    if spec is None:
        comparison = ctx.data("portfolios").get("comparison_dir")
        if not comparison:
            return []
        spec = ctx.path(*Path(comparison).parts, "portfolio-spec.toml")
    minimum = _chronological_minimum(spec)
    if not minimum:
        return []
    shown = Path(spec)
    if shown.is_relative_to(ctx.run_root):
        shown = shown.relative_to(ctx.run_root)
    detail = (
        f"{shown.as_posix()} is a chronological portfolio: it plans cash until its members "
        f"have {minimum} prefix receipts (AllocationPolicy.minimum, about "
        f"{minimum / SESSIONS_PER_YEAR:.0f} years of sessions), and once admitted it waits "
        f"{minimum} live sessions after 2025"
    )
    return [warning("daily", "chronological_cash_period", detail)]


def manifest(ctx):
    """The missing-manifest warning, while `final_protocol` is still to run."""
    if "final_protocol" not in ctx.config.section_of:
        return []
    if ctx.record("final_protocol")["state"] == "done":
        return []
    path = _section(ctx, "final_protocol").get("manifest")
    if path is None:
        detail = (
            "[final_protocol] manifest is not set: after a gate pass the stage reads the "
            "shared-data cache's SHA256SUMS and blocks without one"
        )
    elif not Path(path).is_file():
        detail = f"[final_protocol] manifest {path} is not a file: it blocks after a gate pass"
    else:
        return []
    return [warning("final_protocol", "final_protocol_manifest_missing", detail)]


def runs_dir(ctx):
    """The unset-`runs_dir` warning, when the run may make model calls."""
    spend = _section(ctx, "spend")
    if spend.get("runs_dir") is not None or not spend.get("run_ceiling"):
        return []
    detail = (
        f"[spend] run_ceiling is {spend['run_ceiling']} but [spend] runs_dir is not set: the "
        "global ceiling counts only this run and accounting_roots, the cross-run spend lock "
        "is off, and real_model_calls = true is refused without it"
    )
    return [warning(None, "spend_runs_dir_unset", detail)]


def check(ctx):
    """Every current warning of the run, in a fixed order (read-only)."""
    return [*chronological(ctx), *manifest(ctx), *runs_dir(ctx)]


def _notify(ctx, rows):
    """Send the new warnings; the notifier outcome row (`alert.notify`'s shape)."""
    injected = ctx.services.get("notifier")
    target = _section(ctx, "alert").get("notifier")
    if injected is None and not target:
        return {"target": None, "status": "none", "detail": ""}
    name = getattr(injected, "__name__", "service") if injected is not None else target
    first = rows[0]["reason"]
    line = f"autopilot: warning {rows[0]['stage'] or 'run'}: {first['code']}: {first['detail']}"
    notice = {
        "schema": NOTICE,
        "severity": "warning",
        "at": ctx.now(),
        "run_root": str(ctx.run_root),
        "policy_id": ctx.config.core["policy_id"],
        "line": line + (f" (+{len(rows) - 1} more)" if len(rows) > 1 else ""),
        "problems": copy.deepcopy(rows),
        "notifier": {"target": name, "status": "pending", "detail": ""},
    }
    try:
        function = injected if injected is not None else alert.resolve(target)
        function(notice)
    except Exception as error:  # a notifier is operator code: typed, never a traceback
        return {"target": name, "status": "failed", "detail": f"{name}: {error!r}"}
    return {"target": name, "status": "sent", "detail": ""}


def deliver(ctx, report=None):
    """Record the run's current warnings and send each new one once; returns the new ones.

    `report` (the tick's, in the ops hook) receives `alert_notifier_failed` when the
    notifier fails; `daily` passes none (the hook retries at the end of the tick)."""
    rows = check(ctx)
    path = ops_dir(ctx.run_root) / FILE
    saved = read_json(path, None) or {}
    delivered = set(saved.get("delivered", [])) if saved.get("schema") == SCHEMA else set()
    keyed = [(alert.problem_key(row), row) for row in rows]
    new = [row for key, row in keyed if key not in delivered]
    if not rows and not delivered:
        return []
    document = {"schema": SCHEMA, "warnings": rows, "delivered": sorted(delivered)}
    if document != saved:  # recorded before the notifier is called
        write_json(path, document)
    if not new:
        return []
    outcome = _notify(ctx, new)
    if outcome["status"] == "failed":
        if report is not None:
            report.problems.append(problem("alert_notifier_failed", outcome["detail"]))
        return new
    document["delivered"] = sorted(delivered | {key for key, _ in keyed})
    write_json(path, document)
    return new
