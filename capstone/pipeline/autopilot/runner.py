"""`tick`: run every ready stage in dependency order; `status`; `clear` a STOP.

One tick, under the run's lock file:
- a stage runs when every dependency is `done`; a `done` stage is not run again unless
  it is `recurring`; a stage left `running` by a killed process is run again;
- `blocked` skips the stage's dependents and is retried by the next tick, except a
  dependent that declares the code in `Stage.after_blocked` (a stage that reads
  what a daily stage completed runs after that stage's `data_stale`);
- `stopped` (a STOP) ends the tick, and no later tick runs anything until
  `clear --stage <name>` records who cleared it and why;
- the operations hooks run last, also when the run is stopped;
- `CAPSTONE_LEDGER_DIR` is `<run>/ledger` while a stage or hook runs, and a stage that
  declares `spend` passes the caller's spend guard before its `run`. The guard and its
  ceilings are the caller's: nothing here fixes a number, and a spending stage with no
  guard supplied STOPs rather than runs.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from capstone.locking import exclusive_file
from capstone.pipeline.autopilot.journal import LOCK, Journal, sha256_file
from capstone.pipeline.autopilot.reasons import CORE, Blocked, Stop, record
from capstone.pipeline.autopilot.stage import Context, StageResult, order


@dataclass
class Report:
    """One tick's outcome. `problems` are the root causes (blocked or stopped), in order.

    A row with `status = "warning"` (for example an export's name check) is reported and
    alerted but is not a problem of the tick: it never changes `exit_code` or `line`."""

    ran: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    states: dict = field(default_factory=dict)

    def add(self, stage, status, reason):
        self.problems.append({"stage": stage, "status": status, "reason": reason})

    @property
    def blocking(self):
        """The problems that are not warnings."""
        return [p for p in self.problems if p["status"] != "warning"]

    @property
    def exit_code(self):
        return 2 if self.blocking else 0

    def line(self):
        blocking = self.blocking
        if not blocking:
            done = sum(state == "done" for state in self.states.values())
            warned = len(self.problems)
            return f"autopilot: ok ({done}/{len(self.states)} stages done, ran {len(self.ran)})" + (
                f" ({warned} warning{'s' if warned > 1 else ''})" if warned else ""
            )
        first = next((p for p in blocking if p["status"] == "stopped"), blocking[0])
        reason = first["reason"]
        text = f"autopilot: {first['status']} {first['stage'] or 'run'}: {reason['code']}"
        if reason["detail"]:
            text += f": {reason['detail']}"
        text += f" (cleared by: {reason['cleared_by']})"
        more = len(self.problems) - 1
        return " ".join(text.split()) + (f" (+{more} more)" if more else "")


RUN_ROOT_ENV, POLICY_ENV = "AUTOPILOT_RUN_ROOT", "AUTOPILOT_POLICY_ID"


@contextmanager
def _run_tags(root, policy_id):
    """Name the run root and policy for `log_run` to tag entries with, then restore.

    `CAPSTONE_LEDGER_DIR` is left as the operator set it: autopilot trials belong in the
    project ledger, so a new run root still sees every earlier root's trials.
    """
    values = {RUN_ROOT_ENV: str(Path(root)), POLICY_ENV: policy_id}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: value for key, value in values.items() if value is not None})
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _hash_artifacts(root, artifacts):
    out = {}
    for artifact in artifacts:
        path = Path(artifact)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise Blocked("no_stage_artifact", path)
        out[path.resolve().relative_to(root.resolve()).as_posix()] = sha256_file(path)
    return out


def _changed(root, artifacts):
    for name, digest in sorted(artifacts.items()):
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            return name
    return None


def _finish(journal, stage, rec, status, code, detail, **fields):
    declared = {**CORE, **stage.reasons}
    if code is not None and code not in declared:
        code, detail = "undeclared_reason", f"{code}: {detail}"
    reason = None if code is None else record(code, detail, declared)
    journal.update(stage.name, state=status, finished_at=journal.clock(), reason=reason, **fields)
    journal.event("stage_finished", stage=stage.name, state=status, code=code)
    return reason


def _run(stage, config, journal, services, sha, spend_guard):
    rec = journal.stage(stage.name)
    attempt = rec["attempts"] + 1
    journal.event(
        "stage_started", stage=stage.name, attempt=attempt, resumed=rec["state"] == "running"
    )
    journal.update(
        stage.name,
        state="running",
        attempts=attempt,
        started_at=journal.clock(),
        section_sha256=sha,
    )
    try:
        ctx = Context(config, journal, stage.name, services)
        with _run_tags(config.run_root, config.core["policy_id"]):
            ceiling = None if stage.spend is None else stage.spend(ctx)
            if ceiling is not None:  # before anything is frozen or called
                if spend_guard is None:
                    raise Stop("spend_guard_missing", stage.name)
                spend_guard(ctx, stage.spend_scope or stage.name, ceiling)
            result = stage.run(ctx)
        if not isinstance(result, StageResult):
            raise TypeError(f"{stage.name}.run returned {type(result).__name__}, not StageResult")
        fields = {"crashes": 0}
        if result.status == "done":
            fields |= {
                "artifacts": _hash_artifacts(config.run_root, result.artifacts),
                "data": dict(result.data),
            }
        return result.status, _finish(
            journal, stage, rec, result.status, result.code, result.detail, **fields
        )
    except Blocked as blocker:
        status, code, detail, crashes = blocker.status, blocker.code, blocker.detail, 0
    except (Exception, SystemExit) as error:  # a stage's CLI may exit; that is a crash too
        crashes = rec["crashes"] + 1
        limit = crashes >= config.core["max_crashes"]
        status, code = ("stopped", "stage_crash_limit") if limit else ("blocked", "stage_crashed")
        detail = f"{type(error).__name__}: {error}"
    return status, _finish(journal, stage, rec, status, code, detail, crashes=crashes)


def _tick(config, ordered, journal, services, report, only, spend_guard):
    policy = config.core["policy_id"]
    if journal.doc["policy_id"] is None:
        journal.doc["policy_id"] = policy
        journal.save()
        journal.event("run_started", policy_id=policy, config=str(config.path))
    elif journal.doc["policy_id"] != policy:
        detail = f"journal has {journal.doc['policy_id']!r}, config has {policy!r}"
        report.add(None, "stopped", record("policy_changed", detail, CORE))
        return
    stopped = [s for s in ordered if journal.stage(s.name)["state"] == "stopped"]
    for stage in stopped:
        report.add(stage.name, "stopped", journal.stage(stage.name)["reason"])
    if stopped:
        return
    unusable, blocked_now = set(), {}

    def waits(stage, dep):
        """A dependency blocked in this tick with a code the stage declares in
        `after_blocked` does not hold it back (what the dependency completed is there)."""
        if blocked_now.get(dep) in stage.after_blocked.get(dep, ()):
            return False
        return dep in unusable or journal.stage(dep)["state"] != "done"

    for stage in ordered:
        rec = journal.stage(stage.name)
        if only is not None and stage.name not in only:
            if rec["state"] != "done":
                unusable.add(stage.name)
            continue
        if any(waits(stage, d) for d in stage.depends_on):
            unusable.add(stage.name)
            continue
        sha = config.section_sha256(stage.section)
        if rec.get("section_sha256") not in (None, *config.section_identities(stage.section)):
            reason = record("config_section_changed", f"[{stage.section}]", CORE)
            report.add(stage.name, "blocked", reason)
            unusable.add(stage.name)
            continue
        if rec["state"] == "done" and not stage.recurring:
            changed = _changed(config.run_root, rec["artifacts"])
            if changed is None:
                continue
            status = "stopped"
            reason = _finish(journal, stage, rec, status, "artifact_changed", changed)
        else:
            status, reason = _run(stage, config, journal, services, sha, spend_guard)
            report.ran.append(stage.name)
        if status != "done":
            report.add(stage.name, status, reason)
            unusable.add(stage.name)
        if status == "blocked":
            blocked_now[stage.name] = reason["code"]
        if status == "stopped":
            return


def tick(config, stages, *, services=None, hooks=(), only=None, spend_guard=None):
    """Run one tick; returns its `Report` (`exit_code` 0 = done or progress, 2 otherwise).

    `only` (stage names) is for a unit's own tests: no other stage is run or re-hashed;
    one that is not `done` in the journal just keeps its dependents waiting.

    `spend_guard(ctx, scope, ceiling)` is the caller's check of a spending stage's call
    ceiling; it raises `Stop` (a spend reason) to refuse. It is required for any stage
    whose `spend` hook returns a ceiling."""
    services = dict(services or {})
    ordered = order(stages)
    if only is not None and (unknown := set(only) - {s.name for s in ordered}):
        raise ValueError(f"unknown stages in only: {sorted(unknown)}")
    root, report = config.run_root, Report()
    lock = exclusive_file(root / LOCK)
    try:
        lock.__enter__()
    except RuntimeError:
        report.add(None, "blocked", record("autopilot_locked", root / LOCK, CORE))
        return report
    try:
        journal = Journal(root, clock=services.get("clock"))
        _tick(config, ordered, journal, services, report, only, spend_guard)
        report.states = {s.name: journal.stage(s.name)["state"] for s in ordered}
        for hook in hooks:
            try:
                with _run_tags(root, config.core["policy_id"]):
                    hook(Context(config, journal, None, services), report)
            except Exception as error:
                detail = f"{getattr(hook, '__module__', hook)}: {type(error).__name__}: {error}"
                report.add(None, "blocked", record("ops_hook_failed", detail, CORE))
        return report
    finally:
        lock.__exit__(None, None, None)


def status(config, stages):
    """The journal as one table (read-only, takes no lock)."""
    journal = Journal(config.run_root)
    rows = []
    for stage in order(stages):
        rec = journal.stage(stage.name)
        rows.append(
            {
                "stage": stage.name,
                "unit": stage.unit,
                "depends_on": list(stage.depends_on),
                "state": rec["state"],
                "attempts": rec["attempts"],
                "reason": rec.get("reason"),
                "artifacts": rec["artifacts"],
            }
        )
    return {
        "run_root": str(config.run_root),
        "policy_id": journal.doc["policy_id"],
        "events": journal.verify_events(),
        "stages": rows,
    }


def clear(config, stages, name, *, reviewer, reason):
    """A person clears one STOP; the stage becomes pending and the event log says who and why."""
    if name not in {stage.name for stage in stages}:
        raise ValueError(f"unknown stage {name!r}")
    if not reviewer.strip() or not reason.strip():
        raise ValueError("clear needs a reviewer and a reason")
    with exclusive_file(config.run_root / LOCK, message="another tick owns the run"):
        journal = Journal(config.run_root)
        rec = journal.stage(name)
        if rec["state"] != "stopped":
            raise ValueError(f"stage {name!r} is {rec['state']}, not stopped")
        journal.event("stop_cleared", stage=name, reviewer=reviewer, reason=reason, was=rec)
        journal.update(name, state="pending", crashes=0, reason=None)
