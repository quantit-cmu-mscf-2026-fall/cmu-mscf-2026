"""Model-spend stop-loss across runs, with caller-supplied ceilings.

`guard(ctx, scope, ceiling)` is the arithmetic a person would do by hand before every
freeze, resume or refinement. A run holds one call ceiling per *scope* (for example
`research`: the sum of the registration's `call_limits`), saved in
`<run>/autopilot/spend-guard.json`. It raises a STOP unless

- the run's scopes sum to at most `[spend] run_ceiling` (default 0: nothing may call);
- what every other run has committed, plus this run's, is at most
  `[spend] global_ceiling` (default 0).

Both ceilings are the caller's: nothing in this module fixes a number. The team's agreed
default of 300 calls is a value to put in the config, and `budget_params(config)` returns
the pair to record in the `params` of every `capstone.runlog.log_run` call, so the ledger
shows the budget a run was allowed to spend.

The other runs are discovered, not listed: every folder directly in `[spend] runs_dir`
(except this run and a `<run>-verify` scratch copy of an autopilot run), plus every
`[spend] accounting_roots` entry (runs kept elsewhere).
A run root commits the larger of the `outbound_reservations` in every
`request-accounting.json` under it and the scopes of its own `spend-guard.json`. A root
with neither is refused, never counted as zero, unless it is an autopilot run that has
not passed its first guard yet (`AUTOPILOT.json` and no spend record: the runner guards
before any call, so it has committed nothing). Any unreadable run, or a `runs_dir` that
cannot be listed, is a STOP. Repeating a check with the same scope and ceiling is the
same arithmetic, so a resumed stage passes again.

The read-sum-check-reserve is one critical section across runs: with `runs_dir`
set, the guard holds `<runs_dir>/.spend-guard.lock` (an OS-released file lock, like the
tick's `AUTOPILOT.lock`) from discovering the other runs until this run's
`spend-guard.json` is replaced, so two concurrent freezes cannot both fit the same
headroom. A guard waits up to `LOCK_WAIT` seconds for another run's check; after that
it is blocked (`spend_lock_busy`) and the next tick retries. The operating system
releases a killed holder's lock.

Pass `guard` to `runner.tick(..., spend_guard=guard)`: the runner calls it for every
stage that declares `Stage.spend`, before `run`; a stage that learns its ceiling only
while running calls it itself.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from capstone.locking import exclusive_file
from capstone.pipeline.autopilot.config import SPEND_SECTION
from capstone.pipeline.autopilot.reasons import Blocked, Stop

ACCOUNTING = "request-accounting.json"
GUARD = ("autopilot", "spend-guard.json")
REGISTRATION = ("autopilot", "registration.json")
AUTOPILOT_STATE = "AUTOPILOT.json"  # journal.STATE
VERIFY = "-verify"  # default scratch folder of a verification copy
FROZEN = ("run_ceiling", "global_ceiling")
LOCK = ".spend-guard.lock"  # a file in `runs_dir`: never a run root (only folders are)
LOCK_WAIT = 60.0  # seconds; the section is a folder scan and one small write
LOCK_POLL = 0.05


@contextmanager
def cross_run_lock(policy):
    """Hold `<runs_dir>/.spend-guard.lock`, waiting up to `LOCK_WAIT` seconds for another
    run's check, then `Blocked("spend_lock_busy")`. Without a listable `runs_dir` there is
    no shared folder to lock (the guard itself refuses a missing one)."""
    runs_dir = policy["runs_dir"]
    if runs_dir is None or not Path(runs_dir).is_dir():
        yield
        return
    path = Path(runs_dir) / LOCK
    deadline = time.monotonic() + LOCK_WAIT
    with ExitStack() as held:
        while True:
            try:
                held.enter_context(exclusive_file(path, message=f"{path} is held"))
                break
            except RuntimeError as busy:
                if time.monotonic() >= deadline:
                    raise Blocked("spend_lock_busy", f"{busy} after {LOCK_WAIT:g}s") from busy
                time.sleep(LOCK_POLL)
        yield


def _calls(value, where):
    if type(value) is not int or value < 0:
        raise ValueError(f"{where}: {value!r} is not a call count")
    return value


def _saved(root):
    path = Path(root).joinpath(*GUARD)
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    for scope, ceiling in saved["scopes"].items():
        _calls(ceiling, f"{path}: scope {scope}")
    return saved


def _reserved(root):
    """(sum of `outbound_reservations`, number of accounting files) under one run root."""
    root = Path(root)
    files = sorted(root.rglob(ACCOUNTING)) if root.is_dir() else []
    return sum(
        _calls(json.loads(path.read_text(encoding="utf-8"))["outbound_reservations"], path)
        for path in files
    ), len(files)


def committed(root):
    """Calls one run root counts toward the global ceiling; raises when it has no
    readable accounting and no spend-guard record (an autopilot run before its first
    guard counts 0: it cannot have called yet)."""
    (reserved, files), saved = _reserved(root), _saved(root)
    if not files and saved is None:
        if (Path(root) / AUTOPILOT_STATE).is_file():
            return 0
        raise ValueError(f"{root}: no {ACCOUNTING} and no {'/'.join(GUARD)}")
    return max(reserved, sum(saved["scopes"].values()) if saved else 0)


def other_runs(ctx):
    """Every other run root counts: each folder in `[spend] runs_dir` (discovered) and
    each `[spend] accounting_roots` entry; this run is never counted twice."""
    policy = ctx.config.sections[SPEND_SECTION]
    base = ctx.config.path.resolve().parent
    found = []
    if policy["runs_dir"] is not None:
        runs_dir = Path(policy["runs_dir"])
        if not runs_dir.is_dir():
            raise FileNotFoundError(f"[spend] runs_dir {runs_dir} is not a folder")
        folders = sorted(p for p in runs_dir.iterdir() if p.is_dir())
        runs = {p.name for p in folders if (p / AUTOPILOT_STATE).is_file()}
        # `<run>-verify` beside an autopilot run is a verification scratch
        # (copies of that run), not a run of its own
        found += [
            p for p in folders if not (p.name.endswith(VERIFY) and p.name[: -len(VERIFY)] in runs)
        ]
    found += [base / entry for entry in policy["accounting_roots"]]
    this, out = ctx.run_root.resolve(), []
    for path in (p.resolve() for p in found):
        if path != this and path not in out:
            out.append(path)
    return out


def registration_ceiling(ctx):
    """The `research` scope: the sum of `call_limits` of `<run>/autopilot/registration.json`,
    the document the freeze reads."""
    path = ctx.path(*REGISTRATION)
    try:
        limits = json.loads(path.read_text(encoding="utf-8"))["call_limits"]
        return sum(_calls(count, f"call_limits[{name}]") for name, count in limits.items())
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise Stop("spend_accounting_unreadable", f"{path}: {error}") from error


def budget_params(config):
    """The caller's two ceilings, as the `params` entry of a `log_run` call."""
    policy = config.sections[SPEND_SECTION]
    return {"spend_budget": {key: policy[key] for key in FROZEN}}


def guard(ctx, scope, ceiling):
    """Raise `Stop` unless `ceiling` calls for `scope` fit the ceilings; returns the arithmetic."""
    policy = ctx.config.sections[SPEND_SECTION]
    frozen = {key: policy[key] for key in FROZEN}
    root = ctx.run_root
    with cross_run_lock(policy):  # read, sum, check and reserve as one section
        check = _check_and_reserve(ctx, root, scope, ceiling, frozen)
    ctx.event("spend_guard", **check)
    return check


def _check_and_reserve(ctx, root, scope, ceiling, frozen):
    try:
        _calls(ceiling, f"{scope} ceiling")
        saved = _saved(root) or {"schema_version": 1, "policy": frozen, "scopes": {}}
        scopes = {**saved["scopes"], scope: ceiling}
        others = {other.as_posix(): committed(other) for other in other_runs(ctx)}
        this_run = max(sum(scopes.values()), _reserved(root)[0])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise Stop("spend_accounting_unreadable", f"{type(error).__name__}: {error}") from error
    if saved["policy"] != frozen:
        raise Stop("spend_policy_changed", f"started with {saved['policy']}, config has {frozen}")
    check = {
        "scope": scope,
        "ceiling": ceiling,
        "scopes": scopes,
        "run_total": this_run,
        "run_ceiling": frozen["run_ceiling"],
        "other_runs": others,
        "global_total": this_run + sum(others.values()),
        "global_ceiling": frozen["global_ceiling"],
    }
    if this_run > frozen["run_ceiling"]:
        detail = f"{scopes} = {this_run} > run_ceiling {frozen['run_ceiling']}"
        raise Stop("spend_run_ceiling_exceeded", detail)
    if check["global_total"] > frozen["global_ceiling"]:
        detail = f"{this_run} + other runs {sum(others.values())} > {frozen['global_ceiling']}"
        raise Stop("spend_global_ceiling_exceeded", detail)
    path = root.joinpath(*GUARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({**saved, "scopes": scopes, "last": check}, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)  # visible to every other run before the lock is released
    return check
