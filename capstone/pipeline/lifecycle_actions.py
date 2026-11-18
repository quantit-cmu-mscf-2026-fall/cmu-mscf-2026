"""Registry-driven retirement, suspension and promotion, only on explicit inputs.

Glue over `strategy.registry` (states and roles, hash-chained `deploy/lifecycle.jsonl`),
saved-portfolio membership departures and the shared planners; no broker, no clock
(`at` is the operator's timestamp). Every action re-verifies the lifecycle chain after it
writes. Nothing here runs by itself: each action is a function of an explicit `args`
object (registry root, strategy id, reason, reviewer, timestamp) that a person supplies.

- retire: `registry.set_state(retired)`, then a membership departure
  `{"strategy_id", "kind": "departure", "available_at": <date of at>,
  "reason": "registry_retired"}` in every registered saved portfolio whose pool names the
  strategy (its identity is unchanged: the definition does not hold membership), so the
  portfolio's next allocation decision records the capital transition to cash.
  A fixed-pool portfolio cannot record a departure and is refused before anything is
  written. Repeating a retire completes departures a crash left unwritten; otherwise a
  retired strategy is refused (retired is terminal).
- suspend / suspend with `resume`: `set_state(suspended | active)`; history is kept.
- promote: only on a ConfirmState that promotes (`registry.promote`), read from an
  incubation monitor store entry (`confirm_record`, or the store under `run`) or a
  `{"strategy_id", "confirm_state"}` record.

`close_states` gives a close-out plan the retired / suspended strategies of a bundle.

A monitor's `retire` (or `promote`) decision is only *recommended* (`recommendations`);
nothing here acts on it. The monitor sees attributed returns, not the reasons behind them
(a fill or data fault, a corporate event, a deliberate reallocation); retirement is
terminal and moves capital; and the registry changes only by a reviewed edit (rollback =
revert + re-tag). So a person runs the retire command and the change ships as a registry
commit.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd

from capstone.execution.bundle import load_bundle
from capstone.strategy.registry import (
    INCUBATION_ONLY,
    SAVED_PORTFOLIO_MODULE,
    SPEC_DIR,
    promote,
    read_lifecycle,
    read_registry,
    registry_spec,
    set_state,
)
from capstone.strategy.research import canonical, digest
from capstone.strategy.saved_portfolio import SavedProgramPortfolioSpec

DEPARTURE_REASON = "registry_retired"
ACTIONS = Path("actions") / "events.jsonl"
MONITOR = "monitor"
RECOMMEND = {"retire": "retirement recommended", "promote": "promotion eligible"}
# The CLI behind this command is a planned follow-up; until it lands, call the functions here.
COMMAND = "python -m capstone.pipeline.lifecycle"
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])")


class Refused(ValueError):
    """An operator action refused before (or instead of) any write."""


def _entry(root, strategy_id):
    # verifies the lifecycle chain; a crash-pending last event is the action's to complete
    entries = read_registry(root, resume=True)
    for entry in entries:
        if entry.id == strategy_id:
            return entries, entry
    raise Refused(f"strategy {strategy_id} is not registered in {root}")


def _date(at):
    """The date of an explicit ISO timestamp (`YYYY-MM-DD...`); "NaT", "now", "today" and
    other implicit or invalid values are refused."""
    if not (isinstance(at, str) and ISO_DATE.match(at)):
        raise Refused(f"--at {at!r} is not an explicit ISO date or timestamp")
    try:
        stamp = pd.Timestamp(at)
    except (TypeError, ValueError) as error:
        raise Refused(f"--at {at!r} is not a timestamp") from error
    if pd.isna(stamp):
        raise Refused(f"--at {at!r} is not a timestamp")
    return str(stamp.date())


def _verified(root, args, command, result):
    """Re-verify the chain; record the action under `--run` (status lists it)."""
    events = read_lifecycle(root)
    result = {"command": command, "strategy_id": args.id, **result}
    result |= {"reason": args.reason, "reviewer": args.reviewer, "at": args.at}
    result["lifecycle_head"] = digest(events[-1]) if events else ""
    if args.run is not None:
        path = Path(args.run) / ACTIONS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(result) + "\n")
    return result


def holding_portfolios(root, entries, strategy_id):
    """Registered, unretired saved portfolios whose candidate pool names `strategy_id`."""
    found = []
    for entry in entries:
        if entry.module != SAVED_PORTFOLIO_MODULE or entry.state == "retired":
            continue
        spec = registry_spec(root, entry.id, entry.module)
        if strategy_id not in {member.id for member in spec.specs}:
            continue
        if not spec.chronological:
            raise Refused(
                f"saved portfolio {entry.id} has a retrospective fixed pool (v4) and cannot "
                f"record the departure of {strategy_id}; retire that portfolio first or "
                "re-register it with chronological membership (v5)"
            )
        found.append(spec)
    return found


def retire(args):
    root, sid = Path(args.registry), args.id
    available_at = _date(args.at)
    entries, entry = _entry(root, sid)
    portfolios = holding_portfolios(root, entries, sid)
    departure = {
        "strategy_id": sid,
        "kind": "departure",
        "available_at": available_at,
        "reason": DEPARTURE_REASON,
    }

    def departed(spec):
        return any(e["kind"] == "departure" and e["strategy_id"] == sid for e in spec.membership)

    pending = [spec for spec in portfolios if not departed(spec)]
    for spec in pending:
        last = spec.schedule[-1]["decision_session"] if spec.schedule else ""
        if departure["available_at"] < last:
            raise Refused(
                f"--at {args.at} precedes saved portfolio {spec.id}'s last decision {last}; "
                "a departure cannot rewrite a recorded decision"
            )
    if entry.state == "retired" and not pending:
        raise Refused(f"strategy {sid} is already retired; retired is terminal")
    if entry.state != "retired":
        set_state(root, sid, "retired", reason=args.reason, reviewer=args.reviewer, at=args.at)
    for spec in pending:
        updated = replace(spec, membership=(*spec.membership, departure))
        (root / SPEC_DIR / f"{spec.id}.toml").write_bytes(updated.to_toml().encode("utf-8"))
    result = {"state": "retired", "departures": [spec.id for spec in pending]}
    return _verified(root, args, "retire", result)


def suspend(args):
    root = Path(args.registry)
    state = "active" if args.resume else "suspended"
    _date(args.at)
    _entry(root, args.id)
    set_state(root, args.id, state, reason=args.reason, reviewer=args.reviewer, at=args.at)
    return _verified(root, args, "resume" if args.resume else "suspend", {"state": state})


def confirm_record(path, strategy_id):
    """The `{"strategy_id", "confirm_state"}` record in `path`: an incubation monitor store entry
    (`monitor/<id>.json`, incubation-only) or such a record itself."""
    if not path.is_file():
        raise Refused(
            f"missing confirm record {path}; produce it with: {COMMAND} monitor --run <root> "
            "--role incubation-only"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(record, dict) and {"strategy_id", "role", "state"} <= set(record):
        if record["role"] != INCUBATION_ONLY:
            raise Refused(
                f"{path} is a {record['role']} monitor of {record['strategy_id']}; only an "
                "incubation-only ConfirmState promotes"
            )
        record = {"strategy_id": record["strategy_id"], "confirm_state": record["state"]}
    return record


def promote_entry(args):
    root = Path(args.registry)
    path = args.confirm_record
    if path is None and args.run is not None:
        path = Path(args.run) / MONITOR / f"{args.id}.json"
    if path is None:
        raise Refused(
            "promote needs a confirm record: --confirm-record <store>/<id>.json or --run ROOT "
            "holding the incubation monitor store"
        )
    _date(args.at)
    _entry(root, args.id)
    record = confirm_record(Path(path), args.id)
    entry = promote(root, args.id, confirm_record=record, reviewer=args.reviewer, at=args.at)
    return _verified(root, args, "promote", {"role": entry.role, "confirm_record": str(path)})


def close_states(registry, bundle):
    """{id: state} of the bundle's retired or suspended strategies (and members)."""
    _, _, specs, _ = load_bundle(bundle)
    ids = set()
    for spec in specs:
        ids.add(spec.id)
        if isinstance(spec, SavedProgramPortfolioSpec):
            ids |= {member.id for member in spec.specs}
    states = {e.id: e.state for e in read_registry(registry) if e.state != "active"}
    states = {sid: state for sid, state in states.items() if sid in ids}
    if not states:
        raise Refused(
            f"no retired or suspended strategy of this bundle in the registry at {registry}; "
            "plan without --close"
        )
    return states


def recommendations(run):
    """Monitor decisions in `<run>/monitor` that a person should act on; never acted on."""
    store = Path(run) / MONITOR
    rows = []
    for path in sorted(store.glob("*.json")) if store.is_dir() else ():
        document = json.loads(path.read_text(encoding="utf-8"))
        decision = document.get("state", {}).get("decision")
        if decision not in RECOMMEND:
            continue
        sid = document["strategy_id"]
        command = "retire" if decision == "retire" else "promote"
        action = f"{COMMAND} {command} --registry <repo root> --id {sid} --reason <text> "
        action += f"--reviewer <name> --at <timestamp> --run {run}"
        rows.append(
            {
                "strategy_id": sid,
                "role": document["role"],
                "decision": decision,
                "recommendation": RECOMMEND[decision],
                "action": action,
            }
        )
    return rows
