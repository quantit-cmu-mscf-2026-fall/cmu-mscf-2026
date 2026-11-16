"""Offline recovery evidence using the same frozen plan and dispatcher as normal execution."""

import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path

from capstone.execution.frozen import identify_plan
from capstone.execution.journal import abandon_prepared, execute_plan
from capstone.execution.simulator import SimulatedBroker

SCENARIOS = (
    "partial_restart",
    "reject",
    "timeout_before_accept",
    "timeout_after_accept",
    "cancel",
    "duplicate_ack",
    "disconnect_restart",
    "ambiguous_submission",
)


def fault_proof(plan, account, out, *, market=None):
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    if not plan["orders"]:
        raise ValueError("recovery proof requires a nonempty synthetic plan")
    if not any(o["quantity"] >= 2 for o in plan["orders"]):
        raise ValueError("partial-fill proof needs an instruction of at least two shares")
    results = {}
    for fault in SCENARIOS:
        directory = root / fault
        directory.mkdir()
        results[fault] = run_scenario(fault, plan, account, directory, market=market)
    return results


def run_scenario(fault, plan, account, directory, *, market=None):
    directory = Path(directory)
    if fault == "duplicate_ack":
        return _duplicate_ack(plan, account, directory, market)
    if fault == "disconnect_restart":
        return _disconnect_restart(plan, account, directory, market)
    if fault == "ambiguous_submission":
        return _ambiguous_submission(plan, account, directory, market)
    partial = next(o for o in plan["orders"] if o["quantity"] >= 2)
    broker = _broker(
        plan,
        account,
        market,
        fill_fraction={partial["instruction_id"]: 0.5}
        if fault in {"partial_restart", "cancel"}
        else 1.0,
        fault=fault
        if fault in {"reject", "timeout_before_accept", "timeout_after_accept"}
        else None,
    )
    journal = directory / "journal.sqlite"
    first = execute_plan(plan, broker, journal, simulate=True)
    first_count = broker.submissions
    observed_partial = any(0 < r["filled"] < r["order"]["quantity"] for r in broker.orders.values())
    if fault in {"partial_restart", "cancel"} and not observed_partial:
        raise AssertionError("recovery proof did not produce an actual partial fill")
    broker = _restart(broker, directory)
    if fault == "partial_restart":
        broker.fill(partial["instruction_id"])
        broker.fill_fraction = 1.0
    elif fault == "cancel":
        broker.cancel(partial["instruction_id"])
    second = _settled(plan, broker, journal)
    expected_complete = fault in {"partial_restart", "timeout_after_accept"}
    if second["complete"] != expected_complete:
        raise AssertionError(f"unexpected recovery outcome: {fault}")
    if not expected_complete and broker.submissions != first_count:
        raise AssertionError("uncertain, cancelled or rejected submission was retried")
    if expected_complete and broker.submissions != len(plan["orders"]):
        raise AssertionError("recovery did not submit each instruction exactly once")
    broker.save(directory / "broker.json")
    return {
        "passed": True,
        "initial_complete": first["complete"],
        "observed_partial_fill": observed_partial,
        "final_complete": second["complete"],
        "submissions": broker.submissions,
        "reconciliation": _ledger(journal, broker, second),
    }


def _broker(plan, account, market, **faults):
    return SimulatedBroker(
        account,
        profile=plan["profile"],
        session=plan["execution_session"],
        execution_prices=(market or {}).get("prices"),
        commission_bps=(market or {}).get("commission_bps", 0),
        execution_adjustment_bps=(market or {}).get("execution_adjustment_bps", 0),
        **faults,
    )


def _restart(broker, directory):
    broker.save(directory / "broker.json")
    return SimulatedBroker.load(directory / "broker.json")


def _settled(plan, broker, journal, **kwargs):
    """Run twice: repeated callbacks must not change the reconciled state."""
    once = execute_plan(plan, broker, journal, simulate=True, **kwargs)
    if execute_plan(plan, broker, journal, simulate=True, **kwargs) != once:
        raise AssertionError("duplicate callbacks changed reconciled state")
    return once


def _crash(plan, broker, journal):
    """The dispatcher dies on the dropped link with the first intent left `unknown`."""
    try:
        execute_plan(plan, broker, journal, simulate=True)
    except ConnectionError:
        return _state(journal, plan["orders"][0]["instruction_id"])
    raise AssertionError("simulated connection loss did not reach the dispatcher")


def _state(journal, identifier):
    with closing(sqlite3.connect(journal)) as db:
        return db.execute("SELECT state FROM instructions WHERE id=?", (identifier,)).fetchone()[0]


def _ledger(journal, broker, result):
    """Journal executions equal the broker's distinct ones: no duplicate or fictitious fill."""
    with closing(sqlite3.connect(journal)) as db:
        rows = sorted(db.execute("SELECT id,shares,price,commission FROM executions"))
    reported = {
        m["execution_id"]: (m["execution_id"], m["shares"], m["price"], m["commission"])
        for m in broker.messages
        if m["kind"] == "execution"
    }
    if rows != sorted(reported.values()):
        raise AssertionError("journal executions differ from broker executions")
    if not result["reconciled"]:
        raise AssertionError("journal does not reconcile to the broker")
    return result | {"commissions": sum(row[3] for row in rows)}


def _duplicate_ack(plan, account, directory, market):
    partial = next(o for o in plan["orders"] if o["quantity"] >= 2)["instruction_id"]
    broker = _broker(plan, account, market, fault="duplicate_ack", fill_fraction={partial: 0.5})
    journal = directory / "journal.sqlite"
    execute_plan(plan, broker, journal, simulate=True)
    broker = _restart(broker, directory)
    broker.fill(partial)
    broker.fill_fraction = 1.0
    result = _settled(plan, broker, journal)
    duplicates = len(broker.messages) - len({str(m) for m in broker.messages})
    if not result["complete"] or broker.submissions != len(plan["orders"]) or not duplicates:
        raise AssertionError("duplicate reports changed submissions or the outcome")
    broker.save(directory / "broker.json")
    return {
        "passed": True,
        "duplicate_reports": duplicates,
        "submissions": broker.submissions,
        "reconciliation": _ledger(journal, broker, result),
    }


def _disconnect_restart(plan, account, directory, market):
    first = plan["orders"][0]["instruction_id"]
    broker = _broker(plan, account, market, fault="disconnect_restart", fill_fraction={first: 0.5})
    journal = directory / "journal.sqlite"
    state = _crash(plan, broker, journal)
    record = broker.orders[first]
    missed = record["order"]["quantity"] - record["filled"]
    if missed:
        broker.fill(first, missed)  # Executes while no dispatcher is connected.
    broker = _restart(broker, directory)
    result = _settled(plan, broker, journal)
    if state != "unknown" or not result["complete"]:
        raise AssertionError("restart did not replay the disconnected submission")
    if broker.submissions != len(plan["orders"]):
        raise AssertionError("restart resubmitted an instruction")
    broker.save(directory / "broker.json")
    return {
        "passed": True,
        "state_after_disconnect": state,
        "filled_while_disconnected": missed,
        "submissions": broker.submissions,
        "reconciliation": _ledger(journal, broker, result),
    }


def _refused(plan, broker, journal):
    """A fresh process may not start any other plan while an intent is unresolved."""
    successor = identify_plan(deepcopy(plan) | {"supersedes": plan["plan_id"]})
    try:
        execute_plan(successor, broker, journal, simulate=True)
    except ValueError as error:
        if "prior plan remains unresolved" in str(error):
            return successor
        raise
    raise AssertionError("a new plan was accepted while a submission was unresolved")


def _ambiguous_submission(plan, account, directory, market):
    first = plan["orders"][0]["instruction_id"]
    branches = {}
    # The broker never received the order: reconcile-only marks it not sent.
    folder = directory / "broker_shows_nothing"
    folder.mkdir()
    broker = _broker(plan, account, market, fault="ambiguous_submission")
    journal = folder / "journal.sqlite"
    unknown = _crash(plan, broker, journal)
    broker = _restart(broker, folder)
    successor = _refused(plan, broker, journal)
    if execute_plan(plan, broker, journal, simulate=True)["states"][first] != "unknown":
        raise AssertionError("a restart auto-resolved an unknown submission")
    abandoned = abandon_prepared(plan, journal)
    after_abandon = _state(journal, first)
    if after_abandon != "unknown":
        raise AssertionError("abandon_prepared retired an unknown submission")
    _refused(plan, broker, journal)
    resolved = _settled(plan, broker, journal, reconcile_only=True)
    original = broker.submissions
    outcome = _settled(successor, broker, journal)
    if resolved["states"][first] != "cancelled" or original != 1 or not outcome["complete"]:
        raise AssertionError("an absent submission did not resolve to not-sent")
    branches["broker_shows_nothing"] = {
        "state_before_reconcile": unknown,
        "new_plan_refused": True,
        "abandoned_prepared": abandoned,
        "state_after_abandon": after_abandon,
        "resolved_state": resolved["states"][first],
        "original_submissions": original,
        "successor": _ledger(journal, broker, outcome),
    }
    broker.save(folder / "broker.json")
    # The broker received and filled the order: reconcile-only adopts it.
    folder = directory / "broker_shows_order"
    folder.mkdir()
    broker = _broker(plan, account, market, fault="disconnect_restart")
    journal = folder / "journal.sqlite"
    unknown = _crash(plan, broker, journal)
    broker = _restart(broker, folder)
    _refused(plan, broker, journal)
    resolved = _settled(plan, broker, journal, reconcile_only=True)
    if resolved["states"][first] != "filled" or broker.submissions != 1:
        raise AssertionError("reconciliation did not adopt the broker's order")
    result = _settled(plan, broker, journal)
    if not result["complete"] or broker.submissions != len(plan["orders"]):
        raise AssertionError("adopted submission was retried or the plan stalled")
    branches["broker_shows_order"] = {
        "state_before_reconcile": unknown,
        "new_plan_refused": True,
        "resolved_state": resolved["states"][first],
        "submissions": broker.submissions,
        "reconciliation": _ledger(journal, broker, result),
    }
    broker.save(folder / "broker.json")
    return {
        "passed": True,
        "branches": branches,
        "submissions": broker.submissions,
        "reconciliation": branches["broker_shows_order"]["reconciliation"],
    }
