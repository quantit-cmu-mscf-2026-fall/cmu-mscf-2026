"""Reconcile an offline `simulate` killed before it saved its broker (simulated
broker only). Split from `offline.py`, which stays small."""

from __future__ import annotations

import json
from pathlib import Path

from capstone.execution.frozen import read_plan
from capstone.execution.journal import execute_plan, locked_journal
from capstone.pipeline import offline

SIMULATION_RESOLVED = ("filled", "cancelled", "rejected")


def interrupted_simulation(out, plan_id):
    """{instruction id: state} of a `simulate` killed before it saved its broker, or None.

    The offline broker lives in memory until `simulate` returns, so after a kill the
    journal (`execution.sqlite`) is the only record of what the simulator did."""
    root = Path(out)
    journal = root / "execution.sqlite"
    if (root / "simulated-broker.json").exists() or not journal.is_file():
        return None
    with locked_journal(journal) as db:
        if db.execute("SELECT 1 FROM plans WHERE id=?", (plan_id,)).fetchone() is None:
            return None  # killed before the plan was journaled: nothing happened yet
        return dict(
            db.execute("SELECT id, state FROM instructions WHERE plan=? ORDER BY id", (plan_id,))
        )


def recover_simulation(bundle, plan_path, out, *, market):
    """Reconcile a `simulate` killed before it saved its broker (simulated broker only).

    The broker is rebuilt from the bundle's account and what the journal recorded: every
    execution the simulator reported is adopted (positions, cash, the order as filled).
    An instruction the journal left `submitting` / `unknown` with no execution never
    reached the rebuilt broker, so the journal's reconcile-only path
    (`execute_plan(reconcile_only=True)`, the path `instructions --reconcile-only` runs)
    marks it not sent; no fill is ever invented and nothing is resubmitted. The plan's
    unsent `prepared` intents are then retired by the journal itself. The rebuilt broker
    is saved, so the next `simulate` continues from it. Returns None when there is nothing
    to recover, else `{"plan_id", "adopted", "before", "after"}` (instruction states)."""
    root = Path(out)
    plan = read_plan(plan_path)
    before = interrupted_simulation(root, plan["plan_id"])
    if before is None:
        return None
    if market is None or market.get("session") != plan["execution_session"]:
        raise ValueError("recovering a simulation needs the session's synthetic market")
    # The market goes to the bundle check too, so a recovered live simulation is
    # held to the exported market (`live_market_mismatch`) as a fresh `simulate` is.
    broker = offline._fresh_broker(offline._synthetic(bundle, plan, market), plan, market)
    journal = root / "execution.sqlite"
    with locked_journal(journal) as db:
        payloads = dict(
            db.execute("SELECT id, payload FROM instructions WHERE plan=?", (plan["plan_id"],))
        )
        fills = list(
            db.execute(
                "SELECT e.instruction, e.shares, e.price, e.commission FROM executions e "
                "JOIN instructions i ON i.id=e.instruction WHERE i.plan=? ORDER BY e.id",
                (plan["plan_id"],),
            )
        )
    for identifier, shares, price, commission in fills:  # what the simulator reported
        order = json.loads(payloads[identifier])
        name = int(order["permno"])
        broker.account["positions"][name] = broker.account["positions"].get(name, 0) + shares
        broker.account["cash"] -= shares * price + commission
        record = broker.orders.setdefault(
            identifier, {"order": order, "filled": 0, "terminal": False}
        )
        record["filled"] += abs(shares)
    for record in broker.orders.values():
        record["terminal"] = record["filled"] >= record["order"]["quantity"]
    broker.submissions = len(broker.orders)
    if any(state in ("submitting", "unknown") for state in before.values()):
        execute_plan(plan, broker, journal, simulate=True, reconcile_only=True)
    broker.save(root / "simulated-broker.json")
    with locked_journal(journal) as db:
        after = dict(
            db.execute(
                "SELECT id, state FROM instructions WHERE plan=? ORDER BY id", (plan["plan_id"],)
            )
        )
    adopted = sorted(broker.orders)
    return {"plan_id": plan["plan_id"], "adopted": adopted, "before": before, "after": after}
