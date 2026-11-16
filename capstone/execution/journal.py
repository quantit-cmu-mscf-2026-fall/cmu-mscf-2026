"""Durable intent/execution accounting. An unknown submission is never retried blindly.

The journal only talks to the `broker` object its caller hands it (`identity`, `preflight`,
`snapshot`, `events`, `submit`); it opens no connection itself, and a plan that is not
`transmittable` is refused unless `simulate=True` names the offline `SimulatedBroker`.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from capstone.execution.frozen import account_hash, verify_plan
from capstone.faults import maybe_fault
from capstone.locking import exclusive_file
from capstone.strategy.research import canonical


@contextmanager
def locked_journal(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file(path.with_suffix(".lock"), message="another dispatcher owns this journal"):
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.executescript("""
                    CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, document TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS instructions
                    (id TEXT PRIMARY KEY, plan TEXT NOT NULL,
                     payload TEXT NOT NULL, state TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS executions
                    (id TEXT PRIMARY KEY, instruction TEXT NOT NULL, shares INTEGER NOT NULL,
                     price REAL NOT NULL, commission REAL NOT NULL);
                    CREATE TABLE IF NOT EXISTS events
                    (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS identity (name TEXT PRIMARY KEY);
            """)
            yield db


def _events(db, events):
    for event in events:
        row = db.execute(
            "SELECT payload, state FROM instructions WHERE id=?", (event["instruction_id"],)
        ).fetchone()
        if row is None:
            continue
        order = json.loads(row[0])
        db.execute("INSERT INTO events(payload) VALUES (?)", (canonical(event),))
        if event["kind"] == "execution":
            quantity, price, commission = (
                event["shares"],
                event["price"],
                event.get("commission", 0),
            )
            if isinstance(quantity, bool) or int(quantity) != quantity or quantity == 0:
                raise ValueError("invalid execution quantity")
            if (
                not math.isfinite(price)
                or price <= 0
                or not math.isfinite(commission)
                or commission < 0
            ):
                raise ValueError("invalid execution price or commission")
            if quantity * order["signed_shares"] <= 0:
                raise ValueError("execution direction differs from instruction")
            old = db.execute(
                "SELECT instruction,shares,price,commission FROM executions WHERE id=?",
                (event["execution_id"],),
            ).fetchone()
            values = (event["instruction_id"], int(quantity), float(price), float(commission))
            if old is not None and old != values:
                raise ValueError("conflicting duplicate execution")
            db.execute(
                "INSERT OR IGNORE INTO executions VALUES (?,?,?,?,?)",
                (event["execution_id"], *values),
            )
            filled = db.execute(
                "SELECT COALESCE(SUM(shares),0) FROM executions WHERE instruction=?",
                (event["instruction_id"],),
            ).fetchone()[0]
            if abs(filled) > order["quantity"]:
                raise ValueError("broker overfilled an instruction")
            state = (
                "filled"
                if abs(filled) == order["quantity"]
                else row[1]
                if row[1] in {"cancelled", "rejected"}
                else "partial"
            )
            db.execute(
                "UPDATE instructions SET state=? WHERE id=?", (state, event["instruction_id"])
            )
        elif event["kind"] in {"rejected", "cancelled", "error"}:
            if row[1] != "filled":
                db.execute(
                    "UPDATE instructions SET state=? WHERE id=?",
                    (
                        "unknown" if event["kind"] == "error" else event["kind"],
                        event["instruction_id"],
                    ),
                )
        elif event["kind"] == "acknowledged" and row[1] in {"submitting", "unknown"}:
            db.execute(
                "UPDATE instructions SET state='acknowledged' WHERE id=?",
                (event["instruction_id"],),
            )
    db.commit()


def execute_plan(plan, broker, journal_path, *, simulate=False, reconcile_only=False):
    from capstone.execution.simulator import SimulatedBroker

    verify_plan(plan)
    if simulate and not isinstance(broker, SimulatedBroker):
        raise ValueError("simulation bypass applies only to the offline simulator")
    if not reconcile_only and (
        plan["planning_blockers"] or (not simulate and not plan["transmittable"])
    ):
        raise ValueError("order plan is blocked")
    with locked_journal(journal_path) as db:
        identities = [r[0] for r in db.execute("SELECT name FROM identity")]
        if identities and identities != [broker.identity]:
            raise ValueError("journal belongs to a different broker/profile")
        db.execute("INSERT OR IGNORE INTO identity VALUES (?)", (broker.identity,))
        existing = db.execute(
            "SELECT document FROM plans WHERE id=?", (plan["plan_id"],)
        ).fetchone()
        if existing and existing[0] != canonical(plan):
            raise ValueError("a plan identifier cannot refer to changed instructions")
        if existing is None:
            if reconcile_only:
                raise ValueError("there is no saved execution record to reconcile")
            broker.preflight(plan)
            unresolved = db.execute(
                "SELECT COUNT(*) FROM instructions WHERE state NOT IN "
                "('filled','cancelled','rejected')"
            ).fetchone()[0]
            if unresolved:
                raise ValueError("a prior plan remains unresolved; reconcile it before replanning")
            if account_hash(broker.snapshot()) != plan["account_hash"]:
                raise ValueError("account state changed; rebuild the plan")
            db.execute("INSERT INTO plans VALUES (?,?)", (plan["plan_id"], canonical(plan)))
            for order in plan["orders"]:
                db.execute(
                    "INSERT INTO instructions VALUES (?,?,?,'prepared')",
                    (order["instruction_id"], plan["plan_id"], canonical(order)),
                )
            db.commit()
        _events(db, broker.events())
        if reconcile_only:
            _not_sent(db, plan, broker)
        for order in [] if reconcile_only else plan["orders"]:
            # Reconcile before each new external side effect, including on restart.
            _events(db, broker.events())
            reconcile(db, plan, broker.snapshot())
            state = db.execute(
                "SELECT state FROM instructions WHERE id=?", (order["instruction_id"],)
            ).fetchone()[0]
            if state == "filled":
                continue
            if state != "prepared":
                break
            broker.preflight(plan)
            reconcile(db, plan, broker.snapshot())
            db.execute(
                "UPDATE instructions SET state='submitting' WHERE id=?", (order["instruction_id"],)
            )
            db.commit()  # Durable before the network call. A crash leaves an ambiguous intent.
            maybe_fault("journal:after_submitting")
            try:
                broker.submit(order)
            except (TimeoutError, ConnectionError):
                db.execute(
                    "UPDATE instructions SET state='unknown' WHERE id=?", (order["instruction_id"],)
                )
                db.commit()
                break
            _events(db, broker.events())
            state = db.execute(
                "SELECT state FROM instructions WHERE id=?", (order["instruction_id"],)
            ).fetchone()[0]
            if state != "filled":
                break
        _events(db, broker.events())
        terminal = db.execute(
            "SELECT COUNT(*) FROM instructions WHERE plan=? AND state IN ('cancelled','rejected')",
            (plan["plan_id"],),
        ).fetchone()[0]
        if terminal:
            _abandon(db, plan)
        result = reconcile(db, plan, broker.snapshot())
        result["states"] = dict(
            db.execute(
                "SELECT id,state FROM instructions WHERE plan=? ORDER BY id", (plan["plan_id"],)
            )
        )
        result["complete"] = all(s == "filled" for s in result["states"].values())
        result["plan_id"] = plan["plan_id"]
        return result


def _abandon(db, plan):
    rows = list(
        db.execute(
            "SELECT id FROM instructions WHERE plan=? AND state='prepared'", (plan["plan_id"],)
        )
    )
    for (identifier,) in rows:
        db.execute(
            "INSERT INTO events(payload) VALUES (?)",
            (
                canonical(
                    {
                        "instruction_id": identifier,
                        "kind": "not_submitted",
                        "reason": "plan_retired",
                    }
                ),
            ),
        )
        db.execute("UPDATE instructions SET state='cancelled' WHERE id=?", (identifier,))
    db.commit()
    return len(rows)


def _not_sent(db, plan, broker):
    """Reconcile-only: an uncertain intent the broker positively reports absent was never sent.

    Any broker report about it (even an error) keeps it unresolved, as do brokers without an
    authoritative `confirms_absent` (a real adapter must not claim it without qualified history).
    """
    absent = getattr(broker, "confirms_absent", None)
    if absent is None:
        return
    reported = {event["instruction_id"] for event in broker.events()}
    rows = list(
        db.execute(
            "SELECT id FROM instructions WHERE plan=? AND state IN ('submitting','unknown') "
            "AND id NOT IN (SELECT instruction FROM executions)",
            (plan["plan_id"],),
        )
    )
    for (identifier,) in rows:
        if identifier not in reported and absent(identifier):
            event = {
                "instruction_id": identifier,
                "kind": "not_submitted",
                "reason": "broker_reports_absent",
            }
            db.execute("INSERT INTO events(payload) VALUES (?)", (canonical(event),))
            db.execute("UPDATE instructions SET state='cancelled' WHERE id=?", (identifier,))
    db.commit()


def abandon_prepared(plan, journal_path):
    """Retire unsent intents under the dispatcher lock. Uncertain submissions stay unresolved."""
    verify_plan(plan)
    with locked_journal(journal_path) as db:
        saved = db.execute("SELECT document FROM plans WHERE id=?", (plan["plan_id"],)).fetchone()
        if saved is None or saved[0] != canonical(plan):
            raise ValueError("saved execution plan does not match")
        return _abandon(db, plan)


def reconcile(db, plan, account):
    known = dict(db.execute("SELECT id,state FROM instructions"))
    if any(
        identifier not in known
        or known[identifier] in {"prepared", "filled", "cancelled", "rejected"}
        for identifier in account.get("open_orders", [])
    ):
        raise ValueError("unexpected open orders require reconciliation")
    expected = {int(n): int(q) for n, q in plan["account_positions"].items()}
    cash = plan["account_cash"]
    fills = list(
        db.execute(
            "SELECT i.payload,e.shares,e.price,e.commission FROM executions e "
            "JOIN instructions i ON i.id=e.instruction WHERE i.plan=? ORDER BY e.id",
            (plan["plan_id"],),
        )
    )
    for payload, shares, price, commission in fills:
        name = json.loads(payload)["permno"]
        expected[name] = expected.get(name, 0) + shares
        cash -= shares * price + commission
    actual = {int(n): int(q) for n, q in account["positions"].items() if q}
    expected = {n: q for n, q in expected.items() if q}
    if actual != expected or not math.isclose(account["cash"], cash, abs_tol=0.01, rel_tol=0):
        raise ValueError("broker positions or cash do not reconcile; stop and investigate")
    return {"positions": expected, "cash": cash, "execution_count": len(fills), "reconciled": True}
