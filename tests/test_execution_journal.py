"""The execution journal against the offline simulator: synthetic plans, no broker."""

import sqlite3
from copy import deepcopy

import pytest

from capstone.execution.contracts import execution_contract
from capstone.execution.frozen import account_hash, identify_plan
from capstone.execution.journal import abandon_prepared, execute_plan, locked_journal
from capstone.execution.simulator import SimulatedBroker
from capstone.strategy.reference import ReferenceStrategySpec

SESSION = "2000-01-04"


def fixture_plan(shares=(10,)):
    spec = ReferenceStrategySpec(
        "716bdaf7915c65be", "synthetic-recovery-fixture", "2000-01-03", side="long_only"
    )
    account = {"positions": {}, "cash": 10000.0, "capital": 10000.0, "open_orders": []}
    orders = [
        {
            "permno": n,
            "instrument": {"symbol": f"S{n}"},
            "signed_shares": quantity,
            "quantity": quantity,
            "action": "BUY",
            "reference_price": 100.0,
            "order_type": "MKT",
            "tif": "DAY",
            "outside_rth": False,
        }
        for n, quantity in enumerate(shares, 1)
    ]
    plan = {
        "schema_version": 3,
        "execution_contract": execution_contract(spec),
        "strategy_ids": [spec.id],
        "specifications": {spec.id: spec.to_toml()},
        "account_hash": account_hash(account),
        "account_cash": account["cash"],
        "account_positions": {},
        "profile": "long_only",
        "execution_session": SESSION,
        "planning_blockers": [],
        "validation_blockers": ["synthetic_unadmitted"],
        "broker_blockers": ["simulation_only"],
        "transmittable": False,
        "orders": orders,
    }
    return identify_plan(plan), account


def simulator(account, **changes):
    return SimulatedBroker(account, profile="long_only", session=SESSION, **changes)


def states(path):
    with sqlite3.connect(path) as db:
        return dict(db.execute("SELECT id, state FROM instructions ORDER BY id"))


def test_restart_deduplicates_partial_and_repeated_execution_events(tmp_path):
    plan, account = fixture_plan()
    broker = simulator(account, fill_fraction=0.5)
    path = tmp_path / "journal.db"
    first = execute_plan(plan, broker, path, simulate=True)
    assert not first["complete"] and first["positions"] == {1: 5}
    broker.fill(plan["orders"][0]["instruction_id"])
    second = execute_plan(plan, broker, path, simulate=True)
    third = execute_plan(plan, broker, path, simulate=True)
    assert second == third and second["complete"]
    assert broker.submissions == 1 and second["execution_count"] == 2


def test_duplicate_acknowledgements_are_counted_once(tmp_path):
    plan, account = fixture_plan()
    broker = simulator(account, fault="duplicate_ack")
    result = execute_plan(plan, broker, tmp_path / "journal.db", simulate=True)
    assert result["complete"] and result["execution_count"] == 1 and broker.submissions == 1


@pytest.mark.parametrize("fault", ["timeout_before_accept", "timeout_after_accept", "reject"])
def test_ambiguous_or_rejected_submission_is_not_blindly_retried(tmp_path, fault):
    plan, account = fixture_plan()
    broker = simulator(account, fault=fault)
    path = tmp_path / "journal.db"
    execute_plan(plan, broker, path, simulate=True)
    result = execute_plan(plan, broker, path, simulate=True)
    assert broker.submissions == 1
    assert result["complete"] == (fault == "timeout_after_accept")


def test_a_rejection_retires_the_unsent_orders_of_the_plan(tmp_path):
    plan, account = fixture_plan((10, 5))
    result = execute_plan(
        plan, simulator(account, fault="reject"), tmp_path / "j.db", simulate=True
    )
    assert not result["complete"]
    assert sorted(result["states"].values()) == ["cancelled", "rejected"]


def test_unsent_intents_are_abandoned_under_the_lock_and_uncertain_ones_are_not(tmp_path):
    plan, account = fixture_plan()
    path = tmp_path / "journal.db"
    broker = simulator(account, fault="ambiguous_submission")
    with pytest.raises(ConnectionError):  # the link is down; the intent is already durable
        execute_plan(plan, broker, path, simulate=True)
    assert set(states(path).values()) == {"unknown"}
    assert abandon_prepared(plan, path) == 0  # an uncertain submission stays unresolved
    changed = deepcopy(plan)
    changed["orders"][0]["quantity"] = 11
    with pytest.raises(ValueError, match="does not match"):
        abandon_prepared(identify_plan(changed), path)


def test_reconcile_only_clears_an_intent_the_broker_reports_absent(tmp_path):
    plan, account = fixture_plan()
    path = tmp_path / "journal.db"
    broker = simulator(account, fault="ambiguous_submission")
    with pytest.raises(ConnectionError):  # the link is down; the intent is already durable
        execute_plan(plan, broker, path, simulate=True)
    broker.connected = True  # the link is back; the order never arrived
    result = execute_plan(plan, broker, path, simulate=True, reconcile_only=True)
    assert set(states(path).values()) == {"cancelled"} and result["reconciled"]
    with pytest.raises(ValueError, match="no saved execution record"):
        execute_plan(plan, broker, tmp_path / "empty.db", simulate=True, reconcile_only=True)


def test_manual_account_change_and_untransmittable_plans_are_refused(tmp_path):
    plan, account = fixture_plan()
    broker = simulator(account)
    broker.account["positions"][1] = 1
    with pytest.raises(ValueError, match="state changed"):
        execute_plan(plan, broker, tmp_path / "j.db", simulate=True)
    with pytest.raises(ValueError, match="blocked"):
        execute_plan(plan, simulator(account), tmp_path / "real.db")  # not transmittable
    assert broker.submissions == 0
    changed = deepcopy(plan)
    changed["orders"][0]["quantity"] = 11
    with pytest.raises(ValueError, match="changed"):
        execute_plan(changed, simulator(account), tmp_path / "changed.db", simulate=True)


def test_the_simulation_bypass_applies_only_to_the_offline_simulator(tmp_path):
    plan, account = fixture_plan()

    class Other:
        identity = "other/long_only"

    with pytest.raises(ValueError, match="offline simulator"):
        execute_plan(plan, Other(), tmp_path / "j.db", simulate=True)
    assert not (tmp_path / "j.db").exists()


def test_a_journal_belongs_to_one_broker_profile(tmp_path):
    plan, account = fixture_plan()
    path = tmp_path / "journal.db"
    execute_plan(plan, simulator(account), path, simulate=True)
    other = simulator(account)
    other.identity = "offline-simulator/other"
    with pytest.raises(ValueError, match="different broker"):
        execute_plan(plan, other, path, simulate=True)


def test_an_overfill_is_refused(tmp_path):
    plan, account = fixture_plan()
    broker = simulator(account, fill_fraction=0.5)
    path = tmp_path / "journal.db"
    execute_plan(plan, broker, path, simulate=True)
    key = plan["orders"][0]["instruction_id"]
    broker.messages.append(
        {
            "instruction_id": key,
            "kind": "execution",
            "execution_id": f"{key}/extra",
            "shares": 6,
            "price": 100.0,
        }
    )
    with pytest.raises(ValueError, match="overfilled"):
        execute_plan(plan, broker, path, simulate=True)


def test_exclusive_dispatcher_lock_is_released(tmp_path):
    with locked_journal(tmp_path / "j.db"), pytest.raises(RuntimeError, match="dispatcher"):
        with locked_journal(tmp_path / "j.db"):
            pass
    with locked_journal(tmp_path / "j.db"):
        pass
