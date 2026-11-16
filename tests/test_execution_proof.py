"""Offline recovery evidence: every fault scenario on a synthetic plan, no broker connection."""

import sqlite3
from copy import deepcopy

import pytest

from capstone.execution.contracts import execution_contract
from capstone.execution.frozen import account_hash, identify_plan
from capstone.pipeline import execution_proof
from capstone.pipeline.execution_proof import SCENARIOS, fault_proof, run_scenario
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


def test_every_scenario_passes_and_reconciles_on_a_synthetic_plan(tmp_path):
    plan, account = fixture_plan()
    results = fault_proof(plan, account, tmp_path)
    assert tuple(results) == SCENARIOS and len(results) == 8
    assert all(r["passed"] for r in results.values())
    assert (tmp_path / "partial_restart" / "journal.sqlite").exists()


def test_restart_completes_a_partial_fill_but_cancel_and_reject_do_not_retry(tmp_path):
    plan, account = fixture_plan()
    results = fault_proof(plan, account, tmp_path)
    partial, cancel = results["partial_restart"], results["cancel"]
    assert partial["observed_partial_fill"] and partial["final_complete"]
    assert cancel["observed_partial_fill"] and not cancel["final_complete"]
    assert results["reject"]["submissions"] == 1 and not results["reject"]["final_complete"]
    assert results["timeout_after_accept"]["final_complete"]
    assert not results["timeout_before_accept"]["final_complete"]
    assert results["timeout_before_accept"]["submissions"] == 1


def test_duplicate_reports_are_journaled_once_and_disconnect_replays_the_order(tmp_path):
    plan, account = fixture_plan()
    results = fault_proof(plan, account, tmp_path)
    assert results["duplicate_ack"]["duplicate_reports"] > 0
    assert results["duplicate_ack"]["submissions"] == 1
    assert results["disconnect_restart"]["state_after_disconnect"] == "unknown"
    assert results["disconnect_restart"]["submissions"] == 1


def test_ambiguous_submission_resolves_only_through_reconcile_only(tmp_path):
    plan, account = fixture_plan()
    result = fault_proof(plan, account, tmp_path)["ambiguous_submission"]
    nothing = result["branches"]["broker_shows_nothing"]
    order = result["branches"]["broker_shows_order"]
    assert nothing["state_before_reconcile"] == "unknown" and nothing["new_plan_refused"]
    assert nothing["state_after_abandon"] == "unknown" and nothing["resolved_state"] == "cancelled"
    assert nothing["original_submissions"] == 1
    assert order["resolved_state"] == "filled" and order["new_plan_refused"]
    assert result["submissions"] == len(plan["orders"])


def test_a_later_order_is_partially_filled_after_a_single_share_order(tmp_path):
    plan, account = fixture_plan(shares=(1, 10))
    results = fault_proof(plan, account, tmp_path)
    assert results["partial_restart"]["observed_partial_fill"]
    assert results["cancel"]["observed_partial_fill"]


def test_a_plan_with_no_orders_or_no_splittable_order_is_refused(tmp_path):
    plan, account = fixture_plan()
    empty = identify_plan(deepcopy(plan) | {"orders": []})
    with pytest.raises(ValueError, match="nonempty synthetic plan"):
        fault_proof(empty, account, tmp_path / "a")
    single, account = fixture_plan(shares=(1,))
    with pytest.raises(ValueError, match="at least two shares"):
        fault_proof(single, account, tmp_path / "b")


def test_the_reconciliation_compares_journal_executions_to_the_brokers(tmp_path):
    plan, account = fixture_plan()
    result = run_scenario("partial_restart", plan, account, tmp_path)
    with sqlite3.connect(tmp_path / "journal.sqlite") as db:
        shares = sum(row[0] for row in db.execute("SELECT shares FROM executions"))
    assert shares == plan["orders"][0]["quantity"] == 10
    assert result["reconciliation"]["complete"] and result["reconciliation"]["commissions"] == 0


def test_a_dispatcher_that_retries_after_a_timeout_fails_the_proof(tmp_path, monkeypatch):
    """Mutation: a dispatcher that resubmits an uncertain order must not pass the proof."""
    plan, account = fixture_plan()
    real = execution_proof.execute_plan

    def retrying(plan_, broker, journal, **kwargs):
        result = real(plan_, broker, journal, **kwargs)
        if broker.fault == "timeout_before_accept":
            broker.submissions += 1
        return result

    monkeypatch.setattr(execution_proof, "execute_plan", retrying)
    with pytest.raises(AssertionError, match="retried"):
        run_scenario("timeout_before_accept", plan, account, tmp_path)
