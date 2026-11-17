"""Saved allocation routes through the one share planner and replay engine."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_saved_portfolio import case as _case
from test_saved_portfolio import chronological, departure, portfolio, receipt

from capstone.execution.bundle import freeze_bundle
from capstone.execution.contracts import ExecutionContractBlocked, execution_contract
from capstone.execution.frozen import build_plan, verify_plan, write_plan
from capstone.execution.replay_state import checked_checkpoint, retain_intent_evidence
from capstone.pipeline.replay import ReplayCosts, replay
from capstone.strategy.saved_portfolio import FIXED_POOL_EXECUTION_VERSION as EXECUTION_VERSION
from capstone.strategy.saved_portfolio import MODE


@pytest.fixture
def case():
    return _case.__wrapped__()


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


def run(
    saved, panel, *, start="2000-02-01", end="2000-04-03", prices=None, opens=None, checkpoint=None
):
    prices = panel.fields["close"] * 0 + 10 if prices is None else prices
    return replay(
        saved,
        panel,
        prices if opens is None else opens,
        prices,
        start=start,
        end=end,
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        execution_version=saved.execution_version,
        events=pd.DataFrame(columns=["event_id", "event_date", "permno", "kind"]),
        checkpoint=checkpoint,
    )


def test_v4_is_explicit_and_empty_cash_account_is_a_supported_new_formation(case):
    empty = portfolio(case, members=(), returns=case[2].iloc[:, :0])
    for old in ("fixed-share-next-open-v2", "program-events-v3"):
        with pytest.raises(ExecutionContractBlocked, match="explicit_execution_v4"):
            execution_contract(empty, version=old)
    actual = run(empty, case[0])
    assert actual["initialization_qualifying"] and actual["positions"] == {}
    assert actual["cash"] == 1000 and actual["orders"] == []
    assert actual["checkpoint"]["protocol"] == EXECUTION_VERSION


def test_daily_price_drift_never_refits_or_resizes_surviving_fixed_shares(case, monkeypatch):
    saved = portfolio(case, members=case[1][:1], returns=case[2].iloc[:, :1])
    prices = case[0].fields["close"] * 0 + 10
    prices.loc["2000-02-02":, 103] = 20
    actual = run(saved, case[0], end="2000-02-10", prices=prices)
    assert actual["positions"] == {101: -25, 102: -25, 103: 25, 104: 25}
    assert len(actual["orders"]) == 4
    assert all(row["date"] == "2000-02-01" for row in actual["orders"])


@pytest.mark.parametrize("profile", ["long_only", "long_short"])
def test_removing_the_last_positive_sleeve_closes_book_and_retries_cash_exit(case, profile):
    panel, specs, history = case
    member = replace(specs[0], side=profile)
    history = pd.DataFrame({member.id: history.iloc[:, 0]})
    history.loc["2000-02-01":] = -0.003
    saved = portfolio((panel, (member,), history), mode="median_window_sharpe", side=profile)
    assert saved.members("2000-01-31") and saved.members("2000-02-29") == ()
    prices = panel.fields["close"] * 0 + 10
    opens = prices.copy()
    opens.loc["2000-03-01", 103] = np.nan
    actual = run(saved, panel, end="2000-03-03", prices=prices, opens=opens)
    assert actual["positions"] == {} and actual["cash"] == 1000
    assert any(row.get("status") == "blocked_missing_execution_price" for row in actual["orders"])
    closing = [
        row for row in actual["orders"] if row.get("date") == "2000-03-02" and "shares" in row
    ]
    assert closing and all(row["intent_id"] for row in closing)


def test_appending_future_saved_allocations_preserves_checkpoint_restart(case):
    full = portfolio(case, members=case[1][:1], returns=case[2].iloc[:, :1])
    prefix = portfolio(case, members=case[1][:1], returns=case[2].iloc[:, :1], through="2000-01-31")
    expected = run(full, case[0])
    first = run(prefix, case[0], end="2000-02-10")
    resumed = run(full, case[0], start="2000-02-11", checkpoint=first["checkpoint"])
    assert_frame_equal(pd.concat([first["history"], resumed["history"]]), expected["history"])
    assert first["orders"] + resumed["orders"] == expected["orders"]
    assert resumed["checkpoint"] == expected["checkpoint"]


def freeze(path, saved, panel, monkeypatch, *, as_of="2000-01-31", account=None):
    monkeypatch.setattr(
        "capstone.execution.bundle.release_identity",
        lambda tag: {"commit": "fixture", "dirty": True, "deployment_tag": None},
    )
    return freeze_bundle(
        path,
        panel,
        [saved],
        as_of,
        prices={n: 10 for n in panel.universe},
        instruments={
            n: {"symbol": str(n), "currency": "USD", "sec_type": "STK"} for n in panel.universe
        },
        account=account or {"positions": {}, "cash": 1000, "capital": 1000, "open_orders": []},
        mode=MODE,
        execution_version=saved.execution_version,
        execution_cost_bps=0,
        limits={"max_gross": 1, "max_weight": 1, "max_order_notional": 100000},
    )


def test_frozen_plans_consume_saved_coefficients_without_optimizer_or_future_records(
    case, tmp_path, monkeypatch
):
    complete = portfolio(case, members=case[1][:1], returns=case[2].iloc[:, :1])
    prefix = portfolio(case, members=case[1][:1], returns=case[2].iloc[:, :1], through="2000-01-31")
    first = freeze(tmp_path / "full", complete, case[0], monkeypatch)
    second = freeze(tmp_path / "prefix", prefix, case[0], monkeypatch)
    monkeypatch.setattr(
        "capstone.strategy.robust_allocation.optimize_allocation",
        lambda *a, **k: pytest.fail("planner refit"),
    )
    plan, prior = build_plan(first), build_plan(second)
    assert plan == prior
    assert not plan["formal_admission"] and not plan["transmittable"]
    assert "saved_portfolio_financial_admission_pending" in plan["validation_blockers"]
    assert verify_plan(plan) == plan
    write_plan(plan, tmp_path / "plan.json")
    assert plan["target_shares"] == {"101": -25, "102": -25, "103": 25, "104": 25}
    text = next((tmp_path / "full").glob("strategy-*.toml")).read_text(encoding="utf-8")
    assert "2000-02-29" not in text and "2000-03-31" not in text


def test_rehashed_manual_coefficients_block_before_replay_registration(case, monkeypatch):
    saved = portfolio(case)
    rows = deepcopy(saved.schedule)
    rows[0]["weights"] = dict.fromkeys(rows[0]["pool_ids"], 0.0)
    rows[0]["cash_weight"] = 1.0
    from capstone.strategy.research import digest

    for index, row in enumerate(rows):
        row["previous_state_id"] = None if not index else rows[index - 1]["id"]
        row["id"] = digest({k: v for k, v in row.items() if k != "id"})
    forged = replace(saved, schedule=rows)
    recorded = []
    monkeypatch.setattr(
        "capstone.execution.replay_identity.runlog.log_run", lambda *a, **k: recorded.append(k)
    )
    with pytest.raises(ValueError, match="actual prior funded"):
        run(forged, case[0])
    assert recorded == []


@pytest.mark.parametrize("requested", ["program-events-v3", EXECUTION_VERSION])
def test_checkpoint_protocol_cannot_be_transplanted_across_execution_versions(case, requested):
    from capstone.strategy.research import digest

    saved = portfolio(case, members=case[1][:1], returns=case[2].iloc[:, :1])
    seed = run(saved, case[0], end="2000-02-10")["checkpoint"]
    seed["protocol"] = "program-events-v3" if requested == EXECUTION_VERSION else EXECUTION_VERSION
    seed["checkpoint_id"] = digest({k: v for k, v in seed.items() if k != "checkpoint_id"})
    with pytest.raises(ExecutionContractBlocked, match="checkpoint_protocol_incompatible"):
        checked_checkpoint(requested, seed, pd.DataFrame(), (None, None, None))


def test_retained_formation_metadata_owns_its_nested_values():
    pending = {
        "formation_eligibility": {"s": {"eligible_names": [1]}},
        "construction_evidence": {"source": [2]},
        "source_intent": {"book": {"1": 0.5}},
        "orders": [{"shares": 1}],
    }
    expected = deepcopy(pending)
    row = {}
    retain_intent_evidence(row, pending)
    pending["formation_eligibility"]["s"]["eligible_names"].clear()
    pending["source_intent"]["book"].clear()
    assert row == {
        key: expected[key]
        for key in ("formation_eligibility", "construction_evidence", "source_intent")
    }


def test_forged_saved_weights_cannot_create_a_frozen_optimization_claim(
    case, tmp_path, monkeypatch
):
    from capstone.strategy.research import digest

    saved = portfolio(case, through="2000-01-31")
    rows = deepcopy(saved.schedule)
    rows[0]["weights"] = dict.fromkeys(rows[0]["pool_ids"], 0.0)
    rows[0]["cash_weight"] = 1.0
    rows[0]["id"] = digest({k: v for k, v in rows[0].items() if k != "id"})
    forged = replace(saved, schedule=rows)
    path = tmp_path / "forged"
    with pytest.raises(ValueError, match="actual prior funded"):
        freeze(path, forged, case[0], monkeypatch)
    assert not path.exists()


def test_v4_execution_contract_policy_is_pinned_against_main(case):
    from capstone.strategy.research import digest

    record = execution_contract(portfolio(case), version=EXECUTION_VERSION)
    policy = {k: v for k, v in record.items() if k not in {"id", "implementation"}}
    # Digest generated on main 79dd6bf; runtime-independent (no implementation hashes).
    assert digest(policy) == "04773b290c9bc8691095413c460013b871cf9044ceebe5d953a82c203607cc02"


def _account(plan):
    shares = plan["target_shares"]
    cash = 1000 - sum(10 * quantity for quantity in shares.values())
    return {"positions": shares, "cash": cash, "capital": 1000, "open_orders": []}


def test_v5_earlier_plans_close_retired_exposure(case, tmp_path, monkeypatch):
    panel, (high, _), _ = case
    lower = replace(high, direction="low", mechanism_id="lower", tail=0.25)
    events = [
        receipt(high, "2000-01-28"),
        receipt(lower, "2000-01-28"),
        receipt(high, "2000-02-28"),
        departure(lower, "2000-02-28"),
    ]
    saved = chronological(case, (high, lower), events, through="2000-02-29")
    for old in ("program-events-v4", EXECUTION_VERSION):
        with pytest.raises(ExecutionContractBlocked, match="explicit_execution_v5"):
            execution_contract(saved, version=old)
    first = build_plan(freeze(tmp_path / "jan", saved, panel, monkeypatch))
    held = first["target_shares"]
    # Both books were live: the retiring long-101 / short-104 book flips those signs.
    assert held["101"] > 0 and held["104"] < 0
    second = build_plan(
        freeze(
            tmp_path / "feb", saved, panel, monkeypatch, as_of="2000-02-29", account=_account(first)
        )
    )
    assert verify_plan(second) == second and not second["transmittable"]
    target = second["target_shares"]
    # Only the surviving high/low-tails book remains; the retired tilt is closed by orders.
    assert target["101"] < 0 and target["104"] > 0 and len({abs(q) for q in target.values()}) == 1
    traded = {str(order["permno"]) for order in second["orders"]}
    assert {"101", "104"} <= traded
    decision = saved.decision("2000-02-29")
    assert decision["capital_transition"]["retired"] == {lower.id: first_weight(saved, lower)}


def first_weight(saved, spec):
    return saved.decision("2000-01-31")["weights"][spec.id]


def test_v5_last_member_exit_plans_all_cash(case, tmp_path, monkeypatch):
    panel, (high, _), _ = case
    events = [receipt(high, "2000-01-28"), departure(high, "2000-02-28", "registry_retired")]
    saved = chronological(case, (high,), events, through="2000-02-29")
    first = build_plan(freeze(tmp_path / "jan", saved, panel, monkeypatch))
    assert first["target_shares"]
    second = build_plan(
        freeze(
            tmp_path / "feb", saved, panel, monkeypatch, as_of="2000-02-29", account=_account(first)
        )
    )
    assert saved.decision("2000-02-29")["cash_weight"] == 1
    assert not any(second["target_shares"].values())
    assert {str(order["permno"]) for order in second["orders"]} == set(first["target_shares"])
