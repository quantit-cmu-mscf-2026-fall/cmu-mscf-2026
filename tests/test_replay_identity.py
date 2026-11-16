"""Replay input identity and its ledger registration, on synthetic inputs only."""

import json
from dataclasses import dataclass

import pandas as pd
import pytest
from test_saved_portfolio import case, portfolio  # noqa: F401  (case is a fixture)

from capstone.execution import replay_identity as module
from capstone.execution.contracts import ExecutionContractBlocked, execution_contract
from capstone.execution.replay_identity import (
    allocation_identity,
    completed_replay_metadata,
    cost_identity,
    register_replay,
    replay_identity,
    replay_key,
    table_identity,
)
from capstone.pipeline.market_costs import HistoricalCostSchedule
from capstone.strategy import allocation_state

END = "2000-02-29"


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    short_borrow_bps_annual: float = 3.0
    financing_bps_annual: float = 4.0


def inputs(case, **changes):  # noqa: F811
    panel, (high, _low), _returns = case
    prices = panel.fields["close"]
    base = {
        "spec": high,
        "panel": panel,
        "opens": prices,
        "closes": prices,
        "start": "2000-01-31",
        "end": END,
        "capital": 1000,
        "costs": Costs(),
        "min_notional": 100,
        "execution": execution_contract(high),
        "splits": None,
        "dividends": None,
        "cash_settlements": None,
        "events": None,
        "cost_schedule": None,
    }
    return {**base, **changes}


def test_frame_identity_has_one_definition():
    assert module.frame_identity is allocation_state.frame_identity


def test_report_identity_binds_prices_through_end_and_ignores_later_rows(case):  # noqa: F811
    base = replay_identity(**inputs(case))
    assert replay_identity(**inputs(case)) == base
    prices = case[0].fields["close"]
    early = prices.copy()
    early.loc["2000-02-01", 101] += 1
    assert replay_identity(**inputs(case, opens=early))["report_id"] != base["report_id"]
    late = prices.copy()
    late.loc["2000-03-15", 101] += 1  # after `end`: not an input of this replay
    assert replay_identity(**inputs(case, opens=late)) == base
    assert replay_identity(**inputs(case, min_notional=200))["report_id"] != base["report_id"]
    assert replay_identity(**inputs(case, capital=999))["report_id"] != base["report_id"]


def test_report_identity_is_independent_of_row_and_column_order(case):  # noqa: F811
    prices = case[0].fields["close"]
    shuffled = prices.iloc[::-1].loc[:, [104, 103, 102, 101]]
    base = replay_identity(**inputs(case))["inputs_hash"]
    assert replay_identity(**inputs(case, closes=shuffled))["inputs_hash"] == base


def test_table_identity_ignores_row_order_and_binds_values():
    frame = pd.DataFrame({"event_id": [1, 2, 3], "kind": ["a", "b", "c"]})
    assert table_identity(None) is None
    assert table_identity(frame) == table_identity(frame.iloc[::-1])
    assert table_identity(frame) != table_identity(frame.assign(kind=["a", "b", "x"]))


def test_cost_identity_binds_the_schedule_and_refuses_foreign_ones():
    spreads = pd.DataFrame({101: [1.0, 2.0]}, index=pd.to_datetime(["2000-01-03", "2000-01-04"]))
    schedule = HistoricalCostSchedule(spreads, Costs())
    assert cost_identity(None, END) is None
    full = cost_identity(schedule, END)
    assert full["id"] == schedule.id
    assert "id" not in cost_identity(schedule, END, prefix_only=True)
    doubled = HistoricalCostSchedule(spreads, Costs(), multiplier=2)
    assert cost_identity(doubled, END) != full
    with pytest.raises(ExecutionContractBlocked, match="cost_schedule_identity_unqualified"):
        cost_identity(object(), END)


def test_allocation_identity_is_none_for_a_leaf_and_binds_saved_decisions(case):  # noqa: F811
    assert allocation_identity(case[1][0], END) is None
    book = portfolio(case)
    early = allocation_identity(book, "2000-02-01")
    later = allocation_identity(book, "2000-03-31")
    assert early["decision_prefix_id"] != later["decision_prefix_id"]
    assert set(early["members"]) == {spec.id for spec in book.specs}
    assert early["returns"] != later["returns"]


def test_replay_key_is_the_ledger_round_trip_of_the_params():
    params = {"strategy_id": "x", "costs": {"a": 1.0}, "n": (1, 2)}
    assert replay_key(params) == replay_key(json.loads(json.dumps(params)))


def test_resume_scope_consumes_named_rows_once_and_everything_else_is_logged(monkeypatch):
    logged = []
    monkeypatch.setattr(module.runlog, "log_run", lambda *a, **k: logged.append(k["params"]))
    params = {"strategy_id": "x"}
    module.already_logged([replay_key(params)])  # outside a scope: ignored
    module.log_registered(params)
    assert logged == [params]
    with module.resume_scope():
        module.already_logged([replay_key(params)])
        module.log_registered(params)  # the killed attempt logged it already
        assert logged == [params]
        module.log_registered(params)  # a second trial is a new row
    assert logged == [params, params]
    assert module._RESUMED.get() is None


def test_register_replay_logs_before_results_and_captured_rows_are_not_logged(
    case,  # noqa: F811
    monkeypatch,
):
    logged = []
    monkeypatch.setattr(module.runlog, "log_run", lambda name, **k: logged.append((name, k)))
    run = inputs(case)
    identity = register_replay(run)
    ((name, row),) = logged
    assert name == "reference-replay" and row["tags"] == ["reference-v2", "replay"]
    assert row["params"]["execution_inputs_hash"] == identity["inputs_hash"]
    assert row["params"]["execution_report_id"] == identity["report_id"]
    assert row["params"]["data_version"] == case[0].data_version
    with module.captured() as rows:
        register_replay(run)
    assert len(logged) == 1 and rows == [row["params"]]


def test_completed_replay_metadata_refuses_inputs_that_moved_during_the_replay(case):  # noqa: F811
    run = inputs(case)
    identity = replay_identity(**run)
    metadata = completed_replay_metadata(run, identity)
    assert metadata["execution_report_id"] == identity["report_id"]
    assert metadata["formal_admission"] is False
    moved = case[0].fields["close"].copy()
    moved.loc["2000-02-01", 101] += 1
    with pytest.raises(ExecutionContractBlocked, match="execution_input_changed_during_replay"):
        completed_replay_metadata(replace_inputs(run, opens=moved), identity)
    other = {**run, "execution": {**run["execution"], "version": "program-events-v3"}}
    with pytest.raises(ExecutionContractBlocked, match="execution_source_changed_during_replay"):
        completed_replay_metadata(other, identity)


def replace_inputs(run, **changes):
    return {**run, **changes}
