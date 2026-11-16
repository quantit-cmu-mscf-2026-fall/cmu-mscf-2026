"""Retry semantics, physical event cash and causal snapshot edge cases."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_program_event_replay import case as _case
from test_program_event_replay import events, run
from test_program_event_replay import ledger as _ledger

from capstone.execution.contracts import ExecutionContractBlocked
from capstone.execution.replay_state import read_checkpoint, write_checkpoint
from capstone.pipeline.market_costs import HistoricalCostSchedule
from capstone.pipeline.replay import ReplayCosts, replay
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.research import digest


@pytest.fixture
def case():
    return _case.__wrapped__()


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    return _ledger.__wrapped__(tmp_path, monkeypatch)


def test_open_risk_failure_records_no_fill_and_replans_at_the_completed_close(case):
    panel, spec, prices = case
    opens = prices.copy()
    opens.loc["2020-06-24":, [103, 104]] = 20
    prices.loc["2020-06-24":, [103, 104]] = 20
    result = replay(
        spec,
        panel,
        opens,
        prices,
        start="2020-06-24",
        end="2020-06-26",
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        execution_version="program-events-v3",
        events=events(),
    )
    assert result["orders"][0]["status"] == "blocked_open_risk"
    assert result["history"].loc["2020-06-24", "positions"] == {}
    fills = [row for row in result["orders"] if "shares" in row]
    assert [(row["date"], row["shares"]) for row in fills] == [
        ("2020-06-25", 25),
        ("2020-06-25", 25),
    ]


@pytest.mark.parametrize("failure", ["missing_open", "split", "payment"])
def test_nonformation_exit_retries_never_resize_unrelated_survivors(case, failure):
    panel, spec, prices = case
    exit_day = "2020-06-26" if failure == "payment" else "2020-06-25"
    panel.universe.loc[exit_day:, 104] = False
    prices.loc["2020-06-25":, 103] = 15
    opens = prices.copy()
    table = events()
    if failure == "missing_open":
        opens.loc["2020-06-26", 104] = np.nan
    elif failure == "split":
        prices.loc["2020-06-26":, 104] = opens.loc["2020-06-26":, 104] = 5
        table = events(
            dict(event_id="s", event_date="2020-06-26", permno=104, kind="split", ratio=2)
        )
    else:
        table = events(
            dict(
                event_id="d",
                event_date="2020-06-25",
                permno=103,
                kind="dividend",
                amount=1,
                pay_date="2020-06-29",
            )
        )
    result = replay(
        spec,
        panel,
        opens,
        prices,
        start="2020-06-24",
        end="2020-07-01",
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        execution_version="program-events-v3",
        events=table,
    )
    later = [row for row in result["orders"] if "shares" in row and row["date"] > "2020-06-24"]
    assert len(later) == 1 and later[0]["permno"] == 104
    assert later[0]["shares"] == (-100 if failure == "split" else -50)
    assert result["positions"] == {103: 50}
    for holdings in result["history"].positions:
        assert holdings[103] == 50


def test_fractional_split_uses_only_observed_cash_in_lieu(case):
    _, _, prices = case
    prices.loc["2020-06-25":, 103] = 5
    event = dict(event_id="s", event_date="2020-06-25", permno=103, kind="split", ratio=1.01)
    with pytest.raises(ExecutionContractBlocked, match="cash-in-lieu"):
        run(case, event_table=events(event))
    event["cash_in_lieu_price"] = 5
    result = run(case, event_table=events(event))
    assert result["positions"] == {103: 50, 104: 50}
    assert result["cash"] == 2.5
    assert result["history"].iloc[-1].equity == 752.5


def test_new_scheduled_formation_explicitly_supersedes_deferred_source(case):
    panel, spec, prices = case
    spec = replace(spec, schedule=ProgramSchedule(interval=2))
    panel.fields["close"].loc["2020-06-25":, [101, 102]] = 1e9
    opens = prices.copy()
    opens.loc[["2020-06-24", "2020-06-25"], 103] = np.nan
    result = replay(
        spec,
        panel,
        opens,
        prices,
        start="2020-06-24",
        end="2020-06-26",
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        execution_version="program-events-v3",
        events=events(),
    )
    fills = [row for row in result["orders"] if "shares" in row]
    assert {row["permno"] for row in fills} == {101, 102}
    original = result["orders"][0]["source_intent"]["source_intent_id"]
    assert all(row["superseded_source_intent_id"] == original for row in fills)


def test_resume_does_not_relabel_initially_flat_account_after_later_formation(case):
    panel, spec, prices = case
    case = panel, replace(spec, schedule=ProgramSchedule(interval=3)), prices
    first = run(case, start="2020-06-25", end="2020-06-25")
    resumed = run(case, start="2020-06-26", end="2020-06-26", checkpoint=first["checkpoint"])
    assert not first["initialization_qualifying"] and not resumed["initialization_qualifying"]
    assert resumed["checkpoint"]["formation_sessions"] == ["2020-06-26"]
    third = run(case, start="2020-06-29", end="2020-06-30", checkpoint=resumed["checkpoint"])
    assert third["positions"] and not third["initialization_qualifying"]
    assert (
        third["checkpoint"]["origin_initialization"] == first["checkpoint"]["origin_initialization"]
    )


def test_row_permutation_and_future_observations_do_not_change_completed_checkpoint(case):
    baseline = run(case)
    panel, spec, prices = deepcopy(case)
    panel.fields["close"].loc["2020-07-01":] = 1e20
    prices.loc["2020-07-01":] = 1e20
    panel.fields = {name: frame.iloc[::-1, ::-1] for name, frame in panel.fields.items()}
    panel.universe = panel.universe.iloc[::-1, ::-1]
    actual = run((panel, spec, prices.iloc[::-1, ::-1]))
    assert_frame_equal(actual["history"], baseline["history"])
    assert actual["checkpoint"] == baseline["checkpoint"]


def test_observers_cannot_change_owned_prices_or_nested_history(case):
    baseline = run(case)

    def observer(row):
        row["positions"].clear()
        row["market_values"].clear()
        case[2].loc["2020-06-25"] = 1e20

    observed = run(case, on_completed_session=observer)
    assert_frame_equal(observed["history"], baseline["history"])
    assert observed["checkpoint"] == baseline["checkpoint"]


@pytest.mark.parametrize(
    "field,value", [("cash", []), ("formation_sessions", [{}]), ("terminated", [True])]
)
def test_self_hashed_malformed_checkpoint_is_typed_and_never_resumed(case, field, value):
    seed = run(case, end="2020-06-25")["checkpoint"]
    seed[field] = value
    seed["checkpoint_id"] = digest({k: v for k, v in seed.items() if k != "checkpoint_id"})
    with pytest.raises(ExecutionContractBlocked, match="checkpoint"):
        run(case, start="2020-06-26", checkpoint=seed)


def test_atomic_checkpoint_repetition_and_unresolved_partial_file(case, tmp_path):
    record = run(case, end="2020-06-25")["checkpoint"]
    path = write_checkpoint(record, tmp_path / "seed.json")
    assert write_checkpoint(record, path) == path and read_checkpoint(path) == record
    blocked = tmp_path / "blocked.json"
    blocked.with_suffix(".json.partial").write_text('{"truncated":', encoding="utf-8")
    with pytest.raises(FileExistsError, match="quarantined"):
        write_checkpoint(record, blocked)
    assert not blocked.exists()


def test_short_event_checkpoint_restart_keeps_signed_obligation(case):
    panel, spec, prices = case
    case = panel, replace(spec, side="long_short"), prices
    prices.loc["2020-06-25":, 101] = 9
    table = events(
        dict(
            event_id="short",
            event_date="2020-06-25",
            permno=101,
            kind="dividend",
            amount=1,
            pay_date="2020-06-29",
        )
    )
    full = run(case, event_table=table)
    first = run(case, end="2020-06-25", event_table=table)
    assert first["checkpoint"]["receivables"][0]["amount"] == -25
    second = run(case, start="2020-06-26", event_table=table, checkpoint=first["checkpoint"])
    assert_frame_equal(pd.concat([first["history"], second["history"]]), full["history"])
    assert second["checkpoint"] == full["checkpoint"]


@pytest.mark.parametrize("column", ["pay_date", "observed_pay_date"])
def test_contradictory_terminal_cash_payment_date_blocks(case, column):
    table = events(
        dict(
            event_id="terminal",
            event_date="2020-06-25",
            permno=103,
            kind="settlement",
            amount=10,
            **{column: "2020-06-29"},
        )
    )
    with pytest.raises(ExecutionContractBlocked, match="cash_payment_timing"):
        run(case, event_table=table)


def test_existing_observed_terminal_cash_metadata_and_nat_dividend_column_are_compatible(case):
    panel, _, prices = case
    panel.universe.loc["2020-06-25":, 103] = False
    prices.loc["2020-06-25":, 103] = np.nan
    table = events(
        dict(
            event_id="terminal",
            event_date="2020-06-25",
            permno=103,
            kind="settlement",
            amount=10,
            observed_pay_date="2020-06-24",
            pay_date=pd.NaT,
            reason="observed_cash_terminal_distribution",
        )
    )
    result = run(case, event_table=table)
    assert result["positions"] == {104: 50} and result["cash"] == 500


@pytest.mark.parametrize("forgery", ["orders", "targets", "fees", "missing"])
def test_self_hashed_pending_quantities_must_match_the_shared_planner(case, forgery):
    seed = run(case, end="2020-06-24")["checkpoint"]
    plan = seed["pending"]
    assert plan["source_intent"]["kind"] == "exit_only" and plan["orders"] == []
    if forgery == "orders":
        plan["orders"] = [
            {"permno": 103, "shares": -50, "reference_price": 10.0},
            {"permno": 101, "shares": 50, "reference_price": 10.0},
        ]
        plan["targets"] = plan["projected"] = {"101": 50, "103": 0, "104": 50}
    elif forgery == "targets":
        plan["targets"]["103"] = 999
    elif forgery == "fees":
        plan["estimated_fees"] = 100
    else:
        seed["pending"] = None
    if seed["pending"] is not None:
        plan["intent_id"] = digest({k: v for k, v in plan.items() if k != "intent_id"})
    seed["checkpoint_id"] = digest({k: v for k, v in seed.items() if k != "checkpoint_id"})
    completed = []
    with pytest.raises(ExecutionContractBlocked, match="checkpoint_pending"):
        run(case, start="2020-06-25", checkpoint=seed, on_completed_session=completed.append)
    assert completed == []


def test_rehashed_source_evidence_cannot_change_the_original_construction(case):
    panel, spec, prices = case
    case = panel, replace(spec, schedule=ProgramSchedule(interval=2)), prices
    seed = run(case, end="2020-06-25")["checkpoint"]
    plan, source = seed["pending"], seed["pending"]["source_intent"]
    source["construction_evidence"] = {"fabricated": "qualified"}
    source["source_intent_id"] = digest(
        {k: v for k, v in source.items() if k != "source_intent_id"}
    )
    plan["intent_id"] = digest({k: v for k, v in plan.items() if k != "intent_id"})
    seed["checkpoint_id"] = digest({k: v for k, v in seed.items() if k != "checkpoint_id"})
    with pytest.raises(ExecutionContractBlocked, match="checkpoint_pending_source"):
        run(case, start="2020-06-26", checkpoint=seed)


def test_historical_cost_prefix_can_resume_with_appended_future_quotes(case):
    spread = case[2] * 0 + 2.0
    costs = ReplayCosts(1, 0, 5, 200, 500)
    old_schedule = HistoricalCostSchedule(spread.loc[:"2020-06-25"], costs)
    extended_schedule = HistoricalCostSchedule(spread, costs)
    full = run(case, cost_schedule=extended_schedule)
    first = run(case, end="2020-06-25", cost_schedule=old_schedule)
    resumed = run(
        case, start="2020-06-26", cost_schedule=extended_schedule, checkpoint=first["checkpoint"]
    )
    assert old_schedule.id != extended_schedule.id
    assert_frame_equal(pd.concat([first["history"], resumed["history"]]), full["history"])
    assert resumed["checkpoint"] == full["checkpoint"]
    spread.loc["2020-07-01":] = 1000
    perturbed = run(case, cost_schedule=HistoricalCostSchedule(spread, costs))
    assert perturbed["checkpoint"] == full["checkpoint"]


def test_legacy_cost_identity_keeps_the_original_whole_source_binding(case):
    from capstone.execution.replay_identity import cost_identity

    spread = case[2] * 0 + 2.0
    costs = ReplayCosts(1, 0, 5, 200, 500)
    original = HistoricalCostSchedule(spread.loc[:"2020-06-25"], costs)
    extended = HistoricalCostSchedule(spread, costs)
    assert cost_identity(original, "2020-06-25")["id"] == original.id
    assert cost_identity(original, "2020-06-25") != cost_identity(extended, "2020-06-25")
    assert cost_identity(original, "2020-06-25", prefix_only=True) == cost_identity(
        extended, "2020-06-25", prefix_only=True
    )
