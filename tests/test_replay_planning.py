"""Replay planning, frozen intents and checkpoint recovery on synthetic fixtures.

Not financial evidence; no market data. The last tests run the whole recovery path:
a checkpoint written from a planned session, restored on the next session, and the pending
plan rebuilt by the shared planner.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capstone.execution.contracts import ExecutionContractBlocked, execution_contract
from capstone.execution.daily_planning import (
    calendar_unconfirmed,
    program_replay_plan,
    requalified_entries,
    sticky_source_book,
)
from capstone.execution.replay_state import (
    final_event_metadata,
    initial_account,
    replay_start_context,
    verify_intent,
)
from capstone.factors.evaluation import Panel
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import CalendarWindowSchedule, ProgramSchedule

DATES = pd.bdate_range("2020-06-22", "2020-08-07")
FORMATION, OFF_DAY = pd.Timestamp("2020-06-26"), pd.Timestamp("2020-06-30")
PRICES = {101: 10.0, 102: 20.0, 103: 30.0, 104: 40.0}
EVENT = "program-events-v3"
HELD = "program-events-v5"


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    short_borrow_bps_annual: float = 3.0
    financing_bps_annual: float = 4.0


@pytest.fixture
def panel():
    names = pd.Index([101, 102, 103, 104], name="permno")
    close = pd.DataFrame(
        np.arange(len(DATES))[:, None] * np.array([1.0, 2.0, 3.0, 4.0]) + 10,
        index=DATES,
        columns=names,
    )
    fields = {"close": close, "returns": close.pct_change(fill_method=None)}
    return Panel(
        fields,
        pd.DataFrame(True, index=DATES, columns=names),
        close * 0 + 0.001,
        data_version="fixture",
        calendar=DATES,
        availability={name: pd.Series(DATES, index=DATES) for name in fields},
    )


def program(schedule=None, **over):
    return ProgramStrategySpec(
        expression="delta(close, 1)",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="fixture/mechanism",
        data_version="fixture",
        anchor="2020-06-23",
        tail=0.5,
        schedule=schedule or ProgramSchedule(interval=3),
        **over,
    )


@pytest.fixture
def spec():
    return program()


def plan(spec, panel, day, positions=None, **over):
    options = {
        "min_notional": 1.0,
        "fee_bps": 0.0,
        "terminated": set(),
        "cash": 10000.0,
        "receivables": [],
    }
    return program_replay_plan(
        spec, panel, day, positions or {}, PRICES, 10000.0, **(options | over)
    )


def test_a_legacy_protocol_returns_the_plain_plan_and_refuses_recovery(spec, panel):
    out = plan(spec, panel, FORMATION, execution_version="fixed-share-next-open-v2")
    assert out["decision"] == "2020-06-26"
    assert out["intent"] == "formation_rebalance" and "intent_id" not in out
    with pytest.raises(ExecutionContractBlocked, match="intent_recovery_unqualified"):
        plan(
            spec,
            panel,
            FORMATION,
            execution_version="fixed-share-next-open-v2",
            recovery_requested=True,
        )


def test_the_event_protocol_freezes_a_formation_intent(spec, panel):
    intent = plan(spec, panel, FORMATION, execution_version=EVENT)
    assert verify_intent(intent) == intent
    source = intent["source_intent"]
    assert source["kind"] == "formation" and source["formation_session"] == "2020-06-26"
    assert set(source["book"]) <= {"101", "102", "103", "104"}
    assert "construction_evidence" in intent


def test_between_formations_the_event_protocol_freezes_an_exit_only_intent(spec, panel):
    intent = plan(spec, panel, OFF_DAY, {104: 2}, execution_version=EVENT)
    assert intent["source_intent"]["kind"] == "exit_only"
    assert intent["intent"] == "membership_exit_only"


def test_a_deferred_formation_keeps_its_frozen_book_until_the_next_formation(spec, panel):
    formation = plan(spec, panel, FORMATION, execution_version=EVENT)
    source = formation["source_intent"]
    off = plan(spec, panel, OFF_DAY, execution_version=EVENT, source_intent=source)
    assert off["intent"] == "deferred_formation_rebalance"
    assert off["source_intent"]["formation_session"] == source["formation_session"]
    assert [o["permno"] for o in off["orders"]] == [o["permno"] for o in formation["orders"]]
    later = plan(
        spec, panel, pd.Timestamp("2020-07-01"), execution_version=EVENT, source_intent=source
    )
    assert later["source_intent"]["formation_session"] == "2020-07-01"
    assert later["superseded_source_intent_id"] == source["source_intent_id"]


def test_a_retired_instrument_cannot_be_in_the_target_book(spec, panel):
    with pytest.raises(ExecutionContractBlocked, match="retired_instrument_in_program_targets"):
        plan(spec, panel, FORMATION, execution_version=EVENT, terminated={104})


def formation_source(spec, panel):
    return plan(spec, panel, FORMATION, execution_version=EVENT)["source_intent"]


def test_sticky_source_book_returns_the_frozen_book(spec, panel):
    source = formation_source(spec, panel)
    book = sticky_source_book(source, spec, panel, OFF_DAY)
    assert {int(n): w for n, w in book.items()} == {int(n): w for n, w in source["book"].items()}


@pytest.mark.parametrize(
    "over, day, code",
    [
        ({"kind": "exit_only"}, OFF_DAY, "source_incompatible"),
        ({"strategy_id": "other"}, OFF_DAY, "source_incompatible"),
        ({"formation_session": "2020-06-24"}, OFF_DAY, "source_incompatible"),
        ({}, pd.Timestamp("2020-06-24"), "source_incompatible"),
        ({"book": {"101": 2.0}}, OFF_DAY, "book_invalid"),
    ],
)
def test_sticky_source_book_refuses_inconsistent_sources(spec, panel, over, day, code):
    source = formation_source(spec, panel) | over
    with pytest.raises(ExecutionContractBlocked, match=code):
        sticky_source_book(source, spec, panel, day)


def test_sticky_source_book_drops_names_that_left_the_universe(spec, panel):
    source = formation_source(spec, panel)
    held = int(next(n for n, w in source["book"].items() if w))
    universe = panel.universe.copy()
    universe.loc[OFF_DAY, held] = False
    book = sticky_source_book(source, spec, replace(panel, universe=universe), OFF_DAY)
    assert book[held] == 0
    missing = universe.astype(object)
    missing.loc[OFF_DAY, held] = None
    short = replace(panel, universe=missing)
    with pytest.raises(ExecutionContractBlocked, match="membership_missing"):
        sticky_source_book(source, spec, short, OFF_DAY)


def test_calendar_unconfirmed_needs_a_calendar_window_near_the_calendar_end(panel):
    window = program(CalendarWindowSchedule(), exit_rule="calendar_window_exit")
    assert not calendar_unconfirmed(program(), panel, DATES[-1])
    assert calendar_unconfirmed(window, panel, DATES[-1])
    assert not calendar_unconfirmed(window, panel, DATES[-3])


def test_held_protocols_hold_when_the_calendar_cannot_confirm_a_window(panel):
    window = program(CalendarWindowSchedule(), exit_rule="calendar_window_exit")
    intent = plan(window, panel, DATES[-1], {101: 3}, execution_version=HELD)
    assert intent["intent"] == "calendar_unconfirmed_hold" and intent["orders"] == []
    assert intent["targets"] == {"101": 3}
    assert verify_intent(intent)["source_intent"]["kind"] == "exit_only"


def test_requalified_entries_skip_unqualified_unheld_names(panel):
    spreads = panel.half_spread.copy()
    spreads.loc[FORMATION, 102] = np.nan
    spreads.loc[FORMATION, 103] = -1.0
    panel = replace(panel, half_spread=spreads)
    book = pd.Series({101: 0.2, 102: 0.2, 103: 0.2, 104: 0.2})
    out, excluded = requalified_entries(book, {104: 1}, panel, FORMATION, PRICES, {101})
    assert {e["permno"]: e["reason"] for e in excluded} == {
        101: "deferred_entry_unqualified:retired_instrument",
        102: "deferred_entry_unqualified:missing_qualified_quote",
        103: "deferred_entry_unqualified:missing_qualified_quote",
    }
    assert out[104] == 0.2 and out[[101, 102, 103]].eq(0).all()
    assert book[101] == 0.2
    out, excluded = requalified_entries(
        pd.Series({101: 0.5}), {}, panel, OFF_DAY, PRICES | {101: 0.0}
    )
    assert out[101] == 0 and excluded[0]["reason"].endswith("nonpositive_sizing_price")


# Full recovery path: checkpoint -> restore -> pending plan.


def recovery_inputs(spec, panel):
    closes = panel.fields["close"]
    return {
        "spec": spec,
        "panel": panel,
        "opens": closes,
        "closes": closes,
        "capital": 10000.0,
        "costs": Costs(),
        "min_notional": 1.0,
        "execution": execution_contract(spec, version=EVENT),
        "splits": None,
        "dividends": None,
        "cash_settlements": None,
        "events": None,
        "cost_schedule": None,
    }


def no_events():
    return SimpleNamespace(
        events=pd.DataFrame({"event_date": pd.to_datetime([]), "event_id": []}),
        receivables=[],
        applied=set(),
    )


def test_a_new_account_opens_with_a_frozen_pending_intent(spec, panel):
    inputs = recovery_inputs(spec, panel)
    dates = DATES[DATES.get_loc(FORMATION) + 1 :]
    actions = no_events()
    context = replay_start_context(inputs, None, DATES, dates, actions)
    assert context["initialization"] == "new_account_at_scheduled_formation"
    state = initial_account(inputs, context, None, DATES, dates, actions, lambda day: 0.0)
    assert state["formations"] == ["2020-06-26"] and state["cash"] == 10000.0
    assert verify_intent(state["pending"]) == state["pending"]
    assert context["initialization_qualifying"] is True


def checkpointed(spec, panel, capital_for_plan):
    inputs = recovery_inputs(spec, panel)
    dates = DATES[DATES.get_loc(FORMATION) + 1 :]
    actions = no_events()
    context = replay_start_context(inputs, None, DATES, dates, actions)
    initial_account(inputs, context, None, DATES, dates, actions, lambda day: 0.0)
    # The first session bought 3 shares of 104; the account is marked at that close.
    day = dates[0]
    closes = panel.fields["close"].loc[day].to_dict()
    cash = 10000.0 - 3 * closes[104]
    pending = program_replay_plan(
        spec,
        panel,
        day,
        {104: 3},
        closes,
        capital_for_plan,
        min_notional=1.0,
        fee_bps=0.0,
        terminated=set(),
        recovery_requested=True,
        execution_version=EVENT,
        source_intent=None,
        cash=cash,
        receivables=[],
    )
    meta = final_event_metadata(
        inputs,
        context,
        None,
        DATES,
        day,
        {104: 3},
        cash,
        actions,
        pending,
        set(),
        10000.0,
        ["2020-06-26"],
    )
    return inputs, context, dates, meta, pending, cash


def test_a_checkpoint_restores_through_to_the_pending_plan(spec, panel):
    inputs, context, dates, meta, pending, cash = checkpointed(spec, panel, 10000.0)
    checkpoint = meta["checkpoint"]
    assert meta["seed_checkpoint_id"] is None and checkpoint["session"] == str(dates[0].date())
    resumed, actions = dates[1:], no_events()
    restored = replay_start_context(inputs, checkpoint, DATES, resumed, actions)
    assert restored["initialization"] == "restored_replay_checkpoint"
    state = initial_account(inputs, restored, checkpoint, DATES, resumed, actions, lambda d: 0.0)
    assert state["positions"] == {104: 3} and state["cash"] == cash
    assert state["pending"] == verify_intent(pending)
    assert state["formations"] == ["2020-06-26"] and state["previous"] == 10000.0
    assert restored["origin_session"] == context["origin_session"]


def test_restore_refuses_a_checkpoint_whose_instructions_were_altered(spec, panel):
    inputs, _context, dates, meta, _pending, _cash = checkpointed(spec, panel, 5000.0)
    checkpoint, resumed, actions = meta["checkpoint"], dates[1:], no_events()
    restored = replay_start_context(inputs, checkpoint, DATES, resumed, actions)
    with pytest.raises(ExecutionContractBlocked, match="pending_instructions_incompatible"):
        initial_account(inputs, restored, checkpoint, DATES, resumed, actions, lambda d: 0.0)
