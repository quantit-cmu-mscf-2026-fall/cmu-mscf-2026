"""Daily planning choice on synthetic fixtures; not financial evidence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from capstone.execution.contracts import ExecutionContractBlocked
from capstone.execution.daily_planning import (
    active_programs,
    calendar_window_exit,
    initial_replay_plan,
    lifecycle_plan,
    plan_completed_session,
    program_initialization_due,
    replay_actions,
    window_start_due,
)
from capstone.execution.replay_state import replay_start_context
from capstone.factors.evaluation import Panel
from capstone.strategy.portfolio import PortfolioSpec
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import CalendarWindowSchedule, ProgramSchedule

DATES = pd.bdate_range("2020-06-22", "2020-08-07")
FORMATION, OFF_DAY = pd.Timestamp("2020-06-26"), pd.Timestamp("2020-06-24")
PRICES = {101: 10.0, 102: 20.0, 103: 30.0, 104: 40.0}
WEIGHTS = pd.Series({101: 0.5, 103: 0.5}, name="weight")


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
        close * 0,
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


@pytest.fixture
def window():
    return program(CalendarWindowSchedule(), exit_rule="calendar_window_exit")


def test_active_programs_flatten_portfolios_and_refuse_unknown_members(spec, window):
    portfolio = PortfolioSpec((spec, replace(spec, tail=0.4)), pd.DataFrame())
    assert active_programs(spec, FORMATION) == (spec,)
    assert active_programs([portfolio, window], FORMATION) == (*portfolio.specs, window)
    with pytest.raises(ExecutionContractBlocked, match="unsupported_program_exit_member"):
        active_programs(object(), FORMATION)


def test_a_formation_session_rebalances_to_the_given_book(spec, panel):
    plan = plan_completed_session(spec, panel, FORMATION, {}, PRICES, 1000.0, weights=WEIGHTS)
    assert plan["intent"] == "formation_rebalance"
    assert {o["permno"] for o in plan["orders"]} == {101, 103}


def test_a_formation_without_a_book_is_refused(spec, panel):
    with pytest.raises(ExecutionContractBlocked, match="formation_target_book_missing"):
        plan_completed_session(spec, panel, FORMATION, {}, PRICES, 1000.0)


def test_allocation_change_is_explicit_and_boolean(spec, panel):
    plan = plan_completed_session(
        spec, panel, OFF_DAY, {}, PRICES, 1000.0, weights=WEIGHTS, allocation_changed=True
    )
    assert plan["intent"] == "allocation_rebalance"
    with pytest.raises(ExecutionContractBlocked, match="allocation_transition_contract_invalid"):
        plan_completed_session(
            spec, panel, OFF_DAY, {}, PRICES, 1000.0, weights=WEIGHTS, allocation_changed=1
        )


def test_between_formations_only_exits_are_planned(spec, panel):
    plan = plan_completed_session(spec, panel, OFF_DAY, {101: 5, 104: 2}, PRICES, 1000.0)
    assert plan["intent"] == "membership_exit_only"
    assert plan["projected"] == {101: 0, 104: 2}
    assert [(o["permno"], o["shares"]) for o in plan["orders"]] == [(101, -5)]


def test_a_calendar_window_exit_session_is_labelled(window, panel):
    exit_day = pd.Timestamp("2020-07-03")
    assert calendar_window_exit(window, panel, exit_day)
    assert not calendar_window_exit(window, panel, pd.Timestamp("2020-07-06"))
    plan = plan_completed_session(
        window, panel, exit_day, {101: 5}, PRICES, 1000.0, weights=pd.Series(dtype=float)
    )
    assert plan["intent"] == "calendar_window_exit"


def test_lifecycle_states_require_the_program_route(panel):
    from capstone.factors.evaluation import TradingSpec
    from capstone.strategy.contract import StrategySpec

    legacy = StrategySpec("close", TradingSpec(hold=3), "fixture", "2020-06-23")
    with pytest.raises(ExecutionContractBlocked, match="lifecycle_states_require_program_route"):
        lifecycle_plan(
            legacy, panel, FORMATION, {}, PRICES, 1000.0, weights=None, attribution={}, states={}
        )


def test_retired_members_close_between_decisions(spec, panel):
    other = replace(spec, tail=0.4)
    states = {other.id: "retired"}
    book, plan = lifecycle_plan(
        [spec, other],
        panel,
        OFF_DAY,
        {101: 5},
        PRICES,
        1000.0,
        weights=None,
        attribution={},
        states=states,
    )
    assert plan["intent"] == "lifecycle_exit_only"
    assert plan["projected"] == {101: 0}
    assert [(o["permno"], o["shares"]) for o in plan["orders"]] == [(101, -5)]


def test_lifecycle_decision_uses_only_active_attribution(spec, panel):
    other = replace(spec, tail=0.4)
    attribution = {spec.id: {101: 0.5, 103: 0.5}, other.id: {102: 0.5, 104: 0.5}}
    book, plan = lifecycle_plan(
        [spec, other],
        panel,
        FORMATION,
        {},
        PRICES,
        1000.0,
        weights=None,
        attribution=attribution,
        states={other.id: "retired"},
    )
    assert set(book.index) == {101, 103}
    assert plan["intent"] == "formation_rebalance"
    assert set(plan["projected"]) <= {101, 103}


def test_program_initialization_follows_anchor_and_schedule(spec, panel):
    assert program_initialization_due(spec, panel, FORMATION)
    assert not program_initialization_due(spec, panel, OFF_DAY)
    assert not program_initialization_due(spec, panel, pd.Timestamp("2020-06-22"))


def test_window_start_requires_the_versions_programs_and_anchor(window, spec, panel):
    prior = pd.Timestamp("2020-06-24")
    assert window_start_due(window, panel, prior, "program-events-v6")
    assert not window_start_due(window, panel, prior, "program-events-v4")
    assert not window_start_due(spec, panel, prior, "program-events-v6")
    assert not window_start_due(window, panel, pd.Timestamp("2020-06-22"), "program-events-v6")


def test_initial_replay_plan_invests_only_after_a_formation(spec, panel):
    closes = panel.fields["close"]
    inputs = {
        "spec": spec,
        "panel": panel,
        "closes": closes,
        "capital": 10000.0,
        "min_notional": 1.0,
        "execution": {"route": "program-v1", "version": "fixed-share-next-open-v2"},
    }
    plan = initial_replay_plan(inputs, FORMATION, lambda day: 0.0)
    assert plan["decision"] == "2020-06-26" and plan["orders"]
    assert initial_replay_plan(inputs, OFF_DAY, lambda day: 0.0) is None
    assert initial_replay_plan(inputs, None, lambda day: 0.0) is None
    with pytest.raises(ValueError, match="previous completed-session prices"):
        initial_replay_plan(inputs | {"closes": closes.iloc[:0]}, FORMATION, lambda day: 0.0)


def start_inputs(spec, panel):
    return {
        "spec": spec,
        "panel": panel,
        "execution": {"route": "program-v1", "version": "fixed-share-next-open-v2"},
    }


def test_replay_start_context_initializes_only_after_a_formation(spec, panel):
    for prior, initialize in ((FORMATION, True), (OFF_DAY, False)):
        dates = DATES[DATES.get_loc(prior) + 1 :]
        context = replay_start_context(start_inputs(spec, panel), None, DATES, dates, None)
        assert context["initialize"] is initialize and context["prior"] == prior
        assert context["origin_session"] == str(dates[0].date())
    first = DATES[:1]
    assert (
        replay_start_context(start_inputs(spec, panel), None, DATES, first, None)["prior"] is None
    )


def test_replay_actions_wraps_program_route_failures_only():
    class Actions:
        def apply(self, *args):
            raise ValueError("bad action")

    with pytest.raises(ExecutionContractBlocked, match="corporate_action_transition_unqualified"):
        replay_actions(Actions(), FORMATION, {}, 1.0, True)
    with pytest.raises(ValueError, match="bad action"):
        replay_actions(Actions(), FORMATION, {}, 1.0, False)
