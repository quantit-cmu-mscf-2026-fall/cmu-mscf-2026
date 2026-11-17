"""Formation dispatch regression fixtures; no execution or financial admission proof."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from capstone.factors.evaluation import Panel, TradingSpec
from capstone.factors.research_contracts import canonical
from capstone.strategy.compiled import COMMON, RULES, CompiledStrategySpec
from capstone.strategy.contract import StrategySpec
from capstone.strategy.diversified import DiversifiedStrategySpec
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.reference import BenchmarkSpec, ReferenceStrategySpec
from capstone.strategy.research import contracts
from capstone.strategy.research_spec import ResearchStrategySpec
from capstone.strategy.schedule_dispatch import decision_due


@pytest.fixture
def panel():
    dates = pd.bdate_range("2020-06-22", "2020-07-10")
    frame = pd.DataFrame(1.0, index=dates, columns=[101, 102])
    return Panel(
        {"close": frame},
        frame.astype(bool),
        frame * 0,
        data_version="fixture",
        calendar=dates,
    )


def program():
    return ProgramStrategySpec(
        expression="close",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="controller-owned/family",
        data_version="fixture",
        anchor="2020-06-23",
        schedule=ProgramSchedule(interval=3),
    )


def compiled(method):
    return CompiledStrategySpec(
        method,
        canonical(RULES[method] | COMMON | {"profile": "long_only"}),
        "1" * 64,
        "2" * 64,
        "fixture",
        "2020-06-23",
        "long_only",
    )


def test_program_and_legacy_anchored_calendars(panel):
    legacy = StrategySpec("close", TradingSpec(hold=3), "fixture", "2020-06-23")
    for spec in (program(), legacy):
        assert decision_due(spec, panel, "2020-06-23")
        assert decision_due(spec, panel, "2020-06-26")
        assert not decision_due(spec, panel, "2020-06-24")


def test_compiled_weekly_cannot_fall_through_to_month_end(panel):
    for spec in (
        compiled("weekly_reversal"),
        DiversifiedStrategySpec("weekly_reversal", "fixture", "2020-06-23"),
    ):
        assert decision_due(spec, panel, "2020-06-26")
        assert not decision_due(spec, panel, "2020-06-30")
    assert decision_due(compiled("monthly_low_volatility"), panel, "2020-06-30")


def test_compiled_calendar_window_matches_its_own_scheduler(panel):
    from capstone.strategy.compiled import decision_due as original_due

    spec = compiled("turn_of_month")
    for at in pd.bdate_range("2020-06-23", "2020-07-03"):
        assert decision_due(spec, panel, at) == original_due(spec, panel, at)


def test_reference_research_and_benchmark_known_types(panel):
    claim = next(
        row["id"]
        for row in contracts()["claims"]
        if row["status"] == "executable" and not row["method"].endswith("skip_week")
    )
    reference = ReferenceStrategySpec(claim, "fixture", "2020-06-23")
    research = ResearchStrategySpec("paper-value", "value", "fundamental", "fixture", "2020-06-23")
    cash = BenchmarkSpec("cash", "fixture", "2020-06-23")
    for spec in (reference, research, cash):
        assert decision_due(spec, panel, "2020-06-30")
        assert not decision_due(spec, panel, "2020-06-29")
    momentum = BenchmarkSpec("known_momentum_12_1", "fixture", "2020-06-23")
    assert decision_due(momentum, panel, "2020-06-23")
    assert not decision_due(momentum, panel, "2020-06-24")


def test_unknown_types_fail_closed(panel):
    unknown = SimpleNamespace(method="weekly_reversal", specs=())
    with pytest.raises(ValueError, match="unsupported strategy schedule type"):
        decision_due(unknown, panel, "2020-06-23")


def test_all_recognized_leaves_reject_pre_anchor_and_non_session_decisions(panel):
    claim = next(row["id"] for row in contracts()["claims"] if row["status"] == "executable")
    leaves = (
        program(),
        compiled("turn_of_month"),
        compiled("weekly_reversal"),
        DiversifiedStrategySpec("weekly_reversal", "fixture", "2020-06-23"),
        StrategySpec("close", TradingSpec(hold=3), "fixture", "2020-06-23"),
        ReferenceStrategySpec(claim, "fixture", "2020-06-23"),
        ResearchStrategySpec("paper-value", "value", "fundamental", "fixture", "2020-06-23"),
        BenchmarkSpec("cash", "fixture", "2020-06-23"),
    )
    for leaf in leaves:
        with pytest.raises(ValueError, match="on or after anchor"):
            decision_due(replace(leaf, anchor="2020-07-01"), panel, "2020-06-30")
        with pytest.raises(ValueError, match="observed session"):
            decision_due(leaf, panel, "2020-06-27")


def test_supplied_calendar_cannot_be_bypassed_by_known_benchmark(panel):
    panel.calendar = panel.calendar[panel.calendar != pd.Timestamp("2020-06-30")]
    with pytest.raises(ValueError, match="authoritative calendar"):
        decision_due(BenchmarkSpec("cash", "fixture", "2020-06-23"), panel, "2020-06-30")


def test_calendar_window_entry_past_the_known_calendar_is_not_yet_due():
    from capstone.strategy.program_schedule import CalendarWindowSchedule

    calendar = pd.bdate_range("2020-01-01", "2020-04-01")  # ends one session after March
    schedule = CalendarWindowSchedule(enter_offset=2, exit_offset=5)
    assert not schedule.decision_due(calendar, "2020-01-02", "2020-03-31")
    assert pd.Timestamp("2020-04-02") not in schedule.formations(
        calendar, "2020-01-02", "2020-03-31"
    )


def test_a_decision_is_memoized_inside_an_evaluation_scope(panel, monkeypatch):
    from capstone.strategy import compiled as compiled_module
    from capstone.strategy import input_memo

    calls = []
    original = compiled_module.decision_due
    monkeypatch.setattr(compiled_module, "decision_due", lambda *a: calls.append(1) or original(*a))
    spec = compiled("weekly_reversal")
    with input_memo.scope():
        assert decision_due(spec, panel, "2020-06-26") == decision_due(spec, panel, "2020-06-26")
    assert len(calls) == 1
    decision_due(spec, panel, "2020-06-26")
    decision_due(spec, panel, "2020-06-26")
    assert len(calls) == 3
