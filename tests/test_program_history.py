"""Complete-history program constructions: qualified quotes, window exits, pinned identities."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from capstone.factors.evaluation import Panel
from capstone.strategy.compiled import COMMON, RULES, CompiledStrategySpec
from capstone.strategy.program import ProgramStrategySpec, program_weights
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.schedule_dispatch import decision_due


@pytest.fixture
def panel():
    dates = pd.bdate_range("2020-06-22", periods=12)
    names = pd.Index([101, 102, 103, 104], name="permno")
    close = pd.DataFrame(
        np.arange(12)[:, None] * np.array([1.0, 2.0, 3.0, 4.0]) + 10,
        index=dates,
        columns=names,
    )
    cap = pd.DataFrame(np.tile([10.0, 20.0, 30.0, 70.0], (12, 1)), index=dates, columns=names)
    returns = close.pct_change(fill_method=None)
    fields = {"close": close, "cap": cap, "returns": returns}
    return Panel(
        fields,
        pd.DataFrame(True, index=dates, columns=names),
        cap * 0,
        data_version="qualified-fixture-v1",
        calendar=dates.append(pd.DatetimeIndex(["2020-07-08"])),
        availability={name: pd.Series(dates, index=dates) for name in fields},
    )


@pytest.fixture
def spec():
    return ProgramStrategySpec(
        expression="delta(close, 1)",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="unlisted/arbitrary-mechanism",
        data_version="qualified-fixture-v1",
        anchor="2020-06-23",
        tail=0.5,
        schedule=ProgramSchedule(interval=3),
    )


# Synthetic exchange calendar: Jan 31 (a month end) and Mar 2 (a first session)
# are holidays, so month-end offsets must count sessions, not weekdays.
HOLIDAYS = ["2020-01-20", "2020-01-31", "2020-02-17", "2020-03-02", "2020-04-10"]
ENTRIES = ["2020-01-29", "2020-02-27", "2020-03-30", "2020-04-29"]
EXITS = ["2020-02-05", "2020-03-05", "2020-04-03", "2020-05-05"]


@pytest.fixture
def holiday_panel():
    dates = pd.bdate_range("2020-01-02", "2020-05-29").drop(pd.DatetimeIndex(HOLIDAYS))
    columns = pd.Index([101, 102, 103, 104], name="permno")
    close = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]] * len(dates), index=dates, columns=columns)
    return Panel(
        {"close": close},
        close > 0,
        close * 0,
        data_version="fixture",
        calendar=dates,
        availability={"close": pd.Series(dates, index=dates)},
    )


def calendar_program(**changes):
    from capstone.strategy.program_schedule import CalendarWindowSchedule

    fields = dict(
        expression="close",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="controller-owned/family",
        data_version="fixture",
        anchor=ENTRIES[0],
        tail=1.0,
        schedule=CalendarWindowSchedule(enter_offset=-1, exit_offset=3),
        exit_rule="calendar_window_exit",
    )
    return ProgramStrategySpec(**fields | changes)


def compiled(method):
    from capstone.factors.research_contracts import canonical

    return CompiledStrategySpec(
        method,
        canonical(RULES[method] | COMMON | {"profile": "long_only"}),
        "1" * 64,
        "2" * 64,
        "fixture",
        "2020-06-23",
        "long_only",
    )


def test_calendar_window_v2_enters_and_exits_through_targets(holiday_panel, monkeypatch):
    from capstone.strategy import program as program_module
    from capstone.strategy.compiled import calendar_weights
    from capstone.strategy.contract import target_weights
    from capstone.strategy.program_schedule import CalendarWindowSchedule

    panel, spec = holiday_panel, calendar_program()
    dates = pd.DatetimeIndex(panel.calendar)
    end = pd.Timestamp("2020-05-20")
    schedule = spec.schedule
    assert list(schedule.formations(dates, spec.anchor, end).strftime("%Y-%m-%d")) == ENTRIES
    assert list(schedule.exits(dates, spec.anchor, end).strftime("%Y-%m-%d")) == EXITS
    for at in dates[(dates >= spec.anchor) & (dates <= end)]:
        label = str(at.date())
        assert decision_due(spec, panel, at) == (label in ENTRIES or label in EXITS)

    # Targets: the book formed at each entry is held through the decision before
    # its exit; the exit decision and every session until the next entry are cash.
    legacy = compiled("turn_of_month")
    legacy = replace(legacy, anchor="2020-01-02")
    for at in dates[(dates >= spec.anchor) & (dates <= end)]:
        book = target_weights(spec, panel, at)
        inside = any(
            pd.Timestamp(a) <= at < pd.Timestamp(b) for a, b in zip(ENTRIES, EXITS, strict=True)
        )
        assert book.empty != inside
        if inside:
            assert book.to_dict() == {101: 0.25, 102: 0.25, 103: 0.25, 104: 0.25}
        # Legacy turn_of_month fills at the next close, the program at the next
        # open: the same held sessions are one decision session apart.
        previous = dates[dates.get_loc(at) - 1]
        pd.testing.assert_series_equal(
            book, calendar_weights(legacy, panel, previous), check_index_type=False
        )

    # v1 identities are unchanged (pinned values, computed with a fixed digest).
    monkeypatch.setattr(program_module, "implementation_digest", lambda: "0" * 64)
    assert ProgramSchedule().id == "10c97c226aba273a"
    assert ProgramSchedule("month_end", 1).id == "4295075a75be65bf"
    assert ProgramSchedule("week_end", 1).id == "55a9fa7bc0e7f86e"
    v1 = ProgramStrategySpec(
        expression="close",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="controller-owned/family",
        data_version="fixture",
        anchor="2020-06-23",
        schedule=ProgramSchedule("month_end", 1),
    )
    assert (v1.id, v1.construction_id) == ("8571da77f91faac1", "874f907c2ea1ccd0")
    v2 = calendar_program(implementation="0" * 64)
    assert ProgramStrategySpec.from_toml(v2.to_toml()) == v2

    # Invalid combinations are refused.
    month_end = ProgramSchedule("month_end", 1)
    for changes in (
        {"exit_rule": "membership_or_rebalance"},
        {"schedule": month_end},
        {"schedule": month_end, "exit_rule": "membership_or_rebalance"},
        {"side": "long_short"},
        {"direction": "low"},
        {"tail": 1.5},
    ):
        with pytest.raises(ValueError):
            calendar_program(**changes)
    with pytest.raises(ValueError):
        ProgramSchedule("calendar_window", 1)
    for bad in (
        dict(enter_offset=3, exit_offset=-1),
        dict(enter_offset=0, exit_offset=0),
        dict(enter_offset=-1, exit_offset=30),
        dict(enter_offset=-1, exit_offset=3, interval=2),
        dict(enter_offset=-1, exit_offset=3, version=1),
        dict(enter_offset=True, exit_offset=3),
    ):
        with pytest.raises(ValueError):
            CalendarWindowSchedule(**bad)
    overlapping = CalendarWindowSchedule(enter_offset=-10, exit_offset=10)
    with pytest.raises(ValueError, match="overlap"):
        overlapping.formations(dates, "2020-01-02", "2020-04-30")


def test_v3_excludes_unquoted_names_before_ranking(spec, panel):
    at = pd.Timestamp("2020-06-23")
    spec = replace(spec, side="long_short", construction="fixed_tails_complete_history_v2")
    panel.half_spread.loc[at, 104] = np.nan
    # The v2 book still ranks and holds the unquoted name.
    assert set(program_weights(spec, panel, at).index) == {101, 102, 103, 104}
    v3 = replace(spec, construction="fixed_tails_qualified_v3")
    book = program_weights(v3, panel, at)
    # Ranking {101, 102, 103}: dropping 104 after ranking all four would keep 102 short.
    assert book.to_dict() == {101: -0.5, 103: 0.5}
    evidence = book.attrs["formation_eligibility"][v3.id]
    assert evidence["eligible_names"] == [101, 102, 103]
    assert evidence["exclusions"] == [
        {"field": "half_spread", "name": 104, "reason": "missing_qualified_quote"}
    ]
    panel.fields["close"].loc[at, 102] = 0.0
    evidence = program_weights(v3, panel, at).attrs["formation_eligibility"][v3.id]
    assert evidence["eligible_names"] == [101, 103]
    assert {"field": "close", "name": 102, "reason": "nonpositive_sizing_price"} in evidence[
        "exclusions"
    ]
    assert v3.construction_id != spec.construction_id
    assert ProgramStrategySpec.from_toml(v3.to_toml()) == v3


def test_v2_program_identity_is_pinned(monkeypatch):
    from capstone.strategy import program

    monkeypatch.setattr(program, "implementation_digest", lambda: "0" * 64)
    pinned = ProgramStrategySpec(
        "delta(close, 1)",
        ("1" * 64,),
        "2" * 64,
        "3" * 64,
        "pinned-mechanism",
        "pinned-data-v1",
        "2020-06-23",
        tail=0.5,
        weighting="capitalization",
        schedule=ProgramSchedule(interval=3),
        construction="fixed_tails_complete_history_v2",
    )
    # Pinned values: a new field or identity rule would change them.
    assert (pinned.id, pinned.construction_id) == ("07eda6d9aa7263af", "52fffb691c02332f")


def test_v2_drops_names_without_complete_required_history_and_keeps_the_audit(spec, panel):
    from capstone.strategy.program import FormationEligibilityError

    at = pd.Timestamp("2020-06-23")
    v2 = replace(spec, construction="fixed_tails_complete_history_v2", side="long_short")
    panel.fields["close"].loc[pd.Timestamp("2020-06-22"), 101] = np.nan
    book = program_weights(v2, panel, at)
    evidence = book.attrs["formation_eligibility"][v2.id]
    assert 101 not in book.index and evidence["eligible_names"] == [102, 103, 104]
    assert evidence["exclusions"] == [
        {"field": "close", "name": 101, "reason": "missing_required_history"}
    ]
    assert evidence["returns_based_selection"] is False
    panel.fields["close"].loc[pd.Timestamp("2020-06-22"), 102:104] = np.nan
    with pytest.raises(FormationEligibilityError, match="two eligible") as caught:
        program_weights(v2, panel, at)
    assert caught.value.formation_eligibility["eligible_names"] == []


def test_a_calendar_window_needs_its_exit_rule_and_a_long_only_high_whole_universe():
    from capstone.strategy.program_schedule import CalendarWindowSchedule

    window = CalendarWindowSchedule(enter_offset=-1, exit_offset=3)
    fields = dict(
        expression="close",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="controller-owned/family",
        data_version="fixture",
        anchor="2020-01-29",
    )
    ok = ProgramStrategySpec(**fields, schedule=window, exit_rule="calendar_window_exit", tail=1.0)
    assert ok.tail == 1.0
    with pytest.raises(ValueError, match="calendar_window"):
        ProgramStrategySpec(**fields, schedule=window, tail=1.0)
    with pytest.raises(ValueError, match="tail one"):
        ProgramStrategySpec(**fields, tail=1.0, schedule=ProgramSchedule(interval=3))
