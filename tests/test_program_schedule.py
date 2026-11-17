import pandas as pd
import pytest

from capstone.strategy.program_schedule import (
    CalendarWindowSchedule,
    ProgramSchedule,
    load_schedule,
    session_calendar,
    session_date,
)
from capstone.strategy.research import canonical, digest

CAL = pd.bdate_range("2020-05-01", "2020-09-30")


def test_interval_and_boundary_formations_use_the_supplied_calendar():
    anchor, at = "2020-05-01", "2020-07-15"
    interval = ProgramSchedule(interval=10).formations(CAL, anchor, at)
    assert interval[0] == pd.Timestamp(anchor) and list(
        interval.to_series().diff().dropna().dt.days
    ) == [14] * (len(interval) - 1)
    month_end = ProgramSchedule(kind="month_end", interval=1).formations(CAL, anchor, at)
    assert list(month_end) == [pd.Timestamp("2020-05-29"), pd.Timestamp("2020-06-30")]
    week_end = ProgramSchedule(kind="week_end", interval=1).formations(CAL, anchor, at)
    assert all(d.day_of_week == 4 for d in week_end)
    assert ProgramSchedule(interval=10).decision_due(CAL, anchor, "2020-05-15")
    assert not ProgramSchedule(interval=10).decision_due(CAL, anchor, "2020-05-14")


def test_schedule_identity_and_validation():
    assert ProgramSchedule(interval=21).id == ProgramSchedule(interval=21).id
    assert ProgramSchedule(interval=21).id != ProgramSchedule(interval=20).id
    for bad in (
        dict(kind="event"),
        dict(interval=0),
        dict(interval=True),
        dict(kind="month_end", interval=2),
        dict(version=2),
    ):
        with pytest.raises(ValueError):
            ProgramSchedule(**bad)


def test_calendar_is_required_and_must_cover_the_execution_session():
    with pytest.raises(ValueError, match="authoritative"):
        session_calendar(None, "2020-05-01", "2020-05-04")
    with pytest.raises(ValueError, match="following execution"):
        session_calendar(CAL, "2020-05-01", CAL[-1])
    with pytest.raises(ValueError, match="distinct"):
        session_calendar(CAL.append(CAL[:1]), "2020-05-01", "2020-05-04")
    with pytest.raises(ValueError, match="timezone-free"):
        session_date("2020-05-01 09:30")


def test_calendar_window_enters_before_month_end_and_exits_after():
    schedule = CalendarWindowSchedule()
    at = "2020-07-31"
    entries = schedule.formations(CAL, "2020-05-01", at)
    exits = schedule.exits(CAL, "2020-05-01", at)
    assert list(entries) == [
        pd.Timestamp("2020-05-28"),
        pd.Timestamp("2020-06-29"),
        pd.Timestamp("2020-07-30"),
    ]
    assert list(exits) == [pd.Timestamp("2020-06-03"), pd.Timestamp("2020-07-03")]
    assert schedule.decision_due(CAL, "2020-05-01", "2020-06-29")
    assert schedule.decision_due(CAL, "2020-05-01", "2020-07-03")
    assert not schedule.decision_due(CAL, "2020-05-01", "2020-06-15")
    for bad in (dict(enter_offset=3, exit_offset=3), dict(exit_offset=11), dict(interval=2)):
        with pytest.raises(ValueError):
            CalendarWindowSchedule(**bad)


def test_load_schedule_chooses_the_class_by_version_and_keeps_v1_identity():
    from dataclasses import asdict

    v1, v2 = ProgramSchedule(interval=7), CalendarWindowSchedule()
    assert load_schedule(asdict(v1)) == v1 and load_schedule(asdict(v2)) == v2
    assert type(load_schedule(asdict(v2))) is CalendarWindowSchedule


def test_canonical_rejects_ambiguous_keys_and_digest_is_order_free():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    with pytest.raises(ValueError, match="ambiguous"):
        canonical({1: "x", "1": "y"})
    with pytest.raises(ValueError):
        canonical({"x": float("nan")})
