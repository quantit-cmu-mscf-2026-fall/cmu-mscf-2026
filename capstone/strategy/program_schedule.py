"""Versioned formation schedules over a supplied authoritative session calendar."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from capstone.factors.research_contracts import digest


def session_date(value, label="session") -> pd.Timestamp:
    try:
        at = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a timezone-free session date") from exc
    if pd.isna(at) or at.tz is not None or at != at.normalize():
        raise ValueError(f"{label} must be a timezone-free session date")
    return at


def session_calendar(calendar, anchor, as_of) -> pd.DatetimeIndex:
    if calendar is None:
        raise ValueError("an authoritative session calendar is required")
    dates = pd.DatetimeIndex(calendar)
    if dates.empty or dates.has_duplicates or dates.hasnans:
        raise ValueError("calendar must contain distinct session dates")
    if dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("calendar must contain timezone-free session dates")
    dates = dates.sort_values()
    anchor, at = session_date(anchor, "anchor"), session_date(as_of, "decision")
    if anchor not in dates or at not in dates or at < anchor:
        raise ValueError("calendar must contain anchor and decision on or after anchor")
    if dates[-1] <= at:
        raise ValueError("calendar must contain the following execution session")
    return dates


@dataclass(frozen=True)
class ProgramSchedule:
    kind: str = "interval"
    interval: int = 21
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported program schedule version")
        if type(self.kind) is not str or self.kind not in {"interval", "week_end", "month_end"}:
            raise ValueError("unsupported schedule; event and cohort schedules need a new version")
        if type(self.interval) is not int or not 1 <= self.interval <= 1260:
            raise ValueError("schedule interval must be an integer in [1, 1260]")
        if self.kind != "interval" and self.interval != 1:
            raise ValueError("calendar-boundary schedules must declare interval one")

    @property
    def id(self):
        return digest(asdict(self))[:16]

    def formations(self, calendar, anchor, as_of) -> pd.DatetimeIndex:
        dates = session_calendar(calendar, anchor, as_of)
        at, first = session_date(as_of), dates.get_loc(session_date(anchor))
        if self.kind == "interval":
            return dates[first : dates.get_loc(at) + 1 : self.interval]
        periods = dates.to_period("W-FRI" if self.kind == "week_end" else "M")
        complete = dates[:-1][periods[:-1] != periods[1:]]
        return complete[(complete >= dates[first]) & (complete <= at)]

    def decision_due(self, calendar, anchor, as_of) -> bool:
        return session_date(as_of) in self.formations(calendar, anchor, as_of)


@dataclass(frozen=True)
class CalendarWindowSchedule(ProgramSchedule):
    """Version 2: enter and exit at session offsets from each exchange month end.

    Offset 0 is a month's last session; -1/+3 forms at the second-to-last
    completed session and exits to cash at the third session of the next month.
    A separate class keeps every version-1 schedule and program identity unchanged.
    """

    kind: str = "calendar_window"
    interval: int = 1
    version: int = 2
    enter_offset: int = -1
    exit_offset: int = 3

    def __post_init__(self):
        if type(self.version) is not int or self.version != 2:
            raise ValueError("calendar window schedules are program schedule version 2")
        if type(self.kind) is not str or self.kind != "calendar_window":
            raise ValueError("unsupported version 2 schedule kind")
        if type(self.interval) is not int or self.interval != 1:
            raise ValueError("calendar-boundary schedules must declare interval one")
        offsets = (self.enter_offset, self.exit_offset)
        if any(type(value) is not int or not -10 <= value <= 10 for value in offsets):
            raise ValueError("calendar window offsets must be integer sessions in [-10, 10]")
        if self.enter_offset >= self.exit_offset:
            raise ValueError("calendar window must enter before it exits")

    def _windows(self, calendar, anchor, as_of):
        dates = session_calendar(calendar, anchor, as_of)
        at = dates.get_loc(session_date(as_of))
        # The known exchange calendar (never data) must confirm the month end an
        # entry at a negative offset precedes.
        if len(dates) <= at + max(0, -self.enter_offset) + 1:
            raise ValueError("calendar must contain the month end following an entry decision")
        periods = dates.to_period("M")
        ends = (periods[:-1] != periods[1:]).nonzero()[0]
        enter, leave = ends + self.enter_offset, ends + self.exit_offset
        # An entry past the known calendar cannot be due yet (positive offsets near its end).
        keep = (enter >= 0) & (enter < len(dates))
        enter, leave = enter[keep], leave[keep]
        if (leave[:-1] >= enter[1:]).any():
            raise ValueError("calendar windows overlap; overlapping cohorts are unsupported")
        start = session_date(anchor)
        entries = dates[enter]
        exits = dates[leave[leave < len(dates)]]
        exits = exits[entries[: len(exits)] >= start]
        return entries[(entries >= start) & (entries <= dates[at])], exits[exits <= dates[at]]

    def formations(self, calendar, anchor, as_of) -> pd.DatetimeIndex:
        return self._windows(calendar, anchor, as_of)[0]

    def exits(self, calendar, anchor, as_of) -> pd.DatetimeIndex:
        return self._windows(calendar, anchor, as_of)[1]

    def decision_due(self, calendar, anchor, as_of) -> bool:
        at = session_date(as_of)
        entries, exits = self._windows(calendar, anchor, as_of)
        return at in entries or at in exits


def load_schedule(row) -> ProgramSchedule:
    """Rebuild a schedule from its recorded fields, choosing the class by version."""
    return (CalendarWindowSchedule if row.get("version") == 2 else ProgramSchedule)(**row)
