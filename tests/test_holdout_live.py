"""The post-holdout read route for live-era CRSP.

`holdout_live.guard_live_read` admits only a window that starts after the sealed
holdout's end, read from the holdout configuration (`REAL_PERIOD`), and refuses every
other with a typed reason. `guard_shared_read` and `shared_data.load` are unchanged by it.
Synthetic fixture parquet files only; nothing is synced, no real data is read.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from capstone import holdout_authority as ha
from capstone import holdout_live as live
from capstone import shared_data

END = ha.REAL_PERIOD[1]
START = ha.REAL_PERIOD[0]


def shift(day, days):
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


AFTER = shift(END, 1)  # the first day the route admits


def reason(start, end, name="crsp_daily_live", column="date"):
    with pytest.raises(live.LiveReadRefused) as caught:
        live.guard_live_read(name, start, end, column)
    assert isinstance(caught.value, ha.HoldoutSealed)  # every caller's catch still holds
    assert caught.value.reason in live.LIVE_READ_REASONS
    return caught.value.reason


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (shift(START, 30), shift(START, 60)),  # inside the holdout
        (shift(START, 30), shift(END, 400)),  # starts inside, ends after
        (shift(START, -400), shift(START, 30)),  # starts before, ends inside
        (shift(START, -400), shift(END, 400)),  # spans it
        (shift(START, -400), shift(START, -1)),  # discovery-era only: not this route
        (START, AFTER),  # the holdout's first day
        (END, AFTER),  # the holdout's last day
        (END, None),  # open-ended from the last day
        (AFTER, END),  # starts after, ends inside
        (shift(END, 400), shift(END, -30)),  # ends inside the holdout
    ],
)
def test_a_window_touching_the_holdout_is_refused(start, end):
    assert reason(start, end) == "live_read_touches_holdout"


def test_the_edges_after_the_holdout_are_admitted():
    assert live.guard_live_read("crsp_daily_live", AFTER, AFTER, "date") == (AFTER, AFTER)
    assert live.guard_live_read("crsp_daily_live", AFTER, None, "date") == (AFTER, None)
    later = shift(END, 400)
    assert live.guard_live_read("CRSP_daily_live", later, f"{later} 00:00:00", "date") == (
        later,
        later,
    )
    assert live.guard_live_read("crsp_delist_events", AFTER, None, "event_date")[0] == AFTER


def test_every_other_refusal_is_typed():
    assert reason(None, shift(END, 30)) == "live_read_unbounded"
    assert reason(f"{AFTER}T09:30:00", None) == "live_read_unbounded"
    assert reason("soon", None) == "live_read_unbounded"
    assert reason(shift(END, 30), AFTER) == "live_read_inverted"
    assert reason(AFTER, None, name="comp_funda", column="datadate") == "live_read_not_crsp"
    assert reason(AFTER, None, name="sp500_daily_1990_2025") == "live_read_not_crsp"
    assert reason(AFTER, None, name="../crsp_daily_live") == "live_read_not_crsp"
    assert reason(AFTER, None, name="crsp_x/../crsp_daily_live") == "live_read_not_crsp"
    assert reason(AFTER, None, column="datadate") == "live_read_date_column"
    assert reason(AFTER, None, name="crsp_delist_events") == "live_read_date_column"


def test_the_holdout_end_comes_from_the_holdout_configuration(monkeypatch):
    """Not hard-coded: moving the configured end moves the route's boundary."""
    later = shift(END, 365)
    monkeypatch.setattr(ha, "REAL_PERIOD", (START, later))
    assert reason(AFTER, None) == "live_read_touches_holdout"
    assert live.guard_live_read("crsp_daily_live", shift(later, 1), None, "date")[0] == shift(
        later, 1
    )


def test_the_old_guard_still_refuses_a_live_read(monkeypatch):
    monkeypatch.setenv("CAPSTONE_HOLDOUT_GUARD", "on")
    with pytest.raises(ha.HoldoutSealed):
        ha.guard_shared_read("crsp_daily_live", ["date"], shift(END, 30), "date", False, None)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A synthetic CRSP-shaped fixture file with rows before, inside and after the
    holdout, in a temporary shared-data cache (never synced)."""
    monkeypatch.setenv("CAPSTONE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CAPSTONE_DATA_DIR", str(tmp_path / "nowhere"))
    (tmp_path / "cache").mkdir()
    days = [shift(START, -2), shift(START, 10), END, AFTER, shift(END, 5), shift(END, 40)]
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(days),
            "permno": pd.array([1, 1, 1, 1, 2, 2], dtype="Int64"),
            "dlyret": pd.array([0.01, 0.02, None, 0.03, 0.04, 0.05], dtype="Float64"),
        }
    )
    rows.to_parquet(tmp_path / "cache" / "crsp_daily_live.parquet")
    return tmp_path / "cache"


def test_load_live_reads_only_rows_after_the_holdout(cache):
    frame = shared_data.load_live("crsp_daily_live", start=AFTER, end=None, auto_sync=False)
    assert [str(d.date()) for d in frame["date"]] == [AFTER, shift(END, 5), shift(END, 40)]
    assert frame["dlyret"].dtype == "float64" and frame["permno"].dtype == "int64"
    bounded = shared_data.load_live(
        "crsp_daily_live", ["date", "dlyret"], start=AFTER, end=shift(END, 5), auto_sync=False
    )
    assert list(bounded.columns) == ["date", "dlyret"] and len(bounded) == 2


def test_load_live_refuses_before_any_read_or_sync(cache, monkeypatch):
    def no_sync(*args, **kwargs):
        raise AssertionError("a refused read must not sync")

    monkeypatch.setattr(shared_data, "sync", no_sync)
    for start, end in ((END, None), (shift(START, 10), shift(END, 40)), (AFTER, END)):
        with pytest.raises(live.LiveReadRefused, match="live_read_touches_holdout"):
            shared_data.load_live("crsp_daily_live", start=start, end=end)
    monkeypatch.setenv("CAPSTONE_HOLDOUT_GUARD", "on")
    with pytest.raises(ha.HoldoutSealed):  # the old route: still refused
        shared_data.load("crsp_daily_live", start=AFTER, end=shift(END, 40), auto_sync=False)
