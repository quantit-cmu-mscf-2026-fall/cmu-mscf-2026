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
from holdout_fakes import make_protocol

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


def spend_real_look():
    """Put a real `holdout-look` in the (temporary) ledger, as a spent look would."""
    protocol = make_protocol(dataset=ha.REAL_DATASET)
    ha._log_look(
        protocol,
        {"approved_by": "synthetic gate owner", "cosigned_by": None},
        {"candidates": ["a" * 16], "dates": [ha.REAL_PERIOD[0], END], "id": "r" * 64},
    )


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


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
    spend_real_look()
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


def test_load_live_needs_the_real_look_to_be_spent(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "empty-ledger"))
    import pyarrow.parquet as pq

    monkeypatch.setattr(
        pq, "read_table", lambda *a, **k: pytest.fail("a read happened before the look was spent")
    )
    with pytest.raises(live.LiveReadRefused, match="live_read_look_unspent"):
        shared_data.load_live("crsp_daily_live", start=AFTER, end=None, auto_sync=False)
    assert "live_read_look_unspent" in live.LIVE_READ_REASONS


def test_a_synthetic_look_does_not_open_the_live_route(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "other-ledger"))
    protocol = make_protocol()
    ha._log_look(
        protocol,
        {"approved_by": "synthetic gate owner", "cosigned_by": None},
        {"candidates": ["a" * 16], "dates": [ha.REAL_PERIOD[0], END], "id": "s" * 64},
    )
    with pytest.raises(live.LiveReadRefused, match="live_read_look_unspent"):
        shared_data.load_live("crsp_daily_live", start=AFTER, end=None, auto_sync=False)


def test_load_live_rechecks_the_frame_after_the_read(cache, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    leaked = pa.table({"date": pd.to_datetime([END, AFTER]), "permno": [1, 1]})
    monkeypatch.setattr(pq, "read_table", lambda *a, **k: leaked)
    with pytest.raises(live.LiveReadRefused, match="live_read_touches_holdout"):
        shared_data.load_live("crsp_daily_live", start=AFTER, end=None, auto_sync=False)
