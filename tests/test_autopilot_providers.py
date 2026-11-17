"""The offline market-data provider contracts: fixture and archive-replay providers,
the sealed-holdout refusal and the explicit-configuration rule. Synthetic data only."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capstone.pipeline.autopilot import providers
from capstone.pipeline.autopilot.providers import (
    ArchiveReplayProvider,
    DataInvalid,
    FixtureProvider,
    HoldoutPrices,
    MarketDataMissing,
    ProviderDataError,
    ProviderNotConfigured,
    build,
    in_holdout,
    panel_days,
    require_outside_holdout,
)


def closes_frame(days=("2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07")):
    index = pd.to_datetime(list(days))
    return pd.DataFrame(
        {
            10001: np.linspace(10, 13, 4)[: len(days)],
            10002: np.array([20.0, np.nan, 22.0, 23.0])[: len(days)],
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# fixture provider


def test_from_closes_documents_follow_the_contract():
    closes = closes_frame()
    opens = closes + 0.5
    provider = FixtureProvider.from_closes(closes, opens=opens)
    assert provider.sessions(None) == ["2019-03-04", "2019-03-05", "2019-03-06"]
    assert provider.sessions("2019-03-04") == ["2019-03-05", "2019-03-06"]
    market = provider.market("2019-03-04")
    assert market["session"] == "2019-03-05"  # the execution session
    assert market["prices"] == {"10001": 11.5}  # the NaN open is dropped
    assert provider.marks("2019-03-04") == {
        "session": "2019-03-04",
        "prices": {"10001": 10.0, "10002": 20.0},
    }
    assert provider.events("2019-03-04") is None
    assert provider.panel("2019-03-04") is None
    assert [kind for kind, _ in provider.reads] == ["market", "marks", "events", "panel"]


def test_non_positive_and_missing_prices_are_dropped():
    closes = closes_frame()
    closes.iloc[1, 0] = -1.0
    provider = FixtureProvider.from_closes(closes)
    assert provider.marks("2019-03-05")["prices"] == {}  # NaN and negative: nothing usable


def test_missing_document_is_market_data_missing():
    provider = FixtureProvider({"2019-03-04": {"marks": {"session": "2019-03-04", "prices": {}}}})
    with pytest.raises(MarketDataMissing):
        provider.market("2019-03-04")
    with pytest.raises(MarketDataMissing):
        provider.marks("2019-03-05")
    assert provider.events("2019-03-04") is None


def test_events_come_back_per_session():
    rows = [{"event_id": "e1", "event_date": "2019-03-04", "permno": 10001, "kind": "split"}]
    provider = FixtureProvider.from_closes(closes_frame(), events={"2019-03-04": rows})
    assert provider.events("2019-03-04") == rows
    assert provider.events("2019-03-05") is None


def test_from_folder_reads_documents_and_refuses_a_missing_root(tmp_path):
    with pytest.raises(ProviderNotConfigured):
        FixtureProvider.from_folder(tmp_path / "nope")
    day = tmp_path / "2019-03-04"
    day.mkdir()
    (day / "marks.json").write_text(json.dumps({"session": "2019-03-04", "prices": {"1": 2.0}}))
    provider = FixtureProvider.from_folder(tmp_path)
    assert provider.marks("2019-03-04")["prices"] == {"1": 2.0}
    with pytest.raises(MarketDataMissing):
        provider.market("2019-03-04")


# ---------------------------------------------------------------------------
# the sealed holdout


def test_holdout_bounds_are_inclusive():
    assert not in_holdout("2020-12-31")
    assert in_holdout("2021-01-01")
    assert in_holdout("2025-12-31")
    assert not in_holdout("2026-01-02")


def test_outside_holdout_documents_pass():
    require_outside_holdout(
        "2019-03-04",
        market={"session": "2019-03-05", "prices": {}},
        marks={"session": "2019-03-04", "prices": {}},
        events=[{"event_id": "e", "event_date": "2019-03-04"}],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"market": {"session": "2021-01-04"}},
        {"marks": {"session": "2022-06-01T00:00:00"}},  # a date-like value is normalized
        {"events": [{"event_id": "e", "event_date": "2023-02-01"}]},
        {"panel": {"calendar": ["2020-12-30", "2021-01-04"]}},
        {"panel": {"universe": {"index": ["2024-05-01"], "values": [[1]]}}},
        {"panel": {"fields": {"px": {"index": ["2025-12-31"]}}}},
    ],
)
def test_holdout_dates_stop_the_run(kwargs):
    with pytest.raises(HoldoutPrices) as caught:
        require_outside_holdout("2020-12-30", **kwargs)
    assert not isinstance(caught.value, ProviderDataError)  # a STOP, never a blocker
    assert caught.value.code == "holdout_prices"


def test_non_date_values_are_left_to_the_schema_checks():
    require_outside_holdout("2019-03-04", market={"session": "soon"}, events=["junk", {"x": 1}])
    assert panel_days("not a panel") == []
    assert panel_days({"calendar": "x", "universe": {"index": "y"}}) == []


def test_panel_days_lists_every_row_and_calendar_day():
    panel = {
        "calendar": ["2019-03-04"],
        "universe": {"index": ["2019-03-04"]},
        "half_spread": {"index": ["2019-03-04"]},
        "fields": {"px": {"index": ["2019-03-03"]}},
        "availability": {"a": {"index": ["2019-03-02"]}},
    }
    where = [w for w, _ in panel_days(panel)]
    assert where == [
        "calendar day",
        "`universe` row",
        "`half_spread` row",
        "`field:px` row",
        "`availability:a` row",
    ]


# ---------------------------------------------------------------------------
# archive replay


def write_archive(root, cutoff="2020-12-31", calendar=("2019-03-04", "2019-03-05", "2019-03-06")):
    root.mkdir(parents=True, exist_ok=True)
    (root / "historical-inputs.json").write_text(
        json.dumps({"cutoff": cutoff, "calendar": list(calendar)})
    )
    return root


def with_frames(provider, closes, opens, events=None):
    provider._inputs = SimpleNamespace(closes=closes, opens=opens, events=events)
    return provider


def test_archive_replay_serves_next_open_and_current_close(tmp_path):
    root = write_archive(tmp_path / "arch")
    closes = closes_frame(("2019-03-04", "2019-03-05", "2019-03-06"))
    opens = closes - 1.0
    provider = with_frames(ArchiveReplayProvider(root), closes, opens)
    assert provider.sessions(None) == ["2019-03-04", "2019-03-05"]  # last has no execution day
    assert provider.sessions("2019-03-04") == ["2019-03-05"]
    market = provider.market("2019-03-04")
    assert market["session"] == "2019-03-05"
    assert market["prices"]["10001"] == pytest.approx(opens.loc["2019-03-05", 10001])
    assert provider.marks("2019-03-04")["prices"]["10001"] == pytest.approx(10.0)
    with pytest.raises(MarketDataMissing):
        provider.market("2019-03-06")  # no next session
    with pytest.raises(MarketDataMissing):
        provider.market("2019-03-01")  # not in the calendar
    assert provider.events("2019-03-04") is None


def test_archive_replay_events_are_filtered_by_date(tmp_path):
    root = write_archive(tmp_path / "arch")
    closes = closes_frame(("2019-03-04", "2019-03-05", "2019-03-06"))
    table = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "event_date": pd.to_datetime(["2019-03-04", "2019-03-05"]),
            "permno": [10001, 10002],
            "kind": ["split", "split"],
        }
    )
    provider = with_frames(ArchiveReplayProvider(root), closes, closes, events=table)
    rows = provider.events("2019-03-04")
    assert [r["event_id"] for r in rows] == ["a"]
    assert rows[0]["event_date"] == "2019-03-04"
    assert provider.events("2019-03-06") is None


def test_archive_replay_checks_the_cutoff_before_loading_anything(tmp_path):
    bad = ArchiveReplayProvider(write_archive(tmp_path / "bad", cutoff="2021-06-30"))
    with pytest.raises(ProviderNotConfigured, match="cutoff"):
        bad.sessions(None)
    good = ArchiveReplayProvider(write_archive(tmp_path / "good"))
    with pytest.raises(DataInvalid):
        good.market("2021-02-01")  # after the cutoff, and no frame was ever loaded
    assert good._inputs is None


def test_archive_replay_calendar_never_lists_sessions_after_the_cutoff(tmp_path):
    root = write_archive(tmp_path / "arch", calendar=("2020-12-30", "2020-12-31", "2021-01-04"))
    assert ArchiveReplayProvider(root).sessions(None) == ["2020-12-30"]


def test_archive_replay_refuses_a_folder_that_is_not_an_archive(tmp_path):
    with pytest.raises(ProviderNotConfigured):
        ArchiveReplayProvider(tmp_path).sessions(None)
    (tmp_path / "historical-inputs.json").write_text("{not json")
    with pytest.raises(ProviderNotConfigured, match="unreadable"):
        ArchiveReplayProvider(tmp_path).sessions(None)


# ---------------------------------------------------------------------------
# explicit configuration


def test_build_prefers_the_injected_provider():
    injected = FixtureProvider({})
    section = {"provider": "nonsense", "provider_config": {}}
    assert build(section, {"market_data_provider": injected}) is injected


@pytest.mark.parametrize("kind", ["", "live", "drop_folder_x", "vendor"])
def test_build_refuses_unknown_kinds_by_default(kind):
    with pytest.raises(ProviderNotConfigured):
        build({"provider": kind, "provider_config": {"root": "x"}}, {})


@pytest.mark.parametrize("root", [None, "", "   ", 3])
def test_build_never_falls_back_to_a_default_root(root):
    for kind in ("fixture", "archive_replay"):
        config = {} if root is None else {"root": root}
        with pytest.raises(ProviderNotConfigured, match="root"):
            build({"provider": kind, "provider_config": config}, {})


def test_build_makes_the_named_providers(tmp_path):
    section = {"provider": "fixture", "provider_config": {"root": str(tmp_path)}}
    assert isinstance(build(section, {}), FixtureProvider)
    section = {"provider": "archive_replay", "provider_config": {"root": str(tmp_path)}}
    assert isinstance(build(section, {}), ArchiveReplayProvider)


def test_module_imports_nothing_that_could_reach_a_network():
    source = open(providers.__file__, encoding="utf-8").read()
    for word in ("socket", "urllib", "requests", "http", "getenv", "environ"):
        assert word not in source.replace("a network connection", "")
