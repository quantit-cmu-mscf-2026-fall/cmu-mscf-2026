"""The actual archived-input entry point must validate the whole batch first."""

import json

import pandas as pd
import pytest

from capstone import shared_data as sd
from capstone.pipeline.historical_bundle import load_inputs, runtime_versions
from capstone.pipeline.parquet_guard import ArchivedInputBlocked
from capstone.pipeline.study import file_digest
from capstone.strategy.research import digest


def archive(tmp_path, *, future_role=None, corrupt_role=None):
    dates = pd.DatetimeIndex(pd.to_datetime(["2020-12-30", "2020-12-31"]), name="date")
    frame = pd.DataFrame({101: [1.0, 2.0]}, index=dates)
    availability = pd.Series(dates, index=dates).to_frame("known_at")
    if future_role == "availability:close":
        availability.iloc[-1, 0] = pd.Timestamp("2021-01-04")
    events = pd.DataFrame(
        {
            "event_id": pd.Series(dtype=str),
            "event_date": pd.Series(dtype="datetime64[ns]"),
            "permno": pd.Series(dtype=int),
            "kind": pd.Series(dtype=str),
        }
    )
    frames = {
        "field:close": frame,
        "universe": frame.gt(0),
        "half_spread": frame * 0,
        "opens": frame,
        "closes": frame,
        "events": events,
        "total_returns": frame * 0,
        "half_spread_bps": frame * 0,
        "availability:close": availability,
    }
    # Place the offending frame last: a per-file check after reading would leak.
    if future_role == "opens":
        frames["opens"] = frame.set_axis(
            pd.DatetimeIndex(pd.to_datetime(["2020-12-31", "2021-01-04"]), name="date")
        )
    if future_role:
        frames[future_role] = frames.pop(future_role)
    metadata = {
        "schema_version": 1,
        "cutoff": "2020-12-31",
        "runtime": runtime_versions(),
        "frames": {},
        "data_version": "synthetic-boundary-test",
        "qualification": {},
        "calendar": list(dates.strftime("%Y-%m-%d")),
    }
    for i, (label, values) in enumerate(frames.items()):
        file = tmp_path / f"input-{i}.parquet"
        values.to_parquet(file)
        metadata["frames"][label] = {"file": file.name, "sha256": file_digest(file)}
    if corrupt_role:
        metadata["frames"][corrupt_role]["sha256"] = "0" * 64
    metadata["input_hash"] = digest(metadata)
    (tmp_path / "historical-inputs.json").write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize("future_role", ["opens", "availability:close"])
def test_later_future_frame_blocks_all_reads(tmp_path, monkeypatch, future_role):
    archive(tmp_path, future_role=future_role)
    monkeypatch.setattr(sd, "load", lambda *a, **kw: pytest.fail("read before preflight"))
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery"):
        load_inputs(tmp_path)


def test_later_bad_checksum_blocks_all_reads(tmp_path, monkeypatch):
    archive(tmp_path, corrupt_role="availability:close")
    monkeypatch.setattr(sd, "load", lambda *a, **kw: pytest.fail("read before preflight"))
    with pytest.raises(ValueError, match="integrity"):
        load_inputs(tmp_path)


def test_safe_archive_reads_only_through_the_shared_loader_and_leaves_originals(
    tmp_path, monkeypatch
):
    metadata = archive(tmp_path)
    original = sd.load
    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return original(name, **kwargs)

    monkeypatch.setattr(sd, "load", load)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("direct parquet read"))
    loaded = load_inputs(tmp_path)
    assert loaded.panel.fields["close"].iloc[-1, 0] == 2
    assert len(calls) == len(metadata["frames"])
    assert all(kw["end"] == "2020-12-31" and kw["auto_sync"] is False for _, kw in calls)
    assert {kw["date_column"] for _, kw in calls} == {"date", "event_date"}
    for item in metadata["frames"].values():
        assert file_digest(tmp_path / item["file"]) == item["sha256"]


def test_the_archive_read_restores_the_callers_cache_directory(tmp_path, monkeypatch):
    archive(tmp_path)
    monkeypatch.setenv("CAPSTONE_CACHE_DIR", str(tmp_path / "elsewhere"))
    load_inputs(tmp_path)
    assert sd.cache_dir() == tmp_path / "elsewhere"
    monkeypatch.delenv("CAPSTONE_CACHE_DIR")
    load_inputs(tmp_path)
    assert sd.cache_dir() == sd.DEFAULT_CACHE
