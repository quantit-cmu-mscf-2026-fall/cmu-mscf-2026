import json

import pandas as pd
import pytest
from test_factor_evaluation import fake_rows

from capstone.pipeline import historical as h
from capstone.pipeline.historical_bundle import load_inputs, save_inputs


def build():
    daily = fake_rows(n_stocks=4, n_days=8, start="2000-01-03")
    daily["permco"] = daily.permno
    daily["ticker"] = "T" + daily.permno.astype(str)
    dates = pd.DatetimeIndex(sorted(daily.date.unique()))
    events = pd.DataFrame(
        [
            {
                "event_id": "d1",
                "event_date": "2000-01-05",
                "permno": 10000,
                "kind": "dividend",
                "amount": 1.0,
                "pay_date": "2000-01-12",
            }
        ]
    )
    inputs = h.build_inputs(
        daily,
        dates,
        events,
        data_version="synthetic-bundle",
        qualification={"global_blockers": [], "method_blockers": {}},
    )
    return inputs


def test_an_archive_round_trips_the_inputs(tmp_path):
    inputs = build()
    save_inputs(tmp_path / "archive", inputs)
    loaded = load_inputs(tmp_path / "archive")
    pd.testing.assert_frame_equal(loaded.opens, inputs.opens, check_freq=False)
    pd.testing.assert_frame_equal(loaded.total_returns, inputs.total_returns, check_freq=False)
    pd.testing.assert_frame_equal(
        loaded.panel.fields["close"], inputs.panel.fields["close"], check_freq=False
    )
    assert loaded.panel.data_version == "synthetic-bundle"
    assert list(loaded.panel.calendar) == list(inputs.panel.calendar)
    assert loaded.events.event_id.tolist() == ["d1"]
    assert loaded.symbols is not None and loaded.panel.sizing_prices is loaded.closes


def test_an_archive_is_written_once(tmp_path):
    save_inputs(tmp_path / "archive", build())
    with pytest.raises(FileExistsError, match="overwritten"):
        save_inputs(tmp_path / "archive", build())


def test_edited_metadata_or_changed_frames_are_refused(tmp_path):
    save_inputs(tmp_path / "archive", build())
    root = tmp_path / "archive"
    path = root / "historical-inputs.json"
    metadata = json.loads(path.read_text())
    path.write_text(json.dumps({**metadata, "data_version": "other"}))
    with pytest.raises(ValueError, match="metadata changed"):
        load_inputs(root)
    path.write_text(json.dumps(metadata))
    victim = root / metadata["frames"]["opens"]["file"]
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(ValueError, match="integrity"):
        load_inputs(root)


def test_a_different_runtime_is_refused(tmp_path):
    save_inputs(tmp_path / "archive", build())
    path = tmp_path / "archive" / "historical-inputs.json"
    metadata = json.loads(path.read_text())
    metadata["runtime"] = {**metadata["runtime"], "pandas": "0.0"}
    metadata["input_hash"] = h.digest({k: v for k, v in metadata.items() if k != "input_hash"})
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="pinned environment"):
        load_inputs(tmp_path / "archive")
