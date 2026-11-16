import json

import pandas as pd
import pytest
from test_factor_evaluation import fake_rows

from capstone import shared_data as sd
from capstone.factors.evaluation import COLUMNS, DAILY_FILE
from capstone.pipeline import historical as h
from capstone.pipeline.study import file_digest


def test_real_input_units_and_market_frames_are_explicit():
    daily = fake_rows(n_stocks=4, n_days=8, start="2000-01-03")
    daily["permco"] = daily.permno
    dates = pd.DatetimeIndex(sorted(daily.date.unique()))
    qualification = {"global_blockers": [], "method_blockers": {}}
    inputs = h.build_inputs(
        daily,
        dates,
        pd.DataFrame(columns=h.EVENT_COLUMNS),
        data_version="test",
        qualification=qualification,
    )
    first = daily.iloc[0]
    assert inputs.panel.fields["cap"].loc[first.date, first.permno] == first.dlycap * 1000
    assert isinstance(inputs.panel.fields["mkt_return"], pd.DataFrame)
    assert inputs.opens.loc[first.date, first.permno] == first.dlyopen
    # The qualified path cannot inherit the legacy cross-stock spread fallback or loss proxy.
    missing = daily.copy()
    missing.loc[0, ["dlybid", "dlyask", "dlyret"]] = float("nan")
    strict = h.build_inputs(
        missing,
        dates,
        pd.DataFrame(columns=h.EVENT_COLUMNS),
        data_version="test",
        qualification=qualification,
    )
    assert pd.isna(strict.half_spread_bps.loc[first.date, first.permno])
    assert pd.isna(strict.total_returns.loc[first.date, first.permno])
    broken = daily.copy()
    broken.loc[broken.index[0], "date"] = pd.Timestamp("2021-01-01")
    with pytest.raises(ValueError, match="boundary"):
        h.build_inputs(
            broken,
            dates,
            pd.DataFrame(columns=h.EVENT_COLUMNS),
            data_version="test",
            qualification=qualification,
        )


@pytest.mark.parametrize(
    "event",
    [
        {"kind": "dividend", "amount": 1, "pay_date": "1999-12-31"},
        {"kind": "split", "ratio": 0},
        {"kind": "conversion", "ratio": 1, "amount": 0},
    ],
)
def test_incomplete_event_terms_are_rejected(event):
    row = {"event_id": "event-1", "event_date": "2000-01-03", "permno": 1, **event}
    with pytest.raises(ValueError):
        h.validate_events(pd.DataFrame([row]))


def test_ex_date_and_payment_date_are_distinct_and_ordered():
    event = {
        "event_id": "d1",
        "event_date": "2000-01-03",
        "permno": 1,
        "kind": "dividend",
        "amount": 2.0,
        "pay_date": "2000-01-10",
    }
    assert len(h.validate_events(pd.DataFrame([event]))) == 1
    with pytest.raises(ValueError, match="unique"):
        h.validate_events(pd.DataFrame([event, event]))


def test_missing_sources_block_without_materializing_observations(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_CACHE_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(sd, "load", lambda *a, **kw: calls.append(kw))
    with pytest.raises(h.DataBlocked) as error:
        h.load_inputs()
    assert not calls
    assert not error.value.report["holdout_opened"]
    assert "events_source_missing" in error.value.report["global_blockers"]
    assert error.value.report["method_blockers"]["value"]


def test_shared_loader_receives_cutoff_before_materialization(monkeypatch):
    calls = []
    monkeypatch.setattr(sd, "load", lambda *a, **kw: calls.append(kw))
    h.discovery_read("annual", date_column="datadate")
    assert calls == [
        {
            "columns": None,
            "start": "1990-01-01",
            "end": "2020-12-31",
            "auto_sync": False,
            "date_column": "datadate",
        }
    ]
    with pytest.raises(ValueError, match="holdout"):
        h.discovery_read("daily", end="2021-01-01")
    assert len(calls) == 1


def test_attestation_cannot_override_a_changed_source_file(tmp_path):
    file = tmp_path / "events.parquet"
    file.write_bytes(b"original")
    sha = file_digest(file)
    (tmp_path / "SHA256SUMS").write_text(sha + "  events.parquet\n")
    descriptor = {
        "sources": {
            "events": {
                "dataset": "events",
                "sha256": sha,
                "start": "1990-01-01",
                "end": "2020-12-31",
                "qualifications": sorted(h.REQUIREMENTS["events"]),
            }
        }
    }
    assert h.source_status(descriptor, root=tmp_path)["sources"]["events"]["qualified"]
    file.write_bytes(b"changed")
    assert not h.source_status(descriptor, root=tmp_path)["sources"]["events"]["qualified"]


def test_calendar_reads_only_the_following_session_date(monkeypatch):
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"date": pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-05"])})

    monkeypatch.setattr(sd, "load", read)
    dates = h.read_calendar("verified_exchange_calendar")
    assert list(dates.strftime("%Y-%m-%d")) == ["2020-12-31", "2021-01-04"]
    assert calls[0]["columns"] == ["date"]


def test_calendar_accepts_daily_security_rows_and_is_permutation_invariant(monkeypatch):
    rows = pd.DataFrame({"date": pd.to_datetime(["2021-01-05", "2020-12-31", "2021-01-04"] * 2)})
    monkeypatch.setattr(sd, "load", lambda *args, **kwargs: rows)
    expected = h.read_calendar(h.DAILY_FILE)
    rows = rows.sample(frac=1, random_state=404)
    pd.testing.assert_index_equal(h.read_calendar(h.DAILY_FILE), expected)
    assert list(expected.strftime("%Y-%m-%d")) == ["2020-12-31", "2021-01-04"]
    rows.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError, match="nonmissing"):
        h.read_calendar(h.DAILY_FILE)


def test_missing_descriptor_is_distinct_from_a_verified_price_source(tmp_path, monkeypatch):
    source = tmp_path / (h.DAILY_FILE + ".parquet")
    source.write_bytes(b"verified source")
    sd.write_manifest(tmp_path / sd.MANIFEST, {source.name: file_digest(source)})
    monkeypatch.setenv("CAPSTONE_CACHE_DIR", str(tmp_path))
    report, descriptor = h.preflight()
    assert not report["descriptor_present"] and descriptor == {}
    assert report["source_hashes"]["daily"] == file_digest(source)
    assert "daily_source_missing" not in report["global_blockers"]


def _cache(tmp_path, monkeypatch):
    """A synthetic shared-data cache: a three-session daily file, a calendar, no events."""
    monkeypatch.setenv("CAPSTONE_CACHE_DIR", str(tmp_path))
    rows = fake_rows(n_stocks=3, n_days=3, start="2000-01-03")
    sessions = pd.to_datetime(["1990-01-02", "2020-12-30", "2020-12-31"])
    rows["date"] = rows.date.map(dict(zip(sorted(rows.date.unique()), sessions, strict=True)))
    rows["permco"] = rows.permno
    rows["ticker"] = "T" + rows.permno.astype(str)
    rows[sorted(set(COLUMNS + ["permco", "ticker"]))].to_parquet(
        tmp_path / (DAILY_FILE + ".parquet"), index=False
    )
    calendar = pd.DataFrame({"date": [*sessions, pd.Timestamp("2021-01-04")]})
    calendar.to_parquet(tmp_path / "calendar.parquet", index=False)
    pd.DataFrame(
        {"event_id": ["x"], "event_date": pd.to_datetime(["2020-12-31"]), "permno": [1]}
    ).head(0).assign(kind="dividend").to_parquet(tmp_path / "events.parquet", index=False)
    names = [DAILY_FILE, "calendar", "events"]
    sd.write_manifest(
        tmp_path / sd.MANIFEST,
        {f"{n}.parquet": file_digest(tmp_path / f"{n}.parquet") for n in names},
    )
    sources = {
        role: {
            "dataset": name,
            "sha256": file_digest(tmp_path / f"{name}.parquet"),
            "start": "1990-01-01",
            "end": "2021-01-04",
            "qualifications": sorted(h.REQUIREMENTS[role]),
        }
        for role, name in (("calendar", "calendar"), ("events", "events"))
    }
    path = tmp_path / "historical-sources.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": sources}))
    return path


def test_load_inputs_reads_every_source_through_the_shared_loader(tmp_path, monkeypatch):
    descriptor = _cache(tmp_path, monkeypatch)
    calls = []
    original = sd.load

    def load(*args, **kwargs):
        calls.append((args[0], kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(sd, "load", load)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("direct parquet read"))
    inputs = h.load_inputs(descriptor)
    assert [name for name, _ in calls] == [DAILY_FILE, "calendar", "events"]
    assert all(kw["auto_sync"] is False for _, kw in calls)
    assert calls[0][1]["end"] == "2020-12-31" and calls[2][1]["end"] == "2020-12-31"
    assert inputs.panel.universe.index.max() == pd.Timestamp("2020-12-31")
    assert inputs.qualification["historical_observations_loaded"]
    assert inputs.qualification["event_return_identities"]["verified"]
    assert inputs.panel.data_version == sd.data_version()
    assert not inputs.qualification["holdout_opened"]


def test_a_source_changed_after_the_preflight_refuses_the_load(tmp_path, monkeypatch):
    descriptor = _cache(tmp_path, monkeypatch)
    original = sd.load
    state = {"n": 0}

    def load(*args, **kwargs):
        state["n"] += 1
        frame = original(*args, **kwargs)
        if state["n"] == 3:  # after the last read a sync replaces a source file
            (tmp_path / "events.parquet").write_bytes(b"replaced")
        return frame

    monkeypatch.setattr(sd, "load", load)
    with pytest.raises(ValueError, match="changed while"):
        h.load_inputs(descriptor)
