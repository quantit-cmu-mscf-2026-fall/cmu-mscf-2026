import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from capstone.pipeline.parquet_guard import ArchivedInputBlocked, verify_discovery_dates


def save(tmp_path, frame, *, statistics=True):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pandas(frame), path, write_statistics=statistics)
    return path


@pytest.mark.parametrize("role", ["field:returns", "opens", "company", "symbols"])
def test_discovery_wide_frames_need_no_value_read(tmp_path, monkeypatch, role):
    frame = pd.DataFrame({"101": [1.0, 2.0]}, index=pd.to_datetime(["2020-12-30", "2020-12-31"]))
    path = save(tmp_path, frame)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("materialized rows"))
    assert verify_discovery_dates(path, role)["role"] == role


@pytest.mark.parametrize("role", ["field:close", "opens", "availability:close"])
def test_future_observations_block_at_footer_before_materialization(tmp_path, monkeypatch, role):
    frame = pd.DataFrame(
        {"known_at": pd.to_datetime(["2020-12-31", "2021-01-04"])},
        index=pd.to_datetime(["2020-12-31", "2021-01-04"]),
    )
    path = save(tmp_path, frame)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("materialized rows"))
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery"):
        verify_discovery_dates(path, role)


@pytest.mark.parametrize("name", ["datadate", "published_at", "vintage_at"])
def test_accounting_knowledge_and_observation_bounds_are_checked(tmp_path, name):
    frame = pd.DataFrame(
        {c: pd.to_datetime(["2020-12-31"]) for c in ("datadate", "published_at", "vintage_at")}
    )
    frame[name] = pd.to_datetime(["2021-01-04"])
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery"):
        verify_discovery_dates(save(tmp_path, frame), "fundamentals")


def test_events_check_entitlement_date_not_later_scheduled_payment(tmp_path):
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-12-31"]),
            "pay_date": pd.to_datetime(["2021-01-15"]),
        }
    )
    assert verify_discovery_dates(save(tmp_path, frame), "events")
    frame["event_date"] = pd.to_datetime(["2021-01-04"])
    with pytest.raises(ArchivedInputBlocked):
        verify_discovery_dates(save(tmp_path, frame), "events")


def test_unavailable_statistics_never_infer_safe_dates(tmp_path):
    frame = pd.DataFrame({"101": [1.0]}, index=pd.to_datetime(["2020-12-31"]))
    with pytest.raises(ArchivedInputBlocked, match="bounds_unavailable"):
        verify_discovery_dates(save(tmp_path, frame, statistics=False), "opens")


def test_nested_nondate_columns_cannot_shift_the_checked_physical_column(tmp_path):
    frame = pd.DataFrame({"nested": [{"a": 1, "b": 2}]}, index=pd.to_datetime(["2021-01-04"]))
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery"):
        verify_discovery_dates(save(tmp_path, frame), "opens")


def test_local_future_session_cannot_be_shifted_before_boundary_in_utc(tmp_path):
    frame = pd.DataFrame(
        {"101": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2021-01-01T00:00:00+01:00")]),
    )
    with pytest.raises(ArchivedInputBlocked, match="date_type_unqualified"):
        verify_discovery_dates(save(tmp_path, frame), "opens")


def test_pre_cutoff_observation_with_future_availability_is_blocked(tmp_path):
    frame = pd.DataFrame(
        {"known_at": pd.to_datetime(["2021-01-04"])},
        index=pd.to_datetime(["2020-12-31"]),
    )
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery"):
        verify_discovery_dates(save(tmp_path, frame), "availability:close")


@pytest.mark.parametrize("role", ["unrecognized", "fundamentals", "events"])
def test_missing_date_contracts_block(tmp_path, role):
    with pytest.raises(ArchivedInputBlocked):
        verify_discovery_dates(save(tmp_path, pd.DataFrame({"value": [1.0]})), role)
