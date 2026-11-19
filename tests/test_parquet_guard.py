import datetime
import json
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from capstone.pipeline.parquet_guard import (
    ArchivedInputBlocked,
    discovery_filter,
    verify_discovery_dates,
)


def save(tmp_path, frame, *, statistics=True):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pandas(frame), path, write_statistics=statistics)
    return path


@pytest.mark.parametrize("role", ["field:returns", "opens", "company", "symbols"])
def test_discovery_wide_frames_need_no_value_read(tmp_path, monkeypatch, role):
    frame = pd.DataFrame({"101": [1.0, 2.0]}, index=pd.to_datetime(["2020-12-30", "2020-12-31"]))
    path = save(tmp_path, frame)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("materialized rows"))
    monkeypatch.setattr(pq, "read_table", lambda *a, **kw: pytest.fail("materialized rows"))
    monkeypatch.setattr(ds, "dataset", lambda *a, **kw: pytest.fail("materialized rows"))
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


@pytest.mark.parametrize(
    ("role", "extra"),
    [
        ("field:rdq", "101"),
        ("universe", "date"),
        ("company", "nameendt"),
        ("events", "announce_date"),
        ("events", "ex_date"),
        ("fundamentals", "rdq"),
        ("opens", "pay_date"),  # The scheduled-payment exception belongs only to events.
    ],
)
def test_nonrole_temporal_columns_cannot_hide_boundary_rows(tmp_path, role, extra):
    frame = pd.DataFrame(
        {
            name: pd.to_datetime(["2020-12-31"])
            for name in ("event_date", "datadate", "published_at", "vintage_at")
        },
        index=pd.DatetimeIndex(["2020-12-31"], name="formed"),
    )
    frame[extra] = pd.to_datetime(["2021-01-01"])
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery_boundary"):
        verify_discovery_dates(save(tmp_path, frame), role)


def test_forged_pandas_index_does_not_hide_another_date(tmp_path):
    table = pa.table(
        {"safe": pd.to_datetime(["2020-12-31"]), "date": pd.to_datetime(["2021-01-01"])}
    ).replace_schema_metadata({b"pandas": json.dumps({"index_columns": ["safe"]}).encode()})
    path = tmp_path / "forged.parquet"
    pq.write_table(table, path)
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery_boundary"):
        verify_discovery_dates(path, "universe")


@pytest.mark.parametrize("dtype", [pa.date32(), pa.date64(), pa.timestamp("us")])
def test_additional_arrow_temporal_types_are_checked(tmp_path, dtype):
    table = pa.Table.from_pandas(pd.DataFrame(index=pd.DatetimeIndex(["2020-12-31"])))
    table = table.append_column("extra", pa.array([datetime.datetime(2021, 1, 1)], type=dtype))
    path = tmp_path / "typed.parquet"
    pq.write_table(table, path)
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery_boundary"):
        verify_discovery_dates(path, "opens")


def test_extra_timezone_aware_dates_still_need_qualification(tmp_path):
    frame = pd.DataFrame(
        {"other": pd.to_datetime(["2020-12-31"], utc=True)},
        index=pd.DatetimeIndex(["2020-12-31"]),
    )
    with pytest.raises(ArchivedInputBlocked, match="date_type_unqualified"):
        verify_discovery_dates(save(tmp_path, frame), "opens")


def test_a_later_row_group_cannot_escape_boundary_check(tmp_path):
    frame = pd.DataFrame({"101": [1.0, 2.0]}, index=pd.to_datetime(["2020-12-31", "2021-01-01"]))
    path = tmp_path / "groups.parquet"
    pq.write_table(pa.Table.from_pandas(frame), path, row_group_size=1)
    assert pq.ParquetFile(path).metadata.num_row_groups == 2
    with pytest.raises(ArchivedInputBlocked, match="crosses_discovery_boundary"):
        verify_discovery_dates(path, "opens")


@pytest.mark.parametrize("missing_max", [False, True])
def test_incomplete_or_missing_maximum_is_never_safe(tmp_path, monkeypatch, missing_max):
    path = save(tmp_path, pd.DataFrame(index=pd.DatetimeIndex(["2020-12-31"])))
    file = pq.ParquetFile(path)
    group = file.metadata.row_group(0)
    column = group.column(0)
    # Real parquet footer wrapper: exercise an unavailable bound independently
    # of an absent statistics object (write_statistics=False, covered above).
    stats = SimpleNamespace(null_count=0, has_min_max=missing_max, max=pd.NaT)
    replacement = SimpleNamespace(path_in_schema=column.path_in_schema, statistics=stats)
    metadata = SimpleNamespace(
        num_rows=1,
        num_row_groups=1,
        row_group=lambda i: SimpleNamespace(num_rows=1, column=lambda j: replacement),
    )
    monkeypatch.setattr(
        pq,
        "ParquetFile",
        lambda *a, **kw: SimpleNamespace(
            schema=file.schema, schema_arrow=file.schema_arrow, metadata=metadata
        ),
    )
    reason = "crosses_discovery_boundary" if missing_max else "bounds_unavailable"
    with pytest.raises(ArchivedInputBlocked, match=reason):
        verify_discovery_dates(path, "opens")


@pytest.mark.parametrize("future", [False, True])
def test_nested_date_leaf_is_inspected_and_filtered(tmp_path, future):
    frame = pd.DataFrame(
        {"nested": [{"date": datetime.datetime(2021 if future else 2020, 1, 1)}]},
        index=pd.DatetimeIndex(["2020-12-31"]),
    )
    path = save(tmp_path, frame)
    if future:
        with pytest.raises(ArchivedInputBlocked, match="crosses_discovery_boundary"):
            verify_discovery_dates(path, "opens")
    else:
        evidence = verify_discovery_dates(path, "opens")
        assert "nested.date" in evidence["date_columns"]
        table = pa.Table.from_pandas(frame)
        expression = discovery_filter(path, json.loads(json.dumps(evidence)))
        assert ds.dataset(table).to_table(filter=expression).num_rows == 1
        frame.at[frame.index[0], "nested"] = {"date": datetime.datetime(2021, 1, 1)}
        assert ds.dataset(pa.Table.from_pandas(frame)).to_table(filter=expression).num_rows == 0


def test_repeated_temporal_leaf_cannot_be_silently_ignored(tmp_path):
    frame = pd.DataFrame(
        {"dates": [[datetime.datetime(2021, 1, 1)]]},
        index=pd.DatetimeIndex(["2020-12-31"]),
    )
    with pytest.raises(ArchivedInputBlocked, match="date_type_unqualified"):
        verify_discovery_dates(save(tmp_path, frame), "opens")


def test_filter_covers_every_date_leaf_and_retains_nulls_and_payment_exception(tmp_path):
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-12-31", None, "2020-12-30"]),
            "other": [datetime.date(2020, 12, 31), None, datetime.date(2020, 12, 30)],
            "pay_date": pd.to_datetime(["2021-02-01"] * 3),
            "known_at": pd.Series([pd.NaT] * 3, dtype="datetime64[ns]"),
        }
    )
    path = save(tmp_path, frame)
    evidence = verify_discovery_dates(path, "events")
    assert set(evidence["date_columns"]) == {"event_date", "other", "known_at"}
    expression = discovery_filter(path, evidence)
    pd.testing.assert_frame_equal(pq.read_table(path, filters=expression).to_pandas(), frame)
    frame.loc[2, "other"] = datetime.date(2021, 1, 1)
    filtered = ds.dataset(pa.Table.from_pandas(frame)).to_table(filter=expression).to_pandas()
    pd.testing.assert_frame_equal(filtered, frame.iloc[:2])


@pytest.mark.parametrize("kind", ["list_view", "large_list_view"])
def test_view_typed_date_lists_are_refused(tmp_path, kind):
    view = getattr(pa, kind)(pa.date32())
    table = pa.table(
        {
            "event_date": [datetime.date(2020, 1, 1)],
            "x": pa.array([[datetime.date(2023, 1, 1)]], type=view),
        }
    )
    path = tmp_path / "input.parquet"
    pq.write_table(table, path)
    with pytest.raises(ArchivedInputBlocked, match="archive_date_type_unqualified"):
        verify_discovery_dates(path, "events")
