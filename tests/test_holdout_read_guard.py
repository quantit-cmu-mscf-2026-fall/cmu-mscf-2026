"""The shared-data read guard, on temporary synthetic files; no real data is touched.

`CAPSTONE_HOLDOUT_GUARD` is switched on per test. Every protocol names a `synthetic:`
dataset, so no test can open the real dataset.
"""

import hashlib
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest
from holdout_fakes import approve, make_protocol, open_look

from capstone import holdout_authority as authority
from capstone import shared_data as sd
from capstone import wrds_loader


@pytest.fixture(autouse=True)
def guard_on(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_HOLDOUT_GUARD", "on")
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    cache = tmp_path / "synthetic-cache"
    cache.mkdir()
    path = cache / "synthetic_daily.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-12-31", "2021-01-04", "2025-12-31"]),
            "return": [0.01, 0.02, 0.03],
        }
    ).to_parquet(path, index=False)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    sd.write_manifest(cache / sd.MANIFEST, {path.name: checksum})
    monkeypatch.setenv("CAPSTONE_CACHE_DIR", str(cache))
    monkeypatch.setenv("CAPSTONE_DATA_DIR", str(tmp_path / "never-a-real-drive"))

    def no_sync(*args, **kwargs):
        raise AssertionError("no sync is authorized by these synthetic tests")

    monkeypatch.setattr(sd, "sync", no_sync)
    protocol = make_protocol(
        sources={"synthetic_daily": checksum}, source_version=sd.data_version()
    )
    return protocol, tmp_path / "authority", path


def test_the_guard_is_off_by_default(monkeypatch):
    monkeypatch.delenv("CAPSTONE_HOLDOUT_GUARD")
    assert authority.guard_enabled() is False
    assert authority.guard_shared_read("sp500_daily_1990_2025", None, None, "date", True, None) is (
        False
    )
    authority.require_discovery_query(None)  # a no-op: nothing is sealed until adopted
    monkeypatch.setenv("CAPSTONE_HOLDOUT_GUARD", "0")
    assert authority.guard_enabled() is False


def test_an_authorized_look_reads_each_source_once_within_its_dates(synthetic_source):
    protocol, root, _ = synthetic_source
    approve(protocol, root)
    with open_look(protocol, root) as access:
        frame = sd.load(
            "synthetic_daily",
            start="2021-01-04",
            end="2025-12-31",
            auto_sync=False,
            holdout_access=access,
        )
        assert frame["return"].tolist() == [0.02, 0.03]
        with pytest.raises(authority.HoldoutSealed, match="already materialized"):
            sd.load("synthetic_daily", end="2025-12-31", auto_sync=False, holdout_access=access)
    assert authority.status(protocol, synthetic_root=root)["state"] == "SPENT"
    with pytest.raises(authority.HoldoutSealed, match="active durable"):
        sd.load("synthetic_daily", end="2025-12-31", auto_sync=False, holdout_access=access)


def test_source_version_checksum_scope_and_refresh_refuse(synthetic_source):
    protocol, root, path = synthetic_source
    approve(protocol, root)
    with open_look(protocol, root, end="2021-01-04") as access:
        with pytest.raises(authority.HoldoutSealed, match="dates"):
            sd.load("synthetic_daily", end="2025-12-31", auto_sync=False, holdout_access=access)
        with pytest.raises(authority.HoldoutSealed, match="refresh"):
            sd.load("synthetic_daily", end="2021-01-04", holdout_access=access)
        path.write_bytes(path.read_bytes() + b"tampered")
        with pytest.raises(authority.HoldoutSealed, match="bytes changed"):
            sd.load("synthetic_daily", end="2021-01-04", auto_sync=False, holdout_access=access)
    assert authority.status(protocol, synthetic_root=root)["state"] == "SPENT"


def test_actual_shared_version_changes_cannot_open_authorized_source(synthetic_source):
    protocol, root, path = synthetic_source
    approve(protocol, root)
    sd.write_manifest(path.parent / sd.MANIFEST, {path.name: "0" * 64})
    with open_look(protocol, root) as access:
        with pytest.raises(authority.HoldoutSealed, match="version changed"):
            sd.load("synthetic_daily", end="2025-12-31", auto_sync=False, holdout_access=access)


def test_a_copied_receipt_or_replaced_scope_confers_no_authority(synthetic_source):
    protocol, root, _ = synthetic_source
    approve(protocol, root)
    with open_look(protocol, root, end="2021-01-04") as access:
        with pytest.raises(authority.HoldoutSealed, match="active durable"):
            sd.load(
                "synthetic_daily",
                end="2021-01-04",
                auto_sync=False,
                holdout_access=deepcopy(access),
            )
        with pytest.raises(authority.HoldoutSealed, match="active durable"):
            authority.validate_access(replace(access))
        with pytest.raises(authority.HoldoutSealed, match="frozen dates"):
            sd.load("synthetic_daily", end="2025-12-31", auto_sync=False, holdout_access=access)


@pytest.mark.parametrize(
    "end,columns,date_column",
    [
        (None, None, "date"),
        ("2025-12-31", ["date", "permno"], "date"),
        ("2020-12-31", ["dlyret"], "datadate"),
    ],
)
def test_default_protected_source_is_sealed_before_any_sync_or_metadata(
    monkeypatch, end, columns, date_column
):
    def forbidden(*args, **kwargs):
        raise AssertionError("sealed read reached sync/cache/metadata")

    for name in ("sync", "cache_dir", "data_version"):
        monkeypatch.setattr(sd, name, forbidden)
    with pytest.raises(authority.HoldoutSealed):
        sd.load("sp500_daily_1990_2025", columns=columns, end=end, date_column=date_column)


def test_discovery_reads_stay_open_and_unregistered_files_fail_closed(synthetic_source):
    _, _, path = synthetic_source
    frame = sd.load("synthetic_daily", end="2020-12-31", auto_sync=False)
    assert frame["return"].tolist() == [0.01]
    with pytest.raises(authority.HoldoutSealed):
        sd.load("synthetic_daily", end="2025-12-31", auto_sync=False)
    with pytest.raises(authority.HoldoutSealed):
        sd.load("synthetic_daily", auto_sync=False)


def test_calendar_only_capability_returns_unique_session_labels(synthetic_source):
    _, _, path = synthetic_source
    calendar = path.with_name("sp500_daily_1990_2025.parquet")
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-04", "2021-01-05"]),
            "dlyret": [9, 9, 9, 9],
        }
    ).to_parquet(calendar, index=False)
    frame = sd.load("sp500_daily_1990_2025", columns=["date"], end="2021-01-31", auto_sync=False)
    assert list(frame) == ["date"]
    assert frame.date.tolist() == [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-04")]
    with pytest.raises(authority.HoldoutSealed):
        sd.load(
            "sp500_daily_1990_2025", columns=["date", "dlyret"], end="2021-01-31", auto_sync=False
        )


@pytest.mark.parametrize("function", ["crsp_daily", "sp500_members", "cached_crsp_daily"])
def test_wrds_and_cache_guard_runs_before_queries_or_cache(function, monkeypatch):
    class ForbiddenConnection:
        def raw_sql(self, *args, **kwargs):
            raise AssertionError("protected SQL was sent")

    def forbidden(*args, **kwargs):
        raise AssertionError("protected cache was opened")

    monkeypatch.setattr(wrds_loader, "_cache_path", forbidden)
    kwargs = {"name": "synthetic-probe"} if function == "cached_crsp_daily" else {}
    with pytest.raises(authority.HoldoutSealed):
        getattr(wrds_loader, function)(ForbiddenConnection(), "2020-01-01", "2025-12-31", **kwargs)


@pytest.mark.parametrize("end", ["01/01/2025", "2020-12-31T23:59:59-05:00"])
def test_noncanonical_dates_cannot_bypass_protected_periods(end):
    with pytest.raises(authority.HoldoutSealed):
        authority.require_discovery_query(end)
    with pytest.raises(authority.HoldoutSealed):
        authority.guard_shared_read("sp500_daily_1990_2025", ["dlyret"], end, "date", False, None)


@pytest.mark.parametrize(
    "name", ["SP500_DAILY_1990_2025", "../sp500_daily_1990_2025", "..\\sp500_daily_1990_2025"]
)
def test_case_or_path_aliases_cannot_bypass_the_real_dataset_guard(name):
    with pytest.raises(authority.HoldoutSealed):
        authority.guard_shared_read(name, ["dlyret"], "2025-12-31", "date", False, None)


def test_synthetic_authority_never_opens_a_real_source(synthetic_source):
    protocol, root, _ = synthetic_source
    approve(protocol, root)
    with open_look(protocol, root, end="2021-01-04") as access:
        with pytest.raises(authority.HoldoutSealed, match="real dataset"):
            sd.load(
                "sp500_daily_1990_2025", end="2021-01-04", auto_sync=False, holdout_access=access
            )


def test_a_protected_source_needs_its_authoritative_date_column(synthetic_source):
    protocol, root, _ = synthetic_source
    approve(protocol, root)
    with pytest.raises(authority.HoldoutSealed, match="authoritative date"):
        authority.guard_shared_read("comp_fundq", None, "2020-12-31", "date", False, None)
    with pytest.raises(ValueError, match="unsupported"):
        sd.load("synthetic_daily", end="2020-12-31", auto_sync=False, date_column="ret")


def test_preexisting_wrds_cache_filters_dates_before_frame_materialization(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    path = tmp_path / "synthetic-old-cache.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-31", "2020-06-30", "2021-01-04", "2025-12-31"]),
            "ret": [1, 2, 3, 4],
        }
    ).to_parquet(path, index=False)
    monkeypatch.setattr(wrds_loader, "_cache_path", lambda name: path)
    original = pq.read_table
    seen = []

    def bounded_read(source, **kwargs):
        assert kwargs["filters"] == [
            ("date", ">=", pd.Timestamp("2020-01-01")),
            ("date", "<=", pd.Timestamp("2020-12-31")),
        ]
        frame = original(source, **kwargs)
        seen.append(frame.num_rows)
        return frame

    monkeypatch.setattr(pq, "read_table", bounded_read)
    result = wrds_loader.cached_crsp_daily(None, "2020-01-01", "2020-12-31", name="synthetic")
    assert seen == [1]
    assert result.ret.tolist() == [2]


@pytest.mark.parametrize("kind", ["missing", "string", "timezone"])
def test_preexisting_wrds_cache_requires_bounded_authoritative_date_schema(
    tmp_path, monkeypatch, kind
):
    import pyarrow.parquet as pq

    path = tmp_path / "synthetic-malformed-cache.parquet"
    frame = pd.DataFrame({"ret": [1]})
    if kind == "string":
        frame["date"] = ["2025-12-31"]
    elif kind == "timezone":
        frame["date"] = pd.to_datetime(["2025-12-31"], utc=True)
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(wrds_loader, "_cache_path", lambda name: path)

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed cache reached row materialization")

    monkeypatch.setattr(pq, "read_table", forbidden)
    with pytest.raises(authority.HoldoutSealed, match="date"):
        wrds_loader.cached_crsp_daily(None, "2020-01-01", "2020-12-31", name="synthetic")


def test_without_the_guard_the_cache_and_loader_behave_as_before(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPSTONE_HOLDOUT_GUARD")
    path = tmp_path / "synthetic-cache.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2025-12-31"]), "ret": [4]}).to_parquet(path)
    monkeypatch.setattr(wrds_loader, "_cache_path", lambda name: path)
    frame = wrds_loader.cached_crsp_daily(None, "2020-01-01", "2020-12-31", name="synthetic")
    assert frame.ret.tolist() == [4]  # unchanged behavior: the whole cache, not date-bounded
