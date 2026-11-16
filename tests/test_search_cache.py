"""The search rows cached on disk (`capstone.pipeline.search_cache`).

Synthetic CRSP-shaped rows in a temporary directory: no CRSP, no data_cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.factors.evaluation import HoldoutError, build_panel
from capstone.pipeline import search_cache
from capstone.pipeline.rows import rows_digest, sorted_rows
from capstone.pipeline.search_cache import CachedRows


def market(n_stocks=8, n_days=120, seed=1, start="2015-01-02") -> pd.DataFrame:
    """Synthetic daily rows shaped like the shared file (`evaluation.COLUMNS`)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    ret = rng.normal(0, 0.02, (n_days, n_stocks))
    price = 50 * np.cumprod(1 + ret, axis=0)
    volume = np.exp(rng.normal(13, 0.5, (n_days, n_stocks)))
    rows = []
    for j in range(n_stocks):
        for t, date in enumerate(dates):
            p = price[t, j]
            rows.append(
                {
                    "date": date,
                    "permno": 10000 + j,
                    "in_universe": True,
                    "dlyret": ret[t, j],
                    "dlyretmissflg": None,
                    "dlydelflg": "N",
                    "delactiontype": None,
                    "dlyopen": p,
                    "dlyhigh": p * 1.01,
                    "dlylow": p * 0.99,
                    "dlyprc": p,
                    "dlyvol": volume[t, j],
                    "shrout": 1000.0,
                    "dlycap": p * 1000.0,
                    "dlycumfacpr": 1.0,
                    "dlycumfacshr": 1.0,
                    "dlybid": p * 0.999,
                    "dlyask": p * 1.001,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def rows():
    return market()


@pytest.fixture
def shared(monkeypatch, rows):
    class Shared:
        version = "v1"
        syncs = 0
        calls = []
        frame = rows

        def sync(self):
            self.syncs += 1

        def load(self, name, columns, start, end, auto_sync):
            assert auto_sync is False
            self.calls.append((name, columns, start, end))
            return self.frame[columns].copy()

    source = Shared()
    monkeypatch.setattr(search_cache.sd, "sync", source.sync)
    monkeypatch.setattr(search_cache.sd, "data_version", lambda: source.version)
    monkeypatch.setattr(search_cache.sd, "load", source.load)
    return source


def cache(tmp_path, shared, columns=None, name="daily"):
    return CachedRows(name, list(shared.frame.columns) if columns is None else columns, tmp_path)


def test_a_cached_read_gives_the_same_rows_and_data_version(tmp_path, rows, shared):
    first = cache(tmp_path, shared)("2000-01-03", "2020-12-31")
    again = cache(tmp_path, shared)
    second = again("2000-01-03", "2020-12-31")
    assert len(shared.calls) == 1 and (again.hits, again.misses) == (1, 0)
    assert shared.syncs == 2
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert rows_digest(sorted_rows(second)) == rows_digest(sorted_rows(first))
    a = build_panel(sorted_rows(first), "x")
    b = build_panel(sorted_rows(second), "x")
    assert a.universe.equals(b.universe)


@pytest.mark.parametrize("change", ["version", "start", "end", "file", "columns", "order"])
def test_a_changed_key_rebuilds(tmp_path, shared, change):
    cached = cache(tmp_path, shared)
    cached("2000-01-03", "2020-12-31")
    start, end = "2000-01-03", "2020-12-31"
    if change == "version":
        shared.version = "v2"  # even an existing CachedRows must use the new shared version
        shared.frame = shared.frame.assign(dlyret=0.5)
    elif change == "start":
        start = "2001-01-02"
    elif change == "end":
        end = "2019-12-31"
    elif change == "file":
        cached = cache(tmp_path, shared, name="other_daily")
    elif change == "columns":
        cached = cache(tmp_path, shared, columns=["date", "permno"])
    else:
        cached = cache(tmp_path, shared, columns=list(reversed(shared.frame.columns)))
    out = cached(start, end)
    assert len(shared.calls) == 2
    assert list(out.columns) == cached.columns
    if change == "version":
        assert (out.dlyret == 0.5).all()


@pytest.mark.parametrize("corrupt", ["rows", "json", "parquet", "dtype"])
def test_a_corrupted_file_is_rebuilt_not_served(tmp_path, shared, corrupt):
    cached = cache(tmp_path, shared)
    cached(None, "2020-12-31")
    (data,) = tmp_path.glob("*.parquet")
    (meta,) = tmp_path.glob("*.json")
    if corrupt == "rows":
        shared.frame.head(5).to_parquet(data)
    elif corrupt == "json":
        meta.write_text("{", encoding="utf-8")
    elif corrupt == "parquet":
        data.write_bytes(b"broken")
    else:
        shared.frame.astype({"permno": "float64"}).to_parquet(data)
    out = cached(None, "2020-12-31")
    assert len(shared.calls) == 2
    pd.testing.assert_frame_equal(out, shared.frame, check_exact=True)


@pytest.mark.parametrize("end", [None, "", pd.NaT, float("nan"), "2021-01-01", "2020-12-31 23:59"])
def test_the_cache_refuses_unbounded_or_holdout_end_before_io(tmp_path, shared, end):
    with pytest.raises(HoldoutError):
        cache(tmp_path, shared)("1990-01-01", end)
    assert shared.calls == [] and shared.syncs == 0 and not list(tmp_path.iterdir())


def test_exact_discovery_boundary_is_allowed(tmp_path, shared):
    cache(tmp_path, shared)(None, "2020-12-31")
    assert len(shared.calls) == 1


@pytest.mark.parametrize("date", ["2021-01-01", None])
def test_loader_dates_are_checked_before_writing(tmp_path, shared, date):
    shared.frame = pd.DataFrame({"date": pd.to_datetime([date]), "permno": [1]})
    with pytest.raises(HoldoutError):
        cache(tmp_path, shared)(None, "2020-12-31")
    assert not list(tmp_path.iterdir())


def test_even_a_matching_cache_digest_does_not_authorize_holdout_rows(tmp_path, shared):
    import json

    cache(tmp_path, shared)(None, "2020-12-31")
    (data,) = tmp_path.glob("*.parquet")
    (meta,) = tmp_path.glob("*.json")
    future = shared.frame.assign(date=pd.Timestamp("2021-01-01"))
    future.to_parquet(data)
    loaded = pd.read_parquet(data)
    meta.write_text(
        json.dumps({"digest": rows_digest(loaded), "dtypes": list(map(str, loaded.dtypes))})
    )
    with pytest.raises(HoldoutError):
        cache(tmp_path, shared)(None, "2020-12-31")
    assert len(shared.calls) == 1


def test_a_frame_whose_dtypes_change_is_not_cached(tmp_path, shared):
    shared.frame = pd.DataFrame(
        {
            "date": pd.Series(["2020-01-01"], dtype="datetime64[s]"),
            "permno": pd.Series([1], dtype=object),
        }
    )
    cached = cache(tmp_path, shared)
    cached(None, "2020-12-31")
    out = cached(None, "2020-12-31")
    pd.testing.assert_frame_equal(out, shared.frame, check_exact=True)
    assert len(shared.calls) == 2 and not list(tmp_path.iterdir())


def test_arbitrary_loader_and_source_version_are_no_longer_accepted(tmp_path):
    with pytest.raises(TypeError, match="shared file"):
        CachedRows(lambda start, end: pd.DataFrame(), "v1", tmp_path)


def test_changed_shared_version_during_load_is_not_cached(tmp_path, shared, monkeypatch):
    load = shared.load

    def changed(*args, **kwargs):
        shared.version = "v2"
        return load(*args, **kwargs)

    monkeypatch.setattr(search_cache.sd, "load", changed)
    out = cache(tmp_path, shared)(None, "2020-12-31")
    pd.testing.assert_frame_equal(out, shared.frame)
    assert not list(tmp_path.iterdir())


def test_partial_is_removed_when_publication_fails(tmp_path, shared, monkeypatch):
    def fail(*args):
        raise PermissionError("busy")

    monkeypatch.setattr(search_cache.os, "replace", fail)
    with pytest.raises(PermissionError, match="busy"):
        cache(tmp_path, shared)(None, "2020-12-31")
    assert not list(tmp_path.iterdir())


def test_the_default_directory_is_under_data_cache_or_the_override(monkeypatch, tmp_path):
    monkeypatch.delenv(search_cache.CACHE_ENV, raising=False)
    default = search_cache.default_directory()
    assert default.parent.name == "data_cache"
    monkeypatch.setenv(search_cache.CACHE_ENV, str(tmp_path))
    assert search_cache.default_directory() == tmp_path
