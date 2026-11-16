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


class Loader:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def __call__(self, start, end):
        self.calls.append((start, end))
        return self.rows


@pytest.fixture(scope="module")
def rows():
    return market()


def test_a_cached_read_gives_the_same_rows_and_data_version(tmp_path, rows):
    loader = Loader(rows)
    first = CachedRows(loader, "v1", tmp_path)("2000-01-03", "2007-12-31")
    again = CachedRows(loader, "v1", tmp_path)
    second = again("2000-01-03", "2007-12-31")
    assert len(loader.calls) == 1 and (again.hits, again.misses) == (1, 0)
    assert rows_digest(sorted_rows(second)) == rows_digest(sorted_rows(first))
    assert rows_digest(second) == rows_digest(rows)
    a = build_panel(sorted_rows(first), "x")
    b = build_panel(sorted_rows(second), "x")
    assert a.universe.equals(b.universe)


@pytest.mark.parametrize(
    "change",
    [
        dict(source_version="v2"),
        dict(start="2001-01-02"),
        dict(end="2006-12-29"),
    ],
)
def test_a_changed_key_rebuilds(tmp_path, rows, change):
    loader = Loader(rows)
    CachedRows(loader, "v1", tmp_path)("2000-01-03", "2007-12-31")
    args = dict(source_version="v1", start="2000-01-03", end="2007-12-31") | change
    cached = CachedRows(loader, args["source_version"], tmp_path)
    cached(args["start"], args["end"])
    assert len(loader.calls) == 2 and cached.misses == 1


def test_a_corrupted_file_is_rebuilt_not_served(tmp_path, rows):
    loader = Loader(rows)
    cached = CachedRows(loader, "v1", tmp_path)
    cached("2000-01-03", "2007-12-31")
    (data,) = tmp_path.glob("*.parquet")
    rows.head(5).to_parquet(data)  # other rows under the same key
    out = CachedRows(loader, "v1", tmp_path)("2000-01-03", "2007-12-31")
    assert len(loader.calls) == 2 and rows_digest(out) == rows_digest(rows)


def test_the_cache_refuses_the_holdout_before_reading_anything(tmp_path, rows):
    loader = Loader(rows)
    with pytest.raises(HoldoutError):
        CachedRows(loader, "v1", tmp_path)("1990-01-01", "2021-01-04")
    assert loader.calls == [] and not list(tmp_path.iterdir())


def test_the_default_directory_is_under_data_cache_or_the_override(monkeypatch, tmp_path):
    monkeypatch.delenv(search_cache.CACHE_ENV, raising=False)
    default = search_cache.default_directory()
    assert default.parent.name == "data_cache"
    monkeypatch.setenv(search_cache.CACHE_ENV, str(tmp_path))
    assert search_cache.default_directory() == tmp_path
