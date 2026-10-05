"""wrds_loader: the scoped and CIZ pulls, and how the username is found.

No WRDS connection is used or needed: a recording stub captures each query,
which is the part of this that can actually be wrong. What the database does
with correct SQL is not ours to test.

The username is never typed on a command line, where it would land in shell
history and session logs (#27 review): the pgpass file the wrds package writes
already holds it, in its fourth field.
"""

from __future__ import annotations

import pandas as pd
import pytest

from capstone import wrds_loader

# What main's legacy query selects.
LEGACY_COLUMNS = ["date", "permno", "ticker", "prc", "ret", "vol", "shrout", "cfacpr"]

# The CIZ query aliases back to legacy names and adds OHLC plus the share factor.
CIZ_COLUMNS = [*wrds_loader.CIZ_COLUMN_ALIASES.keys(), "permno", "ticker", "shrout"]


class RecordingConn:
    """Captures every query instead of executing it."""

    def __init__(self, columns=None, price_rows=None, delist_rows=None):
        self.queries: list[str] = []
        self._price = pd.DataFrame(price_rows or [], columns=columns or LEGACY_COLUMNS)
        self._delist = pd.DataFrame(delist_rows or [], columns=["date", "permno", "dlret"])

    def raw_sql(self, query, params=None, date_cols=None):
        self.queries.append(query)
        return self._delist.copy() if self._is_delist(query) else self._price.copy()

    @staticmethod
    def _is_delist(query: str) -> bool:
        return "dsedelist" in query or "stkdelists" in query

    @property
    def price_query(self) -> str:
        return next(q for q in self.queries if not self._is_delist(q))

    @property
    def delist_query(self) -> str:
        return next(q for q in self.queries if self._is_delist(q))


def ciz_conn(**kwargs) -> RecordingConn:
    return RecordingConn(columns=CIZ_COLUMNS, **kwargs)


# --- legacy: membership scoping ---------------------------------------------


def test_wide_pull_is_the_default_and_joins_no_membership():
    conn = RecordingConn()
    wrds_loader.crsp_daily(conn, "2022-01-01", "2023-12-31")
    assert "dsp500list" not in conn.price_query


def test_scoped_pull_joins_the_membership_table():
    conn = RecordingConn()
    wrds_loader.crsp_daily(conn, "2022-01-01", "2023-12-31", sp500_only=True)
    assert "crsp.dsp500list" in conn.price_query


def test_membership_join_is_date_bounded_not_an_id_list():
    """A permno IN-list would readmit a name for every year after it left."""
    conn = RecordingConn()
    wrds_loader.crsp_daily(conn, "2010-01-01", "2026-01-01", sp500_only=True)
    query = conn.price_query
    assert ">= s.start" in query
    assert "coalesce(s.ending" in query


def test_share_and_exchange_filters_survive_scoping():
    """Scoping must not quietly drop the common-share restriction."""
    conn = RecordingConn()
    wrds_loader.crsp_daily(conn, "2022-01-01", "2022-12-31", sp500_only=True)
    assert "n.shrcd in (10, 11)" in conn.price_query
    assert "n.exchcd in (1, 2, 3)" in conn.price_query


# --- CIZ ---------------------------------------------------------------------


def test_ciz_reads_the_v2_daily_table():
    conn = ciz_conn()
    wrds_loader.crsp_daily_ciz(conn, "2020-01-01", "2020-12-31")
    assert "crsp.dsf_v2" in conn.price_query


def test_ciz_aliases_every_column_back_to_the_legacy_name():
    """Callers consume legacy names; the alias is what keeps them working."""
    conn = ciz_conn()
    wrds_loader.crsp_daily_ciz(conn, "2022-01-01", "2022-12-31")
    query = conn.price_query
    for legacy, source in wrds_loader.CIZ_COLUMN_ALIASES.items():
        assert f"v.{source} as {legacy}" in query, f"{source} not aliased to {legacy}"


def test_ciz_applies_all_four_classification_filters():
    """Drop any one and you admit funds, REITs or ADRs - measured on real data."""
    conn = ciz_conn()
    wrds_loader.crsp_daily_ciz(conn, "2022-01-01", "2022-12-31")
    query = conn.price_query
    for column, value in wrds_loader.CIZ_COMMON_SHARE.items():
        assert f"v.{column} = '{value}'" in query
    assert "primaryexch in ('N', 'A', 'Q')" in query


def test_ciz_needs_no_names_table_join():
    """Classification is inline on dsf_v2, so dsenames is gone."""
    conn = ciz_conn()
    wrds_loader.crsp_daily_ciz(conn, "2022-01-01", "2022-12-31")
    assert "dsenames" not in conn.price_query


def test_ciz_scoped_pull_joins_the_v2_membership_table_on_spell_dates():
    conn = ciz_conn()
    wrds_loader.crsp_daily_ciz(conn, "2010-01-01", "2025-12-31", sp500_only=True)
    query = conn.price_query
    assert "crsp.dsp500list_v2" in query
    assert "mbrstartdt" in query and "mbrenddt" in query


def test_ciz_still_merges_the_delisting_return():
    """dlyret does NOT contain it.

    Sampled delisting dates showed dlyret differing from stkdelists.delret in 11
    of 12 cases, with dlydelflg='N'. Dropping this merge would silently remove
    the survivorship correction.
    """
    conn = ciz_conn()
    wrds_loader.crsp_daily_ciz(conn, "2015-01-01", "2020-12-31")
    query = conn.delist_query
    assert "crsp.stkdelists" in query
    assert "d.delret as dlret" in query


def test_ciz_delisting_rows_are_limited_to_permnos_the_price_pull_admitted():
    """The merge is outer, so an unfiltered delisting pull injects strangers."""
    day = pd.Timestamp("2016-03-31")
    member = dict.fromkeys(CIZ_COLUMNS, 1.0) | {"date": day, "permno": 10001, "ticker": "AAA"}
    conn = ciz_conn(
        price_rows=[member],
        delist_rows=[
            {"date": day, "permno": 10001, "dlret": -0.4},
            {"date": day, "permno": 99999, "dlret": -0.9},
        ],
    )
    frame = wrds_loader.crsp_daily_ciz(conn, "2016-01-01", "2016-12-31")
    assert set(frame["permno"]) == {10001}


def test_ciz_membership_returns_the_same_column_names_as_legacy():
    """Callers must not branch on which family produced the membership frame."""
    conn = ciz_conn()
    wrds_loader.sp500_members_ciz(conn, "2010-01-01", "2025-12-31")
    query = conn.queries[0]
    assert "crsp.dsp500list_v2" in query
    assert "mbrstartdt as mbr_start" in query
    assert "mbrenddt as mbr_end" in query


# --- cache keys --------------------------------------------------------------


def test_scope_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """A wide and a scoped pull under one name must not serve each other."""
    seen: list[str] = []
    monkeypatch.setattr(wrds_loader, "_cache_path", lambda n: (seen.append(n), tmp_path / n)[1])

    wrds_loader.cached_crsp_daily(ciz_conn(), "2022-01-01", "2022-12-31", name="probe")
    wrds_loader.cached_crsp_daily(
        ciz_conn(), "2022-01-01", "2022-12-31", name="probe", sp500_only=True
    )
    assert len(set(seen)) == 2, f"scoped and wide pulls shared a cache key: {seen}"


def test_table_family_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """Legacy and CIZ differ by a year; serving one for the other is silent."""
    seen: list[str] = []
    monkeypatch.setattr(wrds_loader, "_cache_path", lambda n: (seen.append(n), tmp_path / n)[1])

    wrds_loader.cached_crsp_daily(ciz_conn(), "2020-01-01", "2020-12-31", name="x", ciz=True)
    wrds_loader.cached_crsp_daily(RecordingConn(), "2020-01-01", "2020-12-31", name="x", ciz=False)
    assert len(set(seen)) == 2, f"CIZ and legacy shared a cache key: {seen}"


def test_ciz_cache_key_cannot_collide_with_a_pre_existing_legacy_cache(tmp_path, monkeypatch):
    """Every cache written before this change sits on the bare `crsp_<name>` path.

    If CIZ used that same bare path, the new default would hand back year-stale
    legacy rows as though they were current — the precise failure the cache key
    exists to prevent.
    """
    seen: list[str] = []
    monkeypatch.setattr(wrds_loader, "_cache_path", lambda n: (seen.append(n), tmp_path / n)[1])

    wrds_loader.cached_crsp_daily(ciz_conn(), "2020-01-01", "2020-12-31", name="legacy_era")
    assert seen == ["crsp_legacy_era_ciz"]
    assert "crsp_legacy_era" not in seen, "CIZ landed on the pre-existing legacy cache path"


def test_cached_pull_defaults_to_ciz(tmp_path, monkeypatch):
    """Legacy is a year stale, so nobody should get it by accident."""
    monkeypatch.setattr(wrds_loader, "_cache_path", lambda n: tmp_path / n)
    conn = ciz_conn()
    wrds_loader.cached_crsp_daily(conn, "2020-01-01", "2020-12-31", name="default_probe")
    assert "crsp.dsf_v2" in conn.price_query


@pytest.mark.parametrize("scoped", [True, False])
def test_ciz_returns_the_same_columns_either_way(scoped):
    """Downstream code must not need to know how the pull was scoped."""
    conn = ciz_conn()
    frame = wrds_loader.crsp_daily_ciz(conn, "2022-01-01", "2022-12-31", sp500_only=scoped)
    assert set(CIZ_COLUMNS) <= set(frame.columns)
    assert "dlret" in frame.columns


# --- username lookup -------------------------------------------------------


def _use_pgpass(monkeypatch, path):
    monkeypatch.setenv("PGPASSFILE", str(path))
    monkeypatch.delenv("WRDS_USERNAME", raising=False)


def test_username_comes_from_the_wrds_line_of_pgpass(tmp_path, monkeypatch):
    pgpass = tmp_path / "pgpass.conf"
    pgpass.write_text(
        "# saved by the wrds package\n"
        "otherhost:5432:db:someone_else:pw\n"
        + wrds_loader.WRDS_HOST
        + r":9737:wrds:test\:user:pa\:ss"
        + "\n"
    )
    _use_pgpass(monkeypatch, pgpass)
    assert wrds_loader.wrds_username() == "test:user"


def test_explicit_then_env_override_pgpass(tmp_path, monkeypatch):
    pgpass = tmp_path / "pgpass.conf"
    pgpass.write_text(f"{wrds_loader.WRDS_HOST}:9737:wrds:from_file:pw\n")
    _use_pgpass(monkeypatch, pgpass)
    monkeypatch.setenv("WRDS_USERNAME", "from_env")
    assert wrds_loader.wrds_username() == "from_env"
    assert wrds_loader.wrds_username("explicit") == "explicit"


def test_no_pgpass_and_no_env_means_no_username(tmp_path, monkeypatch):
    _use_pgpass(monkeypatch, tmp_path / "missing")
    assert wrds_loader.wrds_username() is None
