"""Tests for capstone.agent.tools.crsp.

No live WRDS connection: a hand-rolled fake records the query it was handed and
returns a frame we built ourselves. Every fixture row is invented — CRSP data
may not be redistributed and this repository is public — so the PERMNOs are
out-of-range numbers and the returns are round.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from capstone.agent.tools import crsp
from capstone.agent.tools.crsp import RETURN_COLUMNS, get_crsp_returns


class FakeConn:
    """Stands in for a wrds.Connection, recording how it was called."""

    def __init__(self, frame: pd.DataFrame | None = None, error: Exception | None = None):
        self._frame = _rows() if frame is None else frame
        self._error = error
        self.calls: list[dict] = []
        self.closed = False

    def raw_sql(self, query, params=None, date_cols=None):
        self.calls.append({"query": query, "params": params, "date_cols": date_cols})
        if self._error is not None:
            raise self._error
        return self._frame.copy()

    def close(self):
        self.closed = True


def _rows() -> pd.DataFrame:
    """Two invented securities over two days, deliberately out of order.

    Mirrors `crsp_daily`'s full column set so the tool has something to narrow.
    """
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02"]),
            "permno": [90001.0, 90002.0, 90001.0],
            "ticker": ["AAA", "BBB", "AAA"],
            "prc": [10.0, 20.0, 11.0],
            "ret": [0.01, -0.02, 0.03],
            "vol": [100.0, 200.0, 300.0],
            "shrout": [1000.0, 2000.0, 3000.0],
            "cfacpr": [1.0, 1.0, 1.0],
        }
    )


class TestNormalRetrieval:
    def test_returns_only_the_contract_columns(self):
        out = get_crsp_returns("2020-01-01", "2020-01-31", conn=FakeConn())

        assert list(out.columns) == list(RETURN_COLUMNS)
        assert len(out) == 3
        assert isinstance(out.index, pd.RangeIndex)

    def test_result_is_sorted_by_date_then_permno(self):
        out = get_crsp_returns("2020-01-01", "2020-01-31", conn=FakeConn())

        # The fixture is fed in reverse date order on purpose.
        assert out["date"].is_monotonic_increasing
        assert list(out["permno"]) == [90001, 90002, 90001]

    def test_dtypes_are_stable(self):
        out = get_crsp_returns("2020-01-01", "2020-01-31", conn=FakeConn())

        assert out["date"].dtype == "datetime64[ns]"
        # Float identifiers join wrongly and print as 90001.0.
        assert out["permno"].dtype == "int64"
        assert out["ret"].dtype == "float64"


class TestDateFiltering:
    def test_range_reaches_the_query(self):
        conn = FakeConn()
        get_crsp_returns("2020-01-01", "2020-01-31", conn=conn)

        params = conn.calls[0]["params"]
        assert params["start"] == "2020-01-01"
        assert params["end"] == "2020-01-31"
        assert "d.date between" in conn.calls[0]["query"]

    def test_inverted_range_raises_before_querying(self):
        conn = FakeConn()
        with pytest.raises(ValueError, match="precedes"):
            get_crsp_returns("2020-01-31", "2020-01-01", conn=conn)
        assert conn.calls == []

    def test_single_day_range_is_allowed(self):
        out = get_crsp_returns("2020-01-02", "2020-01-02", conn=FakeConn())
        assert not out.empty


class TestPermnoFiltering:
    def test_permnos_reach_the_query(self):
        conn = FakeConn()
        get_crsp_returns("2020-01-01", "2020-01-31", permnos=[90001, 90002], conn=conn)

        assert conn.calls[0]["params"]["permnos"] == (90001, 90002)
        assert "d.permno in" in conn.calls[0]["query"]

    def test_none_applies_no_permno_filter(self):
        conn = FakeConn()
        get_crsp_returns("2020-01-01", "2020-01-31", permnos=None, conn=conn)

        assert "permnos" not in conn.calls[0]["params"]

    def test_empty_list_returns_empty_without_querying(self):
        conn = FakeConn()
        out = get_crsp_returns("2020-01-01", "2020-01-31", permnos=[], conn=conn)

        # The trap this guards: crsp_daily reads a falsy permnos as "no filter",
        # so forwarding [] would hand back the whole cross-section.
        assert conn.calls == []
        assert out.empty
        assert list(out.columns) == list(RETURN_COLUMNS)


class TestEmptyResults:
    def test_zero_rows_keep_columns_and_dtypes(self):
        out = get_crsp_returns("2020-01-01", "2020-01-31", conn=FakeConn(_rows().iloc[0:0]))

        assert len(out) == 0
        assert list(out.columns) == list(RETURN_COLUMNS)
        assert out["date"].dtype == "datetime64[ns]"
        assert out["permno"].dtype == "int64"
        assert out["ret"].dtype == "float64"

    def test_empty_result_is_a_frame_not_none(self):
        out = get_crsp_returns("2020-01-01", "2020-01-31", conn=FakeConn(_rows().iloc[0:0]))
        assert isinstance(out, pd.DataFrame)


class TestFailuresAndConnectionOwnership:
    def test_query_failure_propagates(self):
        conn = FakeConn(error=RuntimeError("server closed the connection unexpectedly"))

        with pytest.raises(RuntimeError, match="server closed"):
            get_crsp_returns("2020-01-01", "2020-01-31", conn=conn)

    def test_supplied_connection_is_left_open(self):
        conn = FakeConn()
        get_crsp_returns("2020-01-01", "2020-01-31", conn=conn)
        assert not conn.closed

    def test_tool_opened_connection_is_closed(self, monkeypatch):
        conn = FakeConn()
        monkeypatch.setattr(crsp, "connect", lambda: conn)

        get_crsp_returns("2020-01-01", "2020-01-31")
        assert conn.closed

    def test_tool_opened_connection_is_closed_after_a_failure(self, monkeypatch):
        conn = FakeConn(error=RuntimeError("query exploded"))
        monkeypatch.setattr(crsp, "connect", lambda: conn)

        with pytest.raises(RuntimeError):
            get_crsp_returns("2020-01-01", "2020-01-31")
        assert conn.closed


def test_importing_the_tool_does_not_import_wrds():
    """`connect()` imports wrds lazily; this keeps that true through the tool.

    Run in a subprocess because the parent test session may already have wrds
    imported for unrelated reasons.
    """
    code = (
        "import sys; import capstone.agent.tools.crsp; "
        "assert 'wrds' not in sys.modules, 'importing the tool pulled in wrds'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
