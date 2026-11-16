"""Annual fundamentals known at the time (QUANTIT-118).

Synthetic frames only: a business-day calendar, hand-made accounting rows and
fake Compustat/link frames. The shared-data loader is checked with a fake
`sd.load`; nothing reads CRSP or Compustat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone import shared_data as sd
from capstone.factors import fundamentals as fu
from capstone.factors import tree
from capstone.factors.tree import FIELDS, FUNDAMENTAL_FIELDS, ParseError, parse

DATES = pd.bdate_range("2015-01-02", periods=320)
PERMNOS = pd.Index([10000, 10001, 10002])


def one_row(permno, datadate, **values) -> dict:
    base = dict.fromkeys(FUNDAMENTAL_FIELDS, 1.0)
    return {"permno": permno, "datadate": pd.Timestamp(datadate), **base, **values}


def field(rows, name, permno=10000) -> pd.Series:
    return fu.fundamental_fields(pd.DataFrame(rows), DATES, PERMNOS)[name][permno]


# ---------------------------------------------------------------------------
# The field set


def test_the_fundamental_fields_are_a_small_standard_set_outside_the_grammar():
    assert FUNDAMENTAL_FIELDS == ("book", "earnings", "sales", "assets")
    assert fu.FUNDAMENTAL_COLUMNS == ("permno", "datadate", *FUNDAMENTAL_FIELDS)
    # Not grammar fields yet: no tree can name one, and existing fields are unchanged.
    assert not set(FUNDAMENTAL_FIELDS) & set(FIELDS)
    assert FIELDS == (*tree.OHLCV, "returns", "shares", "cap")
    with pytest.raises(ParseError):
        parse("rank(book / cap)")


# ---------------------------------------------------------------------------
# Availability: nothing is visible before it could have been known


def test_a_value_is_never_visible_before_its_availability_date():
    fund = pd.DataFrame(
        [
            one_row(10000, "2015-03-31", book=5.0),
            one_row(10001, "2099-12-31", book=7.0),  # available far past the calendar
        ]
    )
    book = fu.fundamental_fields(fund, DATES, PERMNOS)["book"]
    available = sd.available_from(fund, DATES, report_col=None)
    first = available.iloc[0]
    # 2015-03-31 + 90 days is Tuesday 06-29; strictly after is 06-30.
    assert first == pd.Timestamp("2015-06-30")
    assert book.loc[book.index < first, 10000].isna().all()
    assert (book.loc[book.index >= first, 10000] == 5.0).all()
    assert pd.isna(available.iloc[1])
    assert book[10001].isna().all()
    assert book[10002].isna().all()  # no rows: never a value


def test_the_90_day_mark_itself_is_not_yet_known():
    sales = field([one_row(10000, "2014-12-31", sales=3.0)], "sales")
    mark = pd.Timestamp("2014-12-31") + pd.Timedelta(90, unit="D")  # 2015-03-31, a Tuesday
    assert sales.loc[sales.index <= mark].isna().all()
    assert sales.loc[sales.index > mark].eq(3.0).all()
    assert sales.dropna().index[0] == pd.Timestamp("2015-04-01")


def test_a_late_year_stays_invisible_until_its_day_and_the_old_one_holds():
    earnings = field(
        [one_row(10000, "2013-12-31", earnings=1.0), one_row(10000, "2014-12-31", earnings=2.0)],
        "earnings",
    )
    switch = pd.Timestamp("2015-04-01")
    assert earnings.loc[earnings.index < switch].eq(1.0).all()
    assert earnings.loc[earnings.index >= switch].eq(2.0).all()


def test_each_year_holds_until_the_next_is_available_and_a_gap_is_not_filled():
    fund = pd.DataFrame(
        [
            one_row(10000, "2014-12-31", earnings=1.0),
            one_row(10000, "2015-03-31", earnings=np.nan),  # reported, missing
            one_row(10000, "2015-06-30", earnings=3.0),
        ]
    )
    earnings = fu.fundamental_fields(fund, DATES, PERMNOS)["earnings"][10000]
    a = fu.with_availability(fund, DATES)["available"]
    assert earnings.loc[earnings.index < a.iloc[0]].isna().all()
    assert (earnings.loc[(earnings.index >= a.iloc[0]) & (earnings.index < a.iloc[1])] == 1.0).all()
    # The missing value replaces the old one: no stale fill across a gap.
    assert earnings.loc[(earnings.index >= a.iloc[1]) & (earnings.index < a.iloc[2])].isna().all()
    assert (earnings.loc[earnings.index >= a.iloc[2]] == 3.0).all()


def test_years_available_before_the_panel_start_leave_only_the_latest():
    sales = field(
        [
            one_row(10000, "2013-12-31", sales=1.0),
            one_row(10000, "2014-06-30", sales=2.0),  # the latest available by day one
            one_row(10000, "2012-12-31", sales=9.0),  # an older year, listed last
        ],
        "sales",
    )
    assert (sales == 2.0).all()


def test_an_older_year_never_overrides_a_newer_one():
    # A report date can make an older fiscal year arrive after a newer one.
    fund = pd.DataFrame(
        [
            one_row(10000, "2015-03-31", assets=2.0),
            one_row(10000, "2014-12-31", assets=1.0),
        ]
    )
    out = fu.as_of_fields(fund.assign(available=[DATES[100], DATES[150]]), DATES, PERMNOS)
    assets = out["assets"][10000]
    assert assets.iloc[:100].isna().all()
    assert (assets.iloc[100:] == 2.0).all()


def test_availability_off_the_calendar_is_refused():
    off = DATES[3] + pd.Timedelta(12, unit="h")
    fund = pd.DataFrame([one_row(10000, "2014-12-31")]).assign(available=[off])
    with pytest.raises(ValueError, match="outside the panel's calendar"):
        fu.as_of_fields(fund, DATES, PERMNOS)


def test_fields_share_the_calendar_and_the_requested_permnos():
    rows = [one_row(10000, "2014-12-31"), one_row(99999, "2014-12-31")]  # 99999: not asked for
    out = fu.fundamental_fields(pd.DataFrame(rows), DATES, PERMNOS)
    assert list(out) == list(FUNDAMENTAL_FIELDS)
    for frame in out.values():
        assert frame.index.equals(DATES) and frame.columns.equals(PERMNOS)


def test_no_visible_rows_gives_all_missing_frames():
    out = fu.fundamental_fields(pd.DataFrame([one_row(99999, "2014-12-31")]), DATES, PERMNOS)
    assert all(frame.isna().all().all() for frame in out.values())


# ---------------------------------------------------------------------------
# Refused frames


def test_a_frame_without_its_columns_is_refused():
    fund = pd.DataFrame([one_row(10000, "2014-12-31")]).drop(columns="earnings")
    with pytest.raises(ValueError, match="earnings"):
        fu.fundamental_fields(fund, DATES, PERMNOS)


def test_infinite_values_are_refused():
    fund = pd.DataFrame([one_row(10000, "2014-12-31", sales=np.inf)])
    with pytest.raises(ValueError, match="finite"):
        fu.fundamental_fields(fund, DATES, PERMNOS)


# ---------------------------------------------------------------------------
# Compustat to permno (fake frames)


def fake_funda() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gvkey": ["001", "001", "002", "003"],
            "datadate": pd.to_datetime(["2013-12-31", "2014-12-31", "2014-06-30", "2014-12-31"]),
            "seq": [10.0, np.nan, 20.0, 5.0],
            "ceq": [9.0, 11.0, 19.0, 4.0],
            "pstk": [1.0, 1.0, np.nan, 0.0],
            "pstkrv": [np.nan, 2.0, 3.0, np.nan],
            "pstkl": [0.5, np.nan, np.nan, np.nan],
            "txditc": [1.0, np.nan, 2.0, 0.0],
            "at": [100.0, 120.0, 200.0, 50.0],
            "lt": [80.0, 100.0, 150.0, 40.0],
            "ib": [3.0, 4.0, -1.0, 2.0],
            "sale": [50.0, 60.0, 90.0, 30.0],
        }
    )


def fake_link() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gvkey": ["001", "002", "003"],
            "lpermno": [10000.0, 10001.0, 10002.0],
            "linktype": ["LU", "LC", "LU"],
            "linkprim": ["P", "P", "J"],  # 003's link is not primary: dropped
            "linkdt": pd.to_datetime(["2000-01-01", "2014-07-01", "2000-01-01"]),
            "linkenddt": pd.to_datetime([None, None, None]),
        }
    )


def test_compustat_rows_map_to_permnos_with_book_equity_as_fama_french():
    out = fu.from_compustat(fake_funda(), fake_link()).set_index(["permno", "datadate"])
    # 001 2013: seq 10 - preferred (pstkrv NaN -> pstkl 0.5) + txditc 1 = 10.5
    assert out.loc[(10000, pd.Timestamp("2013-12-31")), "book"] == pytest.approx(10.5)
    # 001 2014: seq NaN -> ceq 11 + pstk 1 = 12; - pstkrv 2; + txditc NaN -> 0 = 10
    assert out.loc[(10000, pd.Timestamp("2014-12-31")), "book"] == pytest.approx(10.0)
    assert out.loc[(10000, pd.Timestamp("2014-12-31")), "earnings"] == 4.0
    assert out.loc[(10000, pd.Timestamp("2014-12-31")), "sales"] == 60.0
    assert out.loc[(10000, pd.Timestamp("2014-12-31")), "assets"] == 120.0
    # 002's fiscal year ended before its link started; 003 has no primary link.
    assert set(out.index.get_level_values("permno")) == {10000}
    assert list(out.columns) == list(FUNDAMENTAL_FIELDS)


def test_a_fiscal_year_after_the_link_ends_is_dropped_and_the_link_end_is_inclusive():
    funda = fake_funda().iloc[:2]
    link = fake_link().iloc[:1].assign(linkenddt=pd.Timestamp("2014-12-31"))
    assert list(fu.from_compustat(funda, link)["datadate"]) == list(funda["datadate"])
    early = link.assign(linkenddt=pd.Timestamp("2014-12-30"))
    assert list(fu.from_compustat(funda, early)["datadate"]) == [pd.Timestamp("2013-12-31")]


def test_a_duplicate_permno_year_keeps_the_last_row():
    funda = pd.concat([fake_funda().iloc[[1]], fake_funda().iloc[[1]].assign(ib=9.0)])
    out = fu.from_compustat(funda, fake_link())
    assert len(out) == 1 and out["earnings"].iloc[0] == 9.0


def test_mapped_rows_become_visible_only_after_their_availability_date():
    rows = fu.from_compustat(fake_funda(), fake_link())
    book = fu.fundamental_fields(rows, DATES, PERMNOS)["book"][10000]
    switch = pd.Timestamp("2015-04-01")  # FY2014: 2014-12-31 + 90 days, then the next day
    np.testing.assert_allclose(book.loc[book.index < switch], 10.5)  # FY2013 from day one
    np.testing.assert_allclose(book.loc[book.index >= switch], 10.0)


def test_loading_goes_through_shared_data_and_keeps_nothing_after_end(monkeypatch):
    asked = []

    def fake_load(name, columns=None, **kwargs):
        asked.append(name)
        frame = {"comp_funda": fake_funda(), "ccm_link": fake_link()}[name]
        assert set(columns) <= set(frame.columns)
        return frame[columns]

    monkeypatch.setattr(sd, "load", fake_load)
    out = fu.load_fundamentals("2014-06-30")
    assert asked == ["comp_funda", "ccm_link"]
    assert list(out["datadate"]) == [pd.Timestamp("2013-12-31")]
    assert set(out["permno"]) == {10000}
