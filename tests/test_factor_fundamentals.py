"""Annual fundamentals known at the time (QUANTIT-118).

Synthetic frames only: a business-day calendar, hand-made accounting rows,
fake Compustat/link frames and `fake_rows` for the daily file. The shared-data
loader is checked with a fake `sd.load`; nothing reads CRSP or Compustat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_factor_evaluation import fake_rows

from capstone import shared_data as sd
from capstone.factors import evaluation as ev
from capstone.factors import fundamentals as fu
from capstone.factors import tree
from capstone.factors.evaluation import build_panel, evaluate_factors, factor_returns
from capstone.factors.interpret import EvaluationError, evaluate
from capstone.factors.tree import (
    FIELDS,
    FUNDAMENTAL_FIELDS,
    STOCK_FIELDS,
    ParseError,
    factor_id,
    features_used,
    identity_key,
    parse,
    unparse,
)

DATES = pd.bdate_range("2015-01-02", periods=320)
PERMNOS = pd.Index([10000, 10001, 10002])


def one_row(permno, datadate, **values) -> dict:
    base = dict.fromkeys(FUNDAMENTAL_FIELDS, 1.0)
    return {"permno": permno, "datadate": pd.Timestamp(datadate), **base, **values}


def field(rows, name, permno=10000) -> pd.Series:
    return fu.fundamental_fields(pd.DataFrame(rows), DATES, PERMNOS)[name][permno]


# ---------------------------------------------------------------------------
# The field set


def test_the_fundamental_fields_are_a_small_standard_set():
    assert FUNDAMENTAL_FIELDS == ("book", "earnings", "sales", "assets")
    assert fu.FUNDAMENTAL_COLUMNS == ("permno", "datadate", *FUNDAMENTAL_FIELDS)
    assert set(FUNDAMENTAL_FIELDS) <= set(FIELDS)
    # The daily stock fields (what edits draw from) are unchanged.
    assert STOCK_FIELDS == (*tree.OHLCV, "returns", "shares", "cap")


@pytest.mark.parametrize(
    "expression",
    [
        "rank(book / cap)",
        "earnings / cap",
        "-delta(assets, 252) / shift(assets, 252)",
        "rank(earnings / book) + rank(sales / assets)",
        "rank(book)",  # a fundamental alone is a per-stock field
    ],
)
def test_a_tree_using_a_fundamental_parses_and_round_trips(expression):
    node = parse(expression)
    assert features_used(node) & set(FUNDAMENTAL_FIELDS)
    again = parse(unparse(node))
    assert again == node
    assert identity_key(again) == identity_key(node)
    assert factor_id(again) == factor_id(node)


@pytest.mark.parametrize("expression", ["rank(bookx)", "ebitda / cap", "rank(ceq)", "book(close)"])
def test_unknown_fields_still_raise(expression):
    with pytest.raises(ParseError):
        parse(expression)


def test_a_factor_from_market_fields_alone_is_still_rejected():
    with pytest.raises(ParseError, match="no stock field"):
        parse("ts_mean(mkt_return, 5)")


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
    # Held until the next year would be available: 2015-06-30 + 90 days is Monday
    # 09-28; strictly after is 09-29. No newer year by then, so missing.
    expiry = pd.Timestamp("2015-09-29")
    assert (sales[sales.index < expiry] == 2.0).all()
    assert sales[sales.index >= expiry].isna().all()


def test_a_year_is_not_carried_past_when_the_next_would_be_available():
    # Only FY2013: visible from before the panel, missing once FY2014 would be
    # (2014-12-31 + 90 days is Tuesday 2015-03-31; strictly after is 04-01).
    book = field([one_row(10000, "2013-12-31", book=5.0)], "book")
    expiry = pd.Timestamp("2015-04-01")
    assert (book[book.index < expiry] == 5.0).all()
    assert book[book.index >= expiry].isna().all()


def test_an_absent_year_reads_the_same_as_a_year_reported_missing():
    old = one_row(10000, "2013-12-31", book=5.0)
    absent = field([old], "book")
    reported = field([old, one_row(10000, "2014-12-31", book=np.nan)], "book")
    pd.testing.assert_series_equal(absent, reported)


def test_a_regular_filer_has_no_gap_between_years():
    rows = [one_row(10000, f"{y}-12-31", book=float(y)) for y in (2013, 2014, 2015)]
    book = field(rows, "book")
    assert book.notna().all()
    assert book.iloc[0] == 2013.0 and book.iloc[-1] == 2014.0


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


def _funda_row(**values) -> pd.DataFrame:
    base = dict.fromkeys(["seq", "ceq", "pstk", "pstkrv", "pstkl", "txditc", "at", "lt"], np.nan)
    return pd.DataFrame([base | values])


@pytest.mark.parametrize(
    ("values", "expected", "step"),
    [
        (dict(seq=10.0, ceq=4.0, pstk=1.0), 10.0 - 1.0, "seq before ceq + pstk"),
        (dict(ceq=4.0, pstk=1.0), 5.0 - 1.0, "ceq + pstk when seq is missing"),
        (dict(ceq=4.0), 4.0, "ceq alone: pstk missing counts 0"),
        (dict(at=100.0, lt=70.0), 30.0, "at - lt when seq and ceq are missing"),
        (dict(seq=10.0, pstkrv=3.0, pstkl=7.0, pstk=9.0), 7.0, "redemption value first"),
        (dict(seq=10.0, pstkl=2.0, pstk=9.0), 8.0, "liquidating value next"),
        (dict(seq=10.0, pstk=2.0), 8.0, "par value last"),
        (dict(seq=10.0), 10.0, "no preferred: 0"),
        (dict(seq=10.0, txditc=1.5), 11.5, "deferred taxes added"),
    ],
)
def test_each_step_of_the_book_equity_fallback(values, expected, step):
    assert fu.book_equity(_funda_row(**values)).iloc[0] == pytest.approx(expected), step


def test_an_ended_link_does_not_carry_its_last_year_forward():
    funda = fake_funda().iloc[:2]  # 001: FY2013 and FY2014
    link = fake_link().iloc[:1].assign(linkenddt=pd.Timestamp("2014-06-30"))
    rows = fu.from_compustat(funda, link)
    assert list(rows["datadate"]) == [pd.Timestamp("2013-12-31")]
    book = fu.fundamental_fields(rows, DATES, PERMNOS)["book"][10000]
    assert book[book.index >= pd.Timestamp("2015-04-01")].isna().all()


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


# ---------------------------------------------------------------------------
# Evaluation

ROWS = fake_rows(n_stocks=6, n_days=320, start="2015-01-02")
CALENDAR = pd.DatetimeIndex(sorted(ROWS["date"].unique()))


def fake_fundamentals(rows: pd.DataFrame, *, seed=0) -> pd.DataFrame:
    """Annual accounting rows for `fake_rows`' permnos: one row per permno and
    December fiscal year end, from two years before the first date to the
    last, deterministic from `seed`."""
    rng = np.random.default_rng([seed, 7])
    dates = pd.DatetimeIndex(rows["date"])
    out = []
    for permno in sorted(rows["permno"].unique()):
        assets = float(np.exp(rng.normal(8.0, 0.5)))
        for year in range(dates.min().year - 2, dates.max().year + 1):
            assets *= float(np.exp(rng.normal(0.05, 0.1)))
            out.append(
                {
                    "permno": permno,
                    "datadate": pd.Timestamp(year, 12, 31),
                    "book": assets * rng.uniform(0.2, 0.6),
                    "earnings": assets * rng.normal(0.05, 0.03),
                    "sales": assets * rng.uniform(0.5, 1.5),
                    "assets": assets,
                }
            )
    return pd.DataFrame(out)


EXISTING = [
    "volume / shares",
    "rank(volume) - rank(ts_mean(volume, 10))",
    "-ts_sum(returns, 5)",
    "ts_corr(returns, mkt_return, 10)",
    "rank(ts_mean(volume, 5)) - rank(delta(close, 3))",
]


def test_a_fundamentals_frame_without_its_columns_is_refused():
    frame = pd.DataFrame([one_row(10000, "2014-12-31")]).drop(columns="earnings")
    with pytest.raises(ValueError, match="earnings"):
        build_panel(ROWS, fundamentals=frame)


def test_existing_trees_evaluate_bit_identically_with_and_without_fundamentals():
    plain = build_panel(ROWS, "v")
    rich = build_panel(ROWS, "v", fundamentals=fake_fundamentals(ROWS))
    assert set(rich.fields) == set(plain.fields) | set(FUNDAMENTAL_FIELDS)
    for name, value in plain.fields.items():
        if isinstance(value, pd.Series):
            pd.testing.assert_series_equal(value, rich.fields[name], check_exact=True)
        else:
            pd.testing.assert_frame_equal(value, rich.fields[name], check_exact=True)
    for expression in EXISTING:
        node = parse(expression)
        pd.testing.assert_frame_equal(
            evaluate(node, plain.fields), evaluate(node, rich.fields), check_exact=True
        )
        for hold in (1, 5):
            a, b = factor_returns(node, plain, hold), factor_returns(node, rich, hold)
            pd.testing.assert_series_equal(a.net, b.net, check_exact=True)
            pd.testing.assert_series_equal(a.turnover, b.turnover, check_exact=True)


def test_a_fundamental_tree_evaluates_to_its_formula():
    fund = fake_fundamentals(ROWS)
    panel = build_panel(ROWS, fundamentals=fund)
    got = evaluate(parse("rank(book / cap)"), panel.fields)
    expected = (panel.fields["book"] / panel.fields["cap"]).rank(axis=1, pct=True)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    # December year ends: the first visible value is 90 days later, in April.
    first = got.dropna(how="all").index[0]
    assert first == sd.available_from(fund, CALENDAR, report_col=None).dropna().min()
    returns = factor_returns(parse("rank(book / cap)"), panel)
    assert returns.net.notna().sum() > 0


def test_a_fundamental_tree_on_a_panel_without_fundamentals_is_a_logged_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    panel = build_panel(ROWS, "v")
    with pytest.raises(EvaluationError, match="book"):
        evaluate(parse("rank(book / cap)"), panel.fields)
    matrix, summary = evaluate_factors({"f": parse("rank(book / cap)")}, panel)
    assert matrix.empty and summary.empty
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 1 and "book" in lines[0]


def test_load_panel_wires_fundamentals_through_shared_data(monkeypatch):
    asked = []

    def fake_load(name, columns=None, start=None, end=None, **kwargs):
        asked.append(name)
        if name == ev.DAILY_FILE:
            return fake_rows(n_stocks=3, n_days=300, start="2015-01-02")
        frame = {"comp_funda": fake_funda(), "ccm_link": fake_link()}[name]
        assert set(columns) <= set(frame.columns)
        return frame[columns]

    monkeypatch.setattr(sd, "load", fake_load)
    monkeypatch.setattr(sd, "data_version", lambda: "test-version")
    plain = ev.load_panel()
    assert asked == [ev.DAILY_FILE]
    assert not set(FUNDAMENTAL_FIELDS) & set(plain.fields)
    panel = ev.load_panel(fundamentals=True)
    assert set(asked[1:]) == {ev.DAILY_FILE, "comp_funda", "ccm_link"}
    book = panel.fields["book"][10000]
    switch = pd.Timestamp("2015-04-01")  # FY2014: 2014-12-31 + 90 days, then the next day
    np.testing.assert_allclose(book.loc[book.index < switch], 10.5)  # FY2013 from day one
    np.testing.assert_allclose(book.loc[book.index >= switch], 10.0)
    assert panel.fields["book"][10001].isna().all()  # its year ended before its link began
