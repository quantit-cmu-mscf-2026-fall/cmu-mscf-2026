from copy import deepcopy

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from capstone.factors.evaluation import Panel
from capstone.pipeline.historical import HistoricalInputs
from capstone.pipeline.prefix_inputs import input_prefix


def fixture():
    days = pd.to_datetime(
        ["1995-12-28", "1995-12-29", "1996-01-02", "1996-01-03", "1996-01-04", "1996-01-05"]
    )
    returns = pd.DataFrame({101: 0.01, 102: 0.02, 103: 999.0}, index=days)
    universe = returns.gt(0)
    universe[103] = False
    universe.loc[days[-1], 103] = True
    panel = Panel(
        {"returns": returns},
        universe,
        returns * 0,
        data_version="synthetic-prefix07",
        calendar=days.append(pd.DatetimeIndex(["1996-01-08"])),
        availability={"returns": pd.Series(days, index=days)},
        company=pd.DataFrame({101: 1001, 102: 1002, 103: 1003}, index=days),
        fundamentals=pd.DataFrame(
            [
                {
                    "permco": 1001,
                    "datadate": "1994-12-31",
                    "published_at": "1996-01-02",
                    "vintage_at": "1996-01-02",
                    "book": 10,
                },
                {
                    "permco": 1001,
                    "datadate": "1994-12-31",
                    "published_at": "1996-01-05",
                    "vintage_at": "1996-01-05",
                    "book": 999,
                },
            ]
        ),
        target_cache={"untrusted": "discard"},
    )
    opens = returns * 0 + 10
    closes = opens.copy()
    closes.loc[days[2], 102] = 11
    events = pd.DataFrame(
        [
            {
                "event_id": "entitlement",
                "event_date": days[3],
                "permno": 102,
                "kind": "dividend",
                "amount": 1.0,
                "pay_date": days[-1],
            },
            {
                "event_id": "late-failure",
                "event_date": days[-1],
                "permno": 102,
                "kind": "unsupported",
                "amount": float("nan"),
                "pay_date": pd.NaT,
            },
        ]
    )
    return HistoricalInputs(
        panel,
        opens,
        closes,
        events,
        returns * 0,
        returns * 0,
        {"source_hashes": {"daily": "a" * 64}, "sources": {}, "global_blockers": []},
    )


def equal(left, right):
    for key in ("opens", "closes", "total_returns", "half_spread_bps", "events"):
        assert_frame_equal(getattr(left, key), getattr(right, key))
    assert_frame_equal(left.panel.universe, right.panel.universe)
    assert_frame_equal(left.panel.company, right.panel.company)
    assert_frame_equal(left.panel.fundamentals, right.panel.fundamentals)
    for key in left.panel.fields:
        assert_frame_equal(left.panel.fields[key], right.panel.fields[key])
        assert_series_equal(left.panel.availability[key], right.panel.availability[key])
    assert left.panel.calendar.equals(right.panel.calendar)
    assert left.qualification == right.qualification


def test_future_failure_values_and_unknown_member_cannot_change_earlier_view():
    original = fixture()
    expected = input_prefix(original, "1996-01-03")
    changed = deepcopy(original)
    for frame in (changed.panel.fields["returns"], changed.opens, changed.closes):
        frame.loc[frame.index > "1996-01-03"] *= -1000
        frame[103] *= -1000
    changed.events.loc[changed.events.event_id == "late-failure", "kind"] = "not-a-supported-event"
    equal(input_prefix(changed, "1996-01-03"), expected)
    assert list(expected.panel.universe) == [101, 102]
    assert expected.panel.fundamentals.book.tolist() == [10]
    assert expected.panel.calendar[-1] == pd.Timestamp("1996-01-04")
    assert expected.panel.target_cache is None


def test_unavailable_numeric_revisions_and_future_dates_are_not_exposed():
    original = fixture()
    original.panel.availability["returns"].loc["1996-01-02"] = pd.Timestamp("1996-01-04")
    earlier = input_prefix(original, "1996-01-03")
    changed = deepcopy(original)
    changed.panel.fields["returns"].loc["1996-01-02"] = 123456
    changed.panel.availability["returns"].loc["1996-01-02"] = pd.Timestamp("1996-01-05")
    equal(earlier, input_prefix(changed, "1996-01-03"))
    assert earlier.panel.fields["returns"].loc["1996-01-02"].isna().all()
    assert pd.isna(earlier.panel.availability["returns"].loc["1996-01-02"])


def test_permutations_and_mutation_do_not_change_source_or_view():
    original = fixture()
    expected = input_prefix(original, "1996-01-03")
    changed = deepcopy(original)
    changed.panel.universe = changed.panel.universe.iloc[::-1, ::-1]
    for name in changed.panel.fields:
        changed.panel.fields[name] = changed.panel.fields[name].iloc[::-1, ::-1]
    changed.events = changed.events.iloc[::-1]
    equal(expected, input_prefix(changed, "1996-01-03"))
    expected.closes.iloc[0, 0] = -1
    expected.qualification["source_hashes"]["daily"] = "changed"
    assert original.closes.iloc[0, 0] == 10
    assert original.qualification["source_hashes"]["daily"] == "a" * 64


def test_payment_metadata_survives_but_later_effective_event_does_not():
    view = input_prefix(fixture(), "1996-01-03")
    assert view.events.event_id.tolist() == ["entitlement"]
    assert view.events.pay_date.iloc[0] == pd.Timestamp("1996-01-05")
    assert not view.qualification["prefix_view"]["source_qualification_recomputed"]
    assert not view.qualification["prefix_view"]["formal_admission"]


def test_conversion_destination_is_retained_even_outside_universe():
    data = fixture()
    data.events = pd.DataFrame(
        [
            {
                "event_id": "conversion",
                "event_date": pd.Timestamp("1996-01-03"),
                "permno": 101,
                "kind": "conversion",
                "new_permno": 103,
                "ratio": 1.0,
                "amount": 0.0,
            }
        ]
    )
    view = input_prefix(data, "1996-01-03")
    assert list(view.opens) == [101, 102, 103]
    assert not view.panel.universe[103].any()


def test_actual_company_schema_excludes_future_links_and_their_records():
    original = fixture()
    extra = pd.DataFrame(
        [
            {
                "permco": 1003,
                "datadate": "1994-12-31",
                "published_at": "1996-01-02",
                "vintage_at": "1996-01-02",
                "book": 99999,
            }
        ]
    )
    original.panel.fundamentals = pd.concat([original.panel.fundamentals, extra], ignore_index=True)
    expected = input_prefix(original, "1996-01-03")
    changed = deepcopy(original)
    changed.panel.company.loc["1996-01-04":, 101] = 1003
    changed.panel.fundamentals.loc[changed.panel.fundamentals.permco == 1003, "book"] = -99999
    equal(expected, input_prefix(changed, "1996-01-03"))
    assert expected.panel.fundamentals.permco.tolist() == [1001]
    assert "permno" not in expected.panel.fundamentals


def test_reverse_identifier_conversion_chain_retains_all_destinations():
    data = fixture()
    for name in (90, 80):
        data.panel.universe[name] = False
        data.panel.fields["returns"][name] = 0.01
        data.panel.company[name] = 1001
        for frame in (data.opens, data.closes, data.total_returns, data.half_spread_bps):
            frame[name] = 10.0
    data.events = pd.DataFrame(
        [
            {
                "event_id": "second",
                "event_date": pd.Timestamp("1996-01-03"),
                "permno": 90,
                "kind": "conversion",
                "new_permno": 80,
                "ratio": 1.0,
                "amount": 0.0,
            },
            {
                "event_id": "first",
                "event_date": pd.Timestamp("1996-01-03"),
                "permno": 101,
                "kind": "conversion",
                "new_permno": 90,
                "ratio": 1.0,
                "amount": 0.0,
            },
        ]
    )
    expected = input_prefix(data, "1996-01-03")
    assert list(expected.opens) == [80, 90, 101, 102]
    data.events = data.events.iloc[::-1]
    equal(expected, input_prefix(data, "1996-01-03"))


def test_missing_availability_remains_unavailable_and_incomplete_accounting_refuses():
    data = fixture()
    data.panel.availability = {}
    view = input_prefix(data, "1996-01-03")
    assert view.panel.fields["returns"].isna().all().all()
    assert "returns" not in view.panel.availability
    data.panel.fundamentals = data.panel.fundamentals.drop(columns="vintage_at")
    with pytest.raises(ValueError, match="publication and vintage"):
        input_prefix(data, "1996-01-03")


@pytest.mark.parametrize("cutoff", ["2021-01-04", "1996-01-06", "1996-01-03T12:00"])
def test_invalid_cutoff_refuses(cutoff):
    with pytest.raises(ValueError):
        input_prefix(fixture(), cutoff)


def test_missing_membership_session_or_next_calendar_label_refuses():
    data = fixture()
    data.panel.universe = data.panel.universe.drop(index="1996-01-02")
    with pytest.raises(ValueError, match="authoritative session history"):
        input_prefix(data, "1996-01-03")
    data = fixture()
    data.panel.calendar = data.panel.calendar[data.panel.calendar <= "1996-01-03"]
    with pytest.raises(ValueError, match="next date-only session"):
        input_prefix(data, "1996-01-03")
