"""CRSP event normalisation and held-action valuation on synthetic rows only."""

import math

import pandas as pd
import pytest

from capstone.pipeline.crsp_actions import (
    HELD_VALUE_RULE,
    held_inputs,
    normalize,
    valued_events,
)
from capstone.pipeline.historical import HistoricalInputs

CALENDAR = pd.bdate_range("1995-12-29", "2000-01-12")


def distribution(**values):
    return {
        "permno": 1,
        "disexdt": pd.Timestamp("2000-01-04"),
        "disseqnbr": 1,
        "distype": "CD",
        "dispaymenttype": "USD",
        "disdivamt": 1.0,
        "disfacshr": 0.0,
        "dispaydt": pd.Timestamp("2000-01-10"),
        **values,
    }


def terminal(**values):
    return {
        "permno": 1,
        "delistingdt": pd.Timestamp("2000-01-03"),
        "deldlydt": pd.Timestamp("2000-01-04"),
        "delpaymenttype": "CASH",
        **values,
    }


def normalized(rows, endings=()):
    return normalize(
        pd.DataFrame(rows, columns=list(distribution())),
        pd.DataFrame(endings, columns=["permno", "delistingdt", "deldlydt", "delpaymenttype"]),
        CALENDAR,
    )


# ------------------------------------------------------------------ normalize


def test_an_observed_cash_distribution_becomes_a_dividend_with_its_own_payment_date():
    events, audit = normalized([distribution()])
    row = events.iloc[0]
    assert (row.kind, row.amount, row.event_id) == ("dividend", 1.0, "crsp-dis:1:2000-01-04:1")
    assert row.pay_date == pd.Timestamp("2000-01-10")
    assert audit == [
        {
            "source_id": "crsp-dis:1:2000-01-04:1",
            "status": "dividend",
            "reason": "observed_usd_distribution",
        }
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"dispaydt": pd.NaT},
        {"disdivamt": float("nan")},
        {"disdivamt": -1.0},
        {"dispaydt": pd.Timestamp("2000-01-03")},
        {"dispaymenttype": "FX"},
        {"distype": "SP"},
    ],
)
def test_incomplete_or_non_cash_terms_remain_explicit_unsupported_events(change):
    events, audit = normalized([distribution(**change)])
    assert events.kind.tolist() == ["unsupported"]
    assert audit[0]["status"] == "unsupported" and audit[0]["reason"]


def test_a_same_security_split_carries_the_ratio_and_a_bad_ratio_is_unsupported():
    events, _ = normalized([distribution(distype="FRS", dispaymenttype="SS", disfacshr=0.5)])
    assert events.kind.tolist() == ["split"] and events.ratio.tolist() == [1.5]
    events, _ = normalized([distribution(distype="FRS", dispaymenttype="SS", disfacshr=-1.0)])
    assert events.kind.tolist() == ["unsupported"]


def test_shares_outstanding_observations_are_audited_but_are_not_events():
    events, audit = normalized([distribution(distype="TSOO")])
    assert events.empty
    assert audit == [
        {"source_id": "crsp-dis:1:2000-01-04:1", "status": "shares_outstanding_observation_only"}
    ]


def test_a_cash_terminal_distribution_is_counted_once_and_delayed_cash_is_blocked():
    row = distribution(distype="CP", disdivamt=101.0, dispaydt=pd.Timestamp("2000-01-03"))
    events, audit = normalized([row], [terminal()])
    assert events.kind.tolist() == ["settlement"] and events.amount.tolist() == [101.0]
    assert len(audit) == 2
    assert {a["status"] for a in audit} == {
        "settlement",
        "represented_by_terminal_distribution",
    }
    delayed = {**row, "dispaydt": pd.Timestamp("2000-01-10")}
    events, _ = normalized([delayed], [terminal()])
    assert set(events.kind) == {"unsupported"}
    assert sorted(events.reason) == [
        "terminal_cash_payment_not_available_on_effective_session",
        "unqualified_terminal_conversion_or_cash_terms",
    ]


def test_a_delisting_without_a_matching_cash_distribution_is_an_unsupported_terminal_event():
    events, audit = normalized([], [terminal()])
    assert events.kind.tolist() == ["unsupported"]
    assert events.reason.tolist() == ["unqualified_terminal_conversion_or_cash_terms"]
    assert events.sequence.tolist() == [1000000]
    assert audit[0]["status"] == "unsupported"


def test_events_outside_the_replay_window_are_audited_not_dropped():
    early = distribution(disexdt=pd.Timestamp("1995-12-29"))
    events, audit = normalized([early])
    assert events.empty and audit[0]["status"] == "outside_replay_window"


def test_normalization_preserves_sequence_and_is_row_order_invariant():
    rows = [distribution(), distribution(permno=2, disseqnbr=2, distype="TSOO")]
    a, audit_a = normalized(rows)
    b, audit_b = normalized(rows[::-1])
    pd.testing.assert_frame_equal(a, b)
    assert audit_a == audit_b and len(audit_a) == 2


def test_every_source_row_is_either_an_event_or_an_audited_disposition():
    rows = [distribution(), distribution(disseqnbr=2, distype="TSOO"), distribution(permno=3)]
    events, audit = normalized(rows, [terminal(permno=9)])
    assert len(audit) == 4 and len(events) == 3


@pytest.mark.parametrize(
    ("rows", "endings", "match"),
    [
        ([distribution(), distribution()], (), "duplicate"),
        ([distribution(disexdt=pd.Timestamp("2021-01-04"))], (), "discovery boundary"),
        ([distribution(disexdt=pd.NaT)], (), "discovery boundary"),
        ([distribution(disseqnbr=0)], (), "positive whole"),
        ([distribution(disseqnbr=1.5)], (), "positive whole"),
        ([distribution()], [terminal(deldlydt=pd.NaT)], "observed effective date"),
        ([distribution()], [terminal(), terminal()], "ambiguous delisting"),
        ([distribution()], [terminal(delistingdt=pd.Timestamp("2021-01-04"))], "discovery"),
    ],
)
def test_ambiguous_or_out_of_bounds_sources_are_refused(rows, endings, match):
    with pytest.raises(ValueError, match=match):
        normalized(rows, endings)


def test_normalization_needs_the_verified_session_calendar():
    frame = pd.DataFrame([distribution()])
    empty = pd.DataFrame(columns=["permno", "delistingdt", "deldlydt", "delpaymenttype"])
    with pytest.raises(ValueError, match="session calendar"):
        normalize(frame, empty, CALENDAR[::-1])
    with pytest.raises(ValueError, match="session calendar"):
        normalize(frame, empty, CALENDAR.append(CALENDAR[:1]))


# -------------------------------------------------------------- valued_events

SESSIONS = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06"])
DAY = SESSIONS[1]


def frame(values):
    return pd.DataFrame({1: values}, index=SESSIONS)


def unsupported(reason="unsupported_distribution_terms", **values):
    row = {
        "event_id": "u1",
        "event_date": DAY,
        "permno": 1,
        "kind": "unsupported",
        "reason": reason,
        "sequence": 2,
    }
    return row | values


def valued(rows, closes, total, **options):
    return valued_events(pd.DataFrame(rows), frame(closes), frame(total), **options)


def test_an_unvalued_distribution_is_priced_from_the_total_return_residual():
    # Prior close 100, close 90, total return +5%: 105 - 90 = 15 per share distributed.
    events, audit = valued([unsupported()], [100.0, 90.0, 90.0, 90.0], [0.0, 0.05, 0.0, 0.0])
    row = events.iloc[0]
    assert (row.kind, row.reason) == ("distribution_value", HELD_VALUE_RULE)
    assert row.amount == pytest.approx(15.0) and row.pay_date == DAY
    assert audit == [
        {"event_ids": ["u1"], "status": "distribution_value", "amount": pytest.approx(15.0)}
    ]


def test_a_terminal_group_pays_the_total_return_value_and_retires_the_position():
    row = unsupported("unqualified_terminal_conversion_or_cash_terms")
    events, _ = valued([row], [100.0, math.nan, math.nan, math.nan], [0.0, -0.4, 0.0, 0.0])
    assert events.kind.tolist() == ["terminal_value"]
    assert events.amount.tolist() == [pytest.approx(60.0)]


def test_observed_dividends_in_the_group_are_removed_from_the_residual():
    dividend = {
        "event_id": "d1",
        "event_date": DAY,
        "permno": 1,
        "kind": "dividend",
        "amount": 1.0,
        "pay_date": DAY,
        "sequence": 1,
    }
    events, _ = valued([dividend, unsupported()], [100.0, 90.0, 90.0, 90.0], [0.0, 0.05, 0.0, 0.0])
    assert events.set_index("event_id").loc["u1", "amount"] == pytest.approx(14.0)


@pytest.mark.parametrize(
    ("closes", "total", "reason"),
    [
        ([100.0, 90.0, 90.0, 90.0], [0.0, -0.2, 0.0, 0.0], "negative_total_return_residual"),
        ([math.nan, 90.0, 90.0, 90.0], [0.0, 0.05, 0.0, 0.0], "total_return_value_unavailable"),
        ([100.0, 90.0, 90.0, 90.0], [0.0, math.nan, 0.0, 0.0], "total_return_value_unavailable"),
    ],
)
def test_an_unpriceable_group_stays_unsupported_with_its_reason(closes, total, reason):
    events, audit = valued([unsupported()], closes, total)
    assert events.kind.tolist() == ["unsupported"] and events.reason.tolist() == [reason]
    assert audit == [{"event_ids": ["u1"], "status": "unsupported", "reason": reason}]


def test_a_reason_that_is_not_valuable_is_left_unsupported():
    events, _ = valued(
        [unsupported("something_else")], [100.0, 90.0, 90.0, 90.0], [0.0, 0.05, 0.0, 0.0]
    )
    assert events.reason.tolist() == ["something_else"]


def test_a_terminal_group_with_a_split_is_mixed_and_unvalued():
    split = {
        "event_id": "s1",
        "event_date": DAY,
        "permno": 1,
        "kind": "split",
        "ratio": 2.0,
        "sequence": 1,
    }
    terminal_row = unsupported("unqualified_terminal_conversion_or_cash_terms")
    events, _ = valued([split, terminal_row], [100.0, 40.0, 40.0, 40.0], [0.0, 0.0, 0.0, 0.0])
    assert events.set_index("event_id").loc["u1", "reason"] == "mixed_event_group_unvalued"


def test_a_relisted_terminal_group_is_unvalued_unless_classified_as_of_its_session():
    row = unsupported("unqualified_terminal_conversion_or_cash_terms")
    closes, total = [100.0, math.nan, 50.0, 50.0], [0.0, -0.4, 0.0, 0.0]
    events, _ = valued([row], closes, total)
    assert events.reason.tolist() == ["relisted_after_terminal_event"]
    events, _ = valued([row], closes, total, as_of=True)
    assert events.kind.tolist() == ["terminal_value"]


def test_the_effective_close_is_attached_to_each_share_event_for_cash_in_lieu():
    split = {"event_id": "s1", "event_date": DAY, "permno": 1, "kind": "split", "ratio": 1.5}
    events, _ = valued([split], [100.0, 66.0, 66.0, 66.0], [0.0, 0.0, 0.0, 0.0])
    assert events.cash_in_lieu_close.tolist() == [66.0]


# ---------------------------------------------------------------- held_inputs


def historical(rows, closes, total):
    prices = frame(closes)
    return HistoricalInputs(None, prices, prices, pd.DataFrame(rows), frame(total), prices * 0, {})


def test_only_held_action_versions_read_the_valued_event_table():
    data = historical([unsupported()], [100.0, 90.0, 90.0, 90.0], [0.0, 0.05, 0.0, 0.0])
    assert held_inputs(data, "program-events-v4") is data
    assert held_inputs(data, "fixed-share-next-open-v2") is data
    held = held_inputs(data, "program-events-v5")
    assert held is not data and held.events.kind.tolist() == ["distribution_value"]
    assert data.events.kind.tolist() == ["unsupported"]


def test_fallback_versions_classify_terminal_events_from_the_event_session_alone():
    row = unsupported("unqualified_terminal_conversion_or_cash_terms")
    data = historical([row], [100.0, math.nan, 50.0, 50.0], [0.0, -0.4, 0.0, 0.0])
    assert held_inputs(data, "program-events-v5").events.reason.tolist() == [
        "relisted_after_terminal_event"
    ]
    assert held_inputs(data, "program-events-v7").events.kind.tolist() == ["terminal_value"]
