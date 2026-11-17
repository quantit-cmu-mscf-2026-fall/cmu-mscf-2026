import pandas as pd
import pytest

from capstone.pipeline.actions import (
    CASH_IN_LIEU_CLOSE,
    CASH_IN_LIEU_DEFAULT,
    FALLBACK_RULE,
    ActionBook,
    action_rules,
    entitlement,
    return_identity,
)
from capstone.pipeline.crsp_actions import HELD_VALUE_RULE, normalize
from capstone.pipeline.event_checks import classify_event_returns, verify_event_returns
from capstone.pipeline.evidence import cash_in_lieu_defaults
from capstone.pipeline.historical import HistoricalInputs


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
        pd.DataFrame(rows),
        pd.DataFrame(endings, columns=["permno", "delistingdt", "deldlydt", "delpaymenttype"]),
        pd.bdate_range("1995-12-29", "2000-01-12"),
    )


def inputs(events, *, close=99.0, total=0.0):
    prices = pd.DataFrame({1: [100.0, close]}, index=pd.to_datetime(["2000-01-03", "2000-01-04"]))
    return HistoricalInputs(None, prices, prices, events, prices * 0 + total, prices * 0, {})


def test_observed_cash_entitlement_has_a_separate_payment_and_signed_short_obligation():
    events, audit = normalized([distribution()])
    book, positions = ActionBook(events), {1: -10}
    action = book.apply("2000-01-04", positions, 100.0)
    assert action["cash"] == 100.0 and book.value == -10.0
    assert book.apply("2000-01-10", positions, 100.0)["cash"] == 90.0
    assert audit[0]["status"] == "dividend"
    assert verify_event_returns(inputs(events))["verified"]


@pytest.mark.parametrize(
    "change",
    [
        {"dispaydt": pd.NaT},
        {"disdivamt": float("nan")},
        {"dispaymenttype": "FX"},
        {"distype": "SP"},
    ],
)
def test_incomplete_or_non_cash_terms_remain_held_position_blockers(change):
    events, audit = normalized([distribution(**change)])
    assert events.kind.tolist() == ["unsupported"] and audit[0]["status"] == "unsupported"
    with pytest.raises(ValueError, match="unresolved held"):
        ActionBook(events).apply("2000-01-04", {1: 10}, 0.0)


def test_cash_terminal_source_is_counted_once_and_delayed_unknown_cash_is_blocked():
    row = distribution(distype="CP", disdivamt=101.0, dispaydt=pd.Timestamp("2000-01-03"))
    events, audit = normalized([row], [terminal()])
    assert events.kind.tolist() == ["settlement"] and len(audit) == 2
    assert audit[0]["status"] == "represented_by_terminal_distribution"
    assert verify_event_returns(inputs(events, close=101.0, total=0.01))["verified"]
    account = {1: 10}
    assert ActionBook(events).apply("2000-01-04", account, 0.0)["cash"] == 1010.0
    assert account == {}
    events, _ = normalized([{**row, "dispaydt": pd.Timestamp("2000-01-10")}], [terminal()])
    assert set(events.kind) == {"unsupported"}


def test_unit_split_identity_does_not_invent_fractional_cash_for_an_actual_account():
    events, _ = normalized([distribution(distype="FRS", dispaymenttype="SS", disfacshr=0.5)])
    assert verify_event_returns(inputs(events, close=100 / 1.5))["verified"]
    account = {1: 2}
    ActionBook(events).apply("2000-01-04", account, 0.0)
    assert account == {1: 3}
    with pytest.raises(ValueError, match="cash-in-lieu"):
        ActionBook(events).apply("2000-01-04", {1: 1}, 0.0)


@pytest.mark.parametrize("shares", [10, -10])
def test_reverse_split_binary_roundoff_is_not_a_fractional_entitlement(shares):
    events, _ = normalized([distribution(distype="FRS", dispaymenttype="SS", disfacshr=-0.9)])
    account = {1: shares}
    ActionBook(events).apply("2000-01-04", account, 0.0)
    assert account == {1: shares // 10}
    with pytest.raises(ValueError, match="cash-in-lieu"):
        ActionBook(events).apply("2000-01-04", {1: 11}, 0.0)


def test_a_small_real_fraction_still_requires_observed_cash_in_lieu():
    events, _ = normalized([distribution(distype="FRS", dispaymenttype="SS", disfacshr=-1e-10)])
    with pytest.raises(ValueError, match="cash-in-lieu"):
        ActionBook(events).apply("2000-01-04", {1: 1}, 0.0)


def test_failed_identity_retains_event_rows_and_blocks_only_when_held():
    events, _ = normalized([distribution()])
    data = inputs(events, close=98.0)
    with pytest.raises(ValueError, match="event identity failed"):
        verify_event_returns(data)
    classified, audit = classify_event_returns(data)
    assert classified.event_id.tolist() == events.event_id.tolist()
    assert classified.kind.tolist() == ["unsupported"] and len(audit) == 1
    assert ActionBook(classified).apply("2000-01-04", {}, 10.0)["cash"] == 10.0
    with pytest.raises(ValueError, match="unresolved held"):
        ActionBook(classified).apply("2000-01-04", {1: 1}, 10.0)


def test_normalization_preserves_sequence_and_is_row_order_invariant():
    rows = [distribution(), distribution(permno=2, disseqnbr=2, distype="TSOO")]
    a, audit_a = normalized(rows)
    b, audit_b = normalized(rows[::-1])
    pd.testing.assert_frame_equal(a, b)
    assert audit_a == audit_b and len(audit_a) == 2
    with pytest.raises(ValueError, match="duplicate"):
        normalized([rows[0], rows[0]])
    with pytest.raises(ValueError, match="discovery boundary"):
        normalized([distribution(disexdt=pd.Timestamp("2021-01-04"))])
    with pytest.raises(ValueError, match="observed effective date"):
        normalized([distribution()], [terminal(deldlydt=pd.NaT)])


def test_cash_in_lieu_default_recorded_and_counted():
    events, _ = normalized([distribution(distype="FRS", dispaymenttype="SS", disfacshr=0.5)])
    opens = pd.Series({1: 40.0})
    # program-events-v3 keeps the blocker.
    with pytest.raises(ValueError, match="cash-in-lieu"):
        ActionBook(events).apply("2000-01-04", {1: 1}, 0.0, opens=opens)
    # program-events-v4: declared default, fraction x following-session open of the survivor.
    book = ActionBook(events, cash_in_lieu=CASH_IN_LIEU_DEFAULT)
    account = {1: 1}
    assert book.apply("2000-01-04", account, 0.0, opens=opens)["cash"] == 20.0
    assert account == {1: 1}
    assert book.history[-1]["cash_in_lieu"] == {
        "rule": "cash_in_lieu_following_open_v1",
        "permno": 1,
        "fraction": 0.5,
        "price": 40.0,
        "amount": 20.0,
    }
    assert cash_in_lieu_defaults(book.history) == {
        "rule": "cash_in_lieu_following_open_v1",
        "count": 1,
        "amount": 20.0,
        "events": [
            {"event_id": book.history[-1]["event_id"], "date": "2000-01-04", "amount": 20.0}
        ],
    }
    # Without a valid open for the surviving security the default cannot apply.
    with pytest.raises(ValueError, match="cash-in-lieu"):
        ActionBook(events, cash_in_lieu=CASH_IN_LIEU_DEFAULT).apply(
            "2000-01-04", {1: 1}, 0.0, opens=pd.Series({1: float("nan")})
        )


# --------------------------------------------------------- the action book itself


def test_entitlement_separates_whole_shares_from_a_real_fraction():
    assert entitlement(10, 1.5) == (15, 0.0)
    assert entitlement(3, 1.5) == (4, pytest.approx(0.5))
    assert entitlement(-10, 0.1) == (-1, 0.0)
    assert entitlement(1, 1e-10)[0] == 0 and entitlement(1, 1e-10)[1] > 0


def test_declared_rules_are_read_from_the_execution_contract_record():
    assert action_rules({}) == {"cash_in_lieu": None, "held_value": None, "fallback": None}
    record = {
        "fractional_corporate_action": CASH_IN_LIEU_DEFAULT,
        "held_unsupported_action": HELD_VALUE_RULE,
        "held_unqualifiable": FALLBACK_RULE,
    }
    assert action_rules(record) == {
        "cash_in_lieu": CASH_IN_LIEU_DEFAULT,
        "held_value": HELD_VALUE_RULE,
        "fallback": FALLBACK_RULE,
    }


def test_unknown_rules_are_refused_and_a_fallback_needs_the_held_value_rule():
    with pytest.raises(ValueError, match="cash-in-lieu"):
        ActionBook(cash_in_lieu="paid_at_noon")
    with pytest.raises(ValueError, match="valuation rule"):
        ActionBook(held_value="guess")
    with pytest.raises(ValueError, match="fallback"):
        ActionBook(fallback=FALLBACK_RULE)
    ActionBook(held_value=HELD_VALUE_RULE, fallback=FALLBACK_RULE)


def test_a_dividend_is_a_receivable_until_its_payment_date_and_applies_once():
    events, _ = normalized([distribution()])
    book, account = ActionBook(events), {1: 10}
    entitled = book.apply("2000-01-04", account, 0.0)
    assert entitled["cash"] == 0.0 and entitled["distribution_pnl"] == 10.0 and book.value == 10.0
    assert book.apply("2000-01-04", account, 0.0)["distribution_pnl"] == 0.0
    paid = book.apply("2000-01-10", account, 0.0)
    assert paid["cash"] == 10.0 and paid["payments_applied"] == 1 and book.value == 0.0
    assert [row["kind"] for row in book.history] == ["dividend", "payment"]


def test_a_name_that_is_not_held_consumes_its_event_without_a_record():
    events, _ = normalized([distribution()])
    book = ActionBook(events)
    assert book.apply("2000-01-04", {2: 5}, 7.0)["cash"] == 7.0
    assert book.history == [] and book.value == 0.0


def test_a_valued_event_needs_its_declared_rule():
    events = pd.DataFrame(
        [
            {
                "event_id": "v1",
                "event_date": pd.Timestamp("2000-01-04"),
                "permno": 1,
                "kind": "terminal_value",
                "amount": 60.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="needs its rule"):
        ActionBook(events).apply("2000-01-04", {1: 2}, 0.0)
    book, account = ActionBook(events, held_value=HELD_VALUE_RULE), {1: 2}
    result = book.apply("2000-01-04", account, 1.0)
    assert result["cash"] == 121.0 and account == {} and result["retired"] == {1}
    assert book.history[0]["basis"] == HELD_VALUE_RULE


def test_a_fractional_entitlement_falls_back_to_the_effective_close_when_the_open_is_missing():
    events, _ = normalized([distribution(distype="FRS", dispaymenttype="SS", disfacshr=0.5)])
    events["cash_in_lieu_close"] = 30.0
    book = ActionBook(events, cash_in_lieu=CASH_IN_LIEU_DEFAULT, held_value=HELD_VALUE_RULE)
    account = {1: 1}
    assert (
        book.apply("2000-01-04", account, 0.0, opens=pd.Series({1: float("nan")}))["cash"] == 15.0
    )
    assert book.history[-1]["cash_in_lieu"]["rule"] == CASH_IN_LIEU_CLOSE


def test_an_unvalued_held_event_exits_by_fallback_before_its_other_events():
    events = pd.DataFrame(
        [
            {
                "event_id": "u1",
                "event_date": pd.Timestamp("2000-01-04"),
                "permno": 1,
                "kind": "unsupported",
                "reason": "negative_total_return_residual",
            }
        ]
    )
    book = ActionBook(events, held_value=HELD_VALUE_RULE, fallback=FALLBACK_RULE)
    result = book.apply("2000-01-04", {1: 3}, 0.0)
    assert result["fallback_exits"] == [
        {
            "event_id": "u1",
            "reason": "negative_total_return_residual",
            "permno": 1,
            "shares": 3,
        }
    ]
    # Without the fallback rule the same held event blocks, naming its reason.
    with pytest.raises(ValueError, match="unresolved held.*negative_total_return_residual"):
        ActionBook(events, held_value=HELD_VALUE_RULE).apply("2000-01-04", {1: 3}, 0.0)


def test_account_events_report_what_reached_positions_between_two_plans():
    events, _ = normalized([distribution(distype="FRS", dispaymenttype="SS", disfacshr=0.5)])
    events["cash_in_lieu_price"] = 40.0
    book = ActionBook(events)
    book.apply("2000-01-04", {1: 3}, 0.0)
    rows = book.account_events("2000-01-03", "2000-01-05")
    assert [row["kind"] for row in rows] == ["split", "cash_in_lieu"]
    assert rows[1]["cash"] == pytest.approx(20.0) and rows[1]["basis"] == "observed"
    assert book.account_events("2000-01-04", "2000-01-05") == []


def test_return_identity_reconciles_a_value_change_with_the_observed_total_return():
    assert return_identity(100.0, 101.0, 0.01) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="do not reconcile"):
        return_identity(100.0, 101.0, 0.02)
    for bad in ((0.0, 1.0, 0.0), (float("nan"), 1.0, 0.0), (1.0, float("inf"), 0.0)):
        with pytest.raises(ValueError, match="finite observed"):
            return_identity(*bad)
