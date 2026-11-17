import pandas as pd
import pytest

from capstone.pipeline.event_checks import (
    classify_event_returns,
    return_identity,
    verify_event_returns,
)
from capstone.pipeline.historical import HistoricalInputs

DAYS = pd.to_datetime(["2000-01-03", "2000-01-04"])


def event(**values):
    return {"event_id": "e1", "event_date": "2000-01-04", "permno": 1, "kind": "dividend", **values}


def inputs(rows, *, close=99.0, total=0.0):
    prices = pd.DataFrame({1: [100.0, close]}, index=DAYS)
    return HistoricalInputs(
        None,
        prices,
        prices,
        pd.DataFrame(rows),
        pd.DataFrame(total, index=DAYS, columns=[1]),
        prices * 0,
        {},
    )


def test_return_identity_returns_the_implied_return_or_refuses():
    assert return_identity(100.0, 101.0, 0.01, tolerance=1e-9) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="do not reconcile"):
        return_identity(100.0, 101.0, 0.02)
    with pytest.raises(ValueError, match="finite"):
        return_identity(0.0, 101.0, 0.01)
    with pytest.raises(ValueError, match="finite"):
        return_identity(100.0, float("nan"), 0.01)


def test_a_cash_dividend_reconciles_with_the_observed_total_return():
    row = event(amount=1.0, pay_date="2000-01-10")
    result = verify_event_returns(inputs([row], close=99.0, total=0.0))
    assert result["verified"]
    assert result["checks"] == [
        {"date": "2000-01-04", "permno": 1, "status": "verified", "return": 0.0}
    ]


def test_a_split_reconciles_through_the_share_ratio():
    row = event(kind="split", ratio=1.5)
    assert verify_event_returns(inputs([row], close=100 / 1.5))["verified"]


def test_a_cash_settlement_reconciles_without_a_closing_price_after_it():
    row = event(kind="settlement", amount=101.0)
    assert verify_event_returns(inputs([row], close=float("nan"), total=0.01))["verified"]


def test_a_stock_conversion_checks_the_destination_instrument():
    prices = pd.DataFrame({1: [100.0, 50.0], 2: [10.0, 20.0]}, index=DAYS)
    row = event(kind="conversion", ratio=1.0, amount=0.0, new_permno=2)
    data = HistoricalInputs(
        None, prices, prices, pd.DataFrame([row]), prices * 0 - 0.8, prices * 0, {}
    )
    assert verify_event_returns(data)["verified"] is True
    assert data.total_returns.loc[DAYS[1], 1] == -0.8  # the unit return it reconciled against


def test_a_return_that_does_not_reconcile_names_the_event():
    row = event(amount=1.0, pay_date="2000-01-10")
    with pytest.raises(ValueError, match="event identity failed"):
        verify_event_returns(inputs([row], close=98.0))


def test_an_unsupported_event_is_listed_not_verified():
    result = verify_event_returns(inputs([event(kind="unsupported")]))
    assert result["checks"][0]["status"] == "unsupported"


def test_failed_identity_retains_the_row_and_marks_it_unsupported():
    row = event(amount=1.0, pay_date="2000-01-10")
    data = inputs([row], close=98.0)
    classified, audit = classify_event_returns(data)
    assert classified.event_id.tolist() == ["e1"]
    assert classified.kind.tolist() == ["unsupported"]
    assert classified.reason.tolist() == ["unverified_observed_return_identity"]
    assert audit[0]["event_ids"] == ["e1"] and audit[0]["permno"] == 1


def test_a_verified_event_is_left_as_it_was():
    row = event(amount=1.0, pay_date="2000-01-10")
    classified, audit = classify_event_returns(inputs([row], close=99.0))
    assert classified.kind.tolist() == ["dividend"] and audit == []
