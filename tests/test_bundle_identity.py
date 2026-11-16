"""Portable account state and frame hashes (`capstone.execution.bundle`), synthetic only."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from capstone.execution.bundle import ACCOUNT_EVENTS_VERSION, frame_hash, portable_account

BASE = {"positions": {"7": 100}, "open_orders": [], "cash": 1000.0, "capital": 2000.0}


def event(event_id, permno, kind, basis="observed", **terms):
    return {
        "event_id": event_id,
        "permno": permno,
        "kind": kind,
        "effective_session": "2000-01-04",
        "basis": basis,
        **terms,
    }


def account_events(*events):
    return {"version": ACCOUNT_EVENTS_VERSION, "events": list(events)}


def test_a_complete_account_is_returned_as_given_and_is_a_copy():
    account = BASE | {"paper_account_verified": True}
    out = portable_account(account)
    assert out == account and out is not account


def test_corporate_events_are_versioned_and_validated():
    good = BASE | {
        "corporate_events": account_events(
            event("conv-7", 7, "conversion", ratio=0.5, new_permno=9),
            event("conv-7:cil", 7, "cash_in_lieu", "declared_default", cash=-2.5),
        )
    }
    assert portable_account(good) == good
    bad = [
        {"version": "account-corporate-events-v0", "events": []},
        account_events(event("s", 7, "split")),  # ratio missing
        account_events(event("s", 7, "split", ratio=2.0, cash=1.0)),  # unsupported term
        account_events(event("s", 7, "split", "declared_default", ratio=2.0)),
        account_events(event("c", 7, "conversion", ratio=1.0, new_permno=7)),
        account_events(event("d", 7, "dividend", cash=1.0)),
        account_events(event("d", 7, "delisting_cash", cash=math.inf)),
        account_events(event("d", 7, "delisting_cash", cash=1.0), event("d", 8, "merger_cash")),
        account_events(event("d", True, "delisting_cash", cash=1.0)),
        account_events(event("d", 7, "delisting_cash", cash=1.0) | {"effective_session": "x"}),
        account_events(event("s", 7, "split", ratio=-1.0)),
        [event("s", 7, "split", ratio=2.0)],
    ]
    for corporate_events in bad:
        with pytest.raises(ValueError, match="corporate event"):
            portable_account(BASE | {"corporate_events": corporate_events})


def test_receivables_are_portable_event_records_and_nothing_else():
    claim = {"event_id": "d1", "amount": 2.0, "pay_date": "2000-01-06"}
    account = BASE | {"receivables": [claim]}
    assert portable_account(account) == account
    with pytest.raises(ValueError, match="unsupported metadata"):
        portable_account(BASE | {"receivables": [claim | {"holder": "SYNTHETIC_PRIVATE"}]})
    with pytest.raises(ValueError, match="unique"):
        portable_account(BASE | {"receivables": [claim, claim]})
    with pytest.raises(ValueError, match="finite"):
        portable_account(BASE | {"receivables": [claim | {"amount": math.nan}]})
    with pytest.raises(ValueError, match="ISO"):
        portable_account(BASE | {"receivables": [claim | {"pay_date": "6 Jan"}]})
    with pytest.raises(ValueError, match="portable event records"):
        portable_account(BASE | {"receivables": {}})


def test_an_incomplete_account_or_unknown_metadata_is_refused():
    with pytest.raises(ValueError, match="incomplete"):
        portable_account({k: v for k, v in BASE.items() if k != "capital"})
    with pytest.raises(ValueError, match="incomplete"):
        portable_account(["not", "a", "dict"])
    with pytest.raises(ValueError, match="unsupported metadata"):
        portable_account(BASE | {"account_id": "SYNTHETIC_PRIVATE"})
    with pytest.raises(ValueError, match="boolean"):
        portable_account(BASE | {"funds_verified": "yes"})


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"positions": {"7": 1.5}}, "whole shares"),
        ({"positions": {"x7": 1}}, "whole shares"),
        ({"positions": {7: True}}, "whole shares"),
        ({"positions": [7]}, "whole shares"),
        ({"open_orders": [1]}, "instruction references"),
        ({"open_orders": "a"}, "instruction references"),
        ({"cash": math.nan}, "finite"),
        ({"capital": True}, "finite"),
        ({"cash": "10"}, "finite"),
        ({"capital": 0}, "positive"),
        ({"capital": -5.0}, "positive"),
    ],
)
def test_account_amounts_positions_and_orders_are_checked(change, match):
    with pytest.raises(ValueError, match=match):
        portable_account(BASE | change)


def test_positions_may_be_keyed_by_integer_or_decimal_string():
    portable_account(BASE | {"positions": {7: 1, "8": -2}})


def test_a_frame_hash_ignores_row_and_column_order_and_follows_content():
    frame = pd.DataFrame(
        {1: [1.0, 2.0], 2: [3.0, 4.0]}, index=pd.to_datetime(["2000-01-03", "2000-01-04"])
    )
    shuffled = frame.iloc[::-1][[2, 1]]
    assert frame_hash(frame) == frame_hash(shuffled)
    changed = frame.copy()
    changed.iloc[0, 0] = 1.0 + 1e-6
    assert frame_hash(changed) != frame_hash(frame)
    assert len(frame_hash(frame)) == 64
    # Beyond 12 digits a difference is noise, not content.
    noisy = frame.copy()
    noisy.iloc[0, 0] = 1.0 + 1e-15
    assert frame_hash(noisy) == frame_hash(frame)
