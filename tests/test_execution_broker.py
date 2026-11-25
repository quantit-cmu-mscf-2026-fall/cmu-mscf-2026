"""The paper runner's broker side: the three-way paper guard, account masking,
weights -> shares, and the fake broker. No broker, no network. Account ids are
made up: paper accounts start "DU", the "live" one does not."""

from __future__ import annotations

import pandas as pd
import pytest

from capstone.execution.broker import (
    PAPER_FLAG,
    Ack,
    FakeBroker,
    PaperGuardError,
    ShareOrder,
    check_paper,
    mask_account,
    to_share_orders,
)

PAPER_ACCOUNT = "DU0000001"
LIVE_ACCOUNT = "U0000001"


def test_account_ids_are_masked_to_their_prefix_and_last_two_digits():
    assert mask_account("DU0000001") == "DU****01"
    assert mask_account("U0000001") == "U0****01"
    assert mask_account("DU12") == "****"
    assert mask_account("") == "****"


def test_the_guard_passes_only_a_paper_account_on_a_paper_port_with_the_flag():
    check_paper(PAPER_ACCOUNT, 4002, {PAPER_FLAG: "1"})
    check_paper(PAPER_ACCOUNT, 7497, {PAPER_FLAG: "1"})


@pytest.mark.parametrize("account", [LIVE_ACCOUNT, "", "du0000001", "XDU0000001"])
def test_the_guard_refuses_an_account_that_is_not_a_paper_account(account):
    with pytest.raises(PaperGuardError, match="paper account"):
        check_paper(account, 4002, {PAPER_FLAG: "1"})


@pytest.mark.parametrize("env", [{}, {PAPER_FLAG: ""}, {PAPER_FLAG: "true"}, {PAPER_FLAG: "0"}])
def test_the_guard_refuses_without_the_flag_set_to_exactly_1(env):
    with pytest.raises(PaperGuardError, match=PAPER_FLAG):
        check_paper(PAPER_ACCOUNT, 4002, env)


@pytest.mark.parametrize("port", [4001, 7496, None, 0, 4003])
def test_the_guard_refuses_a_port_that_is_not_a_paper_port(port):
    with pytest.raises(PaperGuardError, match="port"):
        check_paper(PAPER_ACCOUNT, port, {PAPER_FLAG: "1"})


def test_a_refusal_never_spells_out_the_account_id():
    with pytest.raises(PaperGuardError) as info:
        check_paper(LIVE_ACCOUNT, 4002, {PAPER_FLAG: "1"})
    assert LIVE_ACCOUNT not in str(info.value)
    assert mask_account(LIVE_ACCOUNT) in str(info.value)


def test_weights_become_whole_shares_by_hand():
    trades = pd.Series({10001: 0.10, 10002: -0.0501, 10003: 0.0004, 10004: 0.02})
    symbols = {10001: "AAA", 10002: "BBB", 10003: "CCC", 10004: "DDD"}
    quotes = {"AAA": 50.0, "BBB": 33.0, "CCC": 10.0, "DDD": 400.0}
    got = to_share_orders(trades, symbols, quotes, 100_000.0, min_notional=100.0)
    # AAA: 10,000 / 50 = 200. BBB: -5,010 / 33 = -151.8 -> -152.
    # CCC: 40 / 10 = 4 shares, $40 < $100: skipped. DDD: 2,000 / 400 = 5.
    assert got == (
        ShareOrder(10001, "AAA", 200, 50.0),
        ShareOrder(10002, "BBB", -152, 33.0),
        ShareOrder(10004, "DDD", 5, 400.0),
    )
    assert [o.side for o in got] == ["BUY", "SELL", "BUY"]
    assert got[1].notional == 152 * 33.0


def test_half_shares_round_away_from_zero_and_zero_shares_are_dropped():
    trades = pd.Series({1: 0.00025, 2: -0.00025, 3: 0.00004})
    symbols = {1: "A", 2: "B", 3: "C"}
    got = to_share_orders(
        trades, symbols, {"A": 10.0, "B": 10.0, "C": 10.0}, 100_000.0, min_notional=0.0
    )
    assert [(o.symbol, o.shares) for o in got] == [("A", 3), ("B", -3)]


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan")])
def test_a_missing_or_bad_price_is_refused(price):
    with pytest.raises(ValueError, match="price"):
        to_share_orders(pd.Series({1: 0.1}), {1: "A"}, {"A": price}, 1e5, min_notional=0.0)
    with pytest.raises(ValueError, match="price"):
        to_share_orders(pd.Series({1: 0.1}), {1: "A"}, {}, 1e5, min_notional=0.0)


def test_the_fake_broker_records_what_it_is_sent():
    broker = FakeBroker(PAPER_ACCOUNT, 4002, holdings={"A": 3}, quotes={"A": 10.0})
    order = ShareOrder(1, "A", 2, 10.0)
    acks = broker.place([order])
    assert broker.placed == [order]
    assert acks == (Ack("A", 2, "Filled", 1),)
    assert broker.positions() == {"A": 3}
    assert broker.prices(["A", "Z"]) == {"A": 10.0}
    assert broker.calls == ["place", "positions", "prices"]
