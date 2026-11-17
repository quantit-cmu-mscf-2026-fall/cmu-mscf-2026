"""The offline simulated broker and its injected faults (synthetic orders, no network).

Each fault is one way a real order path goes wrong; a reloaded broker is a
reconnected one, and the simulator's answers are the same every run.
"""

from __future__ import annotations

import socket

import pytest

from capstone.execution.simulator import SimulatedBroker


def account(cash=10_000.0, positions=None):
    return {"cash": cash, "positions": dict(positions or {})}


def broker(**kwargs):
    return SimulatedBroker(
        kwargs.pop("account", account()),
        profile=kwargs.pop("profile", "long_only"),
        session=kwargs.pop("session", "2000-01-04"),
        **kwargs,
    )


def order(key="a", permno=1, action="BUY", quantity=10, price=20.0):
    return {
        "instruction_id": key,
        "permno": permno,
        "action": action,
        "quantity": quantity,
        "reference_price": price,
    }


def kinds(b):
    return [(m["instruction_id"], m["kind"]) for m in b.events()]


def test_a_full_fill_moves_shares_and_cash_at_the_reference_price():
    b = broker()
    b.submit(order(quantity=10, price=20.0))
    assert b.account == {"cash": 10_000.0 - 200.0, "positions": {1: 10}}
    assert kinds(b) == [("a", "acknowledged"), ("a", "execution")]
    assert b.snapshot()["open_orders"] == []
    sell = broker(account=account(positions={1: 10}))
    sell.submit(order(action="SELL", quantity=4, price=25.0))
    assert sell.account == {"cash": 10_100.0, "positions": {1: 6}}


def test_a_partial_fill_leaves_the_order_open_until_it_is_filled_or_cancelled():
    b = broker(fill_fraction=0.5)
    b.submit(order(quantity=10))
    assert b.account["positions"] == {1: 5}
    assert b.snapshot()["open_orders"] == ["a"]
    b.fill("a")  # the rest
    assert b.account["positions"] == {1: 10} and b.snapshot()["open_orders"] == []
    with pytest.raises(ValueError, match="invalid simulated fill"):
        b.fill("a", 1)  # nothing left to fill
    c = broker(fill_fraction={"x": 0.3})
    c.submit(order("x", quantity=10))
    c.submit(order("y", quantity=10))  # unlisted orders fill fully
    assert c.account["positions"] == {1: 13}
    c.cancel("x")
    assert c.snapshot()["open_orders"] == [] and kinds(c)[-1] == ("x", "cancelled")


def test_a_fill_pays_the_execution_price_adjustment_and_commission_per_instrument():
    b = broker(
        execution_prices={1: 20.0, 2: 10.0},
        commission_bps={1: 10, 2: 0},
        execution_adjustment_bps={"1": 100, "2": 0},
    )
    b.submit(order("a", 1, quantity=10, price=19.0))  # the execution price wins over the reference
    fill = [m for m in b.events() if m["kind"] == "execution"][0]
    assert fill["price"] == pytest.approx(20.0 * 1.01) and fill["shares"] == 10
    assert fill["commission"] == pytest.approx(10 * 20.0 * 0.001)
    assert b.account["cash"] == pytest.approx(10_000 - 10 * 20.2 - 0.2)
    b.submit(order("b", 2, quantity=5, price=10.0))
    assert b.account["cash"] == pytest.approx(10_000 - 202.2 - 50.0)


def test_a_sale_is_adjusted_against_the_seller():
    b = broker(account=account(positions={1: 10}), execution_adjustment_bps=100)
    b.submit(order(action="SELL", quantity=10, price=20.0))
    assert b.account["cash"] == pytest.approx(10_000 + 10 * 20.0 * 0.99)


def test_a_cost_without_a_qualified_quote_is_refused_at_fill_time():
    b = broker(commission_bps={2: 5})
    with pytest.raises(ValueError, match="instrument 1"):
        b.submit(order(permno=1))


def test_reject_is_terminal_and_never_fills():
    b = broker(fault="reject")
    b.submit(order())
    assert b.account["positions"] == {} and b.snapshot()["open_orders"] == []
    assert kinds(b) == [("a", "rejected")]


def test_timeout_before_accept_leaves_no_order_at_the_broker():
    b = broker(fault="timeout_before_accept")
    with pytest.raises(TimeoutError):
        b.submit(order())
    assert b.confirms_absent("a") and b.account["positions"] == {} and b.submissions == 1


def test_timeout_after_accept_leaves_a_filled_order_the_caller_never_heard_about():
    b = broker(fault="timeout_after_accept")
    with pytest.raises(TimeoutError):
        b.submit(order())
    assert not b.confirms_absent("a") and b.account["positions"] == {1: 10}
    b.submit(order("b"))  # the fault was one-shot
    assert b.account["positions"] == {1: 20}


def test_duplicate_ack_delivers_every_report_twice():
    b = broker(fault="duplicate_ack")
    b.submit(order())
    messages = b.events()
    assert len(messages) == 4 and messages[0] == messages[1] and messages[2] == messages[3]
    assert len({m.get("execution_id") for m in messages if m["kind"] == "execution"}) == 1
    messages.clear()  # the caller's copy is its own
    assert len(b.events()) == 4


def test_disconnect_restart_lands_the_order_then_drops_until_the_broker_is_reloaded(tmp_path):
    b = broker(fault="disconnect_restart", fill_fraction=0.5)
    with pytest.raises(ConnectionError):
        b.submit(order())
    calls = (b.snapshot, b.events, lambda: b.confirms_absent("a"), lambda: b.submit(order("b")))
    for call in calls:
        with pytest.raises(ConnectionError):
            call()
    b.fill("a")  # executes while nobody is connected
    b.save(tmp_path / "broker.json")
    back = SimulatedBroker.load(tmp_path / "broker.json")
    assert back.account == {"cash": 10_000 - 200.0, "positions": {1: 10}}
    assert back.snapshot()["open_orders"] == [] and not back.confirms_absent("a")
    assert back.submissions == 1 and back.fault is None


def test_ambiguous_submission_is_lost_in_flight_and_the_link_drops(tmp_path):
    b = broker(fault="ambiguous_submission")
    with pytest.raises(TimeoutError, match="ambiguous"):
        b.submit(order())
    with pytest.raises(ConnectionError):
        b.snapshot()
    b.save(tmp_path / "broker.json")
    back = SimulatedBroker.load(tmp_path / "broker.json")
    assert back.confirms_absent("a")  # the order never reached the book
    assert back.account == account() and back.submissions == 1
    back.submit(order())  # reconciliation found nothing: the same intent may now be sent
    assert back.account["positions"] == {1: 10}


def test_the_same_instruction_is_never_submitted_twice():
    b = broker()
    b.submit(order())
    with pytest.raises(AssertionError, match="duplicate external submission"):
        b.submit(order())


def test_a_save_and_load_round_trip_is_exact(tmp_path):
    b = broker(
        fill_fraction={"a": 0.5},
        execution_prices={1: 21.0},
        commission_bps=3,
        execution_adjustment_bps=7,
        fault="duplicate_ack",
    )
    b.submit(order(quantity=9))
    b.save(tmp_path / "one.json")
    back = SimulatedBroker.load(tmp_path / "one.json")
    back.save(tmp_path / "two.json")
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()
    assert back.account == b.account and back.events() == b.events()
    assert back.identity == "offline-simulator/long_only" and back.session == "2000-01-04"


def plan(**overrides):
    values = {
        "execution_session": "2000-01-04",
        "profile": "long_only",
        "projected_shares": {"1": 10},
        "reference_prices": {"1": 20.0},
        "capital": 1000,
        "execution_cost_bps": 0,
        "limits": {"max_gross": 1, "max_weight": 1, "max_order_notional": 1000},
    }
    values.update(overrides)
    return values


def test_preflight_refuses_the_wrong_session_or_profile_before_anything_is_sent():
    b = broker(account=account(1000.0))
    b.preflight(plan())
    with pytest.raises(ValueError, match="session or portfolio profile"):
        b.preflight(plan(execution_session="2000-01-05"))
    with pytest.raises(ValueError, match="session or portfolio profile"):
        b.preflight(plan(profile="long_short"))


def test_preflight_rechecks_cash_at_the_simulator_current_marks():
    b = broker(account=account(1000.0))
    b.preflight(plan())
    dear = broker(account=account(1000.0), execution_prices={1: 150.0})
    with pytest.raises(ValueError, match="frozen execution limits"):
        dear.preflight(plan())
    broker().preflight({"execution_session": "2000-01-04", "profile": "long_only"})


def test_the_simulator_opens_no_connection(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the simulator must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    b = broker(fault="duplicate_ack")
    b.submit(order())
    assert b.account["positions"] == {1: 10}
