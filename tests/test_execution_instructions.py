"""The single send path: the instruction adapter, over a fake paper session.

No network and no vendor client: `FakeSession` is an in-memory `PaperSession`.
The default `main` has no session factory, so it refuses before connecting.
"""

from __future__ import annotations

import ast
import json
import socket
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from capstone.execution import instructions
from capstone.execution.broker import PAPER_FLAG, PaperGuardError
from capstone.execution.contracts import execution_contract
from capstone.execution.frozen import account_hash, identify_plan
from capstone.execution.instructions import InstructionBroker, market_order
from capstone.strategy.reference import ReferenceStrategySpec

ACCOUNT = "DU0000001"
SESSION_DAY = "2000-01-04"


def fixture_plan(**changes):
    spec = ReferenceStrategySpec(
        "716bdaf7915c65be", "synthetic-recovery-fixture", "2000-01-03", side="long_only"
    )
    account = {"positions": {}, "cash": 10000.0, "capital": 10000.0, "open_orders": []}
    order = {
        "permno": 1,
        "instrument": {"symbol": "S1", "con_id": 123},
        "signed_shares": 10,
        "quantity": 10,
        "action": "BUY",
        "reference_price": 100.0,
        "order_type": "MKT",
        "tif": "DAY",
        "outside_rth": False,
    }
    plan = {
        "schema_version": 3,
        "execution_contract": execution_contract(spec),
        "strategy_ids": [spec.id],
        "specifications": {spec.id: spec.to_toml()},
        "account_hash": account_hash(account),
        "account_cash": account["cash"],
        "account_positions": {},
        "profile": "long_only",
        "execution_session": SESSION_DAY,
        "planning_blockers": [],
        "validation_blockers": ["synthetic_unadmitted"],
        "broker_blockers": ["simulation_only"],
        "transmittable": False,
        "orders": [order],
        "capital": 10000,
        "instruments": {"1": {"symbol": "S1", "qualified": True, "con_id": 123}},
    }
    return identify_plan(plan | changes), account


class FakeSession:
    def __init__(self, account_id=ACCOUNT, *, readonly=False, port=4002):
        self.account_id, self.readonly, self.port = account_id, readonly, port
        self.sent, self.cancelled, self.callbacks, self.checks = [], [], [], 0
        self.rows = {"holdings": [], "open": [], "states": [], "fills": []}
        self.cash_values = {"cash": 9000.0, "available": 8000.0}
        self.quotes, self.found, self.next_id = {"S1": 100.0}, [123], 7

    def recheck(self):
        self.checks += 1

    def prices(self, symbols):
        return {s: self.quotes[s] for s in symbols}

    def holdings(self):
        return self.rows["holdings"]

    def cash(self):
        return self.cash_values

    def open_orders(self):
        return self.rows["open"]

    def order_states(self):
        return self.rows["states"]

    def executions(self):
        return self.rows["fills"]

    def contract_ids(self, symbol):
        return self.found

    def next_order_id(self):
        return self.next_id

    def send(self, order):
        self.sent.append(order)

    def cancel(self, ref):
        self.cancelled.append(ref)

    def on_error(self, callback):
        self.callbacks.append(callback)


INSTRUMENTS = {1: {"symbol": "S1", "qualified": True, "con_id": 123}}


@pytest.fixture(autouse=True)
def paper(monkeypatch):
    monkeypatch.setenv(PAPER_FLAG, "1")


def adapter(session=None, context=None):
    session = session or FakeSession()
    return InstructionBroker(session, INSTRUMENTS, context or {"capital": 10000}), session


def instruction():
    return deepcopy(fixture_plan()[0]["orders"][0])


def fill(**changes):
    row = {
        "account": ACCOUNT,
        "ref": instruction()["instruction_id"],
        "execution_id": "fake-execution",
        "side": "BOT",
        "shares": 5,
        "price": 100.0,
        "commission": 0.1,
        "currency": "USD",
    }
    return row | changes


# ---------------------------------------------------------------------------
# The order policy


def test_the_only_order_policy_is_a_whole_share_day_market_order():
    order = market_order(instruction())
    assert order == {
        "action": "BUY",
        "quantity": 10,
        "order_type": "MKT",
        "tif": "DAY",
        "outside_rth": False,
        "ref": instruction()["instruction_id"],
    }


@pytest.mark.parametrize(
    "change, match",
    [
        ({"order_type": "LMT"}, "order policy"),
        ({"tif": "GTC"}, "order policy"),
        ({"outside_rth": True}, "order policy"),
        ({"quantity": 0}, "whole shares"),
        ({"quantity": 1.5}, "whole shares"),
        ({"quantity": True}, "whole shares"),
        ({"action": "SHORT"}, "action"),
    ],
)
def test_any_other_order_policy_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        market_order(instruction() | change)


# ---------------------------------------------------------------------------
# The adapter


def test_submit_checks_the_contract_then_sends_one_identified_order():
    a, session = adapter()
    a.submit(instruction())
    (sent,) = session.sent
    assert (sent["symbol"], sent["contract_id"], sent["quantity"], sent["order_id"]) == (
        "S1",
        123,
        10,
        7,
    )
    assert sent["account"] == ACCOUNT and sent["ref"] == instruction()["instruction_id"]
    assert a.ids == {7: instruction()["instruction_id"]}
    assert a.identity.startswith("paper/") and ACCOUNT not in a.identity


def test_a_changed_contract_is_refused_before_anything_is_sent():
    a, session = adapter()
    session.found = [999]
    with pytest.raises(ValueError, match="contract changed"):
        a.submit(instruction())
    session.found = [123, 124]
    with pytest.raises(ValueError, match="contract changed"):
        a.submit(instruction())
    assert session.sent == []


def test_a_read_only_session_can_neither_send_nor_cancel():
    a, session = adapter(FakeSession(readonly=True))
    with pytest.raises(PaperGuardError, match="read-only"):
        a.submit(instruction())
    with pytest.raises(PaperGuardError, match="read-only"):
        a.cancel("x")
    with pytest.raises(PaperGuardError, match="read-only"):
        a.preflight(fixture_plan()[0])
    assert session.sent == [] and session.cancelled == []


@pytest.mark.parametrize("account, port", [("LIVE0001", 4002), (ACCOUNT, 4001)])
def test_a_non_paper_account_or_port_is_refused_before_sending(account, port):
    a, session = adapter(FakeSession(account, port=port))
    with pytest.raises(PaperGuardError):
        a.submit(instruction())
    with pytest.raises(PaperGuardError):
        a.cancel("x")
    assert session.sent == []


def test_the_paper_flag_is_required_before_sending(monkeypatch):
    monkeypatch.delenv(PAPER_FLAG)
    a, session = adapter()
    with pytest.raises(PaperGuardError, match=PAPER_FLAG):
        a.submit(instruction())
    assert session.sent == []


def test_every_broker_instrument_must_be_uniquely_qualified():
    with pytest.raises(ValueError, match="uniquely qualified"):
        InstructionBroker(FakeSession(), {1: {"symbol": "S1", "qualified": False}}, {})
    twice = {1: INSTRUMENTS[1], 2: INSTRUMENTS[1]}
    with pytest.raises(ValueError, match="uniquely qualified"):
        InstructionBroker(FakeSession(), twice, {})


def test_snapshot_maps_holdings_cash_and_open_orders():
    a, session = adapter()
    session.rows["holdings"] = [
        {"account": ACCOUNT, "kind": "stock", "contract_id": 123, "quantity": 10},
        {"account": "OTHER", "kind": "stock", "contract_id": 555, "quantity": 3},
        {"account": ACCOUNT, "kind": "stock", "contract_id": 555, "quantity": 0},
    ]
    session.rows["open"] = [
        {"account": ACCOUNT, "ref": "b", "order_id": 2},
        {"account": ACCOUNT, "ref": "", "order_id": 3},
        {"account": "OTHER", "ref": "z", "order_id": 4},
    ]
    assert a.snapshot() == {
        "positions": {1: 10},
        "cash": 9000.0,
        "capital": 10000,
        "open_orders": ["b", "unidentified/3"],
    }


@pytest.mark.parametrize(
    "row, match",
    [
        ({"kind": "option", "contract_id": 123, "quantity": 1}, "unmapped"),
        ({"kind": "stock", "contract_id": 999, "quantity": 1}, "unmapped"),
        ({"kind": "stock", "contract_id": 123, "quantity": 1.5}, "fractional"),
    ],
)
def test_an_unmapped_or_fractional_holding_requires_reconciliation(row, match):
    a, session = adapter()
    session.rows["holdings"] = [{"account": ACCOUNT} | row]
    with pytest.raises(ValueError, match=match):
        a.snapshot()


@pytest.mark.parametrize(
    "cash",
    [
        {"cash": 1.0},
        {"cash": 1.0, "available": float("nan")},
        {"cash": 1.0, "available": -1.0},
        {"cash": 1.0, "available": 1.0, "extra": 1.0},
    ],
)
def test_stale_missing_or_negative_cash_is_refused(cash):
    a, session = adapter()
    session.cash_values = cash
    with pytest.raises(ValueError, match="cash and funds|negative"):
        a.snapshot()


def test_events_report_acknowledgements_fills_and_cancellations():
    a, session = adapter()
    ref = instruction()["instruction_id"]
    session.rows["states"] = [
        {"account": ACCOUNT, "ref": ref, "order_id": 7, "status": "Submitted"},
        {"account": ACCOUNT, "ref": "c", "order_id": 8, "status": "Cancelled"},
        {"account": ACCOUNT, "ref": "r", "order_id": 9, "status": "Inactive"},
        {"account": ACCOUNT, "ref": "", "order_id": 10, "status": "Submitted"},
        {"account": "OTHER", "ref": "o", "order_id": 11, "status": "Submitted"},
    ]
    assert [(e["instruction_id"], e["kind"]) for e in a.events()] == [
        (ref, "acknowledged"),
        ("c", "cancelled"),
        ("r", "rejected"),
    ]
    session.rows["fills"] = [fill(), fill(side="SLD", execution_id="e2", account="OTHER")]
    event = a.events()[-1]
    assert event["kind"] == "execution" and event["shares"] == 5 and event["commission"] == 0.1
    session.rows["fills"] = [fill(side="SLD")]
    assert a.events()[-1]["shares"] == -5


@pytest.mark.parametrize(
    "change, match",
    [
        ({"commission": None}, "commission is pending"),
        ({"commission": float("nan")}, "commission is pending"),
        ({"currency": "EUR"}, "commission currency"),
        ({"shares": 1.5}, "quantity or side"),
        ({"side": "SHORT"}, "quantity or side"),
    ],
)
def test_an_unreconcilable_fill_is_refused(change, match):
    a, session = adapter()
    session.rows["fills"] = [fill(**change)]
    with pytest.raises(ValueError, match=match):
        a.events()


def test_asynchronous_refusals_are_attributed_and_never_show_the_account():
    a, session = adapter()
    (callback,) = session.callbacks
    callback(7, 201, "ignored: not ours")
    assert a.messages == []
    a.submit(instruction())
    callback(7, 201, f"refused for {ACCOUNT}")
    (message,) = a.messages
    assert message["kind"] == "error" and message["code"] == 201
    assert ACCOUNT not in message["message"] and "DU****01" in message["message"]
    assert message["instruction_id"] == instruction()["instruction_id"]


def test_cancel_cancels_only_our_open_order_by_reference():
    a, session = adapter()
    session.rows["open"] = [
        {"account": ACCOUNT, "ref": "mine", "order_id": 1},
        {"account": ACCOUNT, "ref": "other", "order_id": 2},
        {"account": "OTHER", "ref": "mine", "order_id": 3},
    ]
    a.cancel("mine")
    assert session.cancelled == ["mine"]


# ---------------------------------------------------------------------------
# Preflight


class Clock(datetime):
    stamp = datetime(2000, 1, 4, 11, 0, tzinfo=ZoneInfo("America/New_York"))

    @classmethod
    def now(cls, tz=None):
        return cls.stamp


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setattr(instructions, "verify_plan", lambda plan: plan)
    monkeypatch.setattr(instructions, "runtime_versions", lambda: {"python": "x"})
    monkeypatch.setattr(instructions, "release_identity", lambda tag: {"deployment_tag": tag})
    monkeypatch.setattr(instructions, "datetime", Clock)
    monkeypatch.setattr(
        instructions, "execution_risk", lambda plan, account, marks: {"blockers": []}
    )
    context = {
        "capital": 10000,
        "session": SESSION_DAY,
        "regular_session_verified": True,
        "permissions_verified": True,
        "funds_verified": True,
        "data_verified": True,
    }
    plan = fixture_plan(
        transmittable=True,
        capital=10000,
        runtime={"python": "x"},
        release={"deployment_tag": "t"},
    )[0]
    return plan, context


def test_preflight_passes_only_with_every_qualification(ready):
    plan, context = ready
    a, session = adapter(context=context)
    a.preflight(plan)
    for key in ("permissions_verified", "data_verified", "funds_verified"):
        bad, _ = adapter(context=context | {key: False})
        with pytest.raises(ValueError, match="qualification"):
            bad.preflight(plan)


def test_preflight_refuses_a_stale_session_a_closed_market_and_a_drifted_release(
    ready, monkeypatch
):
    plan, context = ready
    a, _ = adapter(context=context)
    with pytest.raises(ValueError, match="capital differs"):
        adapter(context=context | {"capital": 1})[0].preflight(plan)
    monkeypatch.setattr(instructions, "runtime_versions", lambda: {"python": "y"})
    with pytest.raises(ValueError, match="differs from the reviewed"):
        a.preflight(plan)
    monkeypatch.setattr(instructions, "runtime_versions", lambda: {"python": "x"})
    Clock.stamp = Clock.stamp.replace(hour=17)
    with pytest.raises(ValueError, match="regular trading session"):
        a.preflight(plan)
    Clock.stamp = Clock.stamp.replace(hour=11, day=5)
    with pytest.raises(ValueError, match="today's session"):
        a.preflight(plan)
    Clock.stamp = Clock.stamp.replace(day=4)


def test_preflight_refuses_a_plan_that_is_not_transmittable_or_breaches_at_current_marks(
    ready, monkeypatch
):
    plan, context = ready
    a, _ = adapter(context=context)
    with pytest.raises(ValueError, match="unresolved"):
        a.preflight(plan | {"transmittable": False})
    monkeypatch.setattr(
        instructions, "execution_risk", lambda plan, account, marks: {"blockers": ["gross"]}
    )
    with pytest.raises(ValueError, match="current marks"):
        a.preflight(plan)


# ---------------------------------------------------------------------------
# main: nothing is connected unless a factory is passed


def write(tmp_path, plan, **context):
    plan_path, context_path = tmp_path / "plan.json", tmp_path / "context.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    context_path.write_text(json.dumps(context), encoding="utf-8")
    return [
        str(plan_path),
        "--context",
        str(context_path),
        "--journal",
        str(tmp_path / "j.db"),
    ]


def trap():
    def factory(*args, **kwargs):
        raise AssertionError("a session was built")

    return factory


def test_an_unqualified_plan_is_refused_before_a_session_is_built(tmp_path):
    plan = fixture_plan()[0]
    rest = write(tmp_path, plan, session=SESSION_DAY, capital=10000)
    with pytest.raises(ValueError, match="fully qualified"):
        instructions.main([*rest, "--paper"], session_factory=trap())
    with pytest.raises(ValueError, match="explicit paper mode"):
        instructions.main(rest, session_factory=trap())


def test_abandoning_requires_reconciliation_only(tmp_path):
    plan = fixture_plan()[0]
    rest = write(tmp_path, plan)
    with pytest.raises(ValueError, match="reconciliation-only"):
        instructions.main([*rest, "--paper", "--abandon-prepared"], session_factory=trap())


def test_the_paper_flag_is_required_before_a_session_is_built(tmp_path, monkeypatch):
    monkeypatch.delenv(PAPER_FLAG)
    plan = fixture_plan()[0]
    rest = write(tmp_path, plan, session=SESSION_DAY, capital=10000)
    with pytest.raises(PaperGuardError, match=PAPER_FLAG):
        instructions.main([*rest, "--paper", "--reconcile-only"], session_factory=trap())


def test_a_mismatched_context_is_refused_before_a_session_is_built(tmp_path):
    plan = fixture_plan()[0]
    rest = write(tmp_path, plan, session="1999-01-01", capital=10000)
    with pytest.raises(ValueError, match="does not match"):
        instructions.main([*rest, "--paper", "--reconcile-only"], session_factory=trap())


def test_the_default_send_path_has_no_session_and_never_touches_the_network(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a network call was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert instructions.main.__kwdefaults__ == {"session_factory": None}
    plan = fixture_plan()[0]
    rest = write(tmp_path, plan, session=SESSION_DAY, capital=10000)
    with pytest.raises(PaperGuardError, match="no paper session is configured"):
        instructions.main([*rest, "--paper", "--reconcile-only"])
    assert not (tmp_path / "j.db").exists()


def test_a_supplied_session_is_read_only_for_reconciliation_and_always_disconnected(tmp_path):
    plan = fixture_plan()[0]
    rest = write(tmp_path, plan, session=SESSION_DAY, capital=10000)
    session = FakeSession(readonly=True)
    session.disconnected = False
    session.disconnect = lambda: setattr(session, "disconnected", True)
    seen = {}

    def factory(env, *, readonly):
        seen["readonly"] = readonly
        return session

    with pytest.raises(ValueError, match="no saved execution record"):
        instructions.main([*rest, "--paper", "--reconcile-only"], session_factory=factory)
    assert seen == {"readonly": True} and session.disconnected and session.sent == []


def test_the_runner_names_this_module_as_the_send_path():
    from capstone.execution.runner import SEND_PATH

    assert "capstone.execution.instructions" in SEND_PATH and "--paper" in SEND_PATH


def test_the_send_path_imports_no_client_library_and_opens_no_socket():
    source = Path(instructions.__file__).read_text(encoding="utf-8")
    for word in ("socket", ".connect(", "getenv"):
        assert word not in source
    allowed = {"capstone", "pandas", "numpy"} | set(sys.stdlib_module_names)
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    assert imported <= allowed, sorted(imported - allowed)
