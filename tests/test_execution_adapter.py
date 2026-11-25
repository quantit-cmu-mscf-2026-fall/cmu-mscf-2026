"""The paper broker adapter against a fake client and a fake SDK module.

No broker, no network, no SDK: the adapter only ever sees `paper_fakes`, and
the real SDK module is never imported. Every connection value comes from a
made-up environment, and no default connects.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from paper_fakes import (
    LIVE_ACCOUNT,
    OTHER_PAPER_ACCOUNT,
    PAPER_ACCOUNT,
    FakeClient,
    fake_api,
)

from capstone.execution import adapter
from capstone.execution.adapter import BrokerAdapter
from capstone.execution.broker import PAPER_FLAG, PaperGuardError, ShareOrder, mask_account

ENV = {
    PAPER_FLAG: "1",
    adapter.HOST_VAR: "paper-host.invalid",
    adapter.PORT_VAR: "4002",
    adapter.CLIENT_ID_VAR: "7",
    adapter.ACCOUNT_VAR: PAPER_ACCOUNT,
}


@pytest.fixture(autouse=True)
def paper_flag(monkeypatch):
    monkeypatch.setenv(PAPER_FLAG, "1")


def connect(client: FakeClient, *, port=4002, readonly=False, expected=None) -> BrokerAdapter:
    return BrokerAdapter.connect(
        "paper-host.invalid",
        port,
        7,
        expected_account=expected,
        readonly=readonly,
        client=client,
        api=fake_api(client),
    )


def from_env(client: FakeClient, env, **kwargs) -> BrokerAdapter:
    return BrokerAdapter.connect_from_env(env, client=client, api=fake_api(client), **kwargs)


@pytest.mark.parametrize("port", [7496, 4001])
def test_a_live_port_is_refused_before_any_connection_or_import(port):
    client = FakeClient()
    with pytest.raises(PaperGuardError, match="port"):
        BrokerAdapter.connect("h", port, 7, client=client)  # no api: would import the SDK
    assert client.connected is None
    assert adapter._SDK_MODULE not in sys.modules


def test_a_login_managing_a_non_paper_account_is_refused_and_disconnected():
    client = FakeClient(accounts=[LIVE_ACCOUNT])
    with pytest.raises(PaperGuardError, match="paper account"):
        connect(client)
    assert client.disconnected


def test_a_login_mixing_paper_and_live_accounts_is_refused():
    client = FakeClient(accounts=[PAPER_ACCOUNT, LIVE_ACCOUNT])
    with pytest.raises(PaperGuardError):
        connect(client, expected=PAPER_ACCOUNT)
    assert client.disconnected


def test_several_paper_accounts_need_the_expected_one_named():
    client = FakeClient(accounts=[OTHER_PAPER_ACCOUNT, PAPER_ACCOUNT])
    with pytest.raises(PaperGuardError, match=adapter.ACCOUNT_VAR):
        connect(client)
    both = [OTHER_PAPER_ACCOUNT, PAPER_ACCOUNT]
    assert connect(FakeClient(accounts=both), expected=PAPER_ACCOUNT).account_id == PAPER_ACCOUNT
    with pytest.raises(PaperGuardError):
        connect(FakeClient(), expected=OTHER_PAPER_ACCOUNT)


def test_a_non_paper_expected_account_is_refused_before_connecting():
    client = FakeClient()
    with pytest.raises(PaperGuardError) as info:
        connect(client, expected=LIVE_ACCOUNT)
    assert client.connected is None
    assert LIVE_ACCOUNT not in str(info.value)


def test_positions_and_prices_come_from_the_connected_paper_account():
    client = FakeClient(
        positions=[(PAPER_ACCOUNT, "AAA", 10), (OTHER_PAPER_ACCOUNT, "BBB", 5)],
        quotes={"AAA": 12.5},
    )
    broker = connect(client, readonly=True)
    assert client.connected == {
        "host": "paper-host.invalid",
        "port": 4002,
        "clientId": 7,
        "readonly": True,
    }
    assert broker.account_id == PAPER_ACCOUNT and broker.port == 4002
    assert broker.positions() == {"AAA": 10}
    assert broker.prices(["AAA"]) == {"AAA": 12.5}


def test_place_translates_share_orders_into_market_orders(monkeypatch):
    monkeypatch.setenv(PAPER_FLAG, "1")
    client = FakeClient()
    broker = connect(client)
    acks = broker.place([ShareOrder(1, "AAA", 200, 50.0), ShareOrder(2, "BBB", -152, 33.0)])
    sent = [
        (c.symbol, c.exchange, c.currency, o.action, o.totalQuantity, o.account, o.tif)
        for c, o in client.placed
    ]
    assert sent == [
        ("AAA", "SMART", "USD", "BUY", 200, PAPER_ACCOUNT, "DAY"),
        ("BBB", "SMART", "USD", "SELL", 152, PAPER_ACCOUNT, "DAY"),
    ]
    assert [(a.symbol, a.shares, a.status) for a in acks] == [
        ("AAA", 200, "Submitted"),
        ("BBB", -152, "Submitted"),
    ]


def test_place_refuses_without_the_flag(monkeypatch):
    client = FakeClient()
    broker = connect(client)
    monkeypatch.delenv(PAPER_FLAG)
    with pytest.raises(PaperGuardError, match=PAPER_FLAG):
        broker.place([ShareOrder(1, "AAA", 1, 50.0)])
    assert client.placed == []


def test_connect_without_the_flag_raises_and_opens_nothing(monkeypatch):
    monkeypatch.delenv(PAPER_FLAG)
    client = FakeClient()
    with pytest.raises(PaperGuardError, match=PAPER_FLAG):
        connect(client)
    assert client.connected is None


def test_place_uses_the_env_mapping_the_connection_was_given(monkeypatch):
    monkeypatch.delenv(PAPER_FLAG)
    client = FakeClient()
    broker = from_env(client, ENV, readonly=False)
    broker.place([ShareOrder(1, "AAA", 1, 50.0)])
    assert len(client.placed) == 1


def test_place_refuses_a_zero_share_order_before_sending_anything():
    client = FakeClient()
    broker = connect(client)
    with pytest.raises(ValueError, match="zero-share"):
        broker.place([ShareOrder(1, "AAA", 5, 50.0), ShareOrder(2, "BBB", 0, 5.0)])
    assert client.placed == []


def test_place_refuses_if_the_login_no_longer_manages_only_paper_accounts(monkeypatch):
    monkeypatch.setenv(PAPER_FLAG, "1")
    client = FakeClient()
    broker = connect(client)
    client.accounts = [LIVE_ACCOUNT]
    with pytest.raises(PaperGuardError, match="paper account"):
        broker.place([ShareOrder(1, "AAA", 1, 50.0)])
    assert client.placed == []


def test_place_refuses_on_a_read_only_connection(monkeypatch):
    monkeypatch.setenv(PAPER_FLAG, "1")
    client = FakeClient()
    with pytest.raises(PaperGuardError, match="read-only"):
        connect(client, readonly=True).place([ShareOrder(1, "AAA", 1, 50.0)])
    assert client.placed == []


def test_historical_bars_run_on_the_rechecked_connection_for_one_stock():
    client = FakeClient(bars={"AAA": [1, 2, 3]}, unknown={"ZZZ"})
    broker = connect(client, readonly=True)
    request = {"endDateTime": "", "durationStr": "5 D", "barSizeSetting": "1 day"}
    assert broker.historical_bars("AAA", request) == [1, 2, 3]
    assert client.requests == [{"symbol": "AAA", **request}]
    with pytest.raises(ValueError, match="ZZZ"):
        broker.historical_bars("ZZZ", request)
    client.accounts = [LIVE_ACCOUNT]
    with pytest.raises(PaperGuardError):
        broker.historical_bars("AAA", request)


# ---------------------------------------------------------------------------
# Connection values come from the environment only, with no connecting defaults


def test_from_env_connects_read_only_with_the_environment_values():
    client = FakeClient()
    broker = from_env(client, ENV)
    assert client.connected == {
        "host": "paper-host.invalid",
        "port": 4002,
        "clientId": 7,
        "readonly": True,
    }
    assert broker.account_id == PAPER_ACCOUNT


@pytest.mark.parametrize(
    "name",
    [PAPER_FLAG, adapter.HOST_VAR, adapter.PORT_VAR, adapter.CLIENT_ID_VAR, adapter.ACCOUNT_VAR],
)
def test_from_env_refuses_when_any_value_is_missing_or_empty(name):
    for broken in ({k: v for k, v in ENV.items() if k != name}, {**ENV, name: ""}):
        client = FakeClient()
        with pytest.raises(PaperGuardError, match=name):
            from_env(client, broken)
        assert client.connected is None


@pytest.mark.parametrize("flag", ["0", "true", " 1", "yes"])
def test_from_env_refuses_unless_the_flag_is_exactly_1(flag):
    client = FakeClient()
    with pytest.raises(PaperGuardError, match=PAPER_FLAG):
        from_env(client, {**ENV, PAPER_FLAG: flag})
    assert client.connected is None


def test_from_env_refuses_a_non_paper_account_without_masking_leaks():
    client = FakeClient(accounts=[LIVE_ACCOUNT])
    with pytest.raises(PaperGuardError) as info:
        from_env(client, {**ENV, adapter.ACCOUNT_VAR: LIVE_ACCOUNT})
    assert client.connected is None
    assert LIVE_ACCOUNT not in str(info.value)
    assert mask_account(LIVE_ACCOUNT) in str(info.value)


@pytest.mark.parametrize("port", ["7496", "4001", "abc", "4002.5"])
def test_from_env_refuses_a_non_paper_or_malformed_port(port):
    client = FakeClient()
    with pytest.raises(PaperGuardError):
        from_env(client, {**ENV, adapter.PORT_VAR: port})
    assert client.connected is None


def test_from_env_refuses_a_malformed_client_id():
    with pytest.raises(PaperGuardError, match=adapter.CLIENT_ID_VAR):
        from_env(FakeClient(), {**ENV, adapter.CLIENT_ID_VAR: "x"})


def test_from_env_reads_the_process_environment_when_none_is_given(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    client = FakeClient()
    with pytest.raises(PaperGuardError, match=PAPER_FLAG):
        BrokerAdapter.connect_from_env(client=client, api=fake_api(client))
    assert client.connected is None


# ---------------------------------------------------------------------------
# The optional extra


def test_a_missing_extra_says_how_to_install_it(monkeypatch):
    def refuse(name, *args, **kwargs):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", refuse)
    with pytest.raises(ImportError, match=r"\.\[paper\]"):
        BrokerAdapter.connect("h", 4002, 7, client=FakeClient())


def test_the_tests_never_import_the_sdk():
    assert adapter._SDK_MODULE not in sys.modules
