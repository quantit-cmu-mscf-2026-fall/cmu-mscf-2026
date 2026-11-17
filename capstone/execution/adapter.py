"""The optional broker adapter for the paper runner: paper accounts only.

This is the one place that touches a broker SDK, and only through the optional
extra (`pip install -e .[paper]`), imported lazily when a real connection is
made. Tests inject a fake client and a fake module, so nothing here is ever
connected in the test suite.

**Everything is refused unless it is provably paper**, in this order, before
anything is imported or opened:

1. `CAPSTONE_PAPER_TRADING` is exactly "1";
2. the configured account id starts with the paper prefix;
3. the port is a paper port.

After connecting, every account the login manages must also be a paper
account, and it is re-checked before every read and every order. A connection
is read-only unless the caller asks otherwise, and `place` re-runs the full
guard (`broker.check_paper`) immediately before sending.

**No defaults that connect.** `connect_from_env` reads host, port, client id and
account from the environment only, and each is required: an unset variable is a
refusal, not a fallback. Account ids are only ever shown masked.

Environment variables:

- `CAPSTONE_PAPER_TRADING`: must be "1"
- `CAPSTONE_PAPER_HOST`, `CAPSTONE_PAPER_PORT`, `CAPSTONE_PAPER_CLIENT_ID`: required
- `CAPSTONE_PAPER_ACCOUNT`: required; the paper account to use
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable, Mapping, Sequence

from capstone.execution.broker import (
    PAPER_FLAG,
    Ack,
    PaperGuardError,
    ShareOrder,
    check_account,
    check_paper,
    check_port,
    mask_account,
)

HOST_VAR = "CAPSTONE_PAPER_HOST"
PORT_VAR = "CAPSTONE_PAPER_PORT"
CLIENT_ID_VAR = "CAPSTONE_PAPER_CLIENT_ID"
ACCOUNT_VAR = "CAPSTONE_PAPER_ACCOUNT"
_SDK_MODULE = "ib_async"  # the optional extra's import name; nothing else names the vendor


def _paper_account(accounts: Sequence[str], expected: str | None) -> str:
    if not accounts:
        raise PaperGuardError("the connected login manages no account")
    for account in accounts:
        check_account(account)
    if expected is not None:
        if expected not in accounts:
            raise PaperGuardError(
                f"expected account {mask_account(expected)} is not managed by this login"
            )
        return expected
    if len(accounts) > 1:
        raise PaperGuardError(
            f"the login manages several paper accounts; name one in {ACCOUNT_VAR}"
        )
    return accounts[0]


class BrokerAdapter:
    """A paper-only broker connection (the `Broker` protocol, plus price history).

    Build it with `connect` or `connect_from_env`. Stocks are SMART-routed USD
    contracts; orders are DAY market orders tagged with the paper account.
    """

    def __init__(self, client, api, *, port: int, account: str, readonly: bool) -> None:
        check_port(port)
        check_account(account)
        self._client, self._api, self._port = client, api, port
        self._account, self._readonly = account, readonly

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        client_id: int,
        *,
        expected_account: str | None = None,
        readonly: bool = True,
        timeout: float = 10.0,
        client=None,
        api=None,
    ) -> BrokerAdapter:
        check_port(port)  # before anything is imported or opened
        if expected_account is not None:
            check_account(expected_account)
        if api is None:
            try:
                api = importlib.import_module(_SDK_MODULE)
            except ImportError as exc:
                raise ImportError("the paper runner needs `pip install -e .[paper]`") from exc
        client = api.IB() if client is None else client
        client.connect(host, port, clientId=client_id, timeout=timeout, readonly=readonly)
        try:
            account = _paper_account(client.managedAccounts(), expected_account)
            return cls(client, api, port=port, account=account, readonly=readonly)
        except BaseException:
            client.disconnect()
            raise

    @classmethod
    def connect_from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        readonly: bool = True,
        client=None,
        api=None,
    ) -> BrokerAdapter:
        """Connect using only the environment; refuse unless it is provably paper.

        The paper flag, host, port, client id and account must all be set (no
        defaults), the account must be paper-prefixed and the port a paper port,
        all checked before the SDK is imported or a connection is opened."""
        env = os.environ if env is None else env
        if env.get(PAPER_FLAG) != "1":
            raise PaperGuardError(f"set {PAPER_FLAG}=1 to use the paper broker")
        values = {}
        for name in (HOST_VAR, PORT_VAR, CLIENT_ID_VAR, ACCOUNT_VAR):
            value = env.get(name, "")
            if not value:
                raise PaperGuardError(f"set {name}: there is no default broker connection")
            values[name] = value
        try:
            port = int(values[PORT_VAR])
            client_id = int(values[CLIENT_ID_VAR])
        except ValueError as exc:
            raise PaperGuardError(f"{PORT_VAR} and {CLIENT_ID_VAR} must be whole numbers") from exc
        return cls.connect(
            values[HOST_VAR],
            port,
            client_id,
            expected_account=values[ACCOUNT_VAR],
            readonly=readonly,
            client=client,
            api=api,
        )

    @property
    def account_id(self) -> str:
        return self._account

    @property
    def port(self) -> int:
        return self._port

    def _recheck(self) -> None:
        _paper_account(self._client.managedAccounts(), self._account)

    def positions(self) -> dict[str, int]:
        self._recheck()
        held: dict[str, int] = {}
        for p in self._client.positions(self._account):
            if p.account == self._account and p.contract.secType == "STK":
                held[p.contract.symbol] = held.get(p.contract.symbol, 0) + round(p.position)
        return {s: n for s, n in sorted(held.items()) if n != 0}

    def _stock(self, symbol: str):
        return self._api.Stock(symbol, "SMART", "USD")

    def prices(self, symbols: Iterable[str]) -> dict[str, float]:
        self._recheck()
        contracts = [self._stock(s) for s in symbols]
        self._client.qualifyContracts(*contracts)
        return {
            t.contract.symbol: float(t.marketPrice()) for t in self._client.reqTickers(*contracts)
        }

    def historical_bars(self, symbol: str, request: Mapping[str, object]) -> list:
        """One stock's historical bars (`request`: the history call's arguments
        after the contract), on this paper-only connection after the account
        re-check. A symbol that does not resolve to one SMART USD stock is refused."""
        self._recheck()
        qualified = [c for c in self._client.qualifyContracts(self._stock(symbol)) if c is not None]
        if len(qualified) != 1 or qualified[0].symbol != symbol:
            raise ValueError(f"{symbol!r} does not resolve to one SMART-routed USD stock")
        return list(self._client.reqHistoricalData(qualified[0], **request))

    def place(self, orders: Sequence[ShareOrder]) -> tuple[Ack, ...]:
        if self._readonly:
            raise PaperGuardError("this connection is read-only (a dry run); it cannot send")
        self._recheck()
        check_paper(self._account, self._port, os.environ)
        acks = []
        for o in orders:
            order = self._api.MarketOrder(o.side, abs(o.shares))
            order.account = self._account
            order.tif = "DAY"
            trade = self._client.placeOrder(self._stock(o.symbol), order)
            acks.append(Ack(o.symbol, o.shares, trade.orderStatus.status, trade.order.orderId))
        return tuple(acks)

    def disconnect(self) -> None:
        self._client.disconnect()
