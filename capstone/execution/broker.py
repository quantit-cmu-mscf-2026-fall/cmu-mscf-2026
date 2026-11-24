"""The paper runner's broker side: the protocol, the paper-only guard and a fake.

**Only a paper account, ever.** A live account can sit one port away from a
paper one, so nothing sends an order unless all three of these hold, checked
by `check_paper` immediately before sending:

1. the account id starts with the paper prefix (`PAPER_PREFIX`);
2. the environment variable `CAPSTONE_PAPER_TRADING` is exactly "1";
3. the port is one of the paper ports (`PAPER_PORTS`).

Account ids are only ever shown masked (`mask_account`): this repository and
its logs are public. `FakeBroker` records every call and order in memory, so
tests never touch a network.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd

PAPER_FLAG = "CAPSTONE_PAPER_TRADING"
PAPER_PORTS = frozenset({4002, 7497})  # the gateway and workstation paper ports
PAPER_PREFIX = "DU"


class PaperGuardError(RuntimeError):
    """Refused: this is not provably the paper account."""


def mask_account(account_id: str) -> str:
    """An account id as logs may show it: 'DU0000001' -> 'DU****01'."""
    if len(account_id) <= 4:
        return "****"
    return f"{account_id[:2]}****{account_id[-2:]}"


def check_port(port: int | None) -> None:
    if port not in PAPER_PORTS:
        raise PaperGuardError(
            f"port {port} is not a paper port {sorted(PAPER_PORTS)}; live ports are never used"
        )


def check_account(account_id: str) -> None:
    if not (isinstance(account_id, str) and account_id.startswith(PAPER_PREFIX)):
        raise PaperGuardError(
            f"account {mask_account(account_id or '')} is not a paper account "
            f"(paper accounts start {PAPER_PREFIX!r})"
        )


def check_paper(account_id: str, port: int | None, env: Mapping[str, str]) -> None:
    """All three paper conditions, or `PaperGuardError`. See the module docstring."""
    check_account(account_id)
    if env.get(PAPER_FLAG) != "1":
        raise PaperGuardError(f"set {PAPER_FLAG}=1 to send orders to the paper account")
    check_port(port)


@dataclass(frozen=True)
class ShareOrder:
    """Trade `shares` (signed, whole) of `symbol`; `price` is the quote it was sized at."""

    permno: int
    symbol: str
    shares: int
    price: float

    @property
    def side(self) -> Literal["BUY", "SELL"]:
        return "BUY" if self.shares > 0 else "SELL"

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.price


@dataclass(frozen=True)
class Ack:
    symbol: str
    shares: int
    status: str
    order_id: int | None = None


class Broker(Protocol):
    @property
    def account_id(self) -> str: ...

    @property
    def port(self) -> int | None: ...

    def positions(self) -> dict[str, int]:
        """Shares held, by symbol."""

    def prices(self, symbols: Iterable[str]) -> dict[str, float]: ...

    def place(self, orders: Sequence[ShareOrder]) -> tuple[Ack, ...]: ...


def _whole(x: float) -> int:
    """Nearest whole number, halves away from zero."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def valid_price(prices: Mapping[str, float], symbol: str) -> float:
    price = prices.get(symbol)
    if price is None or not (math.isfinite(price) and price > 0):
        raise ValueError(f"no valid price for {symbol}: {price!r}")
    return float(price)


def to_share_orders(
    trades: pd.Series,
    symbols: Mapping[int, str],
    prices: Mapping[str, float],
    capital: float,
    *,
    min_notional: float,
) -> tuple[ShareOrder, ...]:
    """Weight trades (permno -> signed weight of `capital`) as whole-share orders,
    sorted by permno. A trade rounding to zero shares, or worth less than
    `min_notional`, is not sent."""
    out = []
    for permno, weight in trades.sort_index().items():
        symbol = symbols[int(permno)]
        price = valid_price(prices, symbol)
        shares = _whole(float(weight) * capital / price)
        order = ShareOrder(int(permno), symbol, shares, price)
        if shares != 0 and order.notional >= min_notional:
            out.append(order)
    return tuple(out)


@dataclass
class FakeBroker:
    """An in-memory broker for tests: it records every call and every order."""

    account_id: str
    port: int | None = 4002
    holdings: dict[str, int] = field(default_factory=dict)
    quotes: dict[str, float] = field(default_factory=dict)
    placed: list[ShareOrder] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def positions(self) -> dict[str, int]:
        self.calls.append("positions")
        return dict(self.holdings)

    def prices(self, symbols: Iterable[str]) -> dict[str, float]:
        self.calls.append("prices")
        return {s: self.quotes[s] for s in symbols if s in self.quotes}

    def place(self, orders: Sequence[ShareOrder]) -> tuple[Ack, ...]:
        self.calls.append("place")
        self.placed.extend(orders)
        return tuple(Ack(o.symbol, o.shares, "Filled", i) for i, o in enumerate(orders, 1))
