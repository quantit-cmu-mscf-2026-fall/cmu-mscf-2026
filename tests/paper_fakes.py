"""A fake broker client and SDK module for the paper adapter's tests.

The account ids are made up: paper accounts start "DU", the "live" one does
not. Nothing here touches a network.
"""

from __future__ import annotations

from types import SimpleNamespace

PAPER_ACCOUNT = "DU0000001"
OTHER_PAPER_ACCOUNT = "DU0000002"
LIVE_ACCOUNT = "U0000001"


class FakeClient:
    def __init__(self, accounts=(PAPER_ACCOUNT,), positions=(), quotes=None, bars=None, unknown=()):
        self.accounts = list(accounts)
        self.held = list(positions)  # (account, symbol, shares)
        self.quotes = dict(quotes or {})
        self.bars = dict(bars or {})  # symbol -> bars
        self.unknown = set(unknown)  # symbols qualifyContracts cannot resolve
        self.requests = []  # every history call: the contract and its arguments
        self.connected = None
        self.placed = []
        self.disconnected = False

    def connect(self, host, port, clientId, timeout=None, readonly=False):
        self.connected = {"host": host, "port": port, "clientId": clientId, "readonly": readonly}

    def disconnect(self):
        self.disconnected = True

    def managedAccounts(self):
        return list(self.accounts)

    def positions(self, account=""):
        return [
            SimpleNamespace(
                account=a,
                contract=SimpleNamespace(symbol=s, secType="STK"),
                position=float(n),
                avgCost=1.0,
            )
            for a, s, n in self.held
            if not account or a == account
        ]

    def qualifyContracts(self, *contracts):
        return [c for c in contracts if c.symbol not in self.unknown]

    def reqHistoricalData(self, contract, **request):
        self.requests.append({"symbol": contract.symbol, **request})
        return list(self.bars.get(contract.symbol, []))

    def reqTickers(self, *contracts):
        return [
            SimpleNamespace(contract=c, marketPrice=lambda c=c: self.quotes[c.symbol])
            for c in contracts
        ]

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        return SimpleNamespace(
            order=SimpleNamespace(orderId=len(self.placed)),
            orderStatus=SimpleNamespace(status="Submitted"),
        )


def fake_api(client: FakeClient) -> SimpleNamespace:
    def stock(symbol, exchange, currency):
        return SimpleNamespace(symbol=symbol, exchange=exchange, currency=currency, secType="STK")

    def market_order(action, quantity):
        return SimpleNamespace(action=action, totalQuantity=quantity, account="", tif="")

    return SimpleNamespace(IB=lambda: client, Stock=stock, MarketOrder=market_order)
