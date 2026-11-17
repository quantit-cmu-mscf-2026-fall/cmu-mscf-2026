"""Deterministic offline broker with partial fills, rejects and ambiguous-submit faults.

This is a simulator, not a broker adapter: it keeps an account in memory, takes
synthetic orders and never opens a connection of any kind.

Faults: reject, timeout_before_accept, timeout_after_accept, duplicate_ack (every report
is delivered twice), disconnect_restart (the order lands, then the link drops until the
broker is reloaded) and ambiguous_submission (the order is lost in flight and the link
drops). A reloaded broker is a reconnected one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from capstone.execution.planning import cost_rate, execution_risk


class SimulatedBroker:
    def __init__(
        self,
        account,
        *,
        profile,
        session,
        fill_fraction=1.0,
        fault=None,
        execution_prices=None,
        commission_bps=0.0,
        execution_adjustment_bps=0.0,
    ):
        self.account = deepcopy(account)
        self.account["positions"] = {int(n): q for n, q in account["positions"].items()}
        self.identity = f"offline-simulator/{profile}"
        self.session = session
        self.fill_fraction, self.fault = fill_fraction, fault
        self.execution_prices = {int(n): float(p) for n, p in (execution_prices or {}).items()}
        self.commission_bps = commission_bps
        self.execution_adjustment_bps = execution_adjustment_bps
        self.orders, self.messages, self.submissions = {}, [], 0
        self.connected = True

    def _online(self):
        if not self.connected:
            raise ConnectionError("simulated broker connection is down")

    def _emit(self, message):
        self.messages.append(message)
        if self.fault == "duplicate_ack":
            self.messages.append(dict(message))

    def confirms_absent(self, key):
        """Authoritative here; a real adapter must not claim this without qualified history."""
        self._online()
        return key not in self.orders

    def preflight(self, plan):
        if (
            plan["execution_session"] != self.session
            or self.identity != f"offline-simulator/{plan['profile']}"
        ):
            raise ValueError("wrong execution session or portfolio profile")
        if "projected_shares" in plan:
            marks = {int(n): p for n, p in plan["reference_prices"].items()}
            marks.update(self.execution_prices)
            if execution_risk(plan, self.snapshot(), marks)["blockers"]:
                raise ValueError("current market or cash violates frozen execution limits")

    def snapshot(self):
        self._online()
        out = deepcopy(self.account)
        out["open_orders"] = [
            key
            for key, record in sorted(self.orders.items())
            if record["filled"] < record["order"]["quantity"] and not record["terminal"]
        ]
        return out

    def events(self):
        self._online()
        return list(self.messages)  # Deliberately repeats callbacks; the journal must deduplicate.

    def submit(self, order):
        key = order["instruction_id"]
        if key in self.orders:
            raise AssertionError("duplicate external submission")
        self._online()
        self.submissions += 1
        if self.fault == "ambiguous_submission":
            self.fault, self.connected = None, False
            raise TimeoutError("simulated ambiguous submission; outcome unknown")
        if self.fault == "timeout_before_accept":
            raise TimeoutError("simulated uncertain delivery")
        self.orders[key] = {"order": deepcopy(order), "filled": 0, "terminal": False}
        if self.fault == "reject":
            self.orders[key]["terminal"] = True
            self._emit({"instruction_id": key, "kind": "rejected", "reason": "synthetic rejection"})
            return
        self._emit({"instruction_id": key, "kind": "acknowledged"})
        fraction = (
            self.fill_fraction.get(key, 1.0)
            if isinstance(self.fill_fraction, Mapping)
            else self.fill_fraction
        )
        amount = min(order["quantity"], int(order["quantity"] * fraction))
        if amount:
            self.fill(key, amount)
        if self.fault == "timeout_after_accept":
            self.fault = None
            raise TimeoutError("simulated disconnect after broker acceptance")
        if self.fault == "disconnect_restart":
            self.fault, self.connected = None, False
            raise ConnectionError("simulated connection loss after broker acceptance")

    def fill(self, key, quantity=None):
        record = self.orders[key]
        order = record["order"]
        quantity = order["quantity"] - record["filled"] if quantity is None else quantity
        if quantity <= 0 or quantity + record["filled"] > order["quantity"]:
            raise ValueError("invalid simulated fill")
        signed = quantity if order["action"] == "BUY" else -quantity
        name = order["permno"]
        raw_price = self.execution_prices.get(name, order["reference_price"])
        price = raw_price * (
            1 + (1 if signed > 0 else -1) * cost_rate(self.execution_adjustment_bps, name)
        )
        commission = abs(signed) * raw_price * cost_rate(self.commission_bps, name)
        self.account["positions"][name] = self.account["positions"].get(name, 0) + signed
        self.account["cash"] -= signed * price + commission
        record["filled"] += quantity
        self._emit(
            {
                "instruction_id": key,
                "kind": "execution",
                "execution_id": f"{key}/{record['filled']}",
                "shares": signed,
                "price": price,
                "commission": commission,
            }
        )

    def cancel(self, key):
        self.orders[key]["terminal"] = True
        self._emit({"instruction_id": key, "kind": "cancelled"})

    def save(self, path):
        data = {
            "execution_prices": self.execution_prices,
            "commission_bps": self.commission_bps,
            "execution_adjustment_bps": self.execution_adjustment_bps,
            "account": self.account,
            "identity": self.identity,
            "session": self.session,
            "fill_fraction": self.fill_fraction,
            "fault": self.fault,
            "orders": self.orders,
            "messages": self.messages,
            "submissions": self.submissions,
        }
        Path(path).write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        broker = cls(
            data["account"],
            profile=data["identity"].split("/", 1)[1],
            session=data["session"],
            fill_fraction=data["fill_fraction"],
            fault=data["fault"],
            execution_prices=data["execution_prices"],
            commission_bps=data["commission_bps"],
            execution_adjustment_bps=data["execution_adjustment_bps"],
        )
        broker.orders, broker.messages, broker.submissions = (
            data["orders"],
            data["messages"],
            data["submissions"],
        )
        return broker
