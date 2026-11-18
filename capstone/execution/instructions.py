"""The single send path: a frozen, journaled plan sent to an explicitly supplied paper session.

    python -m capstone.execution.instructions <plan> --context <file> --journal <file> --paper

`InstructionBroker` is the adapter the journal drives (`identity`, `preflight`,
`snapshot`, `submit`, `events`, `cancel`) over a `PaperSession`: a small
protocol a paper account's connection implements. The session's vendor client
lives outside this repository; importing this module never imports one and
never connects.

**Nothing here builds a connection.** `main` refuses, before anything is
opened, unless all of these hold: `--paper` is given, the plan is fully
qualified (or the run is reconcile-only), `CAPSTONE_PAPER_TRADING` is "1", the
paper context matches the plan, and the caller passed a `session_factory`.
With no factory (the default, and the only thing `python -m` can reach) it
refuses at the last step. The adapter then re-checks, immediately before every
send, that the account is a paper account (`broker.check_paper`), that the
session is not read-only, that the code and runtime are the reviewed release,
that it is today's regular trading session and that fresh marks respect the
frozen limits. Daily source and permission qualification remain an explicit
external prerequisite, recorded in the context file.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from capstone.execution.broker import PaperGuardError, check_paper, mask_account
from capstone.execution.bundle import release_identity, runtime_versions
from capstone.execution.frozen import verify_plan
from capstone.execution.planning import execution_risk

NO_SESSION = (
    "no paper session is configured: this command never builds a broker connection by "
    "default. Call main() with a session_factory from code that owns that connection."
)
CASH_KEYS = {"cash", "available"}
KINDS = {"Cancelled": "cancelled", "ApiCancelled": "cancelled", "Inactive": "rejected"}


class PaperSession(Protocol):
    """What the adapter needs from a connected paper account. Rows are plain mappings."""

    account_id: str
    port: int | None
    readonly: bool

    def recheck(self) -> None:
        """Raise if the connection or the account identity is no longer what it was."""

    def prices(self, symbols: Iterable[str]) -> dict[str, float]: ...

    def holdings(self) -> Sequence[Mapping]:
        """Rows with account, kind ("stock" or other), contract_id, quantity."""

    def cash(self) -> Mapping[str, float]:
        """A fresh USD snapshot with exactly the keys "cash" and "available"."""

    def open_orders(self) -> Sequence[Mapping]:
        """Rows with account, ref (the instruction id, or empty), order_id."""

    def order_states(self) -> Sequence[Mapping]:
        """Open and completed orders: account, ref, order_id, status."""

    def executions(self) -> Sequence[Mapping]:
        """Account, ref, execution_id, side ("BOT"/"SLD"), shares, price, commission
        (None while pending) and currency."""

    def contract_ids(self, symbol: str) -> Sequence[int]: ...

    def next_order_id(self) -> int: ...

    def send(self, order: Mapping) -> None: ...

    def cancel(self, ref: str) -> None: ...

    def on_error(self, callback: Callable[[int, int, str], None]) -> None:
        """Register `callback(order_id, code, message)` for asynchronous refusals."""


def market_order(instruction: Mapping) -> dict:
    """The one order policy the frozen plan may carry: whole-share day market orders."""
    if (
        instruction["order_type"] != "MKT"
        or instruction["tif"] != "DAY"
        or instruction["outside_rth"]
    ):
        raise ValueError("unsupported frozen order policy")
    quantity = instruction["quantity"]
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("broker instructions require positive whole shares")
    if instruction["action"] not in {"BUY", "SELL"}:
        raise ValueError("unsupported order action")
    return {
        "action": instruction["action"],
        "quantity": quantity,
        "order_type": "MKT",
        "tif": "DAY",
        "outside_rth": False,
        "ref": instruction["instruction_id"],
    }


class InstructionBroker:
    def __init__(self, session: PaperSession, instruments: Mapping, context: Mapping):
        self.session, self.context = session, context
        self.instruments = {int(n): item for n, item in instruments.items()}
        self.by_contract = {
            int(item["con_id"]): n
            for n, item in self.instruments.items()
            if item.get("qualified") and item.get("con_id")
        }
        if len(self.by_contract) != len(self.instruments):
            raise ValueError("all broker instruments must be uniquely qualified")
        self.identity = "paper/" + hashlib.sha256(session.account_id.encode()).hexdigest()
        self.messages, self.ids = [], {}
        session.on_error(self._error)

    def _error(self, order_id, code, message):
        if order_id not in self.ids:
            return
        account = self.session.account_id
        clean = str(message).replace(account, mask_account(account))
        self.messages.append(
            {
                "instruction_id": self.ids[order_id],
                "kind": "error",
                "code": code,
                "message": clean,
            }
        )

    def preflight(self, plan):
        verify_plan(plan)
        session = self.session
        if session.readonly:
            raise PaperGuardError("read-only connection cannot submit instructions")
        session.recheck()
        check_paper(session.account_id, session.port, os.environ)
        if self.context.get("capital") != plan["capital"]:
            raise ValueError("paper allocation capital differs from the frozen plan")
        if plan["runtime"] != runtime_versions() or plan["release"] != release_identity(
            plan["release"]["deployment_tag"]
        ):
            raise ValueError("running code or runtime differs from the reviewed frozen release")
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.date().isoformat() != plan["execution_session"]:
            raise ValueError("frozen instruction is not valid for today's session")
        if not (9 * 60 + 30 <= now.hour * 60 + now.minute < 16 * 60):
            raise ValueError("submission is restricted to the regular trading session")
        flags = [
            "regular_session_verified",
            "permissions_verified",
            "funds_verified",
            "data_verified",
        ]
        if plan["profile"] == "long_short":
            flags.append("shorting_verified")
        if self.context.get("session") != plan["execution_session"] or any(
            self.context.get(k) is not True for k in flags
        ):
            raise ValueError(
                "current session, data, permissions, funds and shorting require qualification"
            )
        if not plan["transmittable"]:
            raise ValueError("reference plan has unresolved admission or qualification blockers")
        quotes = session.prices([item["symbol"] for item in self.instruments.values()])
        marks = {n: quotes[item["symbol"]] for n, item in self.instruments.items()}
        risk = execution_risk(plan, self.snapshot(), marks)
        if risk["blockers"]:
            raise ValueError("current marks violate frozen portfolio or order limits")

    def snapshot(self):
        session, positions = self.session, {}
        session.recheck()
        for row in session.holdings():
            if row["account"] != session.account_id or not row["quantity"]:
                continue
            if row["kind"] != "stock" or row["contract_id"] not in self.by_contract:
                raise ValueError("unmapped account holding requires reconciliation")
            if int(row["quantity"]) != row["quantity"]:
                raise ValueError("fractional holdings require a new sizing policy")
            positions[self.by_contract[row["contract_id"]]] = int(row["quantity"])
        values = dict(session.cash())
        if set(values) != CASH_KEYS or not all(math.isfinite(v) for v in values.values()):
            raise ValueError("fresh USD cash and funds are required")
        if values["available"] < 0:
            raise ValueError("available funds are negative")
        opened = [
            row["ref"] or f"unidentified/{row['order_id']}"
            for row in session.open_orders()
            if row["account"] == session.account_id
        ]
        return {
            "positions": positions,
            "cash": values["cash"],
            "capital": float(self.context["capital"]),
            "open_orders": sorted(opened),
        }

    def submit(self, instruction):
        session = self.session
        if session.readonly:
            raise PaperGuardError("read-only connection cannot send")
        session.recheck()
        check_paper(session.account_id, session.port, os.environ)
        item = instruction["instrument"]
        found = list(session.contract_ids(item["symbol"]))
        if found != [item["con_id"]]:
            raise ValueError("instrument contract changed; rebuild the frozen plan")
        order = market_order(instruction) | {
            "account": session.account_id,
            "symbol": item["symbol"],
            "contract_id": item["con_id"],
            "order_id": session.next_order_id(),
        }
        self.ids[order["order_id"]] = instruction["instruction_id"]
        session.send(order)

    def events(self):
        session = self.session
        session.recheck()
        out = list(self.messages)
        for row in session.order_states():
            if row["account"] != session.account_id or not row["ref"]:
                continue
            self.ids[row["order_id"]] = row["ref"]
            out.append(
                {
                    "instruction_id": row["ref"],
                    "kind": KINDS.get(row["status"], "acknowledged"),
                    "broker_status": row["status"],
                }
            )
        for fill in session.executions():
            if fill["account"] != session.account_id or not fill["ref"]:
                continue
            commission = fill["commission"]
            if commission is None or not math.isfinite(commission):
                raise ValueError("execution commission is pending; reconcile before continuing")
            if fill["currency"] != "USD":
                raise ValueError("unsupported execution commission currency; USD is required")
            if int(fill["shares"]) != fill["shares"] or fill["side"] not in {"BOT", "SLD"}:
                raise ValueError("unsupported broker fill quantity or side")
            signed = int(fill["shares"]) * (1 if fill["side"] == "BOT" else -1)
            out.append(
                {
                    "instruction_id": fill["ref"],
                    "kind": "execution",
                    "execution_id": fill["execution_id"],
                    "shares": signed,
                    "price": fill["price"],
                    "commission": commission,
                }
            )
        return out

    def cancel(self, instruction_id):
        session = self.session
        if session.readonly:
            raise PaperGuardError("read-only connection cannot cancel orders")
        session.recheck()
        check_paper(session.account_id, session.port, os.environ)
        if any(
            row["account"] == session.account_id and row["ref"] == instruction_id
            for row in session.open_orders()
        ):
            session.cancel(instruction_id)


def plan_instruments(plan: Mapping) -> dict:
    """Return only the instruments the plan trades or holds."""
    names = set(plan.get("projected_shares", {})) | set(plan.get("account_positions", {}))
    missing = names - set(plan["instruments"])
    if missing:
        raise ValueError("plan trades or holds an instrument it does not describe")
    return {n: plan["instruments"][n] for n in sorted(names, key=int)}


def main(argv=None, *, session_factory: Callable[..., PaperSession] | None = None):
    import argparse
    import json
    from pathlib import Path

    from capstone.execution.frozen import read_plan
    from capstone.execution.journal import abandon_prepared, execute_plan

    parser = argparse.ArgumentParser(
        description="Submit a qualified frozen plan to a paper account"
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--reconcile-only", action="store_true")
    parser.add_argument(
        "--abandon-prepared",
        action="store_true",
        help="with --reconcile-only, retire only never-submitted instructions",
    )
    args = parser.parse_args(argv)
    plan = read_plan(args.plan)
    if args.abandon_prepared and not args.reconcile_only:
        raise ValueError("abandoning unsubmitted instructions requires reconciliation-only mode")
    if not args.paper or (not args.reconcile_only and not plan["transmittable"]):
        raise ValueError(
            "explicit paper mode and a fully qualified plan are required before connecting"
        )
    if os.environ.get("CAPSTONE_PAPER_TRADING") != "1":
        raise PaperGuardError("CAPSTONE_PAPER_TRADING must be 1 before connecting")
    context = json.loads(args.context.read_text(encoding="utf-8"))
    if (
        context.get("session") != plan["execution_session"]
        or context.get("capital") != plan["capital"]
    ):
        raise ValueError("paper context does not match the frozen instruction plan")
    if session_factory is None:
        raise PaperGuardError(NO_SESSION)
    session = session_factory(os.environ, readonly=args.reconcile_only)
    try:
        adapter = InstructionBroker(session, plan_instruments(plan), context)
        result = execute_plan(plan, adapter, args.journal, reconcile_only=args.reconcile_only)
        if args.abandon_prepared:
            count = abandon_prepared(plan, args.journal)
            result = execute_plan(plan, adapter, args.journal, reconcile_only=True)
            result["abandoned_prepared"] = count
        print(json.dumps(result, sort_keys=True))
    finally:
        disconnect = getattr(session, "disconnect", None)
        if disconnect is not None:
            disconnect()


if __name__ == "__main__":
    main()
