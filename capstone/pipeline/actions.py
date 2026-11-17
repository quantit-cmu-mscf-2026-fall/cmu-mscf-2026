"""Observed corporate events, signed entitlements, and payment-date cash accounting."""

from __future__ import annotations

import math

import pandas as pd

from capstone.pipeline.crsp_actions import HELD_VALUE_RULE
from capstone.pipeline.historical import VALUED, validate_events

# Declared default (program-events-v4 only): when a fractional
# split/conversion has no observed cash-in-lieu price, the fraction is paid at the
# effective session's (following-session) open of the surviving security.
CASH_IN_LIEU_DEFAULT = "cash_in_lieu_following_open_v1"
# v5/v7 only (with HELD_VALUE_RULE): when that open is missing too, the fraction is
# paid at the effective session's close (`cash_in_lieu_close`, crsp_actions).
CASH_IN_LIEU_CLOSE = "cash_in_lieu_effective_close_v1"
# contracts.FALLBACK_RULES["held_unqualifiable"].
FALLBACK_RULE = "exit_at_last_observed_close_capped_cost_v1"


def entitlement(shares, ratio):
    """(whole shares, fractional entitlement) of `shares` held through a share ratio."""
    exact = shares * float(ratio)
    if abs(exact - round(exact)) <= 4 * math.ulp(exact):
        exact = float(round(exact))
    return math.trunc(exact), exact - math.trunc(exact)


def action_rules(execution):
    """ActionBook's declared rules from an execution contract record."""
    return {
        "cash_in_lieu": execution.get("fractional_corporate_action"),
        "held_value": execution.get("held_unsupported_action"),
        "fallback": execution.get("held_unqualifiable"),
    }


class ActionBook:
    def __init__(
        self,
        events=None,
        *,
        cash_in_lieu=None,
        current_through=None,
        held_value=None,
        fallback=None,
    ):
        if cash_in_lieu not in {None, CASH_IN_LIEU_DEFAULT}:
            raise ValueError("unknown cash-in-lieu default")
        if held_value not in {None, HELD_VALUE_RULE}:
            raise ValueError("unknown held corporate-action valuation rule")
        if fallback not in {None, FALLBACK_RULE} or fallback and held_value is None:
            raise ValueError("unknown held fallback rule")
        self.held_value, self.fallback = held_value, fallback
        self.events = None if events is None else validate_events(events, current_through)
        self.cash_in_lieu = cash_in_lieu
        self.receivables, self.history = [], []
        self.applied = set()

    @property
    def value(self):
        return sum(row["amount"] for row in self.receivables)

    def _default_cash_in_lieu(self, opens, destination, close):
        price, rule = math.nan, self.cash_in_lieu
        if self.cash_in_lieu is not None and opens is not None and destination in opens:
            price = float(opens[destination])
        if not (math.isfinite(price) and price > 0) and self.held_value is not None:
            price, rule = float(close), CASH_IN_LIEU_CLOSE
        if not math.isfinite(price) or price <= 0:
            raise ValueError("fractional corporate action needs observed cash-in-lieu")
        return price, rule

    def apply(self, day, positions, cash, opens=None):
        """Mutate fixed share positions only on source-defined effective sessions."""
        day = pd.Timestamp(day)
        affected, retired, distribution, exits = set(), set(), 0.0, []
        if self.events is not None:
            today = self.events[self.events.event_date == day]
            if self.fallback:
                exits = self._fallbacks(today, positions)
            exiting = {row["permno"] for row in exits}
            for _, event in today.iterrows():
                if event.event_id in self.applied:
                    continue
                n, kind = int(event.permno), event.kind
                q = 0 if n in exiting else positions.get(n, 0)
                if kind in VALUED and self.held_value is None:
                    raise ValueError(f"valued corporate action needs its rule:{event.event_id}")
                if kind in {"split", "conversion", "settlement", "unsupported", "terminal_value"}:
                    affected.add(n)
                if kind in {"conversion", "settlement", "terminal_value"}:
                    retired.add(n)
                if not q:
                    self.applied.add(event.event_id)
                    continue
                if kind == "unsupported":
                    # v5/v7 name the reason valued_events left it unvalued.
                    detail = f":{event.get('reason')}" if self.held_value else ""
                    raise ValueError(f"unresolved held corporate action:{event.event_id}{detail}")
                record = {
                    "event_id": event.event_id,
                    "date": str(day.date()),
                    "permno": n,
                    "kind": kind,
                    "prior_shares": q,
                }
                if kind in VALUED:
                    record["basis"] = self.held_value
                if kind in {"dividend", "distribution_value"}:
                    amount = q * float(event.amount)
                    self.receivables.append(
                        {
                            "event_id": str(event.event_id),
                            "amount": amount,
                            "pay_date": str(pd.Timestamp(event.pay_date).date()),
                        }
                    )
                    distribution += amount
                    affected.add(n)
                    record.update(amount=amount, pay_date=str(pd.Timestamp(event.pay_date).date()))
                elif kind in {"split", "conversion"}:
                    whole, fraction = entitlement(q, event.ratio)
                    destination = n if kind == "split" else int(event.new_permno)
                    if fraction:
                        price = event.get("cash_in_lieu_price", float("nan"))
                        if not math.isfinite(price) or price < 0:
                            close = event.get("cash_in_lieu_close", math.nan)
                            price, rule = self._default_cash_in_lieu(opens, destination, close)
                            record["cash_in_lieu"] = {
                                "rule": rule,
                                "permno": destination,
                                "fraction": fraction,
                                "price": price,
                                "amount": fraction * price,
                            }
                        cash += fraction * float(price)
                    positions.pop(n)
                    positions[destination] = positions.get(destination, 0) + whole
                    if kind == "conversion":
                        cash += q * float(event.amount)
                    record.update(new_permno=destination, shares=whole)
                elif kind in {"settlement", "terminal_value"}:
                    cash += q * float(event.amount)
                    positions.pop(n)
                    record["amount"] = q * float(event.amount)
                self.history.append(record)
                self.applied.add(event.event_id)
        remaining, paid, payments = [], 0.0, 0
        for claim in self.receivables:
            if pd.Timestamp(claim["pay_date"]) <= day:
                paid += claim["amount"]
                payments += 1
                self.history.append({**claim, "date": str(day.date()), "kind": "payment"})
            else:
                remaining.append(claim)
        self.receivables = remaining
        return {
            "cash": cash + paid,
            "affected": affected,
            "retired": retired,
            "distribution_pnl": distribution,
            "cash_paid": paid,
            "payments_applied": payments,
            "fallback_exits": exits,
        }

    def _fallbacks(self, today, positions):
        """Held names with an `unsupported` event the valuation left unvalued
        exit before any of the session's events for them (the replay books the exit at
        the last observed close). Their other same-session events then find no holding."""
        exits = []
        for event in today[today.kind == "unsupported"].itertuples():
            n = int(event.permno)
            if positions.get(n) and n not in {row["permno"] for row in exits}:
                reason = getattr(event, "reason", None)
                exits.append(
                    {"event_id": str(event.event_id), "reason": reason, "permno": n}
                    | {"shares": positions[n]}
                )
        return exits

    def account_events(self, after, through):
        """Portable `corporate_events` records of the share and cash events applied in
        (after, through]: what reached positions and cash between two plans."""
        start, end = pd.Timestamp(after), pd.Timestamp(through)
        rows = {} if self.events is None else self.events.set_index("event_id")
        found = []
        for record in self.history:
            kind = record["kind"]
            if kind not in {"split", "conversion", "settlement", "terminal_value"}:
                continue
            if not start < pd.Timestamp(record["date"]) <= end:
                continue
            event, q = rows.loc[record["event_id"]], record["prior_shares"]
            base = {
                "event_id": record["event_id"],
                "permno": record["permno"],
                "effective_session": record["date"],
                "basis": "observed",
            }
            if kind in {"settlement", "terminal_value"}:
                basis = {"basis": record["basis"]} if "basis" in record else {}
                cash = {"kind": "delisting_cash", "cash": record["amount"]}
                found.append(base | cash | basis)
                continue
            terms = {"kind": kind, "ratio": float(event.ratio)}
            if kind == "conversion":
                terms["new_permno"] = record["new_permno"]
            found.append(base | terms)
            fraction = entitlement(q, event.ratio)[1]
            if fraction:
                default = record.get("cash_in_lieu")
                price = default["price"] if default else float(event.cash_in_lieu_price)
                found.append(
                    base
                    | {
                        "event_id": f"{record['event_id']}:cash_in_lieu",
                        "kind": "cash_in_lieu",
                        "cash": fraction * price,
                        "basis": "declared_default" if default else "observed",
                    }
                )
            if kind == "conversion" and float(event.amount):
                cash = {"kind": "merger_cash", "cash": q * float(event.amount)}
                found.append(base | {"event_id": f"{record['event_id']}:merger_cash"} | cash)
        return found


def return_identity(previous_value, current_value, expected_return, *, tolerance=1e-8):
    if (
        not all(math.isfinite(v) for v in (previous_value, current_value, expected_return))
        or previous_value <= 0
    ):
        raise ValueError("return identity needs finite observed values")
    actual = current_value / previous_value - 1
    if abs(actual - expected_return) > tolerance:
        raise ValueError("corporate action and observed total return do not reconcile")
    return actual
