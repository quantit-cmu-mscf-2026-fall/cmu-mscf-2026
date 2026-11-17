"""Portable account state and frame hashes for an immutable decision bundle.

A bundle carries only what planning needs from an account: positions, open
order references, cash, capital, receivables and the share and cash events
applied since the last plan. Nothing that identifies an account or a venue is
accepted (`portable_account` refuses unknown keys), and the hash of a frame is
independent of its row and column order. The release and runtime identity a
bundle also records is `capstone.release`.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real

import pandas as pd

# Portable-account record of applied share and cash events, and the terms each event kind carries.
ACCOUNT_EVENTS_VERSION = "account-corporate-events-v1"
EVENT_TERMS = {
    "split": {"ratio"},
    "conversion": {"ratio", "new_permno"},
    "cash_in_lieu": {"cash"},
    "delisting_cash": {"cash"},
    "merger_cash": {"cash"},
}
EVENT_FIELDS = {"event_id", "permno", "kind", "effective_session", "basis"}


def iso_date(value):
    try:
        return str(pd.Timestamp(value).date()) == value
    except (ValueError, TypeError):
        return False


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def instrument(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def corporate_events(record):
    """Validate `{version, events}`: the share and cash events applied since the last plan."""
    if (
        not isinstance(record, dict)
        or set(record) != {"version", "events"}
        or record["version"] != ACCOUNT_EVENTS_VERSION
        or not isinstance(record["events"], list)
    ):
        raise ValueError(f"corporate events must be {ACCOUNT_EVENTS_VERSION} records")
    ids = set()
    for event in record["events"]:
        terms = EVENT_TERMS.get(event.get("kind")) if isinstance(event, dict) else None
        if terms is None or set(event) != EVENT_FIELDS | terms:
            raise ValueError("corporate event kind or terms are unsupported")
        if (
            not isinstance(event["event_id"], str)
            or not event["event_id"]
            or event["event_id"] in ids
        ):
            raise ValueError("corporate event references must be unique")
        ids.add(event["event_id"])
        if not instrument(event["permno"]) or not iso_date(event["effective_session"]):
            raise ValueError("corporate events need an instrument and an ISO effective session")
        if event["basis"] not in {"observed", "declared_default"} or (
            event["basis"] == "declared_default" and event["kind"] != "cash_in_lieu"
        ):
            raise ValueError("corporate event basis is observed, or declared for cash-in-lieu")
        if "ratio" in event and not (finite(event["ratio"]) and event["ratio"] > 0):
            raise ValueError("corporate event share ratio must be positive")
        if "new_permno" in event and (
            not instrument(event["new_permno"]) or event["new_permno"] == event["permno"]
        ):
            raise ValueError("corporate event conversion needs a different instrument")
        if "cash" in event and not finite(event["cash"]):
            raise ValueError("corporate event cash must be finite")


def portable_account(account):
    """Accept only the account state needed by planning, never broker metadata."""
    flags = {
        "paper_account_verified",
        "permissions_verified",
        "funds_verified",
        "data_verified",
        "shorting_verified",
    }
    required = {"positions", "open_orders", "cash", "capital"}
    if not isinstance(account, dict) or not required <= account.keys():
        raise ValueError("portable account state is incomplete")
    if account.keys() - required - flags - {"receivables", "corporate_events"}:
        raise ValueError("portable account state contains unsupported metadata")
    claims = account.get("receivables", [])
    if not isinstance(claims, list):
        raise ValueError("receivables must be portable event records")
    ids = set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {
            "event_id",
            "amount",
            "pay_date",
        }:
            raise ValueError("receivables contain unsupported metadata")
        if (
            not isinstance(claim["event_id"], str)
            or not claim["event_id"]
            or claim["event_id"] in ids
        ):
            raise ValueError("receivable event references must be unique")
        ids.add(claim["event_id"])
        if (
            not isinstance(claim["amount"], Real)
            or isinstance(claim["amount"], bool)
            or not math.isfinite(claim["amount"])
        ):
            raise ValueError("receivable amounts must be finite")
        if not iso_date(claim["pay_date"]):
            raise ValueError("receivable payment dates must be ISO session-independent dates")
    if "corporate_events" in account:
        corporate_events(account["corporate_events"])
    if any(not isinstance(account[k], bool) for k in flags & account.keys()):
        raise ValueError("account qualification flags must be boolean")
    positions = account["positions"]
    if not isinstance(positions, dict) or any(
        not (
            isinstance(n, Integral)
            and not isinstance(n, bool)
            or isinstance(n, str)
            and n.isascii()
            and n.isdecimal()
        )
        or not isinstance(q, Integral)
        or isinstance(q, bool)
        for n, q in positions.items()
    ):
        raise ValueError("account positions require instrument numbers and whole shares")
    if not isinstance(account["open_orders"], list) or any(
        not isinstance(ref, str) for ref in account["open_orders"]
    ):
        raise ValueError("outstanding orders must contain only instruction references")
    for key in ("cash", "capital"):
        value = account[key]
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("account amounts must be finite numbers")
    if account["capital"] <= 0:
        raise ValueError("account capital must be positive")
    return dict(account)


def frame_hash(frame):
    ordered = frame.sort_index().sort_index(axis=1)
    return hashlib.sha256(
        ordered.to_json(orient="split", date_format="iso", double_precision=12).encode()
    ).hexdigest()
