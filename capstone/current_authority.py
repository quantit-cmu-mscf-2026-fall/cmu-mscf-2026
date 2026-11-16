"""Durable, revocable authorizations for current trading features: a contract, inert.

This module records permission; it fetches, downloads and stores no data. It ships
unconfigured: no feed, vendor or signer is named anywhere in the repository, and
without `CAPSTONE_CURRENT_AUTHORITY` naming a store, every call refuses. Who may sign
for a vendor, and whether any current feed may be used at all, is a decision for
Quantit and the team (the sponsor's data rules forbid downloading, copying or
extracting sponsor data); this is only the shape of the record a signer would write.

Current data after the research window is not a holdout: it is read every
session. A named person authorizes a vendor's feed over [start, end] (end may be
open) for a declared set of fields; the record is immutable, tamper-evident
(`record_id` = digest of its content) and may be validated any number of times.
Revocation is appended (never deleted) with an effective date.

Record ids, schema 2: the content hashed is vendor, window, fields, signer and an
`issue` number, not the time `at` (kept in the record as a recorded field). Authorizing
the same content again returns the same record while it is unrevoked; after a revocation
it is issued again as `issue` + 1. Schema 1 records (their id hashes `at`) still verify.

- Attaching a NEW current panel requires an ACTIVE (unrevoked) authorization
  covering [panel start, as_of] (`require_active`).
- Verifying a FROZEN current input (fresh process, later, after revocation)
  needs only that the recorded authorization exists and covered [start, as_of]
  without a revocation effective on or before as_of (`require_valid_at`).
- The research holdout is never opened: any range intersecting 2021-2025 is
  refused, at authorization and at every validation, unless the real holdout's
  one look is SPENT. `holdout_authority` and its once-only store are unchanged.

The store is the directory named by `$CAPSTONE_CURRENT_AUTHORITY`; there is no default.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from capstone.artifacts import atomic_json, canonical, digest
from capstone.holdout_authority import (
    REAL_DATASET,
    REAL_PERIOD,
    HoldoutSealed,
    _day,
    _load,
    _name,
)
from capstone.locking import exclusive_file

PREFIX = "current:"
ROOT_ENV = "CAPSTONE_CURRENT_AUTHORITY"
_VENDOR = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
_KEYS = {"schema_version", "vendor", "dataset_id", "start", "end", "fields", "authorized_by", "at"}
SCHEMA_VERSION = 2  # the record id leaves the recorded time `at` out
_KEYS_BY_SCHEMA = {1: _KEYS, 2: _KEYS | {"issue"}}
MAX_ISSUES = 1000  # revoked re-issues of one authorization before it is refused
_REVOCATION = {"record_id", "effective", "revoked_by", "reason", "at", "id"}


def current_dataset(vendor):
    """`current:<vendor>`; vendors are lowercase `[a-z0-9-]`, at most 40 characters."""
    if not isinstance(vendor, str) or not _VENDOR.fullmatch(vendor):
        raise HoldoutSealed("current vendors are lowercase names: [a-z0-9-], at most 40")
    return PREFIX + vendor


def store_root():
    """The configured store; with none configured no current feed is authorized."""
    root = os.environ.get(ROOT_ENV)
    if not root:
        raise HoldoutSealed(f"no current feed is configured: set {ROOT_ENV} to a store directory")
    return Path(root)


def real_holdout_spent():
    """True only when the host's real 2021-2025 authority record is intact and SPENT."""
    # The same file as holdout_authority._path for REAL_DATASET.
    path = Path.home() / ".capstone" / "holdout-authority" / (digest(REAL_DATASET) + ".json")
    return path.exists() and _load(path)["state"] == "SPENT"


def require_outside_sealed_holdout(start, end):
    """Refuse current dates in 2021-2025 while the research holdout is unspent."""
    last = "9999-12-31" if end is None else _day(end)
    if _day(start) <= REAL_PERIOD[1] and last >= REAL_PERIOD[0] and not real_holdout_spent():
        raise HoldoutSealed(
            "current provenance overlaps the sealed research holdout 2021-2025; current "
            "data must start after 2025-12-31 until the real holdout look is spent"
        )


def _now():
    return datetime.now(UTC).isoformat()


def authorize_current(vendor, *, start, end, fields, authorized_by):
    """Explicit operator action; returns the immutable authorization's record_id."""
    names = sorted(set(fields))
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise HoldoutSealed("a current authorization declares its fields")
    if not _name(authorized_by):
        raise HoldoutSealed("a current authorization names who authorized it")
    first, last = _day(start), None if end is None else _day(end)
    if last is not None and last < first:
        raise HoldoutSealed("current authorization window is reversed")
    require_outside_sealed_holdout(first, last)
    content = {
        "schema_version": SCHEMA_VERSION,
        "vendor": vendor,
        "dataset_id": current_dataset(vendor),
        "start": first,
        "end": last,
        "fields": names,
        "authorized_by": authorized_by,
    }
    root = store_root()
    with exclusive_file(root / "current.lock"):
        for issue in range(1, MAX_ISSUES + 1):
            row = content | {"issue": issue}
            record_id = record_identity(row)
            path = root / "authorizations" / f"{record_id}.json"
            if not path.exists():
                atomic_json(path, row | {"at": _now(), "record_id": record_id})
                return record_id
            authorization(record_id)  # an existing record must verify
            if not revocations(record_id):
                return record_id  # the same authorization, still active: reused
    raise HoldoutSealed(f"more than {MAX_ISSUES} revoked issues of one current authorization")


def record_identity(row):
    """A record's id: the digest of its content. Schema 2 leaves the recorded
    time `at` out, so authorizing the same window twice gives the same record (the
    second call reuses it); schema 1 records hashed `at` too and still verify."""
    if row.get("schema_version") == 1:
        return digest({k: v for k, v in row.items() if k != "record_id"})
    return digest({k: v for k, v in row.items() if k not in ("record_id", "at")})


def revoke_current(record_id, *, revoked_by, reason, effective=None):
    """Append a revocation (effective date, default today UTC); nothing is deleted."""
    authorization(record_id)
    if not all(isinstance(v, str) and v.strip() for v in (revoked_by, reason)):
        raise HoldoutSealed("a revocation names who revoked it and why")
    row = {
        "record_id": record_id,
        "effective": _day(effective or datetime.now(UTC).date()),
        "revoked_by": revoked_by,
        "reason": reason,
        "at": _now(),
    }
    row["id"] = digest(row)
    root = store_root()
    with exclusive_file(root / "current.lock"):
        with (root / "revocations.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return row["id"]


def authorization(record_id):
    """The verified record; a changed or missing file is refused."""
    if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{64}", record_id):
        raise HoldoutSealed("current authorization record_id required")
    path = store_root() / "authorizations" / f"{record_id}.json"
    if not path.exists():
        raise HoldoutSealed("no current authorization with this record_id")
    row = json.loads(path.read_text(encoding="utf-8"))
    keys = _KEYS_BY_SCHEMA.get(row.get("schema_version"))
    if (
        keys is None
        or set(row) - {"record_id"} != keys
        or row.get("record_id") != record_id
        or record_identity(row) != record_id
    ):
        raise HoldoutSealed("current authorization record changed")
    return row


def revocations(record_id):
    path = store_root() / "revocations.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if set(row) != _REVOCATION or row["id"] != digest(
                {k: v for k, v in row.items() if k != "id"}
            ):
                raise HoldoutSealed("current revocation log changed")
            rows.append(row)
    return [row for row in rows if row["record_id"] == record_id]


def require_valid_at(record_id, *, vendor, start, end, fields):
    """Frozen check: the authorization covered [start, end] and was unrevoked at end."""
    row = authorization(record_id)
    first, last = _day(start), _day(end)
    if row["dataset_id"] != current_dataset(vendor):
        raise HoldoutSealed("current authorization belongs to another vendor")
    if not row["start"] <= first <= last <= (row["end"] or "9999-12-31"):
        raise HoldoutSealed("current data is outside its authorized window")
    if set(fields) - set(row["fields"]):
        raise HoldoutSealed("current fields are outside the authorized coverage")
    if any(r["effective"] <= last for r in revocations(record_id)):
        raise HoldoutSealed("current authorization was revoked on or before this session")
    require_outside_sealed_holdout(first, last)
    return row


def require_active(record_id, *, vendor, start, end, fields):
    """New current data: as `require_valid_at`, and no revocation at all."""
    row = require_valid_at(record_id, vendor=vendor, start=start, end=end, fields=fields)
    if revocations(record_id):
        raise HoldoutSealed("current authorization is revoked; new current data is refused")
    return row
