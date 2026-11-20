"""One durable dataset look; gate registrations and output directories cannot renew it.

A protocol is prepared (frozen in memory, no access), authorized by a named person
(the gate owner; a Quantit co-sign is optional), and opened exactly once. Opening
spends the look durably before any reader runs and records it in the run ledger, so a
crashed or uncertain read is never retried and the ledger shows every look.

Nothing in this module opens the real holdout by itself: the real store is reached
only through an explicit `authorize` call by a person, and tests use synthetic
dataset ids with a temporary store.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from capstone import runlog
from capstone.artifacts import atomic_json, canonical, digest
from capstone.locking import exclusive_file
from capstone.release import runtime_versions

REAL_DATASET = "real:crsp-us-equities"
REAL_PERIOD = ("2021-01-01", "2025-12-31")
DISCOVERY_END = "2020-12-31"
LOOK_RUN_NAME = "holdout-look"  # the ledger entry name; tagged LOOK_TAG
LOOK_TAG = "holdout-look"
_PLACEHOLDERS = {"", "unknown", "tbd", "none", "n/a", "open manager decision"}
_ACTIVE = ContextVar("holdout_access", default=None)
_FIELDS = {
    "schema_version",
    "dataset",
    "period",
    "source_version",
    "sources",
    "config_hash",
    "generator_id",
    "procedure_id",
    "candidate_ids",
    "calibration",
    "code_id",
    "runtime_id",
}


class HoldoutSealed(RuntimeError):
    """No exact immutable authorization exists for this final protocol."""


class HoldoutSpent(HoldoutSealed):
    """The dataset's terminal look was already reserved, even if its reader failed."""


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def code_identity():
    root = Path(__file__).resolve().parent  # the capstone package
    return digest(
        {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
    )


def runtime_identity():
    return digest(runtime_versions())


@dataclass(frozen=True)
class FinalHoldoutProtocol:
    """Canonical software/data/selection binding, prepared separately from authorization."""

    payload_json: str

    def __post_init__(self):
        row = self.payload
        if (
            not isinstance(row, dict)
            or set(row) != _FIELDS
            or type(row["schema_version"]) is not int
            or row["schema_version"] != 1
        ):
            raise ValueError("unsupported final holdout protocol")
        if self.payload_json != canonical(row):
            raise ValueError("final holdout protocol must be canonical")
        if not isinstance(row["dataset"], str) or not (
            row["dataset"] == REAL_DATASET or row["dataset"].startswith("synthetic:")
        ):
            raise ValueError("explicit real or synthetic dataset namespace required")
        period = row["period"]
        if (
            not isinstance(period, list)
            or len(period) != 2
            or any(date.fromisoformat(d).isoformat() != d for d in period)
            or period[0] > period[1]
        ):
            raise ValueError("canonical sealed period required")
        if row["dataset"] == REAL_DATASET and tuple(period) != REAL_PERIOD:
            raise ValueError("the real sealed period cannot be shifted or subdivided")
        for name in ("config_hash", "procedure_id", "code_id", "runtime_id"):
            if not _hash(row[name]):
                raise ValueError(f"immutable {name} required")
        for name in ("generator_id", "source_version"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"explicit {name} required")
        sources = row["sources"]
        if (
            not isinstance(sources, dict)
            or not sources
            or any(
                not isinstance(name, str)
                or Path(name).name != name
                or name != name.casefold()
                or not _hash(checksum)
                for name, checksum in sources.items()
            )
        ):
            raise ValueError("exact source names and SHA-256 identities required")
        ids = row["candidate_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                not isinstance(i, str)
                or not re.fullmatch(r"[0-9a-f]{16}(?:-[0-9a-f]{8})?|[0-9a-f]{64}", i)
                for i in ids
            )
            or ids != sorted(set(ids))
        ):
            raise ValueError("distinct immutable candidate identities required")
        calibration = row["calibration"]
        if calibration is not None and (
            not isinstance(calibration, dict)
            or set(calibration)
            != {"evidence_id", "evidence", "generator_id", "procedure_id", "config_hash"}
            or not _hash(calibration["evidence_id"])
            or any(
                calibration[name] != row[name]
                for name in ("generator_id", "procedure_id", "config_hash")
            )
        ):
            raise ValueError("calibration belongs to a different generator, procedure or policy")
        if calibration is not None:
            evidence = calibration["evidence"]
            if (
                not isinstance(evidence, dict)
                or evidence.get("qualified") is not True
                or calibration["evidence_id"] != digest(evidence)
                or any(
                    evidence.get(name) != row[name]
                    for name in ("generator_id", "procedure_id", "config_hash")
                )
            ):
                raise ValueError("calibration evidence identity or qualified scope changed")

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def id(self):
        return digest(self.payload)


def prepare_protocol(
    config,
    *,
    dataset,
    period,
    source_version,
    sources,
    procedure_id,
    candidate_ids,
    calibration=None,
):
    """Freeze a protocol in memory; this grants no access and touches no authority store."""
    return FinalHoldoutProtocol(
        canonical(
            {
                "schema_version": 1,
                "dataset": dataset,
                "period": list(period),
                "source_version": source_version,
                "sources": sources,
                "config_hash": config.config_hash,
                "generator_id": config.generator,
                "procedure_id": procedure_id,
                "candidate_ids": sorted(candidate_ids),
                "calibration": calibration,
                "code_id": code_identity(),
                "runtime_id": runtime_identity(),
            }
        )
    )


def save_protocol(protocol, path):
    """Save reviewable sealed content; another protocol cannot overwrite its bytes."""
    value = {"protocol_id": protocol.id, "protocol": protocol.payload, "authorized": False}
    path = Path(path)
    with exclusive_file(path.with_suffix(path.suffix + ".lock")):
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise HoldoutSealed("frozen final protocol already differs")
        else:
            atomic_json(path, value)
    return protocol.id


def _path(protocol, synthetic_root):
    dataset = protocol.payload["dataset"]
    if dataset == REAL_DATASET:
        if synthetic_root is not None:
            raise HoldoutSealed("real authority cannot use a caller-selected store")
        root = Path.home() / ".capstone" / "holdout-authority"
    else:
        if synthetic_root is None:
            raise HoldoutSealed("synthetic authority needs a separate explicit temporary root")
        root = Path(synthetic_root) / "synthetic"
    # The key excludes config, output, source version and requested sub-period.
    return root / (digest(dataset) + ".json")


def _load(path):
    text = path.read_text()
    row = json.loads(text)
    if text != canonical(row) + "\n":
        raise HoldoutSealed("durable holdout authority bytes changed")
    content = {k: v for k, v in row.items() if k != "record_id"}
    if (
        row.get("record_id") != digest(content)
        or set(content)
        != {"schema_version", "protocol", "protocol_id", "authorization", "state", "receipt"}
        or type(content["schema_version"]) is not int
        or content["schema_version"] != 1
    ):
        raise HoldoutSealed("durable holdout authority changed or is incompatible")
    frozen = FinalHoldoutProtocol(canonical(row["protocol"]))
    approval = row["authorization"]
    if (
        not isinstance(approval, dict)
        or set(approval) != {"protocol_id", "approval_id", "approved_by", "cosigned_by"}
        or not _hash(approval["approval_id"])
        or not _name(approval["approved_by"])
        or (approval["cosigned_by"] is not None and not _name(approval["cosigned_by"]))
    ):
        raise HoldoutSealed("durable operator approval is malformed")
    if (
        frozen.id != row["protocol_id"]
        or row["authorization"]["protocol_id"] != frozen.id
        or row["state"] not in {"AUTHORIZED", "SPENT"}
        or (row["state"] == "SPENT") != (row["receipt"] is not None)
    ):
        raise HoldoutSealed("durable holdout authority binding changed")
    if row["receipt"] is not None:
        receipt = row["receipt"]
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"protocol_id", "candidates", "dates", "reserved_utc", "id"}
            or receipt["protocol_id"] != frozen.id
            or receipt["id"] != digest({k: v for k, v in receipt.items() if k != "id"})
        ):
            raise HoldoutSealed("terminal holdout receipt changed")
    return row


def _write(path, row):
    row = {k: v for k, v in row.items() if k != "record_id"}
    atomic_json(path, row | {"record_id": digest(row)})


def _check(protocol, row):
    if row["protocol_id"] != protocol.id:
        raise HoldoutSealed("dataset already belongs to a different frozen final protocol")
    if (
        protocol.payload["code_id"] != code_identity()
        or protocol.payload["runtime_id"] != runtime_identity()
    ):
        raise HoldoutSealed("frozen final code or runtime is stale")


def status(protocol, *, synthetic_root=None):
    """Read-only inspection; missing state remains sealed and creates nothing."""
    path = _path(protocol, synthetic_root)
    if not path.exists():
        return {"state": "SEALED", "protocol_id": protocol.id}
    row = _load(path)
    _check(protocol, row)
    return {"state": row["state"], "protocol_id": protocol.id, "receipt": row["receipt"]}


def _name(value):
    return isinstance(value, str) and value.strip().casefold() not in _PLACEHOLDERS


def authorize(
    protocol,
    *,
    approved_protocol_id,
    approval_id,
    approved_by,
    cosigned_by=None,
    synthetic_root=None,
):
    """Explicit action by a named person; never called by a data reader.

    `approved_by` names the person who authorizes this one look (the gate owner, a team
    member). `cosigned_by` optionally names a Quantit co-signer. Neither is defaulted
    and a placeholder is refused: the authorizer is a caller-supplied decision.
    """
    if (
        not isinstance(protocol, FinalHoldoutProtocol)
        or approved_protocol_id != protocol.id
        or not _hash(approval_id)
    ):
        raise HoldoutSealed("explicit approval of the exact frozen protocol required")
    if not _name(approved_by) or (cosigned_by is not None and not _name(cosigned_by)):
        raise HoldoutSealed("a named authorizer is required; placeholders are refused")
    if protocol.payload["calibration"] is None:
        raise HoldoutSealed("generator and allocation procedure calibration is pending")
    approval = {
        "protocol_id": protocol.id,
        "approval_id": approval_id,
        "approved_by": approved_by,
        "cosigned_by": cosigned_by,
    }
    path = _path(protocol, synthetic_root)
    with exclusive_file(path.with_suffix(".lock")):
        _refuse_if_ledger_spent(protocol)
        if path.exists():
            row = _load(path)
            _check(protocol, row)
            if row["state"] == "SPENT":
                raise HoldoutSpent("the dataset's final look is terminal")
            if row["authorization"] != approval:
                raise HoldoutSealed("an immutable authorization already exists")
        else:
            row = {
                "schema_version": 1,
                "protocol": protocol.payload,
                "protocol_id": protocol.id,
                "authorization": approval,
                "state": "AUTHORIZED",
                "receipt": None,
            }
            _check(protocol, row)
            _write(path, row)
    return protocol.id


def require_authorized(protocol, *, config_hash, generator_id, synthetic_root=None):
    if not isinstance(protocol, FinalHoldoutProtocol):
        raise HoldoutSealed("the final holdout remains sealed without an explicit frozen protocol")
    if (
        protocol.payload["config_hash"] != config_hash
        or protocol.payload["generator_id"] != generator_id
    ):
        raise HoldoutSealed("final protocol belongs to a different config or generator")
    state = status(protocol, synthetic_root=synthetic_root)["state"]
    if state == "SPENT":
        raise HoldoutSpent("the dataset's final look is terminal")
    if state != "AUTHORIZED":
        raise HoldoutSealed("final protocol has no explicit operator authorization")


def real_look_spent():
    """True when the run ledger already holds a real `holdout-look` for REAL_DATASET.

    The per-HOME authority store cannot see a look taken from another HOME; the ledger
    is the team's shared record, so it is what makes the look once-only across machines.
    """
    return any(
        entry.get("name") == LOOK_RUN_NAME
        and "synthetic" not in (entry.get("tags") or [])
        and isinstance(entry.get("params"), dict)
        and entry["params"].get("dataset") == REAL_DATASET
        for entry in runlog.read_entries()
    )


def _refuse_if_ledger_spent(protocol):
    if protocol.payload["dataset"] == REAL_DATASET and real_look_spent():
        raise HoldoutSpent("the ledger already holds the real holdout's one look")


def _log_look(protocol, approval, receipt):
    row = protocol.payload
    runlog.log_run(
        LOOK_RUN_NAME,
        params={
            "protocol_id": protocol.id,
            "dataset": row["dataset"],
            "data_version": row["source_version"],
            "config_hash": row["config_hash"],
            "generator_id": row["generator_id"],
            "candidates": receipt["candidates"],
            "dates": receipt["dates"],
            "receipt_id": receipt["id"],
            "approved_by": approval["approved_by"],
            "cosigned_by": approval["cosigned_by"],
        },
        tags=[LOOK_TAG] + ([] if row["dataset"] == REAL_DATASET else ["synthetic"]),
        notes="terminal look reserved before the reader ran; no result is recorded here",
    )


def ensure_logged(protocol, *, synthetic_root=None):
    """Recover a spent look whose ledger entry was lost; returns True if it wrote one.

    Idempotent: a receipt already in the ledger is not logged twice. A look that was not
    spent has nothing to record.
    """
    path = _path(protocol, synthetic_root)
    with exclusive_file(path.with_suffix(".lock")):
        if not path.exists():
            return False
        row = _load(path)
        _check(protocol, row)
        if row["state"] != "SPENT":
            return False
        receipt = row["receipt"]
        if any(
            entry.get("name") == LOOK_RUN_NAME
            and entry.get("params", {}).get("receipt_id") == receipt["id"]
            for entry in runlog.read_entries()
        ):
            return False
        _log_look(protocol, row["authorization"], receipt)
        return True


@dataclass(frozen=True)
class _Access:
    receipt_id: str


@dataclass(frozen=True)
class _Look:
    access: _Access
    protocol: FinalHoldoutProtocol
    used_sources: set
    end: str


@contextmanager
def open_once(protocol, *, candidates, start, end, synthetic_root=None):
    """Durably spend before yielding; failed/uncertain readers never receive a retry."""
    row = protocol.payload
    ids = [str(value) for value in candidates]
    if not ids or len(set(ids)) != len(ids) or set(ids) - set(row["candidate_ids"]):
        raise HoldoutSealed("reader candidates are outside the frozen final protocol")
    first, last = str(date.fromisoformat(str(start))), str(date.fromisoformat(str(end)))
    if not row["period"][0] <= first <= last <= row["period"][1]:
        raise HoldoutSealed("reader period is outside the frozen sealed interval")
    path = _path(protocol, synthetic_root)
    with exclusive_file(path.with_suffix(".lock")):
        _refuse_if_ledger_spent(protocol)
        if not path.exists():
            raise HoldoutSealed("the final holdout has no authorization")
        saved = _load(path)
        _check(protocol, saved)
        if saved["state"] == "SPENT":
            raise HoldoutSpent("the dataset's final look is terminal")
        receipt = {
            "protocol_id": protocol.id,
            "candidates": ids,
            "dates": [first, last],
            "reserved_utc": datetime.now(UTC).isoformat(),
        }
        receipt["id"] = digest(receipt)
        saved.update(state="SPENT", receipt=receipt)
        _write(path, saved)
        # Logged after the durable spend and before any reader runs, so the ledger
        # never shows a result-dependent look. A failure here leaves the look spent;
        # `ensure_logged` writes the missing entry.
        _log_look(protocol, saved["authorization"], receipt)
    access = _Access(receipt["id"])
    token = _ACTIVE.set(_Look(access, protocol, set(), last))
    try:
        yield access
    finally:
        _ACTIVE.reset(token)


def _day(value):
    text = str(value)
    try:
        if text[10:] not in {"", " 00:00:00", "T00:00:00"}:
            raise ValueError("date precision")
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise HoldoutSealed("protected reads require bounded timezone-naive date labels") from exc


def validate_access(access, *, dataset=None, source_version=None, end=None):
    look = _ACTIVE.get()
    if not isinstance(access, _Access) or look is None or access is not look.access:
        raise HoldoutSealed(
            "active durable holdout access required; public tokens confer no authority"
        )
    row = look.protocol.payload
    if dataset is not None and row["dataset"] != dataset:
        raise HoldoutSealed("synthetic or foreign authority cannot open this real dataset")
    if source_version is not None and row["source_version"] != source_version:
        raise HoldoutSealed("authorized source version changed")
    if end is not None and _day(end) > look.end:
        raise HoldoutSealed("source read exceeds the terminal look's frozen dates")
    return row


def read_source(access, name, source_version):
    row = validate_access(access, source_version=source_version)
    name = name.casefold()
    used = _ACTIVE.get().used_sources
    if name not in row["sources"] or name in used:
        raise HoldoutSealed("source is unauthorized or already materialized in this terminal look")
    used.add(name)
    return row["sources"][name]


GUARD_ENV = "CAPSTONE_HOLDOUT_GUARD"


def guard_enabled():
    """True when the environment turns on sealing of unauthorized post-discovery reads.

    Default off: enforcing it everywhere is a team decision. With it on, a licensed or
    unregistered shared file read without an access stops at 2020-12-31, and the WRDS
    loader refuses any query or cache past that date.
    """
    return os.environ.get(GUARD_ENV, "").strip().casefold() in {"1", "on", "true", "yes"}


# Licensed shared files (docs/shared_data.md, scripts/pull_wrds.py) and the date
# column a bounded read must filter on; None = undated (no bounded read exists,
# so it is refused). `crsp_*`/`sp500_*` filter on `date` (`event_date` for event
# or action files). They open only under a real-dataset authority.
LICENSED_DATASETS = {
    "comp_funda": "datadate",
    "comp_fundq": "datadate",
    "comp_company": None,
    "ccm_link": "linkdt",
}
LICENSED_PREFIXES = ("crsp_", "sp500_")
# Files whose post-2020 rows are not holdout outcomes; none registered. Every
# other unregistered name (factor, event, descriptor-named files) fails closed:
# without an access it is read only through 2020.
PUBLIC_DATASETS = frozenset()


def protected_dataset(name):
    """REAL_DATASET for a registered licensed file, else None (an unregistered
    file is still guarded by `guard_shared_read` unless it is public)."""
    if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
        raise HoldoutSealed("shared sources require canonical file names, not path aliases")
    name = name.casefold()
    return REAL_DATASET if name in LICENSED_DATASETS or name.startswith(LICENSED_PREFIXES) else None


def guard_shared_read(name, columns, end, date_column, auto_sync, access):
    """Run before sync/import/materialization; calendar capability is projection-only.

    Off (a no-op without an access) unless `CAPSTONE_HOLDOUT_GUARD` is on, so the team's
    existing reads are unchanged until it adopts code-enforced sealing. A read that
    passes an `access` is always validated.
    """
    if access is None and not guard_enabled():
        return False
    dataset = protected_dataset(name)
    name = name.casefold()
    if name in PUBLIC_DATASETS and access is None:
        return False
    if dataset is not None:
        event = "event" in name or "action" in name
        expected = LICENSED_DATASETS.get(name, "event_date" if event else "date")
        if date_column != expected:
            raise HoldoutSealed("protected source requires its authoritative date filter")
    if end is not None and _day(end) <= DISCOVERY_END and access is None:
        return False
    calendar = (
        columns == ["date"]
        and date_column == "date"
        and end is not None
        and _day(end) <= "2021-01-31"
        and access is None
    )
    if calendar:
        if auto_sync:
            raise HoldoutSealed("calendar metadata cannot refresh market sources")
        return True
    validate_access(access, dataset=dataset, end=end)
    if end is None:
        raise HoldoutSealed("source reads require the terminal look's frozen date bound")
    if auto_sync:
        raise HoldoutSealed("authorized final sources cannot refresh during their look")
    return False


def require_discovery_query(end):
    """WRDS refresh/cache is not the frozen final-data path, regardless of credentials.

    Off unless `CAPSTONE_HOLDOUT_GUARD` is on (see `guard_enabled`).
    """
    if not guard_enabled():
        return
    if end is None or _day(end) > DISCOVERY_END:
        raise HoldoutSealed(
            "CRSP holdout queries/caches remain sealed; use separately authorized frozen sources"
        )
