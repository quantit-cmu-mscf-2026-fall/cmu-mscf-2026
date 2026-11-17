"""Portable account state and frame hashes for an immutable decision bundle.

A bundle carries only what planning needs from an account: positions, open
order references, cash, capital, receivables and the share and cash events
applied since the last plan. Nothing that identifies an account or a venue is
accepted (`portable_account` refuses unknown keys), and the hash of a frame is
independent of its row and column order. `freeze_bundle` writes an immutable, hashed
directory of frames, strategy files and account state; `load_bundle` verifies every file
and the runtime before returning it. Nothing here opens a connection to any venue. A panel
reaching past the discovery period is frozen only with an active current-feature
authorization (`capstone.execution.current_features`). The release and runtime identity a
bundle records is `capstone.release`.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from capstone import release
from capstone.execution.contracts import (
    ExecutionContractBlocked,
    execution_contract,
    program_positions,
    validate_program_instruments,
)
from capstone.execution.current_features import (
    CurrentFeatureProvenance,
    panel_frames,
    require_current_route,
)
from capstone.execution.evidence import FINAL, SEARCH_STAGE, admission
from capstone.factors.evaluation import DISCOVERY_END, Panel
from capstone.factors.reference import sessions
from capstone.strategy.contract import load_spec
from capstone.strategy.research import canonical, digest
from capstone.strategy.saved_portfolio import MODE, SavedProgramPortfolioSpec

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


runtime_versions = release.runtime_versions
stamped_identity = release.stamped_identity


def release_identity(deployment_tag=None):
    """`capstone.release.release_identity`, with deploy tags checked by the deploy layer."""
    from capstone.execution.deploy import resolve_tag

    return release.release_identity(deployment_tag, resolve_tag=resolve_tag)


def _record_thresholds(root, evidence):
    """Beside the bundle, the registered-threshold file each verified gate report recorded
    (`registered_thresholds.BUNDLE_FILE`); plans read thresholds from it.

    Only a program-gate report has one, so a bundle without such evidence never imports
    `registered_thresholds`. With one, a missing module is an ImportError: the freeze fails
    closed rather than writing a bundle that has dropped its thresholds.
    """
    reports = {
        spec_id: item["report_path"]
        for spec_id, item in evidence.items()
        if item.get("report_path") and item.get("validation_status") in (SEARCH_STAGE, FINAL)
    }
    if not reports:
        return
    from capstone.pipeline import registered_thresholds  # import cycle via program_gate

    found = {}
    for spec_id, report in reports.items():
        inputs = json.loads(Path(report).read_text(encoding="utf-8"))["inputs"]
        found[spec_id] = registered_thresholds.report_record(Path(report).parent, inputs)
    registered_thresholds.write_bundle_records(root, found)


def freeze_bundle(
    path,
    panel,
    specs,
    as_of,
    *,
    prices,
    instruments,
    account,
    returns=None,
    mode="equal_family",
    limits=None,
    reports=None,
    deployment_tag=None,
    execution_cost_bps=None,
    execution_version="fixed-share-next-open-v2",
):
    root = Path(path)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("a frozen input bundle cannot be overwritten")
    private_account = portable_account(account)
    specs = tuple(specs)
    saved = (
        specs[0] if len(specs) == 1 and isinstance(specs[0], SavedProgramPortfolioSpec) else None
    )
    if saved is not None:
        saved.verify_history(panel, as_of)
        if mode != MODE:
            raise ExecutionContractBlocked("saved_allocation_mode_required")
        if returns is not None and frame_hash(
            returns.loc[returns.index < pd.Timestamp(as_of)]
        ) != frame_hash(saved.returns.loc[saved.returns.index < pd.Timestamp(as_of)]):
            raise ValueError("saved allocation return source differs")
        specs = (saved.frozen(as_of),)
        returns = specs[0].returns
    contract = execution_contract(specs, purpose="frozen", version=execution_version)
    validate_program_instruments(specs, panel)
    if contract["route"] == "program-v1":
        if mode != "equal_family" and saved is None:
            raise ExecutionContractBlocked("variable_program_allocations_unqualified")
        private_account["positions"] = program_positions(private_account["positions"])
    at = pd.Timestamp(as_of)
    calendar = sessions(panel.calendar, at)
    if at not in panel.universe.index:
        raise ValueError("snapshot must contain the completed decision session")
    if panel.universe.index.max() > pd.Timestamp(DISCOVERY_END):
        require_current_route(panel, active=True)  # new current inputs need an active authorization
        if panel.provenance.as_of != str(at.date()):
            raise ValueError("a current bundle freezes its provenance's own session")
    root.mkdir(parents=True, exist_ok=True)
    files, frames = {}, {}

    def write(name, content):
        (root / name).write_text(content, encoding="utf-8", newline="\n")
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()

    if panel.availability is None:
        raise ValueError("reference snapshots require explicit point-in-time availability")
    all_frames = panel_frames(panel)
    for i, (label, original) in enumerate(sorted(all_frames.items())):
        if original.index.has_duplicates or original.columns.has_duplicates:
            raise ValueError("duplicate snapshot observations")
        frame = original.loc[original.index <= at].sort_index().sort_index(axis=1)
        if not frame.index.equals(panel.universe.index[panel.universe.index <= at].sort_values()):
            raise ValueError(f"snapshot dates are not aligned for {label}")
        name = f"frame-{i}.parquet"
        frame.to_parquet(root / name)
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        frames[label] = {"file": name, "content_hash": frame_hash(frame)}
    if returns is None:
        returns = pd.DataFrame(index=pd.DatetimeIndex([]))
    past = returns.loc[returns.index < at].sort_index().sort_index(axis=1)
    past.to_parquet(root / "strategy-returns.parquet")
    files["strategy-returns.parquet"] = hashlib.sha256(
        (root / "strategy-returns.parquet").read_bytes()
    ).hexdigest()
    fundamental_file = None
    if panel.fundamentals is not None:
        fundamental_file = "fundamental-records.parquet"
        records = (
            panel.fundamentals[pd.to_datetime(panel.fundamentals.published_at).dt.normalize() < at]
            .sort_values(["permco", "datadate", "published_at"])
            .reset_index(drop=True)
        )
        records.to_parquet(root / fundamental_file)
        files[fundamental_file] = hashlib.sha256((root / fundamental_file).read_bytes()).hexdigest()
    definitions, evidence = [], {}
    for spec in sorted(specs, key=lambda s: s.id):
        name = f"strategy-{spec.id}.toml"
        write(name, spec.to_toml())
        definitions.append(name)
        evidence[spec.id] = admission(spec, (reports or {}).get(spec.id))
    _record_thresholds(root, evidence)
    metadata = {
        "runtime": runtime_versions(),
        "release": release_identity(deployment_tag),
        "execution_cost_bps": execution_cost_bps,
        "execution_contract": contract,
        "schema_version": 2,
        "fundamentals_file": fundamental_file,
        "fundamentals_hash": files.get(fundamental_file),
        "as_of": str(at.date()),
        "data_version": panel.data_version,
        "calendar": [str(d.date()) for d in calendar],
        "frames": frames,
        "strategy_returns_hash": frame_hash(past),
        "specs": definitions,
        "prices": {str(n): float(p) for n, p in sorted(prices.items())},
        "instruments": {str(n): row for n, row in sorted(instruments.items())},
        "account": private_account,
        "allocation_mode": mode,
        "limits": limits or {"max_gross": 1.0, "max_weight": 0.1, "max_order_notional": 50_000},
        "evidence": evidence,
    }
    if panel.provenance is not None:  # identity-bound; absent (old bytes) for discovery panels
        metadata["current_provenance"] = panel.provenance.to_dict()
    metadata["input_hash"] = digest(metadata)
    if saved is not None:
        metadata["allocation_derivation"] = {
            "status": "verified_at_freeze_from_supplied_funded_returns",
            "decision_prefix_id": saved.prefix_id(as_of),
        }
        metadata["input_hash"] = digest({k: v for k, v in metadata.items() if k != "input_hash"})
    if execution_contract(specs, purpose="frozen", version=execution_version) != contract:
        raise ExecutionContractBlocked("execution_source_changed_during_freeze")
    write("inputs.json", canonical(metadata) + "\n")
    (root / "manifest.json").write_text(
        canonical({"schema_version": 2, "files": files}) + "\n", encoding="utf-8"
    )
    return root


def load_bundle(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("unknown input bundle schema")
    for name, expected in manifest["files"].items():
        item = (root / name).resolve()
        if item.parent != root or hashlib.sha256(item.read_bytes()).hexdigest() != expected:
            raise ValueError("input bundle file integrity failure")
    data = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    if data["runtime"] != runtime_versions():
        raise ValueError("runtime differs from the frozen input bundle")
    if digest({k: v for k, v in data.items() if k != "input_hash"}) != data["input_hash"]:
        raise ValueError("input metadata changed")
    portable_account(data["account"])
    frames = {}
    for label, descriptor in data["frames"].items():
        if descriptor["file"] not in manifest["files"]:
            raise ValueError("frame is not bound by the manifest")
        frame = pd.read_parquet(root / descriptor["file"])
        if frame_hash(frame) != descriptor["content_hash"]:
            raise ValueError("frame content hash mismatch")
        frames[label] = frame
    for name in data["specs"]:
        if name not in manifest["files"]:
            raise ValueError("specification is not bound by the manifest")
    fundamental_file = data.get("fundamentals_file")
    fundamentals = None
    if fundamental_file:
        if (
            fundamental_file not in manifest["files"]
            or data["fundamentals_hash"] != manifest["files"][fundamental_file]
        ):
            raise ValueError("fundamental source is not bound by the manifest")
        fundamentals = pd.read_parquet(root / fundamental_file)
    panel = Panel(
        {n.split(":", 1)[1]: f for n, f in frames.items() if n.startswith("field:")},
        frames["universe"],
        frames["half_spread"],
        frames.get("sic"),
        data["data_version"],
        pd.DatetimeIndex(data["calendar"]),
        {
            n.split(":", 1)[1]: f["available_at"]
            for n, f in frames.items()
            if n.startswith("availability:")
        },
        frames.get("company"),
        fundamentals,
    )
    if "current_provenance" in data:
        panel.provenance = CurrentFeatureProvenance.from_dict(data["current_provenance"])
    returns = pd.read_parquet(root / "strategy-returns.parquet")
    if frame_hash(returns) != data["strategy_returns_hash"]:
        raise ValueError("strategy return history changed")
    specs = []
    for name in data["specs"]:
        text = (root / name).read_text(encoding="utf-8")
        if tomllib.loads(text).get("schema_version") == 7:
            spec = SavedProgramPortfolioSpec.from_toml(text, returns=returns)
            if spec.frozen(data["as_of"]).to_toml() != text or data.get(
                "allocation_derivation"
            ) != {
                "status": "verified_at_freeze_from_supplied_funded_returns",
                "decision_prefix_id": spec.prefix_id(data["as_of"]),
            }:
                raise ExecutionContractBlocked("saved_allocation_frozen_prefix_incompatible")
            specs.append(spec)
        else:
            specs.append(load_spec(root / name))
    version = (data.get("execution_contract") or {}).get("version", "fixed-share-next-open-v2")
    contract = execution_contract(specs, purpose="frozen", version=version)
    saved = data.get("execution_contract")
    if saved != contract and (saved is not None or contract["route"] == "program-v1"):
        raise ExecutionContractBlocked("frozen_execution_contract_incompatible")
    return data, panel, specs, returns
