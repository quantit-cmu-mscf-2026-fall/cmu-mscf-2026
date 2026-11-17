"""Synthetic fixtures shared by the holdout-authority tests: no real dataset, no real store."""

from dataclasses import dataclass

from capstone import holdout_authority as authority
from capstone import runlog
from capstone.artifacts import digest


@dataclass(frozen=True)
class FakeConfig:
    """The two attributes a protocol reads from a gate configuration."""

    config_hash: str = digest("synthetic config")
    generator: str = "synthetic:program-generator-v2"


def make_protocol(
    cfg=None,
    *,
    dataset="synthetic:market-A",
    sources=None,
    source_version="synthetic:v1",
    procedure_id=None,
    calibrated=True,
):
    cfg = cfg or FakeConfig()
    procedure_id = procedure_id or digest("synthetic allocation procedure")
    evidence = {
        "qualified": True,
        "generator_id": cfg.generator,
        "procedure_id": procedure_id,
        "config_hash": cfg.config_hash,
        "basis": "synthetic fixture only",
    }
    calibration = (
        None
        if not calibrated
        else {
            "evidence_id": digest(evidence),
            "evidence": evidence,
            "generator_id": cfg.generator,
            "procedure_id": procedure_id,
            "config_hash": cfg.config_hash,
        }
    )
    return authority.prepare_protocol(
        cfg,
        dataset=dataset,
        period=authority.REAL_PERIOD,
        source_version=source_version,
        sources=sources or {"synthetic_daily": "1" * 64},
        procedure_id=procedure_id,
        candidate_ids=["a" * 16],
        calibration=calibration,
    )


def approve(protocol, root, **changes):
    return authority.authorize(
        protocol,
        approved_protocol_id=protocol.id,
        approval_id=digest("synthetic operator review"),
        **({"approved_by": "synthetic gate owner"} | changes),
        synthetic_root=root,
    )


def open_look(protocol, root, **changes):
    return authority.open_once(
        protocol,
        synthetic_root=root,
        **({"candidates": ["a" * 16], "start": "2021-01-04", "end": "2025-12-31"} | changes),
    )


def looks():
    return [e for e in runlog.read_entries() if e["name"] == authority.LOOK_RUN_NAME]
