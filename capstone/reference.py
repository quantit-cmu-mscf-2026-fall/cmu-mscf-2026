"""Bind current reference construction to reviewed gate artifacts; never invent admission."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from capstone.strategy.recorded import require_current_implementation
from capstone.strategy.reference import METHOD_REVISION
from capstone.strategy.research import contracts, digest, implementation_digest


def generator_name():
    identity = digest(
        {"catalog": contracts(), "implementation": implementation_digest(validation=True)}
    )
    return f"reference-v{METHOD_REVISION}/{identity}"


def admission(spec, report_path=None):
    require_current_implementation(spec)
    result = {
        "strategy_id": spec.id,
        "method_hash": spec.method_hash,
        "admitted": False,
        "blockers": [],
        "report_hash": None,
    }
    if report_path is None:
        result["blockers"].append("reviewed_gate_report_missing")
        return result
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    result["report_hash"] = digest(report)
    header = report.get("header", {})
    if header.get("generator") != generator_name():
        result["blockers"].append("superseded_or_unrelated_validation_generator")
    if header.get("data_version") != spec.data_version:
        result["blockers"].append("validation_data_version_mismatch")
    if spec.claim_id not in report.get("accepted", []):
        result["blockers"].append("family_not_accepted")
    if not header.get("n_holdout_obs", 0):
        result["blockers"].append("formal_validation_incomplete")
    card_name = report.get("scorecard_csv")
    if not isinstance(card_name, str) or Path(card_name).name != card_name:
        result["blockers"].append("invalid_scorecard_reference")
    else:
        card = pd.read_csv(path.parent / card_name, index_col=0)
        if (
            spec.claim_id not in card.index
            or str(card.loc[spec.claim_id, "best"]) != spec.variant_id
        ):
            result["blockers"].append("chosen_variant_mismatch")
    result["admitted"] = not result["blockers"]
    return result
