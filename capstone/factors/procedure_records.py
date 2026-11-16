"""Procedure identity records: register a procedure, its runs and calibration evidence.

Saved in the existing research SQLite artifacts/events and mirrored to the runlog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capstone import runlog
from capstone.factors.procedure_contracts import (
    PROCEDURE_KIND,
    SCOPE_RESOLUTION,
    ProcedureSpec,
    day,
    fields,
    identity,
    limit,
    token,
)
from capstone.factors.research_contracts import canonical, digest
from capstone.factors.research_records import load_artifact, put_artifact
from capstone.locking import exclusive_file

RUN = "procedure_run_v1"
STAGE = "full-procedure-accounting-v1"
ARM_CALLS = "procedure_arm_calls_v1"
CALIBRATION = "procedure_calibration_v1"
INHERITED = "procedure_inherited_reconciliation_v1"
CALIBRATION_LABEL = (
    "scripted-client protocol calibration; configured-model financial calibration is a "
    "separate claim"
)
WHOLE = "whole_procedure_calibration"


def _load(controller, key, kind):
    return load_artifact(controller.db, key, kind=kind)


def _rows(controller, kind):
    keys = {
        json.loads(row[0])["artifact_id"]
        for row in controller.db.execute(
            "SELECT payload FROM research_events WHERE state=? ORDER BY seq",
            ("procedure:" + kind,),
        )
    }
    return [(key, _load(controller, key, kind)) for key in sorted(keys)]


def _registered(controller, key, kind):
    row = _load(controller, key, kind)
    if not controller.db.execute(
        "SELECT 1 FROM research_events WHERE state=? AND payload=?",
        ("procedure:" + kind, canonical({"artifact_id": key})),
    ).fetchone():
        raise ValueError("artifact has no controller-owned registration event")
    return row


def _save(controller, kind, row, scope):
    key = put_artifact(controller.db, kind, row)
    payload = canonical({"artifact_id": key})
    state = "procedure:" + kind
    if not controller.db.execute(
        "SELECT 1 FROM research_events WHERE brief=? AND state=? AND payload=?",
        (scope, state, payload),
    ).fetchone():
        controller.db.execute(
            "INSERT INTO research_events(brief,state,reason,payload) VALUES (?,?,?,?)",
            (scope, state, STAGE, payload),
        )
    return key


def _mirror(key, procedure_id, kind):
    # SQLite is authoritative; replay repairs a crash between commit and this audit mirror.
    with exclusive_file(runlog.ledger_path().with_suffix(".procedure.lock")):
        if not any(
            entry.get("name") == STAGE and entry.get("params", {}).get("artifact_id") == key
            for entry in runlog.read_entries()
        ):
            runlog.log_run(
                STAGE,
                params={"procedure_id": procedure_id, "artifact_id": key, "kind": kind},
                tags=[STAGE, "accounting"],
                notes="Audit metadata; not automatically a statistical trial.",
            )


def inherit_reconciliation(controller, path):
    """Preserve existing row classifications without inventing an effective trial count."""
    raw = Path(path).read_bytes()
    row = json.loads(raw)
    required = {
        "study_hash",
        "source_entries",
        "return_columns",
        "classified_entries",
        "conservative_trial_entries",
        "blockers",
        "reconciled",
        "count_policy",
    }
    if not required <= row.keys() or row["blockers"] or row["reconciled"] is not True:
        raise ValueError("inherited row classifications are not reconciled")
    classified = row["classified_entries"]
    if (
        row["source_entries"] != len(classified)
        or [item["source_index"] for item in classified] != list(range(len(classified)))
        or any(not item.get("reason") for item in classified)
        or any(
            item.get("classification")
            not in {
                "strategy_trial",
                "rerun",
                "diagnostic",
                "methodology",
                "correction",
                "superseded",
            }
            for item in classified
        )
        or row["conservative_trial_entries"]
        != sum(
            item["classification"] in {"strategy_trial", "rerun", "superseded"}
            for item in classified
        )
    ):
        raise ValueError("inherited source classification inventory differs")
    for item in classified:
        identity(item["source_entry_hash"])
    payload = {"source_sha256": hashlib.sha256(raw).hexdigest(), "record": row}
    with controller.transaction():
        return _save(controller, INHERITED, payload, row["study_hash"])


def register_procedure(controller, spec):
    if not isinstance(spec, ProcedureSpec):
        raise ValueError("typed immutable procedure required")
    legacy = spec.to_record()["statistics"]["legacy_scope"]
    _load(controller, legacy["inherited_classification_id"], INHERITED)
    if legacy["state"] == "registered_resolution":
        for key in legacy["evidence_ids"]:
            evidence = _load(controller, key, SCOPE_RESOLUTION)
            if (
                evidence.get("rule_id") != legacy["resolution_rule_id"]
                or evidence.get("inherited_classification_id")
                != legacy["inherited_classification_id"]
                or evidence.get("reported_counts") != legacy["reported_counts"]
                or evidence.get("unresolved_scope_items") != []
                or evidence.get("justification") != legacy["justification"]
                or evidence.get("effective_scope") != legacy["effective_scope"]
            ):
                raise ValueError("legacy effective-scope resolution is not bound to saved evidence")
    with controller.transaction():
        key = _save(controller, PROCEDURE_KIND, spec.to_record(), spec.identity)
    _mirror(key, spec.identity, PROCEDURE_KIND)
    return key


def save_scope_resolution(controller, evidence):
    """Save a legacy-scope resolution's evidence (`scope_evidence`) before registration."""
    with controller.transaction():
        return put_artifact(controller.db, SCOPE_RESOLUTION, evidence)


def register_calibration(controller, path):
    """Import a scripted-client calibration record (a calibration run's output)."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("id") != digest({k: v for k, v in record.items() if k != "id"})
        or record.get("kind") != CALIBRATION
        or record.get("label") != CALIBRATION_LABEL
        or record.get("namespace") != "synthetic"
        or record.get("model_calls") != 0
    ):
        raise ValueError("not a verified scripted-client procedure calibration record")
    identity(record["procedure_id"])
    identity(record["calibration_rule_id"])
    with controller.transaction():
        return _save(controller, CALIBRATION, record, record["procedure_id"])


def calibration_qualified(controller, spec):
    """A registered calibration of this exact procedure identity selected thresholds and, for
    a whole-procedure legacy resolution, ran under the rule the procedure registered."""
    scope = spec.to_record()["statistics"]["legacy_scope"]["effective_scope"] or {}
    return any(
        row["procedure_id"] == spec.identity
        and row.get("selected") is not None
        and (
            scope.get("combination_rule") != WHOLE
            or row["calibration_rule_id"] == scope["calibration_rule_id"]
        )
        for _, row in _rows(controller, CALIBRATION)
    )


def register_run(
    controller, procedure_id, *, run_key, world_id, input_scope_id, namespace, seed, registered_at
):
    _registered(controller, procedure_id, PROCEDURE_KIND)
    token(run_key)
    identity(world_id)
    identity(input_scope_id)
    day(registered_at)
    if namespace not in {"real", "synthetic"} or type(seed) is not int:
        raise ValueError("explicit world namespace and seed required")
    row = {
        "procedure_id": procedure_id,
        "run_key": run_key,
        "world_id": world_id,
        "input_scope_id": input_scope_id,
        "namespace": namespace,
        "seed": seed,
        "registered_at": registered_at,
    }
    row["statistical_scope_id"] = digest(
        {name: row[name] for name in ("procedure_id", "world_id", "namespace")}
    )
    with controller.transaction():
        old = [
            item
            for _, item in _rows(controller, RUN)
            if item["procedure_id"] == procedure_id
            and item["world_id"] == world_id
            and item["namespace"] == namespace
            and item["run_key"] == run_key
        ]
        if old and old[0] != row:
            raise ValueError("logical run already has different frozen inputs")
        key = _save(controller, RUN, row, procedure_id)
    _mirror(key, procedure_id, RUN)
    return key


def _spec(controller, run_id):
    run = _registered(controller, run_id, RUN)
    return run, ProcedureSpec.from_record(
        _registered(controller, run["procedure_id"], PROCEDURE_KIND)
    )


def record_arm_calls(controller, run_id, calls):
    """Whole-procedure model calls per arm: general proposals, repairs and alignment of the
    packets its opportunities own plus its refinements, under one equal registered total."""
    _, spec = _spec(controller, run_id)
    limits = {arm["name"]: arm["model_call_limit"] for arm in spec.to_record()["arms"]}
    if len(set(limits.values())) != 1:
        raise ValueError("comparison arms must register equal model-call totals")
    if set(calls) != set(limits):
        raise ValueError("every registered arm needs its model-call total")
    arms = {}
    for name, used in calls.items():
        fields(used, "general refinement")
        limit(used["general"])
        limit(used["refinement"])
        total = used["general"] + used["refinement"]
        if total > limits[name]:
            raise ValueError(f"arm {name} exceeds its registered model-call total")
        arms[name] = {"limit": limits[name], **used, "total": total}
    with controller.transaction():
        return _save(
            controller,
            ARM_CALLS,
            {"procedure_id": spec.identity, "run_id": run_id, "arms": arms},
            run_id,
        )
