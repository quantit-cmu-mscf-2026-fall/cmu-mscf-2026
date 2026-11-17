"""Procedure records: register a procedure, its runs and calibration evidence, then account
for its financial opportunities, prefixes and reconciliations.

Saved in the existing research SQLite artifacts/events and mirrored to the runlog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capstone import runlog
from capstone.factors.procedure_contracts import (
    KINDS,
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
from capstone.pipeline.study import reconcile as reconcile_source_rows

RUN = "procedure_run_v1"
OPPORTUNITY = "financial_opportunity_v1"
COMPLETION = "financial_opportunity_completion_v1"
PREFIX = "procedure_prefix_v1"
RECONCILIATION = "procedure_reconciliation_v1"
INHERITED = "procedure_inherited_reconciliation_v1"
STAGE = "full-procedure-accounting-v1"
PREFIX_RECEIPT = "financial_completed_prefix_v1"  # prefix_records.RECEIPT (import cycle)
ARM_CALLS = "procedure_arm_calls_v1"
CALIBRATION = "procedure_calibration_v1"
CALIBRATION_LABEL = (
    "scripted-client protocol calibration; configured-model financial calibration is a "
    "separate claim"
)
WHOLE = "whole_procedure_calibration"
# A reconciliation written under ID scheme 2 cites the run ledger's rows without their
# wall-clock field (`ts_utc`), so its id is the same for two runs of one world. A row
# without `id_scheme` is scheme 1 (rows hashed with their clock) and still reads: either
# way an artifact's id is the digest of its own stored payload.
RECONCILIATION_ID_SCHEME = 2
CLOCK_FIELDS = ("ts_utc",)  # the run ledger's wall-clock field


def source_rows(entries):
    """Run-ledger rows as a reconciliation cites them (ID scheme 2): every field but the
    wall clock. The ledger keeps `ts_utc` as a recorded field; it is not an identity input."""
    return [{k: v for k, v in entry.items() if k not in CLOCK_FIELDS} for entry in entries]


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


def _unsealed(controller, run_id, available_at):
    scope = _registered(controller, run_id, RUN)["statistical_scope_id"]
    if any(
        row["statistical_scope_id"] == scope and row["as_of"] >= available_at
        for _, row in _rows(controller, PREFIX)
    ):
        raise ValueError("cannot backdate new evidence into an immutable earlier prefix")


def nothing_shown(before):
    """Record for a trial that saw no financial result."""
    return {"receipt_ids": [], "before": before, "summary_digest": digest({"observed": []})}


def _shown(controller, shown, *, kind, assigned_at, parent_id):
    if shown is None:
        if kind in {"refinement", "rerun"}:
            raise ValueError("feedback opportunity must record what was shown to the agent")
        return nothing_shown(assigned_at)
    fields(shown, "receipt_ids before summary_digest")
    identity(shown["summary_digest"])
    ids = shown["receipt_ids"]
    if shown["before"] != assigned_at or not isinstance(ids, list) or ids != sorted(set(ids)):
        raise ValueError("shown evidence must be the sorted receipts before the assignment")
    for key in ids:
        receipt = load_artifact(controller.db, identity(key), kind=PREFIX_RECEIPT)
        if receipt["opportunity_id"] != parent_id or receipt["available_at"] >= assigned_at:
            raise ValueError("agent was shown a receipt outside its parent's prior evidence")
    if ids and kind not in {"refinement", "rerun"}:
        raise ValueError("initial proposals/comparisons are shown no financial results")
    return dict(shown)


def reserve_opportunity(
    controller,
    run_id,
    *,
    arm,
    method_key,
    kind,
    ordinal,
    assigned_at,
    parent_id=None,
    change_id=None,
    shown_to_agent=None,
):
    run, spec = _spec(controller, run_id)
    owner = spec.owner(method_key)
    arm_row = spec.arm(arm)
    if kind not in KINDS:
        raise ValueError("unregistered financial-opportunity kind")
    limit(ordinal)
    day(assigned_at)
    budget = next(b for b in arm_row["budgets"] if b["owner"] == owner["budget_owner"])
    if ordinal >= budget[kind] or assigned_at < run["registered_at"]:
        raise ValueError("opportunity exceeds its frozen slot/date scope")
    shown = _shown(
        controller, shown_to_agent, kind=kind, assigned_at=assigned_at, parent_id=parent_id
    )
    row = {
        "procedure_id": spec.identity,
        "run_id": run_id,
        "statistical_scope_id": run["statistical_scope_id"],
        "arm": arm,
        "arm_id": digest({"run_id": run_id, "arm": arm}),
        "budget_owner": owner["budget_owner"],
        "family": owner["family"],
        "method_key": owner["method_key"],
        "kind": kind,
        "ordinal": ordinal,
        "assigned_at": assigned_at,
        "parent_id": parent_id,
        "change_id": change_id,
    }
    legacy = dict(row)  # saved before feedback bounds: nothing was ever shown
    row["shown_to_agent"] = shown
    row["logical_slot_id"] = legacy["logical_slot_id"] = digest(
        {k: row[k] for k in ("statistical_scope_id", "arm", "budget_owner", "kind", "ordinal")}
    )
    with controller.transaction():
        old = [
            item
            for _, item in _rows(controller, OPPORTUNITY)
            if item["logical_slot_id"] == row["logical_slot_id"]
        ]
        if old and old[0] != row and (old[0] != legacy or shown != nothing_shown(assigned_at)):
            raise ValueError("logical slot already has a different immutable assignment")
        if old:
            row = old[0]
        else:
            _unsealed(controller, run_id, assigned_at)
            if kind in {"refinement", "rerun"}:
                parent = _registered(controller, parent_id, OPPORTUNITY)
                prior = _completion(controller, parent_id)
                if (
                    parent["run_id"] != run_id
                    or parent["arm"] != arm
                    or parent["budget_owner"] != row["budget_owner"]
                    or parent["family"] != row["family"]
                    or prior is None
                    or prior[1]["as_of"] >= assigned_at
                ):
                    raise ValueError("refinement/rerun needs earlier same-owner evidence")
                identity(change_id)
                bounds = spec.to_record()["feedback"]
                children = [
                    o for _, o in _rows(controller, OPPORTUNITY) if o["parent_id"] == parent_id
                ]
                if kind == "refinement" and len(children) >= bounds["max_refinements_per_parent"]:
                    raise ValueError("parent refinement budget exhausted")
                depth, ancestor = 1, parent
                while ancestor["parent_id"]:
                    depth += 1
                    ancestor = _load(controller, ancestor["parent_id"], OPPORTUNITY)
                if kind == "refinement" and depth > min(
                    bounds["max_refinement_depth"], bounds["max_feedback_rounds"]
                ):
                    raise ValueError("financial feedback depth exhausted")
            elif parent_id is not None or change_id is not None:
                raise ValueError("initial proposals/comparisons cannot acquire a parent")
        key = _save(controller, OPPORTUNITY, row, run_id)
    _mirror(key, spec.identity, OPPORTUNITY)
    return key


def _completion(controller, opportunity_id):
    found = [
        (key, row)
        for key, row in _rows(controller, COMPLETION)
        if row["opportunity_id"] == opportunity_id
    ]
    if len(found) > 1:
        raise ValueError("conflicting saved financial outcomes")
    return found[0] if found else None


def opportunity_record(controller, opportunity_id):
    """Read an actual controller reservation; a freestanding artifact grants no slot."""
    return _registered(controller, opportunity_id, OPPORTUNITY)


def registered_opportunity(controller, run_id, *, arm, method_key, kind, ordinal):
    """Lookup only: aliases and output/run names cannot reserve another budget slot."""
    run, spec = _spec(controller, run_id)
    owner = spec.owner(method_key)
    limit(ordinal)
    for key, row in _rows(controller, OPPORTUNITY):
        if (
            row["statistical_scope_id"] == run["statistical_scope_id"]
            and row["arm"] == arm
            and row["budget_owner"] == owner["budget_owner"]
            and row["method_key"] == owner["method_key"]
            and row["kind"] == kind
            and row["ordinal"] == ordinal
        ):
            return key, row
    raise ValueError("logical opportunity has not been reserved")


def complete_opportunity(
    controller, opportunity_id, *, status, as_of, evidence, return_columns=(), duplicate_of=None
):
    opportunity = _registered(controller, opportunity_id, OPPORTUNITY)
    if status not in {
        "evaluated",
        "malformed",
        "source_blocked",
        "data_blocked",
        "execution_failed",
        "duplicate",
        "not_reached",
    }:
        raise ValueError("typed opportunity outcome required")
    day(as_of)
    if as_of < opportunity["assigned_at"]:
        raise ValueError("outcome predates reservation")
    refs, evidenced_columns, financial_observed = [], set(), False
    for ref in evidence:
        fields(ref, "kind artifact_id")
        saved = _load(controller, ref["artifact_id"], ref["kind"])
        if type(saved.get("financial_observed")) is not bool:
            raise ValueError("saved financial observation state must be explicit")
        if (
            saved.get("opportunity_id") != opportunity_id
            or day(saved.get("available_at")) < opportunity["assigned_at"]
            or day(saved.get("available_at")) > as_of
            or not isinstance(saved.get("return_columns"), list)
            or any(not isinstance(name, str) or not name for name in saved["return_columns"])
            or (saved["return_columns"] and not saved["financial_observed"])
            or (
                saved.get("history_end") is not None
                and day(saved["history_end"]) > saved["available_at"]
            )
        ):
            raise ValueError("saved outcome evidence is future or belongs to another opportunity")
        evidenced_columns.update(saved["return_columns"])
        financial_observed |= saved["financial_observed"]
        refs.append(dict(ref))
    columns = sorted(set(return_columns))
    if (
        any(not isinstance(name, str) or not name for name in columns)
        or not refs
        or set(columns) - evidenced_columns
    ):
        raise ValueError("exact saved evidence and explicit return-column mapping required")
    if (status == "evaluated") != bool(columns):
        raise ValueError("only evaluated outcomes have financial return columns")
    if (columns and not financial_observed) or (
        status in {"duplicate", "not_reached"} and financial_observed
    ):
        raise ValueError("observed finance cannot be relabelled as unused or a free duplicate")
    if (status == "duplicate") != (duplicate_of is not None):
        raise ValueError("duplicates must link an existing completed opportunity")
    if duplicate_of:
        target = _registered(controller, duplicate_of, OPPORTUNITY)
        if (
            target["statistical_scope_id"] != opportunity["statistical_scope_id"]
            or _completion(controller, duplicate_of) is None
        ):
            raise ValueError("duplicate target is not completed in this scope")
    row = {
        "opportunity_id": opportunity_id,
        "status": status,
        "as_of": as_of,
        "evidence": sorted(refs, key=canonical),
        "return_columns": columns,
        "duplicate_of": duplicate_of,
        "financial_observed": financial_observed,
    }
    with controller.transaction():
        old = _completion(controller, opportunity_id)
        if old and old[1] != row:
            raise ValueError("completed opportunity cannot be overwritten")
        if old is None:
            _unsealed(controller, opportunity["run_id"], as_of)
        key = _save(controller, COMPLETION, row, opportunity["run_id"])
    _mirror(key, opportunity["procedure_id"], COMPLETION)
    return key


def record_prefix(controller, run_id, *, as_of):
    run, _ = _spec(controller, run_id)
    day(as_of)
    scope = run["statistical_scope_id"]
    if as_of < run["registered_at"]:
        raise ValueError("prefix predates logical procedure registration")
    with controller.transaction():
        existing = [
            (key, row)
            for key, row in _rows(controller, PREFIX)
            if row["statistical_scope_id"] == scope and row["as_of"] == as_of
        ]
        if existing:
            return existing[0][0]
        opportunities = [
            (key, row)
            for key, row in _rows(controller, OPPORTUNITY)
            if row["statistical_scope_id"] == scope and row["assigned_at"] <= as_of
        ]
        completed = {
            key: value for key, value in _rows(controller, COMPLETION) if value["as_of"] <= as_of
        }
        allowed = {key for key, _ in opportunities}
        row = {
            "procedure_id": run["procedure_id"],
            "statistical_scope_id": scope,
            "as_of": as_of,
            "opportunities": [key for key, _ in opportunities],
            "outcomes": sorted(
                key for key, value in completed.items() if value["opportunity_id"] in allowed
            ),
        }
        return _save(controller, PREFIX, row, scope)


def _reconcile_locked(
    controller,
    run_id,
    *,
    entries,
    return_columns,
    source_decisions,
    classifications,
    previous_id=None,
):
    run, spec = _spec(controller, run_id)
    entries = source_rows(entries)  # ID scheme 2: no wall clock in the identity
    source = reconcile_source_rows(
        {"study_hash": spec.identity}, entries, return_columns, source_decisions
    )
    for item in source["classified_entries"]:
        del item["timestamp"]  # the ledger row keeps it; the reconciliation cites the row
    scope = run["statistical_scope_id"]
    opportunities = {
        key: row
        for key, row in _rows(controller, OPPORTUNITY)
        if row["statistical_scope_id"] == scope
    }
    blockers = list(source["blockers"])
    legacy_scope = spec.to_record()["statistics"]["legacy_scope"]
    if legacy_scope["state"] == "unresolved":
        counts = "/".join(str(n) for n in legacy_scope["reported_counts"])
        blockers.append(f"legacy_effective_statistical_scope_unresolved:{counts}")
    if set(classifications) - opportunities.keys():
        blockers.append("unknown_classified_opportunity")
    financial, bootstrap, mapped = [], set(), set()
    for key, opportunity in opportunities.items():
        completion = _completion(controller, key)
        decision = classifications.get(key, {})
        role = decision.get("classification")
        expected = (
            None
            if completion is None
            else (
                "financial_alternative"
                if completion[1]["financial_observed"]
                and spec.arm(opportunity["arm"])["classification"] == "research"
                else "control"
                if completion[1]["financial_observed"]
                else "duplicate"
                if completion[1]["status"] == "duplicate"
                else "unused"
                if completion[1]["status"] == "not_reached"
                else "nonfinancial_failure"
            )
        )
        if expected is None or role != expected or not decision.get("reason"):
            blockers.append(f"unclassified_or_pending_opportunity:{key}")
        if completion:
            columns = set(completion[1]["return_columns"])
            if columns - set(return_columns):
                blockers.append(f"missing_return_column:{key}")
            mapped.update(columns)
            if role == "financial_alternative" and role == expected:
                financial.append(key)
                bootstrap.update(columns)
    for name in sorted(set(return_columns) - mapped):
        blockers.append(f"return_column_has_no_completed_opportunity:{name}")
    references = set()
    for decision in source_decisions.values():
        ids = set(decision.get("opportunity_ids", []))
        if ids - opportunities.keys():
            blockers.append("source_row_references_unknown_opportunity")
        if decision.get("classification") in {"strategy_trial", "rerun", "superseded"}:
            if not ids or ids - opportunities.keys():
                blockers.append("unreserved_source_trial")
            references.update(ids)
            declared = set(decision.get("variant_ids", []))
            expected = set()
            for key in ids & opportunities.keys():
                completion = _completion(controller, key)
                if completion:
                    expected.update(completion[1]["return_columns"])
            if declared - expected:
                blockers.append("source_trial_return_mapping_mismatch")
    if set(financial) - references:
        blockers.append("financial_opportunity_missing_source_trial_mapping")
    for arm in spec.to_record()["arms"]:
        for budget in arm["budgets"]:
            for kind in KINDS:
                actual = {
                    o["ordinal"]
                    for o in opportunities.values()
                    if o["arm"] == arm["name"]
                    and o["budget_owner"] == budget["owner"]
                    and o["kind"] == kind
                }
                if actual != set(range(budget[kind])):
                    blockers.append(
                        f"unaccounted_planned_slots:{arm['name']}:{budget['owner']}:{kind}"
                    )
    if previous_id:
        prior = _registered(controller, previous_id, RECONCILIATION)
        if prior["statistical_scope_id"] != scope:
            raise ValueError("correction ancestry belongs to another statistical scope")
    inherited_scope = spec.to_record()["statistics"]["legacy_scope"]["effective_scope"]
    total = (
        inherited_scope["prior_effective_trials"] + len(financial)
        if inherited_scope and inherited_scope["combination_rule"] == "add_registered_prior_count"
        else None
    )
    calls = controller.db.execute(
        "SELECT payload FROM research_events WHERE brief=? AND state=? ORDER BY seq DESC LIMIT 1",
        (run_id, "procedure:" + ARM_CALLS),
    ).fetchone()
    row = {
        "id_scheme": RECONCILIATION_ID_SCHEME,
        "procedure_id": spec.identity,
        "statistical_scope_id": scope,
        "counting_rule": spec.to_record()["statistics"]["counting_rule"],
        "arm_model_calls": calls
        and _load(controller, json.loads(calls[0])["artifact_id"], ARM_CALLS),
        "source_inventory_id": digest(entries),
        "source_audit": source,
        "opportunity_ids": sorted(opportunities),
        "classifications": classifications,
        "financial_trial_ids": sorted(financial),
        "current_financial_opportunities": len(financial),
        "inherited_effective_scope": inherited_scope,
        "statistical_trial_count": total,
        "accounting_frontier_id": _frontier(controller, scope),
        "bootstrap_candidate_columns": sorted(bootstrap),
        "procedural_slots": len(opportunities),
        "previous_id": previous_id,
        "blockers": sorted(set(blockers)),
        "reconciled": not blockers,
        "calibration_qualified": calibration_qualified(controller, spec),
        "formal_admission": False,
    }
    latest = controller.db.execute(
        "SELECT payload FROM research_events WHERE brief=? AND state=? ORDER BY seq DESC LIMIT 1",
        (scope, "procedure:" + RECONCILIATION),
    ).fetchone()
    expected_id = digest({"kind": RECONCILIATION, "payload": row})
    if latest:
        latest_id = json.loads(latest[0])["artifact_id"]
        if latest_id != expected_id and previous_id != latest_id:
            raise ValueError("reconciliation correction must extend the latest immutable record")
    key = _save(controller, RECONCILIATION, row, scope)
    return key, spec.identity


def _frontier(controller, scope):
    opportunities = {
        key for key, row in _rows(controller, OPPORTUNITY) if row["statistical_scope_id"] == scope
    }
    return digest(
        {
            "runs": [
                key for key, row in _rows(controller, RUN) if row["statistical_scope_id"] == scope
            ],
            "opportunities": sorted(opportunities),
            "outcomes": [
                key
                for key, row in _rows(controller, COMPLETION)
                if row["opportunity_id"] in opportunities
            ],
        }
    )


def reconcile_opportunities(
    controller,
    run_id,
    *,
    entries,
    return_columns,
    source_decisions,
    classifications,
    previous_id=None,
):
    with controller.transaction():
        key, procedure_id = _reconcile_locked(
            controller,
            run_id,
            entries=entries,
            return_columns=return_columns,
            source_decisions=source_decisions,
            classifications=classifications,
            previous_id=previous_id,
        )
    _mirror(key, procedure_id, RECONCILIATION)
    return key


def require_compatible_reconciliation(
    controller,
    key,
    *,
    procedure_id,
    statistical_scope_id,
    source_inventory_id,
    current_financial_opportunities,
    statistical_trial_count,
    bootstrap_candidate_columns,
):
    with controller.transaction():
        row = _registered(controller, key, RECONCILIATION)
        latest = controller.db.execute(
            "SELECT payload FROM research_events WHERE brief=? AND state=? "
            "ORDER BY seq DESC LIMIT 1",
            (statistical_scope_id, "procedure:" + RECONCILIATION),
        ).fetchone()
        if (
            latest is None
            or json.loads(latest[0])["artifact_id"] != key
            or row["accounting_frontier_id"] != _frontier(controller, statistical_scope_id)
            or row["procedure_id"] != procedure_id
            or row["statistical_scope_id"] != statistical_scope_id
            or row["source_inventory_id"] != source_inventory_id
            or not row["reconciled"]
            or row["blockers"]
            or row["current_financial_opportunities"] != current_financial_opportunities
            or (row["statistical_trial_count"] is None and not _calibrated_scope(row))
            or row["statistical_trial_count"] != statistical_trial_count
            or row["bootstrap_candidate_columns"] != sorted(bootstrap_candidate_columns)
        ):
            raise ValueError("verified procedure-scoped reconciliation is missing or incompatible")
    return row


def latest_reconciliation(controller, run_id):
    """(run row, id of its scope's latest reconciliation or None); reading grants nothing."""
    run = _registered(controller, run_id, RUN)
    latest = controller.db.execute(
        "SELECT payload FROM research_events WHERE brief=? AND state=? ORDER BY seq DESC LIMIT 1",
        (run["statistical_scope_id"], "procedure:" + RECONCILIATION),
    ).fetchone()
    return run, latest and json.loads(latest[0])["artifact_id"]


def _calibrated_scope(row):
    """Whole-procedure calibration replaces the scalar count only once it is qualified."""
    scope = row["inherited_effective_scope"] or {}
    return row["calibration_qualified"] and scope.get("combination_rule") == WHOLE


def opportunities(controller):
    """Registered opportunity rows (feedback bounds); reading grants no slot."""
    return _rows(controller, OPPORTUNITY)
