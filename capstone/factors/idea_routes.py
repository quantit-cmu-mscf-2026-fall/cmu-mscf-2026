"""Controller routing for ideas: catalogue entries and assignment checks.

No model transport, executable program, financial handoff or new storage engine lives here.
Inventory arguments are supplied by the trusted caller, separately from claim payloads.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from capstone.factors.idea_contracts import CLAIM_KINDS, IdeaClaimV2, assess_claim
from capstone.factors.research_contracts import AssignedBrief, canonical, digest
from capstone.factors.research_records import load_artifact, put_artifact
from capstone.factors.research_requests import _plain


def _names(values):
    if not isinstance(values, (tuple, list)) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError("expected an explicit sequence of nonempty names")
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class IdeaCatalogueEntry:
    brief: AssignedBrief
    kind: str
    source_revisions: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.brief, AssignedBrief) or self.kind not in CLAIM_KINDS:
            raise ValueError("controller-owned brief and supported idea kind required")
        for name in ("source_revisions", "required_inputs", "aliases"):
            object.__setattr__(self, name, _names(getattr(self, name)))
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in self.source_revisions
        ):
            raise ValueError("approved source revision identity is invalid")

    def to_dict(self):
        return {**asdict(self), "version": 2, "constructor": "program"}

    @classmethod
    def from_dict(cls, payload):
        fields = dict(payload)
        if type(fields.pop("version", None)) is not int or payload["version"] != 2:
            raise ValueError("unsupported idea catalogue version")
        if fields.pop("constructor", None) != "program":
            raise ValueError("unsupported idea constructor route")
        fields["brief"] = AssignedBrief(**fields["brief"])
        return cls(**fields)


def catalogue_rows(catalogue):
    """Freeze typed additions while preserving the canonical V1 manifest unchanged."""
    rows = [
        entry.to_dict() if isinstance(entry, IdeaCatalogueEntry) else entry for entry in catalogue
    ]
    rows = json.loads(canonical(rows))
    names = {row["brief"]["packet_id"]: row["brief"]["packet_id"] for row in rows}
    for row in rows:
        if row.get("version") == 2:
            entry = IdeaCatalogueEntry.from_dict(row)
            for alias in entry.aliases:
                if alias in names and names[alias] != entry.brief.packet_id:
                    raise ValueError("catalogue alias is ambiguous")
                names[alias] = entry.brief.packet_id
    return rows


def _entry(controller, packet):
    if packet not in controller.catalogue:
        matches = [
            key
            for key, (_, entry, _) in controller.catalogue.items()
            if isinstance(entry, IdeaCatalogueEntry) and packet in entry.aliases
        ]
        if len(matches) != 1:
            raise ValueError("unknown or ambiguous idea alias")
        packet = matches[0]
    _, entry, _ = controller.catalogue[packet]
    if not isinstance(entry, IdeaCatalogueEntry):
        raise ValueError("legacy catalogue entry requires the V1 controller route")
    return packet, entry


def _scope(controller, entry):
    return digest(
        {
            "policy": controller.policy.identity,
            "family": entry.brief.family,
            "track": entry.brief.track,
        }
    )


def reserve_idea(controller, packet):
    packet, _ = _entry(controller, packet)
    return controller.reserve(packet)


def idea_binding(controller, packet):
    """Load exact saved identities and independent status axes without reassessment."""
    packet, _ = _entry(controller, packet)
    key = controller._latest(packet, "idea_binding_id")
    binding = load_artifact(controller.db, key, kind="idea_binding_v1")
    return {
        **binding,
        "binding_id": key,
        "assessment": load_artifact(
            controller.db, binding["assessment_record_id"], kind="idea_assessment_v1"
        ),
    }


def assess_idea(
    controller,
    packet,
    claim,
    *,
    archive,
    reviewed_methods,
    qualified_inputs=(),
    input_inventory_id=None,
):
    """Record quote integrity/specification/data axes; never authorize execution."""
    return _record_assessment(
        controller,
        packet,
        claim,
        archive=archive,
        reviewed_methods=reviewed_methods,
        qualified_inputs=qualified_inputs,
        input_inventory_id=input_inventory_id,
    )


def assess_entry_claim(
    controller,
    entry,
    claim,
    *,
    archive,
    reviewed_methods,
    qualified_inputs=(),
    input_inventory_id=None,
):
    """Assignment checks plus assessment axes; records nothing."""
    if not isinstance(claim, IdeaClaimV2) or claim.kind != entry.kind:
        raise ValueError("claim kind differs from its controller-owned assignment")
    if claim.method_key and claim.method_key != entry.brief.method_key:
        raise ValueError("claim cannot change its assigned canonical method")
    refs = set(claim.method_sources) | {quote.revision_id for quote in claim.source_statements}
    if refs - set(entry.source_revisions):
        raise ValueError("claim cites an unapproved source revision")
    if not isinstance(qualified_inputs, (tuple, list, set, frozenset)):
        raise ValueError("qualified inputs must be an explicit collection")
    if set(qualified_inputs) - controller.inputs:
        raise ValueError("qualification exceeds frozen controller input capabilities")
    effective = replace(
        claim, required_inputs=tuple(set(claim.required_inputs) | set(entry.required_inputs))
    )
    assessment = assess_claim(
        effective,
        archive,
        budget_scope_id=_scope(controller, entry),
        reviewed_methods=reviewed_methods,
        qualified_inputs=qualified_inputs,
        input_inventory_id=input_inventory_id,
    )
    return effective, assessment


def _record_assessment(
    controller,
    packet,
    claim,
    *,
    archive,
    reviewed_methods,
    qualified_inputs=(),
    input_inventory_id=None,
    completing_attempt=False,
    in_transaction=False,
):
    packet, entry = _entry(controller, packet)
    effective, assessment = assess_entry_claim(
        controller,
        entry,
        claim,
        archive=archive,
        reviewed_methods=reviewed_methods,
        qualified_inputs=qualified_inputs,
        input_inventory_id=input_inventory_id,
    )
    with nullcontext() if in_transaction else controller.transaction():
        phase = controller.state(packet)
        controller._require(
            packet, "assigned", "idea_assessed", "idea_repair_requested", "idea_repair_exhausted"
        )
        try:
            previous = idea_binding(controller, packet)
        except ValueError as exc:
            if phase not in ("assigned", "idea_repair_requested"):
                raise ValueError("saved idea binding is missing or invalid") from exc
            previous = None
        changed = previous is not None and previous["declared_claim_id"] != claim.identity
        if changed and (phase != "idea_repair_requested" or not completing_attempt):
            raise ValueError("changed claim requires a counted controller repair")
        if previous is None and phase != "assigned" and not completing_attempt:
            raise ValueError("a response must complete the pending attempt before assessment")
        declared_id = put_artifact(controller.db, "idea_claim_v2", asdict(claim))
        claim_id = put_artifact(controller.db, "idea_claim_v2", asdict(effective))
        methods_id = put_artifact(
            controller.db, "reviewed_source_inventory_v1", asdict(reviewed_methods)
        )
        assessment_id = put_artifact(controller.db, "idea_assessment_v1", asdict(assessment))
        binding = {
            "packet_id": packet,
            "budget_scope_id": assessment.budget_scope_id,
            "declared_claim_id": claim.identity,
            "claim_id": effective.identity,
            "declared_claim_record_id": declared_id,
            "claim_record_id": claim_id,
            "methods_inventory_record_id": methods_id,
            "assessment_record_id": assessment_id,
        }
        binding_id = put_artifact(controller.db, "idea_binding_v1", binding)
        if previous is None or previous["binding_id"] != binding_id:
            next_phase = "idea_assessed" if phase == "assigned" else phase
            controller._event(
                packet,
                next_phase,
                "separate trusted assessment axes retained",
                {"idea_binding_id": binding_id},
            )
    return idea_binding(controller, packet)


def _journal_identity(controller, journal):
    saved = journal.db.execute("SELECT payload FROM request_manifest").fetchone()
    if (
        saved is None
        or saved[0] != canonical(journal.manifest)
        or journal.manifest.get("controller") != controller.identity
    ):
        raise ValueError("request journal does not bind this controller")
    path = journal.db.execute("PRAGMA database_list").fetchone()[2]
    if not path:
        raise ValueError("a durable request journal is required")
    return digest({"manifest": journal.manifest, "path": str(Path(path).resolve())})


def _attempt(controller, packet, attempt_id):
    for event in controller.events():
        payload = event["payload"]
        if event["brief"] == packet and payload.get("idea_attempt_id") == attempt_id:
            key = payload.get("idea_attempt_record_id")
            if key:
                return load_artifact(controller.db, key, kind="idea_attempt_v1")
    raise ValueError("unknown controller attempt")


def _completed_attempt(controller, packet, attempt_id):
    for event in controller.events():
        payload = event["payload"]
        if event["brief"] == packet and payload.get("idea_attempt_id") == attempt_id:
            key = payload.get("idea_outcome_id")
            if key:
                return load_artifact(controller.db, key, kind="idea_attempt_outcome_v1")
    return None


def begin_idea_attempt(controller, packet, *, frozen_input_id, request, journal):
    """Freeze one generation request before the existing BudgetedClient sends it."""
    packet, _ = _entry(controller, packet)
    if (
        not isinstance(frozen_input_id, str)
        or len(frozen_input_id) != 64
        or any(c not in "0123456789abcdef" for c in frozen_input_id)
    ):
        raise ValueError("a frozen generation input identity is required")
    context = request_context(controller, packet)
    repairs = controller.db.execute(
        "SELECT repairs FROM research_briefs WHERE id=?", (packet,)
    ).fetchone()[0]
    attempt = {
        "attempt_id": context["task_id"],
        "controller_id": controller.identity,
        "packet_id": packet,
        "repair": repairs,
        "frozen_input_id": frozen_input_id,
        "journal_id": _journal_identity(controller, journal),
        "bucket": context["bucket"],
        "logical_request_id": digest(
            {"bucket": context["bucket"], "task": context["task_id"], "ordinal": 0}
        ),
        "request_digest": digest(request),
    }
    with controller.transaction():
        try:
            previous = _attempt(controller, packet, attempt["attempt_id"])
        except ValueError:
            previous = None
        if previous is not None:
            if previous != attempt:
                raise ValueError("frozen attempt request, input or journal changed")
            return attempt["attempt_id"]
        key = put_artifact(controller.db, "idea_attempt_v1", attempt)
        controller._event(
            packet,
            controller.state(packet),
            "generation request frozen before transport reservation",
            {"idea_attempt_id": attempt["attempt_id"], "idea_attempt_record_id": key},
        )
    return attempt["attempt_id"]


def complete_idea_attempt(
    controller,
    packet,
    *,
    attempt_id,
    frozen_input_id,
    journal,
    request_id,
    response,
    parse_response,
    archive,
    reviewed_methods,
    qualified_inputs=(),
    input_inventory_id=None,
):
    """Parse the checked checkpoint once; an old receipt cannot finish a later repair."""
    packet, _ = _entry(controller, packet)
    attempt = _attempt(controller, packet, attempt_id)
    if (
        attempt["controller_id"] != controller.identity
        or attempt["packet_id"] != packet
        or attempt["frozen_input_id"] != frozen_input_id
        or attempt["journal_id"] != _journal_identity(controller, journal)
    ):
        raise ValueError("attempt controller, input or journal does not match")
    row = journal.db.execute(
        "SELECT logical,bucket,attempt,request,response,state FROM model_requests WHERE id=?",
        (request_id,),
    ).fetchone()
    if (
        row is None
        or row[0] != attempt["logical_request_id"]
        or row[1] != attempt["bucket"]
        or request_id != digest({"logical": row[0], "attempt": row[2]})
        or digest(json.loads(row[3])) != attempt["request_digest"]
        or row[5] != "completed"
    ):
        raise ValueError("completed journal request does not match the frozen attempt")
    checkpoint = journal._response_checked(request_id, row[4])
    receipt = digest(json.loads(row[4]))
    if canonical(_plain(response)) != row[4]:
        raise ValueError("response does not match the saved journal receipt")
    previous = _completed_attempt(controller, packet, attempt_id)
    if previous is not None:
        if previous["request_id"] != request_id or previous["response_receipt"] != receipt:
            raise ValueError("completed attempt receipt changed")
        return previous
    if attempt_id != request_context(controller, packet)["task_id"]:
        raise ValueError("response does not belong to the pending attempt")
    try:
        proposal = parse_response(checkpoint)
    except ValueError as exc:
        claim = None
        failure = {"kind": "schema_invalid", "reason": str(exc)}
    else:
        claim = proposal if isinstance(proposal, IdeaClaimV2) else proposal.claim
        if not isinstance(claim, IdeaClaimV2):
            raise TypeError("generation parser must return a typed claim or proposal")
        if not isinstance(proposal, IdeaClaimV2) and proposal.input_id != frozen_input_id:
            raise ValueError("parsed proposal does not bind the frozen input")
        failure = None
    with controller.transaction():
        previous = _completed_attempt(controller, packet, attempt_id)
        if previous is not None:
            return previous
        if attempt_id != request_context(controller, packet)["task_id"]:
            raise ValueError("response does not belong to the pending attempt")
        binding = None
        if claim is not None:
            binding = _record_assessment(
                controller,
                packet,
                claim,
                archive=archive,
                reviewed_methods=reviewed_methods,
                qualified_inputs=qualified_inputs,
                input_inventory_id=input_inventory_id,
                completing_attempt=True,
                in_transaction=True,
            )
        outcome = {
            "attempt_id": attempt_id,
            "request_id": request_id,
            "response_receipt": receipt,
            "frozen_input_id": frozen_input_id,
            "claim_id": None if binding is None else binding["claim_id"],
            "idea_binding_id": None if binding is None else binding["binding_id"],
            "failure": failure,
        }
        key = put_artifact(controller.db, "idea_attempt_outcome_v1", outcome)
        controller._event(
            packet,
            "idea_assessed" if failure is None else "idea_schema_failed",
            "completed journal receipt parsed and retained",
            {"idea_attempt_id": attempt_id, "idea_outcome_id": key},
        )
    return outcome


def record_idea_feedback(controller, packet, *, kind, reason):
    """Trusted typed feedback binds the latest completed attempt and exact receipt."""
    packet, _ = _entry(controller, packet)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("nonempty feedback reason required")
    if kind not in (
        "schema_invalid",
        "evidence_invalid",
        "incomplete_specification",
        "construction_failed",
    ):
        raise ValueError("unsupported typed repair feedback")
    outcome = load_artifact(
        controller.db, controller._latest(packet, "idea_outcome_id"), kind="idea_attempt_outcome_v1"
    )
    with controller.transaction():
        return put_artifact(
            controller.db,
            "idea_feedback_v1",
            {
                "attempt_id": outcome["attempt_id"],
                "response_receipt": outcome["response_receipt"],
                "claim_id": outcome["claim_id"],
                "kind": kind,
                "reason": reason,
            },
        )


def request_idea_repair(controller, packet, *, feedback_id, reason):
    """Grant at most the existing policy allowance, replaying identical feedback idempotently."""
    packet, _ = _entry(controller, packet)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a nonempty repair reason is required")
    with controller.transaction():
        prior = [
            event
            for event in controller.events()
            if event["brief"] == packet and event["payload"].get("idea_feedback_id") == feedback_id
        ]
        if prior:
            return controller.state(packet)
        controller._require(packet, "idea_assessed", "idea_schema_failed")
        outcome = load_artifact(
            controller.db,
            controller._latest(packet, "idea_outcome_id"),
            kind="idea_attempt_outcome_v1",
        )
        feedback = load_artifact(controller.db, feedback_id, kind="idea_feedback_v1")
        if any(
            feedback.get(name) != outcome[name]
            for name in ("attempt_id", "response_receipt", "claim_id")
        ):
            raise ValueError("repair feedback does not bind the current attempt receipt")
        if feedback.get("kind") not in (
            "schema_invalid",
            "evidence_invalid",
            "incomplete_specification",
            "construction_failed",
        ):
            raise ValueError("unsupported typed repair feedback")
        repairs = controller.db.execute(
            "SELECT repairs FROM research_briefs WHERE id=?", (packet,)
        ).fetchone()[0]
        phase = "idea_repair_exhausted"
        if repairs < controller.policy.max_repairs:
            controller.db.execute(
                "UPDATE research_briefs SET repairs=repairs+1 WHERE id=?", (packet,)
            )
            phase = "idea_repair_requested"
        controller._event(packet, phase, reason, {"idea_feedback_id": feedback_id})
        return phase


def request_context(controller, packet, *, phase="claim"):
    """Stable RequestJournal routing only; the caller supplies its existing BudgetedClient."""
    packet, entry = _entry(controller, packet)
    if phase != "claim":
        raise ValueError("program routing awaits the integrated schema6 construction contract")
    controller._require(packet, "assigned", "idea_assessed", "idea_repair_requested")
    repairs = controller.db.execute(
        "SELECT repairs FROM research_briefs WHERE id=?", (packet,)
    ).fetchone()[0]
    return {
        "bucket": f"{entry.brief.family}:{entry.brief.track}",
        "task_id": digest(
            {
                "scope": _scope(controller, entry),
                "controller": controller.identity,
                "packet": packet,
                "method": entry.brief.method_key,
                "phase": phase,
                "repair": repairs,
            }
        ),
    }
