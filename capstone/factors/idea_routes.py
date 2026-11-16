"""Controller routing for ideas: catalogue entries and assignment checks.

No model transport, executable program, financial handoff or new storage engine lives here.
Inventory arguments are supplied by the trusted caller, separately from claim payloads.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace

from capstone.factors.idea_contracts import CLAIM_KINDS, IdeaClaimV2, assess_claim
from capstone.factors.research_contracts import AssignedBrief, canonical, digest
from capstone.factors.research_records import load_artifact, put_artifact


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
