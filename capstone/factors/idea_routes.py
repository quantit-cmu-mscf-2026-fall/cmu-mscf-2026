"""Controller routing for ideas: catalogue entries and assignment checks.

No model transport, executable program, financial handoff or new storage engine lives here.
Inventory arguments are supplied by the trusted caller, separately from claim payloads.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

from capstone.factors.idea_contracts import CLAIM_KINDS, IdeaClaimV2, assess_claim
from capstone.factors.research_contracts import AssignedBrief, canonical, digest


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
