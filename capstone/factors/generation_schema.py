"""One frozen strict tool schema; source and constructor authority belong to its caller.

Strict tool use rejects numeric bounds and maxItems, so ranges, quote positions and
source-less citations are enforced by parse_generation_payload and the claim checks.
"""

import json

from capstone.factors.idea_contracts import QUOTE_PURPOSES

TOOL_NAME = "propose_source_grounded_program"
CLAIM_TEXT = ("kind", "mechanism", "prediction", "horizon", "universe", "falsifier", "mathematics")
CLAIM_LISTS = (
    "method_sources",
    "unresolved_rules",
    "adaptation_notes",
    "aliases",
)
PROGRAM_FIELDS = ("expression", "side", "tail", "weighting", "schedule", "direction")
PROVENANCE_RULES = PROGRAM_FIELDS


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def array(items):
    return {"type": "array", "items": items}


DESCRIPTIONS = {
    "unresolved_rules": "Only rules that cannot be set here and block construction. "
    "Never controller-owned globals (point_in_time_universe, exchange_calendar, "
    "feature_availability, index membership) and never researcher choices.",
    "declared_defaults": "Researcher choices the source leaves open, each with the chosen "
    "value and its basis. Recorded as labeled adaptations; they do not block.",
    "assumptions": "Researcher assumptions cited by method provenance; not source facts.",
    "adaptation_notes": "Departures from the source method; required for adaptations.",
    "required_inputs": "Each input as {name, description} when it is one of the available "
    "inputs, else {unavailable, description} naming it in free text. Never relabel an "
    "unavailable input as an available name.",
}


def available_inputs(frozen):
    """Canonical input names frozen into the prompt: the only names a claim may use."""
    return json.loads(frozen.prompt_json)["available_inputs"]


def generation_tool(frozen):
    quote = obj(
        {
            "revision_id": {"type": "string"},
            "span_id": {"type": "string"},
            "quote": {
                "type": "string",
                "description": "Exact contiguous substring of the span text, verbatim.",
            },
            "start": {
                "type": "integer",
                "description": "0-based character index of quote in the span text; a quote "
                "occurring exactly once in its span is located by the controller.",
            },
            "purpose": {"type": "string", "enum": list(QUOTE_PURPOSES)},
        }
    )
    claim = obj(
        {
            **{name: {"type": "string"} for name in CLAIM_TEXT},
            **{name: array({"type": "string"}) for name in CLAIM_LISTS},
            "source_statements": array(quote),
            "assumptions": array(
                obj({"text": {"type": "string"}, "rationale": {"type": "string"}})
            ),
            # Strict schemas: two closed objects under anyOf, never an enum in a union.
            "required_inputs": array(
                {
                    "anyOf": [
                        obj(
                            {
                                "name": {"type": "string", "enum": available_inputs(frozen)},
                                "description": {"type": "string"},
                            }
                        ),
                        obj({"unavailable": {"type": "string"}, "description": {"type": "string"}}),
                    ]
                }
            ),
            "declared_defaults": array(
                obj(
                    {
                        "rule": {"type": "string"},
                        "value": {"type": "string"},
                        "basis": {"type": "string"},
                    }
                )
            ),
        }
    )
    for name, text in DESCRIPTIONS.items():
        claim["properties"][name]["description"] = text
    claim["properties"]["kind"]["enum"] = [frozen.kind]
    if frozen.source_revisions:
        quote["properties"]["revision_id"]["enum"] = list(frozen.source_revisions)
        claim["properties"]["method_sources"]["items"]["enum"] = list(frozen.source_revisions)
    context = frozen.context
    fields = {
        "kind": {"type": "string", "enum": sorted({s.kind for s in context.schedules})},
        "interval": {
            "type": "integer",
            "enum": sorted({s.interval for s in context.schedules}),
        },
    }
    windows = [s for s in context.schedules if s.kind == "calendar_window"]
    if windows:
        # Offsets exist only on version-2 calendar windows; other schedules send null.
        for name in ("enter_offset", "exit_offset"):
            values = sorted({getattr(s, name) for s in windows})
            # Strict tool schemas reject an enum under a type union (API 400: "Enum value
            # -1 does not match declared type"); a nullable enum is an anyOf.
            fields[name] = {"anyOf": [{"type": "integer", "enum": values}, {"type": "null"}]}
    schedule = obj(fields)
    program = obj(
        {
            "expression": {"type": "string"},
            "side": {"type": "string", "enum": list(context.sides)},
            "tail": {"type": "number"},
            "weighting": {"type": "string", "enum": list(context.weightings)},
            "schedule": schedule,
            "direction": {"type": "string", "enum": list(context.directions)},
        }
    )
    program["type"] = ["object", "null"]
    provenance = obj(
        {
            "rule": {"type": "string", "enum": list(PROVENANCE_RULES)},
            "reasoning": {"type": "string"},
            "quote_indexes": array({"type": "integer"}),
            "assumption_indexes": array({"type": "integer"}),
        }
    )
    return {
        "name": TOOL_NAME,
        "description": "Propose one claim and one construction or retain explicit blockers.",
        "input_schema": obj(
            {
                "claim": claim,
                "program": program,
                "method_provenance": array(provenance),
                "construction_blockers": array({"type": "string"}),
            }
        ),
    }
