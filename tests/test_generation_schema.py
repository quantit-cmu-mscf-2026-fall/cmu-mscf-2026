"""The generation tool schema: closed objects, caller-frozen choices, no executable fields."""

import json
from types import SimpleNamespace

from capstone.factors.generation_schema import (
    CLAIM_LISTS,
    CLAIM_TEXT,
    PROGRAM_FIELDS,
    PROVENANCE_RULES,
    TOOL_NAME,
    available_inputs,
    generation_tool,
)
from capstone.factors.idea_contracts import QUOTE_PURPOSES

REVISION = "a" * 64


def schedule(kind, interval, enter=None, exit_=None):
    return SimpleNamespace(kind=kind, interval=interval, enter_offset=enter, exit_offset=exit_)


def frozen(schedules=None, revisions=(REVISION,), kind="reference"):
    context = SimpleNamespace(
        schedules=schedules or (schedule("month_end", 1),),
        sides=("long_only",),
        weightings=("equal", "capitalization"),
        directions=("high",),
    )
    prompt = json.dumps({"available_inputs": ["adjusted_prices", "membership"]})
    return SimpleNamespace(
        kind=kind, source_revisions=revisions, context=context, prompt_json=prompt
    )


def closed_objects(node):
    """Yield every object schema below node."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "object" in (node.get("type") or ()):
            yield node
        for value in node.values():
            yield from closed_objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from closed_objects(value)


def test_available_inputs_come_from_the_frozen_prompt_only():
    assert available_inputs(frozen()) == ["adjusted_prices", "membership"]


def test_every_object_is_closed_and_every_property_required():
    tool = generation_tool(frozen())
    assert tool["name"] == TOOL_NAME
    objects = list(closed_objects(tool["input_schema"]))
    assert len(objects) > 8
    for node in objects:
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
    json.dumps(tool, sort_keys=True)  # a strict tool schema must be plain JSON


def test_schema_carries_no_numeric_bounds_or_executable_fields():
    text = json.dumps(generation_tool(frozen()))
    for forbidden in ("maxItems", "minimum", "maximum", "minItems", "code", "python"):
        assert forbidden not in text
    properties = generation_tool(frozen())["input_schema"]["properties"]
    assert set(properties) == {"claim", "program", "method_provenance", "construction_blockers"}
    assert set(properties["program"]["properties"]) == set(PROGRAM_FIELDS)
    assert set(PROVENANCE_RULES) == set(PROGRAM_FIELDS)


def test_claim_fields_follow_the_frozen_assignment():
    tool = generation_tool(frozen(revisions=(REVISION, "b" * 64)))
    claim = tool["input_schema"]["properties"]["claim"]["properties"]
    assert claim["kind"]["enum"] == ["reference"]
    assert all(name in claim for name in (*CLAIM_TEXT, *CLAIM_LISTS))
    quote = claim["source_statements"]["items"]["properties"]
    assert quote["revision_id"]["enum"] == [REVISION, "b" * 64]
    assert quote["purpose"]["enum"] == list(QUOTE_PURPOSES)
    assert claim["method_sources"]["items"]["enum"] == [REVISION, "b" * 64]
    # An original idea has no approved revision to cite.
    original = generation_tool(frozen(revisions=(), kind="exploratory_original"))
    claim = original["input_schema"]["properties"]["claim"]["properties"]
    assert "enum" not in claim["source_statements"]["items"]["properties"]["revision_id"]
    assert claim["kind"]["enum"] == ["exploratory_original"]


def test_required_inputs_are_a_closed_choice_between_available_and_unavailable():
    claim = generation_tool(frozen())["input_schema"]["properties"]["claim"]["properties"]
    named, unavailable = claim["required_inputs"]["items"]["anyOf"]
    assert named["properties"]["name"]["enum"] == ["adjusted_prices", "membership"]
    assert set(unavailable["properties"]) == {"unavailable", "description"}
    assert (
        "unresolved_rules" in claim
        and "Never controller-owned" in (claim["unresolved_rules"]["description"])
    )


def test_program_choices_are_only_those_the_caller_froze():
    properties = generation_tool(frozen())["input_schema"]["properties"]["program"]
    assert properties["type"] == ["object", "null"]
    fields = properties["properties"]
    assert fields["weighting"]["enum"] == ["equal", "capitalization"]
    assert fields["side"]["enum"] == ["long_only"] and fields["direction"]["enum"] == ["high"]
    assert set(fields["schedule"]["properties"]) == {"kind", "interval"}
    assert fields["schedule"]["properties"]["kind"]["enum"] == ["month_end"]


def test_calendar_window_offsets_are_nullable_enums_not_a_type_union():
    windows = (schedule("month_end", 1), schedule("calendar_window", 1, -1, 2))
    fields = generation_tool(frozen(windows))["input_schema"]["properties"]["program"][
        "properties"
    ]["schedule"]["properties"]
    assert fields["kind"]["enum"] == ["calendar_window", "month_end"]
    assert fields["enter_offset"] == {
        "anyOf": [{"type": "integer", "enum": [-1]}, {"type": "null"}]
    }
    assert fields["exit_offset"]["anyOf"][0]["enum"] == [2]
