import json
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
from test_generation_freeze import setup_input  # noqa: E402

from capstone.factors.generation import (
    SYSTEM,
    parse_generation_payload,
    parse_generation_response,
)
from capstone.factors.generation_schema import PROGRAM_FIELDS, TOOL_NAME, generation_tool
from capstone.factors.idea_contracts import (
    DeclaredDefault,
    ReviewedSourceInventory,
    assess_claim,
)
from capstone.factors.llm import FactorConfig, call_tool, tool_input
from capstone.factors.research_contracts import canonical, digest
from capstone.factors.research_requests import _plain, _response
from capstone.strategy.program_schedule import ProgramSchedule


def payload(kind="exploratory_original"):
    return {
        "claim": {
            "kind": kind,
            "mechanism": "Temporary demand pressure reverses.",
            "prediction": "Prior losers outperform next month.",
            "horizon": "One month.",
            "universe": "sp500",
            "falsifier": "No incremental net return.",
            "mathematics": "score(i,t) = -return(i,t-1)",
            "required_inputs": [{"name": "returns", "description": "Daily total returns."}],
            "source_statements": [],
            "assumptions": [
                {
                    "text": "Temporary demand is assumed.",
                    "rationale": "This is a testable researcher conjecture.",
                }
            ],
            "method_sources": [],
            "unresolved_rules": [],
            "declared_defaults": [],
            "adaptation_notes": [],
            "aliases": [],
        },
        "program": {
            "expression": "-shift(returns, 1)",
            "side": "long_only",
            "tail": 0.2,
            "weighting": "equal",
            "schedule": {"kind": "month_end", "interval": 1},
            "direction": "high",
        },
        "method_provenance": [
            {
                "rule": rule,
                "reasoning": "Explicit proposed construction.",
                "quote_indexes": [],
                "assumption_indexes": [0],
            }
            for rule in PROGRAM_FIELDS
        ],
        "construction_blockers": [],
    }


def response(value):
    return NS(
        content=[
            NS(type="thinking", thinking="Private reasoning"),
            NS(type="tool_use", name=TOOL_NAME, input=value),
        ],
        usage=NS(input_tokens=3, output_tokens=4),
    )


def test_declared_defaults_parse_into_the_claim(tmp_path):
    _, frozen = setup_input(tmp_path)
    value = payload()
    value["claim"]["declared_defaults"] = [
        {"rule": "formation lookback", "value": "252 sessions", "basis": "Source is silent."}
    ]
    proposal = parse_generation_payload(value, frozen)
    assert proposal.claim.semantics_version == 4
    assert proposal.claim.declared_defaults == (
        DeclaredDefault("formation lookback", "252 sessions", "Source is silent."),
    )
    del value["claim"]["declared_defaults"]
    with pytest.raises(ValueError, match="unapproved fields"):
        parse_generation_payload(value, frozen)


def test_original_generation_and_cached_client_response_shape_need_no_paper_or_calls(tmp_path):
    archive, frozen = setup_input(tmp_path)
    proposal = parse_generation_response(response(payload()), frozen)
    cached = _response(canonical(_plain(response(payload()))))
    assert parse_generation_response(cached, frozen) == proposal
    assert proposal.claim.method_key == "assigned-method"
    assert proposal.claim.source_statements == ()
    assert proposal.program.expression == "-shift(returns, 1)"
    assert not hasattr(frozen.context, "claim_id")
    assert frozen.context.data_version == "root-data-v1"
    assert proposal.input_id == frozen.identity
    assessment = assess_claim(
        proposal.claim,
        archive,
        budget_scope_id="trusted-assignment",
        reviewed_methods=ReviewedSourceInventory("trusted-review"),
    )
    assert assessment.data_status == "needs_data" and assessment.financial_status == "pending"


def test_quote_and_assumption_indexes_bind_raw_records_before_claim_canonical_sorting(tmp_path):
    archive, frozen = setup_input(
        tmp_path, kind="reference", text="First source paragraph.\n\nSecond paragraph."
    )
    artifact = archive.load(frozen.source_revisions[0])
    value = payload("reference")
    quotes = [
        {
            "revision_id": artifact.revision_id,
            "span_id": span.span_id,
            "quote": span.text,
            "start": 0,
            "purpose": "method",
        }
        for span in artifact.spans
    ]
    quotes.sort(key=digest, reverse=True)
    assumptions = [{"text": name, "rationale": "Explicit conjecture."} for name in ("A", "B")]
    assumptions.sort(key=digest, reverse=True)
    value["claim"].update(
        source_statements=quotes, assumptions=assumptions, method_sources=[artifact.revision_id]
    )
    for row in value["method_provenance"]:
        row["quote_indexes"] = [0]
    proposal = parse_generation_payload(value, frozen)
    raw_quote_id, raw_assumption_id = digest(quotes[0]), digest(assumptions[0])
    assert digest(asdict(proposal.claim.source_statements[0])) != raw_quote_id
    assert digest(asdict(proposal.claim.assumptions[0])) != raw_assumption_id
    assert all(
        row.quote_ids == (raw_quote_id,) and row.assumption_ids == (raw_assumption_id,)
        for row in proposal.method_provenance
    )
    assert "quote_indexes" not in canonical(asdict(proposal))
    assessment = assess_claim(
        proposal.claim,
        archive,
        budget_scope_id="trusted-assignment",
        reviewed_methods=ReviewedSourceInventory("no-methods-review"),
    )
    assert assessment.evidence_status == "quote_integrity_verified"
    assert assessment.completeness_status == "incomplete"


@pytest.mark.parametrize("index", [True, False, -1, 2, 0.0, "0"])
def test_unstable_or_invalid_provenance_positions_are_rejected(tmp_path, index):
    _, frozen = setup_input(tmp_path)
    value = payload()
    value["method_provenance"][0]["assumption_indexes"] = [index]
    with pytest.raises(ValueError, match="index"):
        parse_generation_payload(value, frozen)


@pytest.mark.parametrize(
    "location,key",
    [
        ("claim", "full_methods"),
        ("claim", "method_key"),
        ("claim", "budget_scope_id"),
        ("claim", "family"),
        ("claim", "qualified_inputs"),
        ("program", "claim_id"),
        ("program", "data_version"),
        ("program", "anchor"),
        ("program", "financial_admission"),
        ("program", "gate_report"),
        ("program", "constructor"),
    ],
)
def test_model_cannot_supply_trusted_scope_bindings_or_authority_fields(tmp_path, location, key):
    _, frozen = setup_input(tmp_path)
    value = payload()
    value[location][key] = "model-attempted-authority"
    with pytest.raises(ValueError, match="unapproved"):
        parse_generation_payload(value, frozen)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_claim",
        "extra_top",
        "extra_quote",
        "wrong_kind",
        "unknown_expression",
        "missing_provenance",
    ],
)
def test_malformed_single_proposals_are_rejected_without_repair_loops(tmp_path, mutation):
    _, frozen = setup_input(tmp_path)
    value = payload()
    if mutation == "missing_claim":
        del value["claim"]
    elif mutation == "extra_top":
        value["claims"] = [value["claim"]]
    elif mutation == "extra_quote":
        value["claim"]["assumptions"][0]["source_fact"] = True
    elif mutation == "wrong_kind":
        value["claim"]["kind"] = "reference"
    elif mutation == "unknown_expression":
        value["program"]["expression"] = "open('not-executed')"
    else:
        value["method_provenance"].pop()
    with pytest.raises(ValueError):
        parse_generation_payload(value, frozen)


@pytest.mark.parametrize(
    "field,value",
    [
        ("side", "long_short"),
        ("weighting", "capitalization"),
        ("direction", "low"),
        ("tail", True),
        ("tail", float("nan")),
        ("tail", 0.6),
        ("schedule", {"kind": "interval", "interval": 21}),
        ("schedule", {"kind": "month_end", "interval": 1, "version": 2}),
    ],
)
def test_proposed_constructor_choices_cannot_exceed_frozen_root_scope(tmp_path, field, value):
    _, frozen = setup_input(tmp_path)
    candidate = payload()
    candidate["program"][field] = value
    with pytest.raises(ValueError):
        parse_generation_payload(candidate, frozen)


def test_root_can_explicitly_allow_additional_supported_constructor_choices(tmp_path):
    _, frozen = setup_input(
        tmp_path,
        sides=("long_short",),
        directions=("low",),
        weightings=("capitalization",),
        schedules=(ProgramSchedule("interval", 5),),
    )
    value = payload()
    value["program"].update(
        side="long_short",
        direction="low",
        weighting="capitalization",
        schedule={"kind": "interval", "interval": 5},
    )
    proposal = parse_generation_payload(value, frozen)
    assert proposal.program.side == "long_short" and proposal.program.schedule.interval == 5


def test_frozen_quote_mismatch_and_unapproved_revision_are_not_semantic_assessment(tmp_path):
    archive, frozen = setup_input(tmp_path, kind="reference", text="An exact archived statement.")
    artifact = archive.load(frozen.source_revisions[0])
    value = payload("reference")
    quote = {
        "revision_id": artifact.revision_id,
        "span_id": artifact.spans[0].span_id,
        "quote": "Invented words.",
        "start": 0,
        "purpose": "method",
    }
    value["claim"]["source_statements"] = [quote]
    with pytest.raises(ValueError, match="frozen archived"):
        parse_generation_payload(value, frozen)
    quote.update(quote=artifact.spans[0].text, revision_id="0" * 64)
    with pytest.raises(ValueError, match="frozen archived"):
        parse_generation_payload(value, frozen)


def test_verbatim_quote_with_miscounted_start_is_located_only_when_unique(tmp_path):
    # Slice-01 calendar-boundary and network-reference: every quote was verbatim and
    # unique in its span, but starts were off by one or estimated (15045 for 9019).
    text = "Researchers note a strategy may change exposure around month boundaries. Again: twice."
    archive, frozen = setup_input(tmp_path, kind="reference", text=text)
    artifact = archive.load(frozen.source_revisions[0])
    span = artifact.spans[0]
    exact = "a strategy may change exposure around month boundaries"
    value = payload("reference")
    quote = {
        "revision_id": artifact.revision_id,
        "span_id": span.span_id,
        "quote": exact,
        "start": span.text.index(exact) + 1,
        "purpose": "method",
    }
    value["claim"]["source_statements"] = [quote]
    proposal = parse_generation_payload(value, frozen)
    located = proposal.claim.source_statements[0]
    assert span.text[located.start : located.end] == exact
    quote.update(start=15045)
    assert parse_generation_payload(value, frozen).claim == proposal.claim
    quote.update(quote="e", start=0)  # ambiguous: occurs many times, wrong start
    with pytest.raises(ValueError, match="frozen archived"):
        parse_generation_payload(value, frozen)
    assert "exact contiguous substring" in SYSTEM and "locates it" in SYSTEM
    schema = generation_tool(frozen)["input_schema"]["properties"]["claim"]
    statement = schema["properties"]["source_statements"]["items"]["properties"]
    assert "exactly once" in statement["start"]["description"]


def test_unsupported_construction_retains_claim_dependencies_without_inventing_proxy(tmp_path):
    _, frozen = setup_input(tmp_path)
    value = payload()
    value["program"] = None
    value["method_provenance"] = []
    value["construction_blockers"] = ["Option-expiry cohorts are not supported."]
    value["claim"]["required_inputs"].append(
        {"unavailable": "dated option quotes", "description": "Expiry-dated option prices."}
    )
    value["claim"]["unresolved_rules"] = ["Exact expiry exit remains unresolved."]
    proposal = parse_generation_payload(value, frozen)
    assert proposal.program is None
    assert "unavailable:dated option quotes" in proposal.claim.required_inputs
    assert proposal.claim.unresolved_rules == ("Exact expiry exit remains unresolved.",)
    assert proposal.construction_blockers == ("Option-expiry cohorts are not supported.",)
    value["construction_blockers"] = []
    with pytest.raises(ValueError, match="explicit blockers"):
        parse_generation_payload(value, frozen)


def test_response_extractor_preserves_legacy_first_match_but_new_path_requires_exactly_one(
    tmp_path,
):
    _, frozen = setup_input(tmp_path)
    raw = response(payload())
    raw.content.append(NS(type="tool_use", name="extra_tool", input={}))
    assert tool_input(raw, TOOL_NAME) == payload()
    with pytest.raises(ValueError, match="exactly one"):
        parse_generation_response(raw, frozen)
    for malformed in (
        NS(content=[]),
        NS(content=None),
        NS(content=[NS(type="tool_use", name="wrong")]),
        NS(content=[NS(type="tool_use", name=TOOL_NAME, input=[])]),
    ):
        with pytest.raises(ValueError):
            parse_generation_response(malformed, frozen)


def test_existing_call_tool_uses_shared_extractor_and_preserves_first_match():
    raw = response(payload())
    raw.content.append(NS(type="tool_use", name=TOOL_NAME, input={"later": "ignored"}))
    client = NS(messages=NS(create=lambda **request: raw))
    assert (
        call_tool(
            client, FactorConfig(model="test-model"), system="s", user="u", tool={"name": TOOL_NAME}
        )
        == payload()
    )


def test_calendar_window_proposals_are_parsed_against_the_frozen_schedules(tmp_path):
    from capstone.strategy.program_schedule import CalendarWindowSchedule

    window = CalendarWindowSchedule(enter_offset=-1, exit_offset=3)
    month_end = ProgramSchedule("month_end", 1)
    _, frozen = setup_input(tmp_path, schedules=(month_end, window))
    schema = generation_tool(frozen)["input_schema"]["properties"]["program"]["properties"]
    assert set(schema["schedule"]["properties"]) == {
        "kind",
        "interval",
        "enter_offset",
        "exit_offset",
    }
    value = payload()
    value["program"] |= {
        "expression": "close",
        "tail": 1.0,
        "schedule": {
            "kind": "calendar_window",
            "interval": 1,
            "enter_offset": -1,
            "exit_offset": 3,
        },
    }
    proposal = parse_generation_payload(value, frozen)
    assert proposal.program.schedule == window and proposal.program.tail == 1.0
    value["program"]["schedule"] = {
        "kind": "month_end",
        "interval": 1,
        "enter_offset": None,
        "exit_offset": None,
    }
    with pytest.raises(ValueError, match="frozen root constructor choices"):
        parse_generation_payload(value, frozen)
    value["program"]["tail"] = 0.2
    assert parse_generation_payload(value, frozen).program.schedule == month_end
    # A context without calendar windows keeps the v1 schedule object unchanged.
    _, plain = setup_input(tmp_path / "plain")
    schema = generation_tool(plain)["input_schema"]["properties"]["program"]["properties"]
    assert set(schema["schedule"]["properties"]) == {"kind", "interval"}


MEMBERSHIP_TEXT = "point-in-time S&P 500 index membership"
# Qualified names of a frozen archive inventory.
QUALIFIED_NAMES = (
    "cap",
    "close",
    "exchange_calendar",
    "feature_availability",
    "high",
    "low",
    "mkt_return",
    "open",
    "point_in_time_universe",
    "returns",
    "shares",
    "volume",
)


def test_required_inputs_are_canonical_names_or_explicitly_unavailable(tmp_path):
    _, frozen = setup_input(tmp_path, qualified_inputs=("fundamentals_lag", "returns"))
    prompt = json.loads(frozen.prompt_json)
    assert "unavailable" in SYSTEM and "unavailable" in prompt["input_requirements_notice"]
    claim_schema = generation_tool(frozen)["input_schema"]["properties"]["claim"]
    items = claim_schema["properties"]["required_inputs"]["items"]
    named, unavailable = items["anyOf"]
    assert named["required"] == ["name", "description"] and not named["additionalProperties"]
    assert unavailable["required"] == ["unavailable", "description"]
    assert not unavailable["additionalProperties"]
    assert "enum" not in unavailable["properties"]["unavailable"]
    names = named["properties"]["name"]["enum"]
    expected = {"fundamentals_lag", "point_in_time_universe", "exchange_calendar"}
    expected |= {"feature_availability", "returns", "close", "book", "mkt_return"}
    assert expected <= set(names) and names == sorted(names)
    assert prompt["available_inputs"] == names
    value = payload()
    value["claim"]["required_inputs"] = [
        {"name": "fundamentals_lag", "description": "Qualified lag."},
        {"unavailable": "dated analyst forecasts", "description": "Not in the inventory."},
    ]
    claim = parse_generation_payload(value, frozen).claim
    assert claim.required_inputs == ("fundamentals_lag", "unavailable:dated analyst forecasts")
    for bad in (
        ["returns"],  # plain strings were the v4 shape
        [{"name": "S&P 500 membership", "description": "Not a canonical name."}],
        [{"unavailable": " ", "description": "Blank."}],
        [{"name": "returns", "unavailable": "x", "description": "Both forms."}],
        [{"name": "returns"}],
    ):
        value["claim"]["required_inputs"] = bad
        with pytest.raises(ValueError, match="required input"):
            parse_generation_payload(value, frozen)


@pytest.mark.parametrize(
    "item, data_status, missing",
    [
        ({"name": "point_in_time_universe", "description": MEMBERSHIP_TEXT}, "ready", ()),
        # The same idea in free text is declared unavailable, never guessed.
        (
            {"unavailable": MEMBERSHIP_TEXT, "description": "Index membership by date."},
            "needs_data",
            (f"unavailable:{MEMBERSHIP_TEXT}",),
        ),
    ],
)
def test_membership_wording_is_ready_only_under_its_canonical_name(
    tmp_path, item, data_status, missing
):
    archive, frozen = setup_input(tmp_path, qualified_inputs=QUALIFIED_NAMES)
    value = payload()
    value["claim"]["required_inputs"] = [
        {"name": "returns", "description": "Daily total returns."},
        {"name": "exchange_calendar", "description": "Rebalance sessions."},
        item,
    ]
    claim = parse_generation_payload(value, frozen).claim
    assessment = assess_claim(
        claim,
        archive,
        budget_scope_id="trusted-assignment",
        reviewed_methods=ReviewedSourceInventory("trusted-review"),
        qualified_inputs=frozenset(QUALIFIED_NAMES),
        input_inventory_id="frozen-inventory",
    )
    assert assessment.completeness_status == "specification_ready"
    assert assessment.data_status == data_status and assessment.missing_inputs == missing
