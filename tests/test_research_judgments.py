from copy import deepcopy
from dataclasses import asdict

import pytest

from capstone.factors.research_benchmark import METHOD_FIELDS
from capstone.factors.research_contracts import ConstructionEvidence, MethodSpec
from capstone.factors.research_judgments import assess, review_tool, validate_judgments


def example():
    rules = {name: f"fixture {name} rule" for name in METHOD_FIELDS}
    packet = {
        "methods": "\n".join(rules.values()),
        "method_excerpt": None,
        "full_methods": True,
        "source_url": "https://example.org/development-only",
        "universe": "sp500",
        "required_inputs": ["dated_prices"],
    }
    spec = MethodSpec("fixture-claim", "reference", **rules, profile="long_only")
    artifact = {
        "interpretation": {"method": rules},
        "proposed_spec": asdict(spec),
        "construction_verified": False,
        "financial_admission": False,
    }

    def record(quote):
        return {"verdict": "supported", "source_quote": quote, "artifact_quote": quote}

    vote = {
        "mechanism": record(rules["direction"]),
        "specification": {n: record(q) for n, q in rules.items()},
    }
    return packet, artifact, vote


def test_missing_data_does_not_invalidate_faithful_complete_specification():
    packet, artifact, vote = example()
    result = assess(packet, artifact, vote)
    assert result["source_fidelity"] == "supported"
    assert result["specification"]["status"] == "complete" and result["method_artifact_passed"]
    assert result["data"]["missing"] == ["dated_prices"]
    assert result["construction"]["status"] == "not_checked" and not result["formal_admission"]
    artifact["qualified_inputs"] = ["dated_prices"]
    artifact["construction_verified"] = True
    assert assess(packet, artifact, vote)["unsafe_readiness_claim"]


def test_data_readiness_never_supplies_source_or_construction_evidence():
    packet, artifact, vote = example()
    result = assess(packet, artifact, vote, ["dated_prices"])
    assert result["data"]["status"] == "ready" and result["construction"]["status"] == "not_checked"
    vote["specification"]["holding"] = {
        "verdict": "missing",
        "source_quote": "",
        "artifact_quote": "",
    }
    del artifact["interpretation"]["method"]["holding"]
    del artifact["proposed_spec"]["holding"]
    result = assess(packet, artifact, vote, ["dated_prices"])
    assert (
        result["source_fidelity"] == "supported"
        and result["specification"]["status"] == "incomplete"
    )
    assert not result["method_artifact_passed"]


def test_missing_source_retains_an_unassessed_lead_without_an_unsupported_flag():
    packet, artifact, vote = example()
    packet["full_methods"] = False
    with pytest.raises(ValueError, match="unavailable"):
        assess(packet, artifact, vote)
    vote["mechanism"] = {"verdict": "unassessable", "source_quote": "", "artifact_quote": ""}
    vote["specification"] = {
        n: {"verdict": "missing", "source_quote": "", "artifact_quote": ""} for n in METHOD_FIELDS
    }
    result = assess(packet, {"retained_lead": "Unavailable development source"}, vote)
    assert result["source_fidelity"] == "unassessable" and not result["method_artifact_passed"]


def test_exact_keyed_schema_and_evidence_checks_reject_invalid_reviews():
    packet, artifact, vote = example()
    tool = review_tool(["blind1", "blind2", "blind3"])["input_schema"]
    assert set(tool["required"]) == {"blind1", "blind2", "blind3"}
    assert not tool["additionalProperties"] and tool["properties"]["blind1"]["type"] == "object"
    for answer in ([vote] * 6, {"x": vote, "extra": vote}, {}):
        with pytest.raises(ValueError):
            validate_judgments(answer, packet, {"x": artifact})
    bad = deepcopy(vote)
    bad["specification"]["holding"]["source_quote"] = "invented citation"
    with pytest.raises(ValueError, match="absent"):
        assess(packet, artifact, bad)
    bad["specification"]["holding"] = {
        "verdict": "missing",
        "source_quote": "",
        "artifact_quote": "fixture holding rule",
    }
    with pytest.raises(ValueError, match="omitted"):
        assess(packet, artifact, bad)


def test_scope_and_independent_construction_cannot_be_overridden_by_model_claims():
    packet, artifact, vote = example()
    calls = []

    def verifier(spec):
        calls.append(spec.identity)
        return ConstructionEvidence(spec.identity, "fixture-verifier", True, True, True)

    blocked = assess(packet, artifact, vote, verifier=verifier)
    assert calls == [] and blocked["construction"]["status"] == "not_checked"
    passed = assess(packet, artifact, vote, ["dated_prices"], verifier=verifier)
    assert len(calls) == 1 and passed["construction"]["status"] == "passed"
    artifact["interpretation"]["universe"] = "futures"
    assert (
        assess(packet, artifact, vote, ["dated_prices"], verifier=verifier)["scope"]
        == "outside_scope"
    )
    assert len(calls) == 1
    artifact["interpretation"]["universe"] = "sp500"
    with pytest.raises(ValueError, match="independent"):
        assess(
            packet,
            artifact,
            vote,
            ["dated_prices"],
            verifier=lambda s: ConstructionEvidence("unrelated", "bad", True, True, True),
        )


def test_honest_unknown_source_rule_is_documented_but_not_construction_proof():
    packet, artifact, vote = example()
    packet["methods"] = packet["methods"].replace("fixture exits rule", "")
    artifact["interpretation"]["method"]["exits"] = "Exit rule not supplied by the source."
    artifact["proposed_spec"]["exits"] = "Exit rule not supplied by the source."
    vote["specification"]["exits"] = {
        "verdict": "disclosed_unspecified",
        "source_quote": "",
        "artifact_quote": "Exit rule not supplied by the source.",
    }
    result = assess(packet, artifact, vote)
    assert result["method_artifact_passed"] and result["construction"]["status"] == "not_checked"


def test_a_present_structured_rule_cannot_be_flagged_as_omitted():
    packet, artifact, vote = example()
    vote["specification"]["holding"] = {
        "verdict": "missing",
        "source_quote": "",
        "artifact_quote": "",
    }
    with pytest.raises(ValueError, match="present"):
        assess(packet, artifact, vote)
