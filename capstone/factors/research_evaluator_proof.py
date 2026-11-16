"""Synthetic development examples, never scored-paper or historical return evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

from capstone.factors.llm import FactorConfig
from capstone.factors.research_contracts import ConstructionEvidence, MethodSpec, digest
from capstone.factors.research_judgments import (
    METHOD_FIELDS,
    assess,
    validate_judgments,
)
from capstone.factors.research_proof import construction_handoff_checks
from capstone.factors.research_requests import RequestJournal
from capstone.factors.research_review_v2 import review_packet
from capstone.pipeline.budget import BudgetedClient

RULES = {
    "direction": "Buy five-session losers and short five-session winners.",
    "selection": "Include both stocks in the synthetic universe.",
    "weighting": "Demean the negative sum of five session returns; gross exposure is one.",
    "formation": "Form targets after each completed session.",
    "holding": "Hold until the next session's rebalance.",
    "exits": "Close or replace positions at the next rebalance.",
    "availability": "Only completed-session returns and dated universe records are permitted.",
}


def fixtures():
    packet = {
        "id": "development-only",
        "source_order": 0,
        "family": "liquidity_supply",
        "title": "Synthetic two-stock method; no research finding",
        "source_url": "https://example.org/synthetic-evaluator-development",
        "method_location": "self-authored development rules",
        "method_excerpt": None,
        "methods": "\n".join(RULES.values()),
        "full_methods": True,
        "required_inputs": ["returns", "dated_universe"],
        "universe": "sp500",
    }
    spec = MethodSpec("synthetic-evaluator-claim", "reference", **RULES, profile="long_short")
    artifact = {
        "status": "needs_data",
        "interpretation": {"method": RULES.copy()},
        "proposed_spec": asdict(spec),
        "construction_verified": False,
        "financial_admission": False,
    }
    vote = {
        "mechanism": {
            "verdict": "supported",
            "source_quote": RULES["direction"],
            "artifact_quote": RULES["direction"],
        },
        "specification": {
            n: {"verdict": "supported", "source_quote": q, "artifact_quote": q}
            for n, q in RULES.items()
        },
    }

    def case(name, expected, qualified=()):
        return {
            "name": name,
            "packet": deepcopy(packet),
            "artifact": deepcopy(artifact),
            "judgment": deepcopy(vote),
            "qualified": list(qualified),
            "expected": expected,
        }

    # Expected answers are authored independently of evaluator output.
    cases = [
        case(
            "complete_missing_data",
            ["supported", "complete", "missing_inputs", "in_scope", False],
        ),
        case(
            "ready_without_construction",
            ["supported", "complete", "ready", "in_scope", False],
            packet["required_inputs"],
        ),
        case(
            "missing_holding",
            ["supported", "incomplete", "ready", "in_scope", False],
            packet["required_inputs"],
        ),
        case(
            "wrong_direction",
            ["unsupported", "incomplete", "missing_inputs", "in_scope", False],
        ),
        case(
            "disclosed_unknown",
            ["supported", "complete", "missing_inputs", "in_scope", False],
        ),
        case(
            "outside_scope",
            ["supported", "complete", "ready", "outside_scope", False],
            packet["required_inputs"],
        ),
        case(
            "fabricated_readiness",
            ["supported", "complete", "missing_inputs", "in_scope", True],
        ),
        case(
            "unavailable_method",
            ["unassessable", "incomplete", "missing_inputs", "in_scope", False],
        ),
    ]
    missing, wrong, unknown, outside, forged, unavailable = cases[2:]
    del missing["artifact"]["interpretation"]["method"]["holding"]
    del missing["artifact"]["proposed_spec"]["holding"]
    missing["judgment"]["specification"]["holding"] = {
        "verdict": "missing",
        "source_quote": "",
        "artifact_quote": "",
    }
    opposite = "Buy winners and short losers instead."
    wrong["artifact"]["interpretation"]["method"]["direction"] = opposite
    wrong["artifact"]["proposed_spec"]["direction"] = opposite
    wrong["judgment"]["mechanism"].update(verdict="contradicted", artifact_quote=opposite)
    wrong["judgment"]["specification"]["direction"].update(
        verdict="contradicted", artifact_quote=opposite
    )
    unknown["packet"]["methods"] = "\n".join(q for n, q in RULES.items() if n != "exits")
    disclosure = "The supplied source does not specify an exit rule."
    unknown["artifact"]["interpretation"]["method"]["exits"] = disclosure
    unknown["artifact"]["proposed_spec"]["exits"] = disclosure
    unknown["judgment"]["specification"]["exits"] = {
        "verdict": "disclosed_unspecified",
        "source_quote": "",
        "artifact_quote": disclosure,
    }
    outside["packet"]["universe"] = "futures"
    forged["artifact"].update(
        qualified_inputs=packet["required_inputs"],
        construction_verified=True,
        financial_admission=True,
    )
    unavailable["packet"].update(full_methods=False, methods="Primary methods unavailable.")
    unavailable["artifact"] = {
        "status": "methods_review_needed",
        "retained_lead": "Unverified synthetic lead",
    }
    unavailable["judgment"] = {
        "mechanism": {
            "verdict": "unassessable",
            "source_quote": "",
            "artifact_quote": "",
        },
        "specification": {
            n: {"verdict": "missing", "source_quote": "", "artifact_quote": ""}
            for n in METHOD_FIELDS
        },
    }
    partial = case(
        "underspecified_holding",
        ["supported", "incomplete", "ready", "in_scope", False],
        packet["required_inputs"],
    )
    partial_rule = "Hold for an unspecified interval."
    partial["artifact"]["interpretation"]["method"]["holding"] = partial_rule
    partial["artifact"]["proposed_spec"]["holding"] = partial_rule
    partial["judgment"]["specification"]["holding"].update(
        verdict="underspecified", artifact_quote=partial_rule
    )
    cases.append(partial)
    return cases


class ScriptedReviewer:
    """Returns a supplied oracle record; does not simulate semantic model ability."""

    max_retries = 0

    def __init__(self, vote, *, invalid=0):
        self.messages = self
        self.vote, self.invalid, self.calls = vote, invalid, 0
        self.contexts = []

    def create(self, **kwargs):
        self.calls += 1
        context = json.loads(kwargs["messages"][0]["content"])
        self.contexts.append(context)
        answer = {row["blind_id"]: deepcopy(self.vote) for row in context["artifacts"]}
        if self.calls <= self.invalid:
            answer["extra-artifact"] = deepcopy(self.vote)
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="review_methods_v2", input=answer)],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def run_proof(root):
    cases = fixtures()
    evidence, rows = {}, []
    for case in cases:
        result = assess(case["packet"], case["artifact"], case["judgment"], case["qualified"])
        observed = [
            result["source_fidelity"],
            result["specification"]["status"],
            result["data"]["status"],
            result["scope"],
            result["unsafe_readiness_claim"],
        ]
        assert observed == case["expected"], case["name"]
        assert result["construction"]["status"] == "not_checked" and not result["formal_admission"]
        rows.append({"case": case["name"], "expected": case["expected"], "observed": observed})
    evidence["independent_axis_answers"] = True
    case = cases[0]
    for bad in (
        [case["judgment"]] * 6,
        {"blind": case["judgment"], "extra": case["judgment"]},
    ):
        try:
            validate_judgments(bad, case["packet"], {"blind": case["artifact"]})
        except ValueError:
            continue
        raise AssertionError("malformed review accepted")
    evidence["invalid_cardinality_rejected"] = True
    client = ScriptedReviewer(case["judgment"], invalid=1)
    frozen = {
        "kind": "offline-evaluator-v2",
        "fixtures": digest(cases),
        "call_limits": {"review": 2},
    }
    config = FactorConfig(model="fixture", alignment_model="fixture-reviewer")
    outputs = {arm: case["artifact"] for arm in "ABC"}
    with RequestJournal(root / "requests.db", frozen) as journal:
        before = client.calls
        wrapped = BudgetedClient(
            client, 2, journal=journal, bucket="review", task_id="bounded-schema-repair"
        )
        first = review_packet(case["packet"], outputs, wrapped, config, 0, version=2)
        wrapped = BudgetedClient(
            client, 2, journal=journal, bucket="review", task_id="bounded-schema-repair"
        )
        assert review_packet(case["packet"], outputs, wrapped, config, 0, version=2) == first
        assert client.calls - before in (0, 2) and journal.count("review") == 2
    evidence["bounded_repair_and_saved_response_replay"] = True
    broken = ScriptedReviewer(case["judgment"], invalid=99)
    try:
        review_packet(case["packet"], outputs, broken, config, 0, version=2)
    except ValueError:
        assert broken.calls == 2
    else:
        raise AssertionError("unbounded repair or invalid acceptance")
    evidence["all_invalid_reviews_stop_after_one_repair"] = True
    verified = []
    engine_checks = construction_handoff_checks(root)

    def independent(spec):
        matches = all(getattr(spec, n) == RULES[n] for n in METHOD_FIELDS)
        verified.append(spec.identity)
        return ConstructionEvidence(
            spec.identity,
            "existing-engine-development-proof",
            True,
            True,
            matches and all(engine_checks.values()),
        )

    result = assess(
        case["packet"],
        case["artifact"],
        case["judgment"],
        ["returns", "dated_universe"],
        verifier=independent,
    )
    assert result["construction"]["status"] == "passed" and len(verified) == 1
    assert not result["formal_admission"]
    evidence["matching_existing_constructor_evidence_required"] = True
    failed = assess(
        case["packet"],
        case["artifact"],
        case["judgment"],
        ["returns", "dated_universe"],
        verifier=lambda s: ConstructionEvidence(s.identity, "failed-example", True, True, False),
    )
    assert failed["method_artifact_passed"] and failed["construction"]["status"] == "failed"
    assert not failed["formal_admission"]
    evidence["failed_construction_cannot_be_overridden"] = True
    evidence["method_artifact_can_pass_while_data_blocked"] = (
        first["A"]["method_artifact_passed"] and first["A"]["data"]["status"] == "missing_inputs"
    )
    return {
        "mode": "offline_scripted_development_only",
        "review_version": 2,
        "checks": evidence | engine_checks,
        "cases": rows,
        "fixture_hash": digest(cases),
        "model_requests": 0,
        "financial_admissions": 0,
        "limitation": "Scripted checks only; semantic model reliability is unestablished.",
    }
