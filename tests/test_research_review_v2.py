import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_research_judgments import example

from capstone.factors import research_review as legacy
from capstone.factors.llm import FactorConfig
from capstone.factors.research_contracts import ConstructionEvidence
from capstone.factors.research_judgments import assess
from capstone.factors.research_review_v2 import review_packet, summarize

CONFIG = FactorConfig(model="scripted", alignment_model="scripted-reviewer")


def fixtures():
    packet, artifact, judgment = example()
    artifact["status"] = "needs_data"
    packet.update(
        id="fixture",
        source_order=0,
        family="liquidity_supply",
        title="Synthetic method",
        method_location="fixture rules",
    )
    return [dict(packet=packet, artifact=artifact, judgment=judgment)]


class ScriptedReviewer:
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
            content=[SimpleNamespace(type="tool_use", name="review_methods_v2", input=answer)]
        )


def test_blinding_row_permutations_and_rotation_do_not_change_judgments():
    case = fixtures()[0]
    outputs = {arm: case["artifact"] for arm in "ABC"}
    client = ScriptedReviewer(case["judgment"])
    first = review_packet(case["packet"], outputs, client, CONFIG, 0, version=2)
    permuted = {arm: dict(reversed(list(case["artifact"].items()))) for arm in "CBA"}
    second = review_packet(case["packet"], permuted, client, CONFIG, 0, version=2)
    third = review_packet(case["packet"], outputs, client, CONFIG, 1, version=2)
    assert first == second == third
    assert client.contexts[0] == client.contexts[1]
    before = [r["blind_id"] for r in client.contexts[0]["artifacts"]]
    after = [r["blind_id"] for r in client.contexts[2]["artifacts"]]
    assert before != after and set(before) == set(after)
    assert "qualified_inputs" not in client.contexts[0] and "answer_key" not in client.contexts[0]


def test_constructor_fault_is_not_a_model_schema_repair_or_a_financial_pass():
    case = fixtures()[0]
    client = ScriptedReviewer(case["judgment"])
    with pytest.raises(ValueError, match="inventory"):
        review_packet(
            case["packet"],
            {"A": case["artifact"]},
            client,
            CONFIG,
            0,
            version=2,
            qualified_inputs="returns",
        )
    assert client.calls == 0

    def broken(spec):
        raise ValueError("independent verifier fault")

    with pytest.raises(ValueError, match="independent verifier fault"):
        review_packet(
            case["packet"],
            {"A": case["artifact"]},
            client,
            CONFIG,
            0,
            version=2,
            qualified_inputs=case["packet"]["required_inputs"],
            verifiers={"A": broken},
        )
    assert client.calls == 1
    result = assess(
        case["packet"],
        case["artifact"],
        case["judgment"],
        case["packet"]["required_inputs"],
        verifier=lambda spec: ConstructionEvidence(
            spec.identity, "failed-example", True, True, False
        ),
    )
    assert result["method_artifact_passed"] and result["construction"]["status"] == "failed"
    assert not result["formal_admission"]


def test_unsupported_source_and_untyped_construction_cannot_inherit_positive_judgments():
    case = fixtures()[0]
    case["artifact"]["interpretation"]["source_url"] = "https://example.org/different-source"
    assert (
        assess(case["packet"], case["artifact"], case["judgment"])["source_fidelity"]
        == "unsupported"
    )
    del case["artifact"]["interpretation"]["source_url"]
    with pytest.raises(ValueError, match="typed evidence"):
        assess(
            case["packet"],
            case["artifact"],
            case["judgment"],
            case["packet"]["required_inputs"],
            verifier=lambda s: ConstructionEvidence(s.identity, "untyped", "true", True, True),
        )


def test_v2_metrics_do_not_reject_missing_data_or_reuse_v1_success_criteria():
    case = fixtures()[0]
    client = ScriptedReviewer(case["judgment"])
    outputs = {arm: case["artifact"] for arm in "ABC"}
    votes = review_packet(case["packet"], outputs, client, CONFIG, 0, version=2)
    results = {arm: {case["packet"]["id"]: value} for arm, value in outputs.items()}
    reviews = {case["packet"]["id"]: [deepcopy(votes), deepcopy(votes)]}
    benchmark = {"review_version": 2, "packets": [case["packet"]]}
    accounting = {"requests": [{"bucket": a} for a in "ABC"]}
    summary = summarize(benchmark, results, reviews, accounting, integrity_passed=True)
    assert (
        summary["complete_matched_study"] and summary["decision"] == "evaluator_v2_diagnostics_only"
    )
    for row in summary["metrics"].values():
        assert row["verified_method_artifacts"] == row["retained_data_blocked_specs"] == 1
        assert row["data_ready"] == row["verified_constructions"] == 0
        assert row["calls_per_verified_method"] == 1
    reviews[case["packet"]["id"]][1]["B"]["source_fidelity"] = "unassessable"
    reviews[case["packet"]["id"]][1]["B"]["method_artifact_passed"] = False
    changed = summarize(benchmark, results, reviews, accounting, integrity_passed=True)
    assert (
        changed["review_disagreements"]
        and changed["metrics"]["B"]["verified_method_artifacts"] == 0
    )
    assert not changed["formal_admission"]
    reviews[case["packet"]["id"]][0]["A"]["review_version"] = 1
    with pytest.raises(ValueError, match="cannot share"):
        summarize(benchmark, results, reviews, accounting, integrity_passed=True)


def test_v1_summary_and_review_default_delegate_without_migration(monkeypatch):
    benchmark = {"packets": []}
    results, reviews, accounting = {a: {} for a in "ABC"}, {}, {"requests": []}
    expected = legacy.summarize(benchmark, results, reviews, accounting, integrity_passed=True)
    assert summarize(benchmark, results, reviews, accounting, integrity_passed=True) == expected
    monkeypatch.setattr(legacy, "review_packet", lambda *args: "legacy unchanged")
    assert review_packet({}, {}, None, CONFIG, 0) == "legacy unchanged"
    with pytest.raises(ValueError, match="version"):
        review_packet({}, {}, None, CONFIG, 0, version=3)


def test_schema_repair_is_bounded_and_does_not_repeat_constructor_work():
    case = fixtures()[0]
    client = ScriptedReviewer(case["judgment"], invalid=1)
    result = review_packet(case["packet"], {"A": case["artifact"]}, client, CONFIG, 0, version=2)
    assert client.calls == 2 and result["A"]["method_artifact_passed"]
    assert "schema_repair" in client.contexts[1]
    failing = ScriptedReviewer(case["judgment"], invalid=2)
    with pytest.raises(ValueError, match="schema/evidence"):
        review_packet(case["packet"], {"A": case["artifact"]}, failing, CONFIG, 0, version=2)
    assert failing.calls == 2
