import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from capstone.factors.llm import FactorConfig
from capstone.factors.research_benchmark import (
    load_benchmark,
    paper_packet,
    pilot_configs,
    run_controlled,
    validate_intake,
)
from capstone.factors.research_review import FLAGS, review_packet, summarize

BENCHMARK = Path(__file__).resolve().parents[1] / "config/research_benchmark.json"


def benchmark():
    return load_benchmark(BENCHMARK)


def intake(packet):
    return {
        "observation": packet["methods"],
        "knowledge": "primary method",
        "justification": "economic mechanism",
        "specification": packet["methods"],
        "falsification_condition": "no effect in a prespecified test",
        "source_url": packet["source_url"],
        "method_location": packet["method_location"],
        "universe": packet["universe"],
        "required_inputs": packet["required_inputs"],
        "method": {
            n: "honestly unspecified by this fixture"
            for n in (
                "direction",
                "selection",
                "weighting",
                "formation",
                "holding",
                "exits",
                "availability",
            )
        },
        "variant": "reference",
        "profile": "long_short",
        "deviations": [],
    }


class ReviewClient:
    def __init__(self, invalid_first=False):
        self.messages, self.calls, self.contexts = self, 0, []
        self.invalid_first = invalid_first

    def create(self, **kwargs):
        self.calls += 1
        context = json.loads(kwargs["messages"][0]["content"])
        self.contexts.append(context)
        judgments = [
            {"blind_id": r["blind_id"], **dict.fromkeys(FLAGS, False), "reason": "fixture"}
            for r in context["artifacts"]
        ]
        if self.invalid_first and self.calls == 1:
            judgments[0]["grounded"] = 1
        return NS(
            content=[NS(type="tool_use", name="review_methods", input={"judgments": judgments})],
            usage=NS(input_tokens=1, output_tokens=1),
        )


def test_frozen_packet_inventory_and_arm_configs_preserve_boundaries():
    data = benchmark()
    assert len(data["packets"]) == 8 and sum(p["full_methods"] for p in data["packets"]) == 7
    assert data["qualified_inputs"] == [] and sum(data["call_limits"].values()) == 120
    for p in data["packets"]:
        assert "expected" not in paper_packet(p)["methods"]
    config = FactorConfig(alignment_model="fixture-judge")
    configs = pilot_configs(config)
    assert configs["A"].model == config.model and configs["C"].min_alignment == 1
    assert configs["B"].min_alignment == 0 and configs["B"].max_repairs == 1
    assert config.factors_per_hypothesis == 4


def test_packet_row_permutations_preserve_explicit_source_order(tmp_path):
    data = benchmark()
    data["packets"] = list(reversed(data["packets"]))
    path = tmp_path / "permuted.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_benchmark(path) == benchmark()


def test_blinding_rotates_labels_and_repairs_schema_only_once():
    p = benchmark()["packets"][0]
    outputs = {"A": {"status": "one"}, "B": {"status": "two"}, "C": {"status": "three"}}
    client = ReviewClient(invalid_first=True)
    config = FactorConfig(alignment_model="fixture")
    assert set(review_packet(p, outputs, client, config, 0)) == set(outputs)
    assert client.calls == 2 and "schema_repair" in client.contexts[1]
    review_packet(p, outputs, client, config, 1)
    before = [a["blind_id"] for a in client.contexts[0]["artifacts"]]
    after = [a["blind_id"] for a in client.contexts[-1]["artifacts"]]
    assert set(before) == set(after) and before != after
    assert not any(k in client.contexts[0] for k in outputs)


def test_empty_or_incomplete_study_is_inconclusive_not_a_passing_controller():
    data = benchmark()
    summary = summarize(
        data, {"A": {}, "B": {}, "C": {}}, {}, {"requests": []}, integrity_passed=True
    )
    assert summary["decision"] == "inconclusive"
    assert summary["criteria"]["relative_call_cost"] is None
    assert not summary["formal_admission"] and summary["new_financial_trials"] == 0


def test_registered_decision_requires_complete_matched_evidence_and_defined_cost():
    data = benchmark()
    results = {
        arm: {
            p["id"]: {"status": "prototype_only", "hypotheses": [{"specification": "fixture"}]}
            for p in data["packets"]
        }
        for arm in ("A", "B", "C")
    }
    reviews = {}
    for p in data["packets"]:
        votes = {arm: {**dict.fromkeys(FLAGS, False), "reason": "fixture"} for arm in results}
        votes["A"].update(grounded=p["id"] == "lehmann", complete_method=True)
        votes["B"].update(grounded=p["full_methods"], complete_method=True)
        reviews[p["id"]] = [json.loads(json.dumps(votes)), json.loads(json.dumps(votes))]
    accounting = {"requests": [{"bucket": arm, "usage": {}} for arm in results for _ in range(8)]}
    summary = summarize(data, results, reviews, accounting, integrity_passed=True)
    assert summary["decision"] == "supports_larger_exploratory_study"
    results["B"]["links"]["status"] = "unsupported_claim"
    blocked = summarize(data, results, reviews, accounting, integrity_passed=True)
    assert blocked["metrics"]["B"]["unsupported_claims"] == 1
    assert blocked["decision"] != "supports_larger_exploratory_study"
    results["B"]["links"]["status"] = "prototype_only"
    results["A"]["lehmann"]["hypotheses"] = []
    empty = summarize(data, results, reviews, accounting, integrity_passed=True)
    assert empty["metrics"]["A"]["verified_method_artifacts"] == 0
    assert empty["decision"] == "inconclusive"
    results["A"]["lehmann"]["hypotheses"] = [{"specification": "fixture"}]
    reviews["links"][1]["B"]["unsupported_claim"] = True
    summary = summarize(data, results, reviews, accounting, integrity_passed=True)
    assert summary["decision"] == "inconclusive" and summary["decisive_review_disagreement"]
    assert not summary["formal_admission"]


class RecordingController:
    """Just enough of the controller's interface to see what the run asks of it."""

    def __init__(self):
        self.status, self.calls = {}, []

    def state(self, key):
        return self.status.get(key)

    def reserve(self, key):
        self.status[key] = "assigned"
        return True

    def ground(self, key, claim):
        self.calls.append(("ground", key, claim.full_methods))
        self.status[key] = "methods_review_needed"


def test_a_packet_without_full_methods_is_a_retained_lead_and_needs_no_model():
    unavailable = next(p for p in benchmark()["packets"] if not p["full_methods"])
    controller = RecordingController()
    result = run_controlled(unavailable, None, None, FactorConfig(), controller)
    assert result["status"] == "methods_review_needed"
    assert result["retained_lead"] == unavailable["title"]
    assert not result["construction_verified"] and not result["financial_admission"]
    assert controller.calls == [("ground", unavailable["id"], False)]


def test_a_rich_intake_must_be_complete_and_cannot_invent_a_profile_or_method():
    packet = benchmark()["packets"][0]
    answer = intake(packet)
    validate_intake(answer)
    for bad in (
        answer | {"profile": "anything"},
        answer | {"method": None},
        answer | {"required_inputs": "close"},
        answer | {"variant": "novel"},
        {k: v for k, v in answer.items() if k != "deviations"},
        answer | {"observation": "  "},
    ):
        with pytest.raises(ValueError):
            validate_intake(bad)
