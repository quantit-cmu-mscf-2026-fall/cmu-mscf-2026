import json
from types import SimpleNamespace

import pytest

from capstone.factors.llm import FactorConfig
from capstone.factors.research_review import FLAGS, review_packet, summarize

CONFIG = FactorConfig(model="generator", alignment_model="reviewer")
ARMS = ("A", "B", "C")


class FakeClient:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(
            type="tool_use", name=kwargs["tools"][0]["name"], input=self.answers.pop(0)
        )
        return SimpleNamespace(content=[block])


def packet(i=0):
    return {
        "id": f"p{i}",
        "family": f"family-{i % 4}",
        "source_order": i,
        "title": "Synthetic paper",
        "source_url": "https://example.org/paper",
        "methods": "m",
        "method_excerpt": "x",
        "method_location": "s2",
        "full_methods": True,
        "required_inputs": ["returns"],
        "universe": "sp500",
        "expected": {"status": "ok"},
    }


def judgments_for(client_call, **override):
    context = json.loads(client_call["messages"][0]["content"])
    rows = []
    for item in context["artifacts"]:
        row = {flag: False for flag in FLAGS} | {"blind_id": item["blind_id"], "reason": "r"}
        rows.append(row | override)
    return rows


class Answering(FakeClient):
    """Answers each call from the blind ids actually shown to the reviewer."""

    def __init__(self, **override):
        super().__init__()
        self.override = override

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(
            type="tool_use",
            name="review_methods",
            input={"judgments": judgments_for(kwargs, **self.override)},
        )
        return SimpleNamespace(content=[block])


def test_reviewer_sees_blind_ids_never_arm_labels_and_judgments_map_back():
    outputs = {arm: {"note": f"artifact-{arm}"} for arm in ARMS}
    client = Answering(grounded=True)
    result = review_packet(packet(), outputs, client, CONFIG, pass_index=0)
    assert set(result) == set(ARMS)
    assert all(row["grounded"] for row in result.values())
    call = client.calls[0]
    assert call["model"] == "reviewer"
    shown = json.loads(call["messages"][0]["content"])
    assert {item["blind_id"] for item in shown["artifacts"]}.isdisjoint(ARMS)
    assert "arm" not in call["messages"][0]["content"].lower().replace("artifact", "")
    assert shown["qualified_inputs"] == []


def test_presentation_order_rotates_with_the_pass():
    outputs = {arm: {"note": arm} for arm in ARMS}
    orders = []
    for index in (0, 1):
        client = Answering()
        review_packet(packet(), outputs, client, CONFIG, pass_index=index)
        shown = json.loads(client.calls[0]["messages"][0]["content"])
        orders.append([item["blind_id"] for item in shown["artifacts"]])
    assert sorted(orders[0]) == sorted(orders[1]) and orders[0] != orders[1]


def test_incomplete_or_non_boolean_review_is_repaired_once_then_refused():
    outputs = {arm: {"note": arm} for arm in ARMS}
    bad = {"judgments": [{"blind_id": "nope"}]}
    client = Answering()
    answer = client.create

    def create(**kwargs):
        if not client.calls:
            client.calls.append(kwargs)
            block = SimpleNamespace(type="tool_use", name="review_methods", input=bad)
            return SimpleNamespace(content=[block])
        return answer(**kwargs)

    client.create = create
    repaired = review_packet(packet(), outputs, client, CONFIG, pass_index=0)
    assert set(repaired) == set(ARMS)
    assert "schema_repair" in client.calls[1]["messages"][0]["content"]
    with pytest.raises(ValueError, match="blinded artifact"):
        review_packet(packet(), outputs, FakeClient(bad, bad), CONFIG, pass_index=0)
    with pytest.raises(ValueError, match="booleans"):
        review_packet(packet(), outputs, Answering(grounded="yes"), CONFIG, pass_index=0)


def flags(**on):
    return {flag: on.get(flag, False) for flag in FLAGS} | {"reason": "r"}


def study(arm_b_better):
    packets = [packet(i) for i in range(8)]
    results = {arm: {} for arm in ARMS}
    reviews = {}
    for p in packets:
        for arm in ARMS:
            results[arm][p["id"]] = {
                "status": "ok",
                "hypotheses": ["h"],
                "proposals": [{"expression": "rank(close)"}],
            }
        good = flags(grounded=True, complete_method=True)
        weak = flags()
        by_arm = {"A": weak if arm_b_better else good, "B": good, "C": weak}
        if arm_b_better and int(p["id"][1:]) >= 4:
            by_arm["A"] = good
        reviews[p["id"]] = [by_arm, by_arm]
    accounting = {
        "requests": [
            {"bucket": arm, "usage": {"input_tokens": 3, "output_tokens": 2}} for arm in ARMS
        ]
        * 4
    }
    return {"packets": packets}, results, reviews, accounting


def test_summary_applies_registered_criteria_and_never_admits():
    benchmark, results, reviews, accounting = study(arm_b_better=True)
    outcome = summarize(benchmark, results, reviews, accounting, integrity_passed=True)
    assert outcome["complete_matched_study"] and outcome["formal_admission"] is False
    assert outcome["controller_default"] is False and outcome["new_financial_trials"] == 0
    assert outcome["metrics"]["B"]["grounded_families"] == [f"family-{i}" for i in range(4)]
    assert outcome["metrics"]["B"]["executable_strategies"] == 0
    assert outcome["metrics"]["A"]["repeated_structures"] == 7
    assert outcome["decision"] == "does_not_support_larger_study_under_registered_criteria"
    assert not outcome["criteria"]["additional_grounded_family"]


def test_incomplete_study_or_failed_integrity_is_not_a_decision():
    benchmark, results, reviews, accounting = study(arm_b_better=False)
    del results["C"]["p7"]
    outcome = summarize(benchmark, results, reviews, accounting, integrity_passed=True)
    assert outcome["decision"] == "inconclusive" and not outcome["complete_matched_study"]
    benchmark, results, reviews, accounting = study(arm_b_better=False)
    failed = summarize(benchmark, results, reviews, accounting, integrity_passed=False)
    assert failed["decision"] == "does_not_support_larger_study_under_registered_criteria"
    assert failed["criteria"]["integrity"] is False


def test_a_disagreement_that_changes_a_criterion_makes_the_pilot_inconclusive():
    benchmark, results, reviews, accounting = study(arm_b_better=False)
    first, second = reviews["p0"]
    second = {**second, "B": flags(unsupported_claim=True)}
    reviews["p0"] = [first, second]
    outcome = summarize(benchmark, results, reviews, accounting, integrity_passed=True)
    assert outcome["review_disagreements"]
    assert outcome["decisive_review_disagreement"] and outcome["decision"] == "inconclusive"
