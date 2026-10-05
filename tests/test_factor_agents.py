"""Hypothesis extraction and factor proposal, against a scripted fake model.

No network and no API key: `FakeClient` answers each call with the next
scripted tool input and records what it was asked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capstone.factors import store
from capstone.factors.hypotheses import extract
from capstone.factors.llm import FactorConfig, grammar_help, load_config
from capstone.factors.proposer import propose
from capstone.factors.tree import BINARY_PARAMETER_FUNCS, BINARY_WINDOW_FUNCS, FUNCS
from capstone.factors.zoo import ALPHA101_EXPRESSIONS


class FakeClient:
    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        tool = kwargs["tools"][0]["name"]
        block = SimpleNamespace(type="tool_use", name=tool, input=self.answers.pop(0))
        return SimpleNamespace(content=[block])


PAPER = {
    "key": "arxiv:2502.16789",
    "title": "AlphaAgent",
    "abstract": "LLM agents mine alphas with originality and complexity regularization.",
    "summary_en": "Regularized exploration counters alpha decay.",
    "url": "https://arxiv.org/abs/2502.16789",
}

HYPOTHESIS = {
    "observation": "Stocks that fell over a week tend to rebound.",
    "knowledge": "Short-term reversal.",
    "justification": "Liquidity providers are paid to absorb order imbalances.",
    "specification": "Minus the 5-day return.",
    "falsification_condition": "No negative relation between past 5-day and next-day returns.",
    "kind": "market",
}

CONFIG = FactorConfig(model="test-model", factors_per_hypothesis=4, max_repairs=1)


@pytest.fixture
def con(tmp_path):
    con = store.connect(tmp_path / "factors.db")
    store.add_paper(con, PAPER["key"], PAPER["title"])
    yield con
    con.close()


@pytest.fixture
def hid(con):
    return store.add_hypothesis(con, store.Hypothesis(paper_key=PAPER["key"], **HYPOTHESIS))


# ---------------------------------------------------------------------------
# Parameters and prompt


def test_shipped_config_loads_and_typos_raise(tmp_path):
    assert load_config().prompt_version
    bad = tmp_path / "bad.toml"
    bad.write_text('model = "x"\nmax_repair = 3\n')
    with pytest.raises(ValueError, match="max_repair"):
        load_config(bad)


def test_grammar_help_names_every_function_the_parser_accepts():
    text = grammar_help()
    for name in [*FUNCS, *BINARY_WINDOW_FUNCS, *BINARY_PARAMETER_FUNCS]:
        assert name in text, name


# ---------------------------------------------------------------------------
# Paper -> hypotheses


def test_extract_forces_the_tool_and_sends_the_abstract():
    client = FakeClient({"hypotheses": [HYPOTHESIS]})
    (hypothesis,) = extract(PAPER, client, CONFIG)

    call = client.calls[0]
    assert call["tool_choice"] == {"type": "auto"}
    assert [t["name"] for t in call["tools"]] == ["record_hypotheses"]
    assert call["tools"][0]["strict"] is True
    assert "record_hypotheses" in call["system"]
    assert call["model"] == "test-model"
    assert PAPER["abstract"] in call["messages"][0]["content"]
    assert hypothesis.paper_key == PAPER["key"]


def test_extract_drops_incomplete_hypotheses_and_caps_the_count():
    incomplete = {**HYPOTHESIS, "justification": ""}
    many = [incomplete] + [{**HYPOTHESIS, "specification": f"variant {i}"} for i in range(5)]
    out = extract(PAPER, FakeClient({"hypotheses": many}), FactorConfig(hypotheses_per_paper=3))
    assert [h.specification for h in out] == ["variant 0", "variant 1"]


def test_nothing_testable_is_an_empty_answer():
    assert extract(PAPER, FakeClient({"hypotheses": []}), CONFIG) == []


# ---------------------------------------------------------------------------
# Hypothesis -> factors


def _factors(*expressions: str) -> dict:
    return {"factors": [{"expression": e, "rationale": "r"} for e in expressions]}


def test_original_factor_is_stored_and_a_published_alpha_is_rejected(con, hid):
    client = FakeClient(
        _factors(
            "-(close / shift(close, 5) - 1) * ts_rank(volume, 20)",
            ALPHA101_EXPRESSIONS["alpha_101"],
        )
    )
    stored, copied = propose(con, hid, client, CONFIG)

    assert stored.status == "stored"
    assert copied.status == "rejected"
    assert "alpha_101" in copied.reason
    assert len(store.factors(con)) == 1


def test_parse_errors_go_back_to_the_model_and_the_fix_is_stored(con, hid):
    client = FakeClient(
        _factors("ts_mean(close, 2.5) / volume"),
        _factors("ts_mean(close, 3) / ts_mean(volume, 10)"),
    )
    first, fixed = propose(con, hid, client, CONFIG)

    assert first.status == "rejected" and "parse error" in first.reason
    assert fixed.status == "stored"
    assert "ts_mean(close, 2.5) / volume" in client.calls[1]["messages"][0]["content"]
    assert [p["status"] for p in store.proposals(con)] == ["rejected", "stored"]


def test_repairs_stop_after_max_repairs(con, hid):
    client = FakeClient(_factors("unknown(close)"), _factors("still_unknown(close)"))
    outcomes = propose(con, hid, client, CONFIG)
    assert [o.status for o in outcomes] == ["rejected", "rejected"]
    assert len(client.calls) == CONFIG.max_repairs + 1


def test_retuned_variant_of_a_stored_factor_is_rejected_but_a_repeat_is_a_duplicate(con, hid):
    original = "rank(delta(volume, 3)) * -(close / shift(close, 5) - 1)"
    propose(con, hid, FakeClient(_factors(original)), CONFIG)

    retuned = "rank(delta(volume, 7)) * -(close / shift(close, 10) - 1)"
    repeat = "-(close / shift(close, 5) - 1) * rank(delta(volume, 3))"
    variant, again = propose(con, hid, FakeClient(_factors(retuned, repeat)), CONFIG)

    assert variant.status == "rejected" and "stored factor" in variant.reason
    assert again.status == "duplicate"
    assert len(store.factors(con)) == 1


def test_every_stored_row_records_the_model_and_prompt_version(con, hid):
    propose(con, hid, FakeClient(_factors("ts_rank(high - low, 15) / ts_mean(volume, 5)")), CONFIG)
    (row,) = store.proposals(con)
    assert (row["model"], row["prompt_version"]) == ("test-model", CONFIG.prompt_version)


def test_unknown_hypothesis_raises(con):
    with pytest.raises(KeyError):
        propose(con, "nope", FakeClient(), CONFIG)


def _closed(schema: dict) -> bool:
    """Strict tools need every object closed and every property required."""
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False or set(schema["required"]) != set(props):
            return False
        return all(_closed(sub) for sub in props.values())
    if schema.get("type") == "array":
        return _closed(schema["items"])
    return True


def test_tool_schemas_are_valid_for_strict_mode():
    from capstone.factors.hypotheses import _tool
    from capstone.factors.proposer import TOOL

    assert _closed(_tool(3)["input_schema"])
    assert _closed(TOOL["input_schema"])


def test_a_reply_without_the_tool_call_raises():
    from capstone.factors.llm import call_tool

    class TextOnly:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="no")])

    with pytest.raises(ValueError, match="did not call"):
        call_tool(TextOnly(), CONFIG, system="s", user="u", tool={"name": "t", "input_schema": {}})


# ---------------------------------------------------------------------------
# Complexity, kinds and usage


def test_a_formula_over_the_node_limit_is_rejected_as_too_complex(con, hid):
    small = FactorConfig(model="test-model", max_nodes=8, max_repairs=0)
    big = "rank(ts_mean(close, 5) / ts_mean(volume, 5)) * ts_rank(high - low, 10)"
    (outcome,) = propose(con, hid, FakeClient(_factors(big)), small)
    assert outcome.status == "rejected"
    assert "too complex" in outcome.reason and "limit 8" in outcome.reason


def test_the_prompt_states_the_node_limit():
    from capstone.factors.proposer import system_prompt

    assert "at most 17 tree nodes" in system_prompt(FactorConfig(max_nodes=17))


def test_shipped_limit_admits_every_alpha101_member():
    from capstone.factors.tree import node_count
    from capstone.factors.zoo import default_zoo

    assert max(node_count(tree) for _, tree in default_zoo()) <= load_config().max_nodes


def test_extract_keeps_method_hypotheses_and_drops_unknown_kinds():
    method = {**HYPOTHESIS, "kind": "method", "specification": "Simpler factors decay less."}
    unknown = {**HYPOTHESIS, "kind": "vibes", "specification": "other"}
    out = extract(PAPER, FakeClient({"hypotheses": [HYPOTHESIS, method, unknown]}), CONFIG)
    assert [h.kind for h in out] == ["market", "method"]


def test_call_tool_counts_calls_and_tokens():
    from collections import Counter

    from capstone.factors.llm import call_tool

    class Metered(FakeClient):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            response.usage = SimpleNamespace(input_tokens=1200, output_tokens=300)
            return response

    usage = Counter()
    client = Metered({"hypotheses": []}, {"hypotheses": []})
    for _ in range(2):
        call_tool(client, CONFIG, system="s", user="u", tool={"name": "t"}, usage=usage)
    assert usage == {"calls": 2, "input_tokens": 2400, "output_tokens": 600}


# ---------------------------------------------------------------------------
# Model


def test_shipped_config_runs_opus_5_5_at_high_effort():
    config = load_config()
    assert (config.model, config.effort) == ("claude-opus-5-5", "high")
    assert FactorConfig().model == config.model


def test_call_tool_sends_the_configured_effort():
    from capstone.factors.llm import call_tool

    client = FakeClient({"hypotheses": []})
    call_tool(client, CONFIG, system="s", user="u", tool={"name": "t"})
    assert client.calls[0]["output_config"] == {"effort": CONFIG.effort}


def test_unknown_effort_is_rejected():
    with pytest.raises(ValueError, match="effort"):
        FactorConfig(effort="extreme")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"stop_reason": "refusal", "stop_details": SimpleNamespace(category="cyber")}, "declined"),
        ({"model": "some-other-model"}, "not the configured"),
    ],
)
def test_a_refusal_or_another_model_stops_the_run(extra, message):
    # Rows are stored under config.model and the holdout depends on its
    # cutoff, so neither may be stored as if the configured model answered.
    from capstone.factors.llm import call_tool

    class Answers(FakeClient):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            for key, value in extra.items():
                setattr(response, key, value)
            return response

    with pytest.raises(ValueError, match=message):
        call_tool(Answers({"hypotheses": []}), CONFIG, system="s", user="u", tool={"name": "t"})
