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


def test_grammar_help_explains_the_market_field():
    help_text = grammar_help()
    assert "mkt_return" in help_text and "value-weighted market return" in help_text
    assert "per-stock field" in help_text


# ---------------------------------------------------------------------------
# Hypothesis-formula alignment (QUANTIT-83)


def _verdict(variables=True, direction=True, horizon=True, reason="ok"):
    return {"variables": variables, "direction": direction, "horizon": horizon, "reason": reason}


def test_alignment_score_is_the_share_of_checks_passed_and_uses_the_judge_model():
    from capstone.factors.alignment import judge

    config = FactorConfig(model="proposer-model", alignment_model="judge-model")
    hypothesis = {**HYPOTHESIS, "id": "h1"}
    client = FakeClient(_verdict(), _verdict(direction=False, reason="sign flipped"))
    assert judge(client, config, hypothesis, "-ts_sum(returns, 5)").score == 1.0
    half = judge(client, config, hypothesis, "ts_sum(returns, 5)")
    assert half.score == pytest.approx(2 / 3)
    assert half.reason == "fails direction: sign flipped"
    assert {call["model"] for call in client.calls} == {"judge-model"}
    assert "Minus the 5-day return." in client.calls[0]["messages"][0]["content"]


def test_a_misaligned_formula_is_rejected_and_its_score_kept(con, hid):
    strict = FactorConfig(model="m", alignment_model="j", min_alignment=1.0, max_repairs=0)
    answers = _factors("-ts_sum(returns, 5)", "ts_mean(volume / shares, 21) / log(cap)")
    client = FakeClient(answers, _verdict(), _verdict(direction=False, reason="sign"))
    kept, dropped = propose(con, hid, client, strict)
    assert kept.status == "stored" and dropped.status == "rejected"
    assert dropped.reason.startswith("misaligned: 0.67 < 1.0; fails direction")
    rows = con.execute("SELECT status, alignment, alignment_model FROM proposals").fetchall()
    assert {(r["status"], round(r["alignment"], 2), r["alignment_model"]) for r in rows} == {
        ("stored", 1.0, "j"),
        ("rejected", 0.67, "j"),
    }


def test_without_a_judge_model_no_alignment_call_is_made(con, hid):
    client = FakeClient(_factors("-ts_sum(returns, 5)"))
    (outcome,) = propose(con, hid, client, CONFIG)
    assert outcome.status == "stored" and len(client.calls) == 1


def test_shipped_config_records_alignment_but_rejects_nothing_until_calibrated():
    config = load_config()
    assert config.alignment_model == "claude-haiku-4-5"
    assert config.min_alignment == 0.0


def test_a_store_from_before_alignment_gains_the_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    lines = store.SCHEMA.splitlines()
    old.executescript("\n".join(line for line in lines if "alignment" not in line))
    assert "alignment" not in {row[1] for row in old.execute("PRAGMA table_info(proposals)")}
    old.commit()
    old.close()
    con = store.connect(path)
    columns = {row[1] for row in con.execute("PRAGMA table_info(proposals)")}
    assert {"alignment", "alignment_reason", "alignment_model"} <= columns


def test_judge_tokens_are_counted_apart_from_the_proposer():
    from collections import Counter

    from capstone.factors.alignment import judge

    class Metered(FakeClient):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            response.usage = SimpleNamespace(input_tokens=500, output_tokens=40)
            return response

    usage = Counter()
    config = FactorConfig(alignment_model="judge-model")
    judge(Metered(_verdict()), config, {**HYPOTHESIS, "id": "h"}, "-returns", usage=usage)
    expected = {"alignment_calls": 1, "alignment_input_tokens": 500, "alignment_output_tokens": 40}
    assert usage == expected
