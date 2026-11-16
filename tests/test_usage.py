"""Prompt caching and the price table (`capstone.pipeline.usage`, `llm.make_client`)."""

from __future__ import annotations

import sys
import types

import pytest

from capstone.factors import llm
from capstone.pipeline import usage
from capstone.pipeline.budget import BudgetedClient


class _Capture:
    def __init__(self, max_retries=0):
        self.max_retries = max_retries
        self.kwargs = []
        self.messages = self

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return "ok"


def test_prompt_caching_marks_the_system_text_without_changing_it():
    inner = _Capture()
    client = usage.PromptCaching(inner)
    system = "You write factors.\n\nAnswer only by calling the record_factors tool, exactly once."
    messages = [{"role": "user", "content": "Hypothesis ..."}]
    client.messages.create(model="m", system=system, messages=messages, tools=[{"name": "t"}])
    sent = inner.kwargs[0]
    assert sent["system"] == [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    assert sent["messages"] == messages and sent["tools"] == [{"name": "t"}]
    blocks = [{"type": "text", "text": "x"}]
    client.messages.create(model="m", system=blocks, messages=messages)
    assert inner.kwargs[1]["system"] is blocks


def test_make_client_wraps_for_prompt_caching_only_when_asked(monkeypatch):
    fake = types.SimpleNamespace(Anthropic=_Capture)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    plain = llm.make_client()
    assert isinstance(plain, _Capture) and plain.max_retries == 0
    cached = llm.make_client(prompt_caching=True)
    assert isinstance(cached, usage.PromptCaching) and cached.client.max_retries == 0
    BudgetedClient(cached, 1)  # the shared budget accepts the wrapped client


@pytest.mark.parametrize("wrapper", [usage.PromptCaching, usage.UsageRecorder])
def test_wrappers_do_not_hide_sdk_retries_from_the_budget(wrapper):
    with pytest.raises(ValueError, match="hidden SDK retries"):
        BudgetedClient(wrapper(_Capture(max_retries=2)), 1)
    BudgetedClient(wrapper(_Capture()), 1)


def test_the_models_are_priced():
    assert usage.PRICES["claude-opus-5-5"] == (4.0, 20.0)
    assert usage.PRICES["claude-haiku-5-5"] == (0.10, 0.50)


def test_cost_estimate_prices_tokens_and_names_unpriced_models():
    records = [
        {"model": "claude-opus-5", "input_tokens": 1_000_000, "output_tokens": 100_000},
        {"model": "mystery", "input_tokens": 5, "output_tokens": 5},
    ]
    out = usage.cost_estimate(records)
    assert out["usd"] == 7.5
    assert out["unpriced"] == ["mystery"]
    assert out["by_model"]["claude-opus-5"]["calls"] == 1
