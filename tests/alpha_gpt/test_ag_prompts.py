"""Tests for capstone.alpha_gpt.prompts."""

from __future__ import annotations

import inspect
import re

import pytest

from capstone.alpha_gpt import prompts
from capstone.alpha_gpt.dsl import parse
from capstone.alpha_gpt.operators import OPERATORS
from capstone.alpha_gpt.prompts import (
    PriorRound,
    noise_tstat,
    render_analyst,
    render_quant_developer,
)
from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.trials import LedgerContext, TrialEvaluator

FIELDS = ["open", "high", "low", "close", "volume", "vwap", "returns"]
PLACEHOLDER = re.compile(r"\$[a-z_]+")


def _qd(**kwargs):
    kwargs.setdefault("idea", "stocks that fell sharply yesterday bounce today")
    kwargs.setdefault("fields", FIELDS)
    kwargs.setdefault("groups", ["sector"])
    kwargs.setdefault("n_alphas", 5)
    return render_quant_developer(**kwargs)


class TestQuantDeveloper:
    def test_first_round_prompt(self):
        prompt = _qd()
        assert not PLACEHOLDER.search(prompt)
        assert "stocks that fell sharply yesterday bounce today" in prompt
        assert "Propose exactly 5 distinct expressions" in prompt
        assert "integer literal in [2, 126]" in prompt
        assert "Where this idea came from" not in prompt
        for spec in OPERATORS.values():
            assert spec.signature in prompt

    def test_group_operators_hidden_without_groups(self):
        prompt = _qd(groups=[])
        assert "grouped_demean" not in prompt
        assert "none (grouped_* operators are unavailable)" in prompt

    def test_later_round_carries_feedback_and_evaluated_list(self):
        prior = PriorRound(
            round=1,
            analyst_summary="Momentum framing had the wrong sign.",
            evaluated=["neg(returns)", "ts_mean(returns, 5)"],
        )
        prompt = _qd(idea="short-term reversal", prior=prior)
        assert "after round 1" in prompt
        assert "Momentum framing had the wrong sign." in prompt
        assert "- ts_mean(returns, 5)" in prompt
        assert not PLACEHOLDER.search(prompt)

    def test_format_example_expressions_are_valid(self):
        prompt = _qd()
        examples = re.findall(r'"expression": "([^".]+)"', prompt)
        assert len(examples) == 2
        for expression in examples:
            parse(expression, fields=FIELDS, groups=["sector"])


class TestAnalyst:
    @pytest.fixture
    def records(self, reversal_panel, ledger):
        splits = SplitSpec.from_fractions(reversal_panel.dates)
        evaluator = TrialEvaluator(
            reversal_panel, splits, LedgerContext(ledger_name="t", run_id="r", seed=0)
        )
        for i, expression in enumerate(["neg(returns)", "ts_mean(volume, 10)"], start=1):
            node = parse(expression, fields=reversal_panel.fields, groups=reversal_panel.groups)
            evaluator.evaluate(node, trial_id=f"r1a{i}", round=1, raw_expression=expression)
        return splits, evaluator.records

    def test_prompt_has_table_and_protocol_but_no_test_period(self, records):
        splits, rows = records
        as_dict = splits.as_dict()
        prompt = render_analyst(
            idea="reversal",
            records=rows,
            train_range=" to ".join(as_dict["train"]),
            valid_range=" to ".join(as_dict["valid"]),
            cost_bps=0.0,
        )
        assert not PLACEHOLDER.search(prompt)
        assert "| r1a1 | 1 | `neg(returns)` |" in prompt
        assert "`ts_mean(volume, 10)`" in prompt
        assert "2 distinct alphas" in prompt
        assert f"t near {noise_tstat(2):.2f}" in prompt
        for date in as_dict["test"]:
            assert date not in prompt

    def test_renderers_cannot_be_handed_test_data(self):
        for renderer in (render_analyst, render_quant_developer):
            params = set(inspect.signature(renderer).parameters)
            assert not params & {"panel", "splits", "test", "test_range"}

    def test_missing_values_raise_instead_of_sending_a_partial_prompt(self):
        with pytest.raises(KeyError):
            prompts._render("analyst", idea="x")


def test_noise_tstat_grows_with_trials():
    assert noise_tstat(1) == 0.0
    assert 1.0 < noise_tstat(10) < noise_tstat(100) < 3.0
