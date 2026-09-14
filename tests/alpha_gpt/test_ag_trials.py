"""Tests for capstone.alpha_gpt.trials."""

from __future__ import annotations

import dataclasses

import pytest

from capstone.alpha_gpt import trials
from capstone.alpha_gpt.dsl import parse
from capstone.alpha_gpt.metrics import split_metrics
from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.trials import (
    Duplicate,
    LedgerContext,
    TrialBudgetExhausted,
    TrialEvaluator,
    TrialRecord,
    evaluate_on_test,
)

CONTEXT = LedgerContext(
    ledger_name="alpha_gpt_test", run_id="run-1", seed=0, tags=("alpha_gpt", "baseline")
)


def _node(panel, expression):
    return parse(expression, fields=panel.fields, groups=panel.groups)


@pytest.fixture
def setup(reversal_panel):
    splits = SplitSpec.from_fractions(reversal_panel.dates)
    return reversal_panel, splits, TrialEvaluator(reversal_panel, splits, CONTEXT)


def test_each_evaluation_is_one_ledger_entry(setup, ledger):
    panel, _, evaluator = setup
    record = evaluator.evaluate(
        _node(panel, "neg(returns)"), trial_id="r1a1", round=1, raw_expression="neg(returns)"
    )

    (entry,) = ledger()
    assert isinstance(record, TrialRecord)
    assert entry["name"] == "alpha_gpt_test"
    assert entry["seed"] == 0
    assert entry["tags"] == ["alpha_gpt", "baseline", "search"]
    assert entry["params"]["expression"] == "neg(returns)"
    assert entry["params"]["run_id"] == "run-1"
    assert entry["params"]["panel"]["pattern"] == "reversal"
    assert entry["metrics"]["valid_ic_tstat"] == pytest.approx(record.valid.ic_tstat)
    assert not any(key.startswith("test_") for key in entry["metrics"])


def test_record_metrics_match_a_direct_computation(setup, ledger):
    panel, splits, evaluator = setup
    record = evaluator.evaluate(
        _node(panel, "neg(returns)"), trial_id="r1a1", round=1, raw_expression="neg(returns)"
    )
    search = splits.search_panel(panel)
    returns = search.fields["returns"]
    expected = split_metrics(-returns, returns, splits.dates(search.dates, "valid"), split="valid")
    # approx: the DSL path can sum in a different order (differences ~1e-16).
    assert record.valid.as_flat("v") == pytest.approx(expected.as_flat("v"))
    assert record.valid.ic_tstat > 3


def test_nothing_is_returned_unless_it_was_logged(setup, monkeypatch):
    panel, _, evaluator = setup

    def broken_ledger(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(trials, "log_run", broken_ledger)
    with pytest.raises(OSError):
        evaluator.evaluate(
            _node(panel, "neg(returns)"), trial_id="r1a1", round=1, raw_expression=""
        )
    assert evaluator.n_trials == 0


def test_duplicates_are_not_logged_again(setup, ledger):
    panel, _, evaluator = setup
    first = evaluator.evaluate(
        _node(panel, "add(close, open)"), trial_id="r1a1", round=1, raw_expression="a"
    )
    again = evaluator.evaluate(
        _node(panel, "add(open, close)"), trial_id="r2a1", round=2, raw_expression="b"
    )

    assert isinstance(first, TrialRecord)
    assert again == Duplicate(expression="add(close, open)", raw_expression="b", of="r1a1")
    assert evaluator.n_trials == 1
    assert len(ledger()) == 1
    assert evaluator.seen(_node(panel, "add(open, close)")) == "r1a1"


def test_trial_budget(setup, ledger):
    panel, splits, _ = setup
    evaluator = TrialEvaluator(panel, splits, CONTEXT, max_trials=2)
    evaluator.evaluate(_node(panel, "neg(returns)"), trial_id="a", round=1, raw_expression="")
    evaluator.evaluate(_node(panel, "returns"), trial_id="b", round=1, raw_expression="")
    with pytest.raises(TrialBudgetExhausted):
        evaluator.evaluate(_node(panel, "neg(close)"), trial_id="c", round=1, raw_expression="")
    assert len(ledger()) == 2


def test_search_never_holds_test_data(setup):
    _, splits, evaluator = setup
    assert evaluator.panel.dates.max() < splits.test[0]
    assert "test" not in {field.name for field in dataclasses.fields(TrialRecord)}


def test_evaluate_on_test_logs_before_returning(setup, ledger):
    panel, splits, _ = setup
    metrics = evaluate_on_test(
        _node(panel, "neg(returns)"), panel, splits, CONTEXT, trial_id="r1a1"
    )

    (entry,) = ledger()
    assert metrics.split == "test"
    assert entry["tags"] == ["alpha_gpt", "baseline", "test"]
    assert entry["metrics"]["test_ic_tstat"] == pytest.approx(metrics.ic_tstat)
    assert metrics.ic_tstat > 3
