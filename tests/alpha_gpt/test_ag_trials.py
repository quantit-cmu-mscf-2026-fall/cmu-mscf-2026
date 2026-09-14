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
    evaluate_holdout,
    sample_weights,
)

CONTEXT = LedgerContext(
    ledger_name="alpha_gpt_test", run_id="run-1", seed=0, tags=("alpha_gpt", "baseline")
)


def _node(panel, expression):
    return parse(expression, fields=panel.fields, groups=panel.groups)


@pytest.fixture
def setup(reversal_panel):
    splits = SplitSpec.from_fractions(reversal_panel.dates)
    search = splits.search_panel(reversal_panel)
    train = splits.dates(search.dates, "development")
    evaluator = TrialEvaluator(
        search, train, CONTEXT, test_start=splits.test[0], scope={"stage": "final"}
    )
    return reversal_panel, splits, search, train, evaluator


def test_each_evaluation_is_one_ledger_entry(setup, ledger):
    _, _, search, train, evaluator = setup
    record = evaluator.evaluate(
        _node(search, "neg(returns)"), trial_id="r1a1", round=1, raw_expression="neg(returns)"
    )

    (entry,) = ledger()
    assert isinstance(record, TrialRecord)
    assert entry["name"] == "alpha_gpt_test"
    assert entry["seed"] == 0
    assert entry["tags"] == ["alpha_gpt", "baseline", "search"]
    params = entry["params"]
    assert params["expression"] == "neg(returns)"
    assert params["run_id"] == "run-1"
    assert params["stage"] == "final"
    assert params["panel"]["pattern"] == "reversal"
    assert params["train_dates"][2] == len(train)
    assert params["horizon"] == 1
    assert entry["metrics"]["train_ic_tstat"] == pytest.approx(record.train.ic_tstat)
    assert not any(key.startswith(("valid_", "test_")) for key in entry["metrics"])


def test_record_metrics_match_a_direct_weighted_computation(setup, ledger):
    _, _, search, train, evaluator = setup
    record = evaluator.evaluate(
        _node(search, "neg(returns)"), trial_id="r1a1", round=1, raw_expression="neg(returns)"
    )
    returns = search.fields["returns"]
    expected = split_metrics(
        -returns, returns, train, split="train", weights=sample_weights(search.dates, 1)
    )
    # approx: the DSL path can sum in a different order (differences ~1e-16).
    assert record.train.as_flat("t") == pytest.approx(expected.as_flat("t"))
    assert record.train.ic_tstat > 3


def test_nothing_is_returned_unless_it_was_logged(setup, monkeypatch):
    _, _, search, _, evaluator = setup

    def broken_ledger(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(trials, "log_run", broken_ledger)
    with pytest.raises(OSError):
        evaluator.evaluate(
            _node(search, "neg(returns)"), trial_id="r1a1", round=1, raw_expression=""
        )
    assert evaluator.n_trials == 0


def test_duplicates_are_not_logged_again(setup, ledger):
    _, _, search, _, evaluator = setup
    first = evaluator.evaluate(
        _node(search, "add(close, open)"), trial_id="r1a1", round=1, raw_expression="a"
    )
    again = evaluator.evaluate(
        _node(search, "add(open, close)"), trial_id="r2a1", round=2, raw_expression="b"
    )

    assert isinstance(first, TrialRecord)
    assert again == Duplicate(expression="add(close, open)", raw_expression="b", of="r1a1")
    assert evaluator.n_trials == 1
    assert len(ledger()) == 1
    assert evaluator.seen(_node(search, "add(open, close)")) == "r1a1"


def test_trial_budget(setup, ledger):
    _, splits, search, train, _ = setup
    evaluator = TrialEvaluator(search, train, CONTEXT, test_start=splits.test[0], max_trials=2)
    evaluator.evaluate(_node(search, "neg(returns)"), trial_id="a", round=1, raw_expression="")
    evaluator.evaluate(_node(search, "returns"), trial_id="b", round=1, raw_expression="")
    with pytest.raises(TrialBudgetExhausted):
        evaluator.evaluate(_node(search, "neg(close)"), trial_id="c", round=1, raw_expression="")
    assert len(ledger()) == 2


def test_refuses_a_panel_that_still_contains_test_dates(setup):
    panel, splits, _, train, _ = setup
    with pytest.raises(ValueError, match="TEST dates"):
        TrialEvaluator(panel, train, CONTEXT, test_start=splits.test[0])


def test_records_carry_train_metrics_only():
    names = {field.name for field in dataclasses.fields(TrialRecord)}
    assert "train" in names
    assert not names & {"valid", "test", "held_out"}


def test_holdout_evaluations_are_logged_before_returning(setup, ledger):
    panel, splits, search, train, _ = setup
    node = _node(search, "neg(returns)")
    fold = evaluate_holdout(
        node,
        search,
        train[-100:],
        CONTEXT,
        phase="cv_valid",
        trial_id="r1a1",
        scope={"stage": "cv", "fold": 2},
    )
    test = evaluate_holdout(
        node, panel, splits.dates(panel.dates, "test"), CONTEXT, phase="test", trial_id="r1a1"
    )

    fold_entry, test_entry = ledger()
    assert fold_entry["tags"][-1] == "cv_valid" and test_entry["tags"][-1] == "test"
    assert fold_entry["params"]["fold"] == 2
    assert fold_entry["params"]["held_out_dates"][2] == 100
    assert fold_entry["metrics"]["valid_ic_tstat"] == pytest.approx(fold.ic_tstat)
    assert test_entry["metrics"]["test_ic_tstat"] == pytest.approx(test.ic_tstat)
    assert fold.split == "valid" and test.split == "test"
    assert test.ic_tstat > 3


def test_unknown_holdout_phase_is_rejected(setup):
    _, _, search, train, _ = setup
    with pytest.raises(ValueError, match="phase"):
        evaluate_holdout(
            _node(search, "returns"), search, train, CONTEXT, phase="search", trial_id="x"
        )


def test_sample_weights_are_unit_at_horizon_one(reversal_panel):
    weights = sample_weights(reversal_panel.dates, 1)
    assert len(weights) == len(reversal_panel.dates) - 1
    assert (weights == 1.0).all()
