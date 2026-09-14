"""Evaluating an alpha is a trial, and every trial is in the ledger before anyone sees it.

`TrialEvaluator` is the only place the search phase turns an expression into
numbers. It computes TRAIN and VALIDATION metrics and appends the ledger entry
(`capstone.runlog.log_run`) before returning, so no caller — the loop, the
Analyst, a human — can see a result that was not counted.

It is built on the search panel (TEST removed), and `TrialRecord` has no TEST
field, so test-period performance cannot leak into selection or into a prompt.
TEST is evaluated separately and once, by `evaluate_on_test`, which also logs
before returning.

Re-proposing an expression already evaluated in this run (same canonical form,
same data and splits) is not a new hypothesis test: the numbers would be
identical. It returns a `Duplicate` and writes nothing to the ledger.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from capstone.alpha_gpt.dsl import Node, canonical, evaluate
from capstone.alpha_gpt.metrics import SplitMetrics, split_metrics
from capstone.alpha_gpt.panel import RETURNS_FIELD, OHLCVPanel
from capstone.alpha_gpt.splits import SplitSpec
from capstone.runlog import log_run


class TrialBudgetExhausted(RuntimeError):
    """The run has evaluated `max_trials` distinct alphas."""


@dataclass(frozen=True)
class TrialRecord:
    """One evaluated alpha. Deliberately has no TEST metrics."""

    trial_id: str
    round: int
    expression: str
    raw_expression: str
    rationale: str
    train: SplitMetrics
    valid: SplitMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "round": self.round,
            "expression": self.expression,
            "raw_expression": self.raw_expression,
            "rationale": self.rationale,
            **self.train.as_flat("train"),
            **self.valid.as_flat("valid"),
        }


@dataclass(frozen=True)
class Duplicate:
    """An expression this run already evaluated, as trial `of`."""

    expression: str
    raw_expression: str
    of: str


@dataclass(frozen=True)
class LedgerContext:
    """What every ledger entry of one run shares."""

    ledger_name: str
    run_id: str
    seed: int | None
    tags: Sequence[str] = ("alpha_gpt",)
    extra_params: Mapping[str, Any] | None = None


class TrialEvaluator:
    """Evaluate alphas on TRAIN and VALIDATION, logging each distinct one as a trial."""

    def __init__(
        self,
        panel: OHLCVPanel,
        splits: SplitSpec,
        context: LedgerContext,
        *,
        cost_bps: float = 0.0,
        min_assets: int = 20,
        max_trials: int | None = None,
    ) -> None:
        self.panel = splits.search_panel(panel)
        self.splits = splits
        self.context = context
        self.cost_bps = cost_bps
        self.min_assets = min_assets
        self.max_trials = max_trials
        self.records: list[TrialRecord] = []
        self._seen: dict[str, str] = {}
        self._train_dates = splits.dates(self.panel.dates, "train")
        self._valid_dates = splits.dates(self.panel.dates, "valid")

    @property
    def n_trials(self) -> int:
        return len(self.records)

    def seen(self, node: Node) -> str | None:
        """trial_id of an earlier evaluation of the same canonical alpha, if any."""
        return self._seen.get(canonical(node))

    def evaluate(
        self,
        node: Node,
        *,
        trial_id: str,
        round: int,
        raw_expression: str,
        rationale: str = "",
    ) -> TrialRecord | Duplicate:
        expression = canonical(node)
        if expression in self._seen:
            return Duplicate(expression, raw_expression, of=self._seen[expression])
        if self.max_trials is not None and self.n_trials >= self.max_trials:
            raise TrialBudgetExhausted(f"max_trials={self.max_trials} reached")

        signal = evaluate(node, self.panel)
        returns = self.panel.fields[RETURNS_FIELD]
        train = split_metrics(
            signal, returns, self._train_dates, split="train", **self._metric_kwargs()
        )
        valid = split_metrics(
            signal, returns, self._valid_dates, split="valid", **self._metric_kwargs()
        )

        # Logged before the record exists anywhere a caller could read it.
        _log(
            self.context,
            phase="search",
            params={
                "round": round,
                "trial_id": trial_id,
                "expression": expression,
                "panel": dict(self.panel.descriptor),
                "splits": self.splits.as_dict(),
                "cost_bps": self.cost_bps,
                "min_assets": self.min_assets,
            },
            metrics={**train.as_flat("train"), **valid.as_flat("valid")},
        )

        record = TrialRecord(trial_id, round, expression, raw_expression, rationale, train, valid)
        self._seen[expression] = trial_id
        self.records.append(record)
        return record

    def _metric_kwargs(self) -> dict[str, Any]:
        return {"cost_bps": self.cost_bps, "min_assets": self.min_assets}


def evaluate_on_test(
    node: Node,
    panel: OHLCVPanel,
    splits: SplitSpec,
    context: LedgerContext,
    *,
    trial_id: str,
    cost_bps: float = 0.0,
    min_assets: int = 20,
) -> SplitMetrics:
    """TEST metrics for one selected alpha on the full panel, logged before returning."""
    signal = evaluate(node, panel)
    metrics = split_metrics(
        signal,
        panel.fields[RETURNS_FIELD],
        splits.dates(panel.dates, "test"),
        split="test",
        cost_bps=cost_bps,
        min_assets=min_assets,
    )
    _log(
        context,
        phase="test",
        params={
            "trial_id": trial_id,
            "expression": canonical(node),
            "panel": dict(panel.descriptor),
            "splits": splits.as_dict(),
            "cost_bps": cost_bps,
            "min_assets": min_assets,
        },
        metrics=metrics.as_flat("test"),
    )
    return metrics


def _log(
    context: LedgerContext, *, phase: str, params: dict[str, Any], metrics: dict[str, Any]
) -> None:
    log_run(
        context.ledger_name,
        params={
            "run_id": context.run_id,
            "phase": phase,
            **dict(context.extra_params or {}),
            **params,
        },
        metrics=metrics,
        seed=context.seed,
        tags=[*context.tags, phase],
    )
