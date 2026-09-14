"""Evaluating an alpha is a trial, and every trial is in the ledger before anyone sees it.

`TrialEvaluator` is the only place a discovery procedure turns an expression into
numbers. It scores the procedure's TRAIN dates and appends the ledger entry
(`capstone.runlog.log_run`) before returning, so no caller — the loop, the
Analyst, a human — can see a result that was not counted.

A procedure only ever holds a panel view: TEST removed (`SplitSpec.search_panel`)
and, inside cross-validation, its held-out fold masked (`OHLCVPanel.mask`). The
evaluator refuses a panel that still contains TEST dates, and `TrialRecord` has
TRAIN metrics only. Held-out scores — cross-validation folds and TEST — come from
`evaluate_holdout`, which also logs before returning and which the procedure
itself never calls.

Every score uses the same `Scoring` settings and the same label-uniqueness
sample weights (`capstone.cv.average_uniqueness`), in TRAIN and held out alike.

Re-proposing an expression this procedure already evaluated (same canonical form,
same data and dates) is not a new hypothesis test: the numbers would be
identical. It returns a `Duplicate` and writes nothing to the ledger.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from capstone.alpha_gpt.dsl import Node, canonical, evaluate
from capstone.alpha_gpt.metrics import SplitMetrics, split_metrics
from capstone.alpha_gpt.panel import RETURNS_FIELD, OHLCVPanel
from capstone.cv import average_uniqueness, label_end_times
from capstone.runlog import log_run

#: Ledger phase -> the prefix its metrics are logged under.
HOLDOUT_PHASES = {"cv_valid": "valid", "test": "test"}


class TrialBudgetExhausted(RuntimeError):
    """The procedure has evaluated `max_trials` distinct alphas."""


@dataclass(frozen=True)
class Scoring:
    """How every alpha in a run is scored — identically on TRAIN, VALIDATION and TEST."""

    horizon: int = 1
    cost_bps: float = 0.0
    min_assets: int = 20

    def as_dict(self) -> dict[str, Any]:
        return {"horizon": self.horizon, "cost_bps": self.cost_bps, "min_assets": self.min_assets}


DEFAULT_SCORING = Scoring()


def sample_weights(dates: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """Label-uniqueness weights (AFML ch. 4) for every date with a complete label."""
    return average_uniqueness(label_end_times(dates, horizon), dates)


def date_span(dates: pd.DatetimeIndex) -> list[Any]:
    """[first ISO date, last ISO date, count] — how date sets are recorded."""
    if len(dates) == 0:
        return [None, None, 0]
    return [dates[0].date().isoformat(), dates[-1].date().isoformat(), len(dates)]


@dataclass(frozen=True)
class TrialRecord:
    """One evaluated alpha, with TRAIN metrics only."""

    trial_id: str
    round: int
    expression: str
    raw_expression: str
    rationale: str
    train: SplitMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "round": self.round,
            "expression": self.expression,
            "raw_expression": self.raw_expression,
            "rationale": self.rationale,
            **self.train.as_flat("train"),
        }


@dataclass(frozen=True)
class Duplicate:
    """An expression this procedure already evaluated, as trial `of`."""

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
    """Score alphas on a procedure's TRAIN dates, logging each distinct one as a trial."""

    def __init__(
        self,
        panel: OHLCVPanel,
        train_dates: pd.DatetimeIndex,
        context: LedgerContext,
        *,
        test_start: pd.Timestamp,
        scoring: Scoring = DEFAULT_SCORING,
        weights: pd.Series | None = None,
        scope: Mapping[str, Any] | None = None,
        max_trials: int | None = None,
    ) -> None:
        if (panel.dates >= pd.Timestamp(test_start)).any():
            raise ValueError("a search panel must not contain TEST dates")
        self.panel = panel
        self.train_dates = pd.DatetimeIndex(train_dates)
        self.context = context
        self.scoring = scoring
        self.weights = (
            weights if weights is not None else sample_weights(panel.dates, scoring.horizon)
        )
        self.scope = dict(scope or {})
        self.max_trials = max_trials
        self.records: list[TrialRecord] = []
        self._seen: dict[str, str] = {}

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

        train = _score(node, self.panel, self.train_dates, "train", self.scoring, self.weights)
        # Logged before the record exists anywhere a caller could read it.
        _log(
            self.context,
            phase="search",
            params={
                **self.scope,
                "round": round,
                "trial_id": trial_id,
                "expression": expression,
                "panel": dict(self.panel.descriptor),
                "train_dates": date_span(self.train_dates),
                **self.scoring.as_dict(),
            },
            metrics=train.as_flat("train"),
        )

        record = TrialRecord(trial_id, round, expression, raw_expression, rationale, train)
        self._seen[expression] = trial_id
        self.records.append(record)
        return record


def evaluate_holdout(
    node: Node,
    panel: OHLCVPanel,
    dates: pd.DatetimeIndex,
    context: LedgerContext,
    *,
    phase: str,
    trial_id: str,
    scoring: Scoring = DEFAULT_SCORING,
    weights: pd.Series | None = None,
    scope: Mapping[str, Any] | None = None,
) -> SplitMetrics:
    """Score one selected alpha on held-out dates (a CV fold or TEST), logged before returning."""
    if phase not in HOLDOUT_PHASES:
        raise ValueError(f"phase must be one of {sorted(HOLDOUT_PHASES)}, got {phase!r}")
    prefix = HOLDOUT_PHASES[phase]
    if weights is None:
        weights = sample_weights(panel.dates, scoring.horizon)
    metrics = _score(node, panel, pd.DatetimeIndex(dates), prefix, scoring, weights)
    _log(
        context,
        phase=phase,
        params={
            **dict(scope or {}),
            "trial_id": trial_id,
            "expression": canonical(node),
            "panel": dict(panel.descriptor),
            "held_out_dates": date_span(pd.DatetimeIndex(dates)),
            **scoring.as_dict(),
        },
        metrics=metrics.as_flat(prefix),
    )
    return metrics


def _score(
    node: Node,
    panel: OHLCVPanel,
    dates: pd.DatetimeIndex,
    split: str,
    scoring: Scoring,
    weights: pd.Series,
) -> SplitMetrics:
    return split_metrics(
        evaluate(node, panel),
        panel.fields[RETURNS_FIELD],
        dates,
        split=split,
        cost_bps=scoring.cost_bps,
        min_assets=scoring.min_assets,
        horizon=scoring.horizon,
        weights=weights,
    )


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
