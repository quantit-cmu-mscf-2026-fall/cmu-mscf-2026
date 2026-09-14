"""The Alpha-GPT discovery procedure, cross-validated with purged k-fold, then TEST once.

The procedure (`run_procedure`), per round:
  1. Quant Developer turns the current idea into N alpha expressions.
  2. Each valid, new expression is scored on the procedure's TRAIN dates and
     logged to the ledger at that moment (`TrialEvaluator`).
  3. Unless it is the last round, the Analyst reads the TRAIN results and writes
     a revised idea for the next round.
  Finally the best alphas by TRAIN IC t-stat are selected.

Everything in that procedure can overfit — including the Analyst steering on its
own results and the selection on them. So the procedure itself is what gets
validated (López de Prado, AFML ch. 7):

- `cross_validate_procedure` splits DEVELOPMENT with `capstone.cv.PurgedKFold`.
  For each fold it runs the whole procedure on a panel in which the fold is
  masked, on the purged and embargoed TRAIN dates around it, then scores the
  selected alphas on the held-out fold (VALIDATION). The gap between in-sample
  and held-out scores is the procedure's overfitting, measured.
- `run_final_procedure` runs it once on all of DEVELOPMENT, and
  `finalize_on_test` scores that selection on TEST — once per run and, via the
  ledger, once per dataset and split.

Seams for later phases: an Idea Polisher goes before round 1's prompt, RAG
context into the Quant Developer prompt, genetic search between proposal and
evaluation (every child through `TrialEvaluator.evaluate`, fitness on TRAIN),
and the BH-FDR gate on held-out p-values, before `finalize_on_test`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from capstone.alpha_gpt.dsl import DEFAULT_LIMITS, DSLError, Limits, parse
from capstone.alpha_gpt.metrics import SplitMetrics
from capstone.alpha_gpt.panel import OHLCVPanel
from capstone.alpha_gpt.prompts import PriorRound, render_analyst, render_quant_developer
from capstone.alpha_gpt.roles import (
    LLM,
    AnalystReview,
    SeedProposal,
    call_role,
    check_analyst_review,
    check_seed_proposal,
)
from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.trials import (
    DEFAULT_SCORING,
    Duplicate,
    LedgerContext,
    Scoring,
    TrialBudgetExhausted,
    TrialEvaluator,
    TrialRecord,
    date_span,
    evaluate_holdout,
    sample_weights,
)
from capstone.cv import PurgedKFold
from capstone.evaluate import deflated_sharpe_ratio
from capstone.runlog import read_entries

FINAL_TEST_FILE = "final_test.json"


@dataclass(frozen=True)
class LoopConfig:
    n_rounds: int = 3
    n_alphas: int = 5
    max_attempts: int = 3
    top_k: int = 1
    max_trials: int = 100


@dataclass(frozen=True)
class RoundRecord:
    round: int
    idea: str
    trials: list[TrialRecord]
    duplicates: list[Duplicate]
    problems: list[str]
    review: AnalystReview | None


@dataclass(frozen=True)
class ProcedureResult:
    scope: dict[str, Any]
    rounds: list[RoundRecord]
    trials: list[TrialRecord]
    selected: list[TrialRecord]
    budget_exhausted: bool = False


@dataclass(frozen=True)
class HeldOutScore:
    """A selected alpha's in-sample TRAIN metrics next to its held-out metrics."""

    trial_id: str
    expression: str
    train: SplitMetrics
    held_out: SplitMetrics

    def as_dict(self, prefix: str) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "expression": self.expression,
            **self.train.as_flat("train"),
            **self.held_out.as_flat(prefix),
        }


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_span: list[Any]
    valid_span: list[Any]
    procedure: ProcedureResult
    held_out: list[HeldOutScore]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_dates": self.train_span,
            "valid_dates": self.valid_span,
            "n_trials": len(self.procedure.trials),
            "selected": [score.as_dict("valid") for score in self.held_out],
        }


@dataclass(frozen=True)
class CVResult:
    folds: list[FoldResult]

    @property
    def n_trials(self) -> int:
        return sum(len(fold.procedure.trials) for fold in self.folds)

    def summary(self) -> dict[str, Any]:
        """Out-of-fold performance of the procedure's top pick, fold by fold and pooled."""
        tops = [fold.held_out[0] for fold in self.folds if fold.held_out]
        held_ic = np.array([top.held_out.ic_mean for top in tops], dtype=float)
        k = int(np.isfinite(held_ic).sum())
        mean_ic = float(np.nanmean(held_ic)) if k else math.nan
        spread = float(np.nanstd(held_ic, ddof=1)) if k >= 2 else math.nan
        t_across = mean_ic / spread * math.sqrt(k) if k >= 2 and spread > 0 else math.nan
        return _finite_or_none(
            {
                "n_folds": len(self.folds),
                "held_out_ic_mean": mean_ic,
                "held_out_ic_t_across_folds": t_across,
                "share_of_folds_positive": float(np.mean(held_ic[np.isfinite(held_ic)] > 0))
                if k
                else math.nan,
                "mean_train_t": _mean([top.train.ic_tstat for top in tops]),
                "mean_held_out_t": _mean([top.held_out.ic_tstat for top in tops]),
            }
        )


@dataclass(frozen=True)
class FinalReport:
    run_id: str
    entries: list[HeldOutScore] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "selected": [e.as_dict("test") for e in self.entries]}


class RunFiles:
    """Append-only JSONL files in a run directory (gitignored; never the ledger)."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, entry: dict[str, Any]) -> None:
        with (self.run_dir / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def run_procedure(
    idea: str,
    panel: OHLCVPanel,
    train_dates: pd.DatetimeIndex,
    cfg: LoopConfig,
    *,
    llm: LLM,
    context: LedgerContext,
    test_start: pd.Timestamp,
    run_dir: Path,
    scoring: Scoring = DEFAULT_SCORING,
    weights: pd.Series | None = None,
    scope: Mapping[str, Any] | None = None,
    held_out_removed: bool = False,
    limits: Limits = DEFAULT_LIMITS,
) -> ProcedureResult:
    """Seed + Analyst rounds on `train_dates` of `panel`, then select by TRAIN t."""
    evaluator = TrialEvaluator(
        panel,
        train_dates,
        context,
        test_start=test_start,
        scoring=scoring,
        weights=weights,
        scope=scope,
        max_trials=cfg.max_trials,
    )
    fields = list(panel.fields)
    groups = list(panel.groups)
    first, last, count = date_span(evaluator.train_dates)
    train_description = f"{first} to {last}, {count} dates" + (
        ", with held-out periods removed" if held_out_removed else ""
    )
    files = RunFiles(run_dir)

    rounds: list[RoundRecord] = []
    prior: PriorRound | None = None
    current_idea = idea
    budget_exhausted = False

    for k in range(1, cfg.n_rounds + 1):
        files.append("events", {"event": "round_start", "round": k, "idea": current_idea})
        already = {record.expression for record in evaluator.records}
        proposal = call_role(
            render_quant_developer(
                idea=current_idea,
                fields=fields,
                groups=groups,
                n_alphas=cfg.n_alphas,
                horizon=scoring.horizon,
                limits=limits,
                prior=prior,
            ),
            SeedProposal,
            llm=llm,
            role="quant_developer",
            check=lambda reply, already=already: check_seed_proposal(
                reply,
                fields=fields,
                groups=groups,
                n_alphas=cfg.n_alphas,
                already_evaluated=already,
                limits=limits,
            ),
            max_attempts=cfg.max_attempts,
            record=lambda entry, k=k: files.append("llm_calls", {"round": k, **entry}),
        )

        trials: list[TrialRecord] = []
        duplicates: list[Duplicate] = []
        for i, alpha in enumerate(proposal.reply.alphas[: cfg.n_alphas], start=1):
            try:
                node = parse(alpha.expression, fields=fields, groups=groups, limits=limits)
            except DSLError as exc:
                files.append(
                    "events",
                    {
                        "event": "rejected",
                        "round": k,
                        "expression": alpha.expression,
                        "error": str(exc),
                    },
                )
                continue
            try:
                outcome = evaluator.evaluate(
                    node,
                    trial_id=f"r{k}a{i}",
                    round=k,
                    raw_expression=alpha.expression,
                    rationale=alpha.rationale,
                )
            except TrialBudgetExhausted:
                budget_exhausted = True
                files.append("events", {"event": "budget_exhausted", "round": k})
                break
            if isinstance(outcome, Duplicate):
                duplicates.append(outcome)
                files.append("events", {"event": "duplicate", "round": k, **vars(outcome)})
            else:
                trials.append(outcome)
                files.append("trials", outcome.as_dict())

        review = None
        is_last = k == cfg.n_rounds or budget_exhausted
        if not is_last and evaluator.records:
            known_ids = {record.trial_id for record in evaluator.records}
            analysis = call_role(
                render_analyst(
                    idea=current_idea,
                    records=evaluator.records,
                    train_description=train_description,
                    cost_bps=scoring.cost_bps,
                    horizon=scoring.horizon,
                ),
                AnalystReview,
                llm=llm,
                role="analyst",
                check=lambda reply, known_ids=known_ids: check_analyst_review(
                    reply, known_ids=known_ids
                ),
                max_attempts=cfg.max_attempts,
                record=lambda entry, k=k: files.append("llm_calls", {"round": k, **entry}),
            )
            review = analysis.reply
            files.append(
                "reviews", {"round": k, **review.model_dump(), "problems": analysis.problems}
            )
            prior = PriorRound(
                round=k,
                analyst_summary=review.summary,
                evaluated=[record.expression for record in evaluator.records],
            )

        rounds.append(
            RoundRecord(k, current_idea, trials, duplicates, list(proposal.problems), review)
        )
        if review is not None:
            current_idea = review.revised_idea
        if budget_exhausted:
            break

    selected = select_best(evaluator.records, cfg.top_k)
    files.append(
        "events", {"event": "selected", "trial_ids": [record.trial_id for record in selected]}
    )
    return ProcedureResult(
        scope=dict(scope or {}),
        rounds=rounds,
        trials=list(evaluator.records),
        selected=selected,
        budget_exhausted=budget_exhausted,
    )


def cross_validate_procedure(
    idea: str,
    panel: OHLCVPanel,
    splits: SplitSpec,
    cfg: LoopConfig,
    cv: PurgedKFold,
    *,
    llm: LLM,
    context: LedgerContext,
    run_dir: Path,
    scoring: Scoring = DEFAULT_SCORING,
    limits: Limits = DEFAULT_LIMITS,
) -> CVResult:
    """Run the whole procedure once per purged, embargoed fold; score picks on the fold."""
    if cv.horizon != scoring.horizon:
        raise ValueError(f"cv horizon {cv.horizon} differs from scoring horizon {scoring.horizon}")
    splits.check_horizon(scoring.horizon)

    search = splits.search_panel(panel)
    weights = sample_weights(search.dates, scoring.horizon)
    development = splits.dates(search.dates, "development")
    files = RunFiles(run_dir)

    folds = []
    for j, (train, valid) in enumerate(cv.split(development), start=1):
        # Hide the fold and every return its labels are built from.
        label_end = search.dates[search.dates.get_loc(valid[-1]) + scoring.horizon]
        view = search.mask(valid[0], label_end)
        scope = {"stage": "cv", "fold": j, "splits": splits.as_dict()}
        procedure = run_procedure(
            idea,
            view,
            train,
            cfg,
            llm=llm,
            context=context,
            test_start=splits.test[0],
            run_dir=Path(run_dir) / f"fold{j}",
            scoring=scoring,
            weights=weights,
            scope=scope,
            held_out_removed=True,
            limits=limits,
        )

        held_out = []
        for record in procedure.selected:
            node = parse(record.expression, fields=search.fields, groups=search.groups)
            score = evaluate_holdout(
                node,
                search,
                valid,
                context,
                phase="cv_valid",
                trial_id=record.trial_id,
                scoring=scoring,
                weights=weights,
                scope=scope,
            )
            held_out.append(HeldOutScore(record.trial_id, record.expression, record.train, score))

        fold = FoldResult(j, date_span(train), date_span(valid), procedure, held_out)
        files.append("cv_folds", fold.as_dict())
        folds.append(fold)
    return CVResult(folds)


def run_final_procedure(
    idea: str,
    panel: OHLCVPanel,
    splits: SplitSpec,
    cfg: LoopConfig,
    *,
    llm: LLM,
    context: LedgerContext,
    run_dir: Path,
    scoring: Scoring = DEFAULT_SCORING,
    limits: Limits = DEFAULT_LIMITS,
) -> ProcedureResult:
    """The procedure on all of DEVELOPMENT; its selection is what TEST evaluates."""
    splits.check_horizon(scoring.horizon)
    search = splits.search_panel(panel)
    return run_procedure(
        idea,
        search,
        splits.dates(search.dates, "development"),
        cfg,
        llm=llm,
        context=context,
        test_start=splits.test[0],
        run_dir=Path(run_dir) / "final",
        scoring=scoring,
        weights=sample_weights(search.dates, scoring.horizon),
        scope={"stage": "final", "splits": splits.as_dict()},
        limits=limits,
    )


def select_best(records: list[TrialRecord], k: int) -> list[TrialRecord]:
    """Top `k` by signed TRAIN IC t-stat; NaN last, ties broken by expression.

    Signed on purpose: an alpha with strongly negative IC is not silently
    flipped — its negation has to be proposed and evaluated as its own trial.
    """

    def key(record: TrialRecord) -> tuple[bool, float, str]:
        t = record.train.ic_tstat
        return (math.isnan(t), -t if not math.isnan(t) else 0.0, record.expression)

    return sorted(records, key=key)[:k]


def finalize_on_test(
    final: ProcedureResult,
    panel: OHLCVPanel,
    splits: SplitSpec,
    *,
    context: LedgerContext,
    run_dir: Path,
    scoring: Scoring = DEFAULT_SCORING,
    reuse_test: bool = False,
) -> FinalReport:
    """Evaluate the final selection on TEST, once per run directory and once per dataset.

    Two locks, both checked before anything is computed:

    - the run directory: a second call for the same run refuses;
    - the ledger: if any earlier run already evaluated TEST on the same data and
      split dates, this refuses unless `reuse_test=True`, in which case every
      TEST entry is tagged `test_reuse` so the peek stays visible in the record.
    """
    path = Path(run_dir) / FINAL_TEST_FILE
    if path.exists():
        raise FileExistsError(f"TEST was already evaluated for this run: {path}")

    previous = previous_test_runs(panel, splits)
    if previous and not reuse_test:
        raise TestAlreadyEvaluated(
            f"TEST on this data and split was already evaluated by run(s) {previous}; "
            "pass reuse_test=True (--reuse-test) to evaluate it again as a recorded reuse"
        )
    if previous:
        context = replace(context, tags=(*context.tags, "test_reuse"))

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise FileExistsError(f"TEST was already evaluated for this run: {path}") from exc

    with handle:
        weights = sample_weights(panel.dates, scoring.horizon)
        test_dates = splits.dates(panel.dates, "test")
        entries = []
        for record in final.selected:
            node = parse(record.expression, fields=panel.fields, groups=panel.groups)
            test = evaluate_holdout(
                node,
                panel,
                test_dates,
                context,
                phase="test",
                trial_id=record.trial_id,
                scoring=scoring,
                weights=weights,
                # The split dates are what the cross-run TEST lock matches on.
                scope={**final.scope, "splits": splits.as_dict()},
            )
            entries.append(HeldOutScore(record.trial_id, record.expression, record.train, test))
        report = FinalReport(context.run_id, entries)
        json.dump(report.as_dict(), handle, indent=2)
    return report


def experiment_summary(
    cv_result: CVResult, final: ProcedureResult, report: FinalReport | None
) -> dict[str, Any]:
    """In-sample -> cross-validated -> TEST, plus the family size behind every number."""
    family_size = cv_result.n_trials + len(final.trials)
    summary: dict[str, Any] = {
        "family_size": family_size,
        "cross_validation": cv_result.summary(),
        "in_sample": None,
        "test": None,
    }
    if final.selected:
        best = final.selected[0]
        dsr = (
            deflated_sharpe_ratio(
                best.train.ls_sharpe, n_trials=family_size, n_obs=best.train.n_days
            )
            if math.isfinite(best.train.ls_sharpe) and best.train.n_days >= 2
            else math.nan
        )
        summary["in_sample"] = _finite_or_none(
            {
                "trial_id": best.trial_id,
                "expression": best.expression,
                "train_ic_mean": best.train.ic_mean,
                "train_ic_tstat": best.train.ic_tstat,
                "train_ls_sharpe": best.train.ls_sharpe,
                "deflated_sharpe_ratio": dsr,
            }
        )
    if report is not None and report.entries:
        summary["test"] = report.entries[0].as_dict("test")
    return summary


class TestAlreadyEvaluated(RuntimeError):
    """TEST on this data and split was already looked at by an earlier run."""

    __test__ = False  # not a pytest test class, despite the name


def previous_test_runs(panel: OHLCVPanel, splits: SplitSpec) -> list[str]:
    """run_ids of ledger entries that evaluated TEST on the same data and split dates."""
    descriptor = _without_masks(panel.descriptor)
    split_dates = splits.as_dict()
    runs = set()
    for entry in read_entries():
        params = entry.get("params") or {}
        if (
            params.get("phase") == "test"
            and params.get("splits") == split_dates
            and _without_masks(params.get("panel") or {}) == descriptor
        ):
            runs.add(str(params.get("run_id")))
    return sorted(runs)


def _without_masks(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in descriptor.items() if key != "masked"}


def _mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def _finite_or_none(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in values.items()
    }
