"""The baseline Alpha-GPT loop: Seed + Analyst, then one TEST evaluation.

Per round:
  1. Quant Developer turns the current idea into N alpha expressions.
  2. Each valid, not-yet-seen expression is evaluated on TRAIN and VALIDATION
     and logged to the ledger at that moment (`TrialEvaluator`).
  3. Unless it is the last round, the Analyst reads the results table and
     writes a revised idea, which becomes the next round's idea.

After the last round the best alphas by VALIDATION IC t-stat are selected, and
`finalize_on_test` evaluates them on TEST exactly once per run directory.

Seams for the later phases: an Idea Polisher goes before round 1's prompt, RAG
context into the Quant Developer prompt, genetic search between proposal and
evaluation (every child through `TrialEvaluator.evaluate`), and the BH-FDR
gate inside `select_best`, before `finalize_on_test`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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
    Duplicate,
    LedgerContext,
    TrialBudgetExhausted,
    TrialEvaluator,
    TrialRecord,
    evaluate_on_test,
)
from capstone.runlog import read_entries

FINAL_TEST_FILE = "final_test.json"


@dataclass(frozen=True)
class LoopConfig:
    n_rounds: int = 3
    n_alphas: int = 5
    max_attempts: int = 3
    top_k: int = 1
    cost_bps: float = 0.0
    max_trials: int = 100
    min_assets: int = 20


@dataclass(frozen=True)
class RoundRecord:
    round: int
    idea: str
    trials: list[TrialRecord]
    duplicates: list[Duplicate]
    problems: list[str]
    review: AnalystReview | None


@dataclass(frozen=True)
class LoopResult:
    run_id: str
    rounds: list[RoundRecord]
    trials: list[TrialRecord]
    selected: list[TrialRecord]
    budget_exhausted: bool = False


@dataclass(frozen=True)
class FinalEntry:
    trial_id: str
    expression: str
    valid: SplitMetrics
    test: SplitMetrics


@dataclass(frozen=True)
class FinalReport:
    run_id: str
    n_search_trials: int
    entries: list[FinalEntry] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "n_search_trials": self.n_search_trials,
            "selected": [
                {
                    "trial_id": e.trial_id,
                    "expression": e.expression,
                    **e.valid.as_flat("valid"),
                    **e.test.as_flat("test"),
                }
                for e in self.entries
            ],
        }


class RunFiles:
    """Append-only JSONL files in the run directory (gitignored; never the ledger)."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, entry: dict[str, Any]) -> None:
        with (self.run_dir / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def run_seed_analyst_loop(
    idea: str,
    panel: OHLCVPanel,
    splits: SplitSpec,
    cfg: LoopConfig,
    *,
    llm: LLM,
    context: LedgerContext,
    run_dir: Path,
    limits: Limits = DEFAULT_LIMITS,
) -> LoopResult:
    evaluator = TrialEvaluator(
        panel,
        splits,
        context,
        cost_bps=cfg.cost_bps,
        min_assets=cfg.min_assets,
        max_trials=cfg.max_trials,
    )
    fields = list(evaluator.panel.fields)
    groups = list(evaluator.panel.groups)
    ranges = splits.as_dict()
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
                    train_range=" to ".join(ranges["train"]),
                    valid_range=" to ".join(ranges["valid"]),
                    cost_bps=cfg.cost_bps,
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
    return LoopResult(
        run_id=context.run_id,
        rounds=rounds,
        trials=list(evaluator.records),
        selected=selected,
        budget_exhausted=budget_exhausted,
    )


def select_best(records: list[TrialRecord], k: int) -> list[TrialRecord]:
    """Top `k` by signed VALIDATION IC t-stat; NaN last, ties broken by expression.

    Signed on purpose: an alpha with strongly negative IC is not silently
    flipped — its negation has to be proposed and evaluated as its own trial.
    """

    def key(record: TrialRecord) -> tuple[bool, float, str]:
        t = record.valid.ic_tstat
        return (math.isnan(t), -t if not math.isnan(t) else 0.0, record.expression)

    return sorted(records, key=key)[:k]


def finalize_on_test(
    result: LoopResult,
    panel: OHLCVPanel,
    splits: SplitSpec,
    cfg: LoopConfig,
    *,
    context: LedgerContext,
    run_dir: Path,
    reuse_test: bool = False,
) -> FinalReport:
    """Evaluate the selected alphas on TEST, once per run directory and once per dataset.

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

    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise FileExistsError(f"TEST was already evaluated for this run: {path}") from exc

    with handle:
        entries = []
        for record in result.selected:
            node = parse(record.expression, fields=panel.fields, groups=panel.groups)
            test = evaluate_on_test(
                node,
                panel,
                splits,
                context,
                trial_id=record.trial_id,
                cost_bps=cfg.cost_bps,
                min_assets=cfg.min_assets,
            )
            entries.append(FinalEntry(record.trial_id, record.expression, record.valid, test))
        report = FinalReport(result.run_id, len(result.trials), entries)
        json.dump(report.as_dict(), handle, indent=2)
    return report


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
