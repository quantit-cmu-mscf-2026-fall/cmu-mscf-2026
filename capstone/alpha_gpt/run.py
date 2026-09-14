"""Run the Alpha-GPT baseline from a TOML config.

    python -m capstone.alpha_gpt.run --config experiments/alpha_gpt/baseline_synth.toml

Cross-validates the discovery procedure with purged k-fold inside DEVELOPMENT,
runs it once more on all of DEVELOPMENT, then evaluates that selection on TEST
once. Writes a run directory under experiments/alpha_gpt/out/ (gitignored) and
appends every evaluation to the run ledger. Real Claude is called through
`claude -p`; tests pass a scripted `llm` instead.
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from capstone.alpha_gpt.loop import (
    CVResult,
    FinalReport,
    LoopConfig,
    ProcedureResult,
    cross_validate_procedure,
    experiment_summary,
    finalize_on_test,
    run_final_procedure,
)
from capstone.alpha_gpt.panel import OHLCVPanel
from capstone.alpha_gpt.prompts import noise_tstat
from capstone.alpha_gpt.roles import LLM, claude_llm
from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel
from capstone.alpha_gpt.trials import LedgerContext, Scoring
from capstone.cv import PurgedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "alpha_gpt" / "out"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Section):
    source: Literal["synthetic"] = "synthetic"
    n_dates: int = Field(1260, ge=200)
    n_assets: int = Field(100, ge=20)
    n_sectors: int = Field(10, ge=1)
    pattern: Literal["none", "reversal"] = "none"
    strength: float = Field(0.03, ge=0.0, lt=1.0)


class SplitsConfig(_Section):
    test_frac: float = Field(0.2, gt=0.0, lt=1.0)
    embargo_days: int = Field(5, ge=1)


class CVConfig(_Section):
    n_splits: int = Field(4, ge=2)
    embargo_pct: float = Field(0.01, ge=0.0, lt=1.0)


class ScoringConfig(_Section):
    label_horizon: int = Field(1, ge=1)
    cost_bps: float = Field(0.0, ge=0.0)
    min_assets: int = Field(20, ge=2)


class LoopSection(_Section):
    n_rounds: int = Field(3, ge=1)
    n_alphas: int = Field(5, ge=1, le=20)
    max_attempts: int = Field(3, ge=1)
    top_k: int = Field(1, ge=1)
    max_trials: int = Field(100, ge=1)


class LLMConfig(_Section):
    timeout_s: float = 300.0
    model: str = ""


class BaselineConfig(_Section):
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    idea: str = Field(min_length=1)
    seed: int = 0
    data: DataConfig = DataConfig()
    splits: SplitsConfig = SplitsConfig()
    cv: CVConfig = CVConfig()
    scoring: ScoringConfig = ScoringConfig()
    loop: LoopSection = LoopSection()
    llm: LLMConfig = LLMConfig()


def load_config(path: str | Path) -> BaselineConfig:
    with Path(path).open("rb") as fh:
        return BaselineConfig.model_validate(tomllib.load(fh))


def build_panel(cfg: BaselineConfig) -> OHLCVPanel:
    data = cfg.data
    return make_ohlcv_panel(
        n_dates=data.n_dates,
        n_assets=data.n_assets,
        n_sectors=data.n_sectors,
        pattern=data.pattern,
        strength=data.strength,
        seed=cfg.seed,
    )


def main(argv: Sequence[str] | None = None, *, llm: LLM | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m capstone.alpha_gpt.run", description=__doc__)
    parser.add_argument("--config", required=True, help="TOML config file")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="parent of run dirs")
    parser.add_argument("--run-id", default=None, help="defaults to <name>-<UTC timestamp>")
    parser.add_argument(
        "--skip-test", action="store_true", help="stop before TEST; do not touch it"
    )
    parser.add_argument(
        "--reuse-test",
        action="store_true",
        help="evaluate TEST even if an earlier run did on this data (tagged test_reuse)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    panel = build_panel(cfg)
    splits = SplitSpec.from_fractions(panel.dates, **cfg.splits.model_dump())
    scoring = Scoring(
        horizon=cfg.scoring.label_horizon,
        cost_bps=cfg.scoring.cost_bps,
        min_assets=cfg.scoring.min_assets,
    )
    splits.check_horizon(scoring.horizon)
    cv = PurgedKFold(cfg.cv.n_splits, horizon=scoring.horizon, embargo_pct=cfg.cv.embargo_pct)

    run_id = args.run_id or f"{cfg.name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": cfg.model_dump(),
                "panel": dict(panel.descriptor),
                "splits": splits.as_dict(),
            },
            indent=2,
        )
    )

    context = LedgerContext(
        ledger_name="alpha_gpt_baseline",
        run_id=run_id,
        seed=cfg.seed,
        tags=("alpha_gpt", "baseline", cfg.name),
        extra_params={"config_name": cfg.name, "idea": cfg.idea},
    )
    loop_cfg = LoopConfig(**cfg.loop.model_dump())
    chosen = llm or claude_llm(timeout=cfg.llm.timeout_s, model=cfg.llm.model or None)

    cv_result = cross_validate_procedure(
        cfg.idea,
        panel,
        splits,
        loop_cfg,
        cv,
        llm=chosen,
        context=context,
        run_dir=run_dir,
        scoring=scoring,
    )
    print(format_cv_summary(cv_result))

    final = run_final_procedure(
        cfg.idea,
        panel,
        splits,
        loop_cfg,
        llm=chosen,
        context=context,
        run_dir=run_dir,
        scoring=scoring,
    )
    print(format_procedure(final, "final procedure on all of DEVELOPMENT"))

    report = None
    if not args.skip_test:
        report = finalize_on_test(
            final,
            panel,
            splits,
            context=context,
            run_dir=run_dir,
            scoring=scoring,
            reuse_test=args.reuse_test,
        )
        print(format_test_summary(report))

    summary = experiment_summary(cv_result, final, report)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(format_overview(summary))
    print(f"\nrun directory: {run_dir}")
    return 0


def format_procedure(result: ProcedureResult, title: str) -> str:
    lines = [f"\n== {title} =="]
    for rnd in result.rounds:
        lines.append(f"-- round {rnd.round} --")
        lines.append(f"idea: {rnd.idea}")
        for record in rnd.trials:
            lines.append(
                f"  {record.trial_id:6s} train t={record.train.ic_tstat:6.2f} "
                f"ic={record.train.ic_mean:+.4f} ls_sharpe={record.train.ls_sharpe:5.2f}  "
                f"{record.expression}"
            )
        if rnd.duplicates:
            lines.append(f"  duplicates not re-evaluated: {[d.expression for d in rnd.duplicates]}")
        if rnd.problems:
            lines.append(f"  unresolved reply problems: {len(rnd.problems)}")
        if rnd.review is not None:
            lines.append(f"  analyst: {rnd.review.summary}")
    n = len(result.trials)
    lines.append(
        f"trials: {n}; best-of-{n} noise benchmark t ~ {noise_tstat(n):.2f}"
        + ("; trial budget exhausted" if result.budget_exhausted else "")
    )
    for record in result.selected:
        lines.append(
            f"selected {record.trial_id}: {record.expression} (train t={record.train.ic_tstat:.2f})"
        )
    return "\n".join(lines)


def format_cv_summary(cv_result: CVResult) -> str:
    lines = [
        "\n== cross-validation: each fold held out from the whole procedure (purged k-fold) =="
    ]
    for fold in cv_result.folds:
        train, valid = fold.train_span, fold.valid_span
        lines.append(
            f"fold {fold.fold}: train {train[0]}..{train[1]} ({train[2]} dates), "
            f"held out {valid[0]}..{valid[1]} ({valid[2]} dates), "
            f"{len(fold.procedure.trials)} trials"
        )
        for score in fold.held_out:
            lines.append(
                f"  {score.expression}\n"
                f"    train t={score.train.ic_tstat:.2f} -> held-out "
                f"ic={score.held_out.ic_mean:+.4f} t={score.held_out.ic_tstat:.2f}"
            )
    s = cv_result.summary()
    lines.append(
        f"held-out ic mean {_fmt(s['held_out_ic_mean'], '+.4f')} vs mean train ic "
        f"{_fmt(s['mean_train_ic'], '+.4f')} "
        f"({_fmt(s['held_out_share_of_train_ic'], '.0%')} retained; "
        f"t across folds {_fmt(s['held_out_ic_t_across_folds'], '.2f')}, "
        f"positive in {_fmt(s['share_of_folds_positive'], '.0%')} of folds)"
    )
    return "\n".join(lines)


def format_test_summary(report: FinalReport) -> str:
    lines = ["\n== TEST (evaluated once) =="]
    for entry in report.entries:
        t = entry.held_out
        lines.append(
            f"  {entry.trial_id}: {entry.expression}\n"
            f"    ic={t.ic_mean:+.4f} icir={t.icir:.3f} t={t.ic_tstat:.2f} p={t.ic_pvalue:.4f} "
            f"ls_sharpe={t.ls_sharpe:.2f} turnover={t.turnover:.3f} days={t.n_days}"
        )
    return "\n".join(lines)


def format_overview(summary: dict[str, Any]) -> str:
    in_sample = summary["in_sample"] or {}
    cv = summary["cross_validation"]
    test = summary["test"]
    return "\n".join(
        [
            "\n== overfitting check ==",
            f"family size (search trials in every fold and the final run): "
            f"{summary['family_size']}",
            f"in-sample, final selection: train ic = "
            f"{_fmt(in_sample.get('train_ic_mean'), '+.4f')} "
            f"(t = {_fmt(in_sample.get('train_ic_tstat'), '.2f')}), deflated Sharpe ratio = "
            f"{_fmt(in_sample.get('deflated_sharpe_ratio'), '.3f')}",
            f"cross-validated procedure: mean train ic = {_fmt(cv['mean_train_ic'], '+.4f')} -> "
            f"mean held-out ic = {_fmt(cv['held_out_ic_mean'], '+.4f')} "
            f"({_fmt(cv['held_out_share_of_train_ic'], '.0%')} of train IC retained)",
            f"TEST: ic = {_fmt(test['test_ic_mean'], '+.4f')}, "
            f"t = {_fmt(test['test_ic_tstat'], '.2f')}, "
            f"p = {_fmt(test['test_ic_pvalue'], '.4f')}"
            if test
            else "TEST: not evaluated (--skip-test)",
            "(compare IC across these lines, not t: t grows with the number of dates, "
            "and a training set is longer than a held-out fold)",
        ]
    )


def _fmt(value: float | None, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return format(value, spec)


if __name__ == "__main__":
    raise SystemExit(main())
