"""Run the Alpha-GPT baseline from a TOML config.

    python -m capstone.alpha_gpt.run --config experiments/alpha_gpt/baseline_synth.toml

Writes a run directory under experiments/alpha_gpt/out/ (gitignored) with the
config, every LLM prompt and reply, trials, Analyst reviews and the final TEST
report, and appends every evaluation to the run ledger. Real Claude is called
through `claude -p`; tests pass a scripted `llm` instead.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from capstone.alpha_gpt.loop import (
    FinalReport,
    LoopConfig,
    LoopResult,
    finalize_on_test,
    run_seed_analyst_loop,
)
from capstone.alpha_gpt.panel import OHLCVPanel
from capstone.alpha_gpt.prompts import noise_tstat
from capstone.alpha_gpt.roles import LLM, claude_llm
from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel
from capstone.alpha_gpt.trials import LedgerContext

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
    train_frac: float = 0.6
    valid_frac: float = 0.2
    embargo_days: int = 5


class LoopSection(_Section):
    n_rounds: int = Field(3, ge=1)
    n_alphas: int = Field(5, ge=1, le=20)
    max_attempts: int = Field(3, ge=1)
    top_k: int = Field(1, ge=1)
    cost_bps: float = Field(0.0, ge=0.0)
    max_trials: int = Field(100, ge=1)
    min_assets: int = Field(20, ge=2)


class LLMConfig(_Section):
    timeout_s: float = 300.0
    model: str = ""


class BaselineConfig(_Section):
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    idea: str = Field(min_length=1)
    seed: int = 0
    data: DataConfig = DataConfig()
    splits: SplitsConfig = SplitsConfig()
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
        "--skip-test", action="store_true", help="stop after selection; do not touch TEST"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    panel = build_panel(cfg)
    splits = SplitSpec.from_fractions(panel.dates, **cfg.splits.model_dump())
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
    model = claude_llm(timeout=cfg.llm.timeout_s, model=cfg.llm.model or None)
    result = run_seed_analyst_loop(
        cfg.idea, panel, splits, loop_cfg, llm=llm or model, context=context, run_dir=run_dir
    )
    print(format_search_summary(result, splits))

    if not args.skip_test:
        report = finalize_on_test(result, panel, splits, loop_cfg, context=context, run_dir=run_dir)
        print(format_test_summary(report))
    print(f"\nrun directory: {run_dir}")
    return 0


def format_search_summary(result: LoopResult, splits: SplitSpec) -> str:
    ranges = splits.as_dict()
    lines = [
        f"run {result.run_id}",
        f"TRAIN {ranges['train'][0]}..{ranges['train'][1]}  "
        f"VALID {ranges['valid'][0]}..{ranges['valid'][1]}",
    ]
    for rnd in result.rounds:
        lines.append(f"\n== round {rnd.round} ==")
        lines.append(f"idea: {rnd.idea}")
        for record in rnd.trials:
            lines.append(
                f"  {record.trial_id:6s} train t={record.train.ic_tstat:6.2f}  "
                f"valid t={record.valid.ic_tstat:6.2f} ic={record.valid.ic_mean:+.4f} "
                f"ls_sharpe={record.valid.ls_sharpe:5.2f}  {record.expression}"
            )
        if rnd.duplicates:
            lines.append(f"  duplicates not re-evaluated: {[d.expression for d in rnd.duplicates]}")
        if rnd.problems:
            lines.append(f"  unresolved reply problems: {len(rnd.problems)}")
        if rnd.review is not None:
            lines.append(f"  analyst: {rnd.review.summary}")
    n = len(result.trials)
    lines.append(
        f"\nsearch trials: {n}; best-of-{n} noise benchmark t ~ {noise_tstat(n):.2f}"
        + ("; trial budget exhausted" if result.budget_exhausted else "")
    )
    for record in result.selected:
        lines.append(
            f"selected {record.trial_id}: {record.expression} (valid t={record.valid.ic_tstat:.2f})"
        )
    return "\n".join(lines)


def format_test_summary(report: FinalReport) -> str:
    lines = ["\n== TEST (evaluated once) =="]
    for entry in report.entries:
        t = entry.test
        lines.append(
            f"  {entry.trial_id}: {entry.expression}\n"
            f"    ic={t.ic_mean:+.4f} icir={t.icir:.3f} t={t.ic_tstat:.2f} p={t.ic_pvalue:.4f} "
            f"ls_sharpe={t.ls_sharpe:.2f} turnover={t.turnover:.3f} days={t.n_days}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
