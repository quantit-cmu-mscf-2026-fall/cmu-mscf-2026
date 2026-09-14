"""Render the role prompts from `prompts/*.md`.

Templates use `string.Template` ($name placeholders) rather than str.format, so
the JSON examples in them need no brace escaping, and `substitute` raises on a
missing value instead of sending a half-filled prompt.

What the LLM may see is decided by these function signatures. Neither renderer
accepts a panel, a SplitSpec or anything carrying held-out dates or metrics:
the Analyst gets a description of its TRAIN dates and `TrialRecord`s, which
carry TRAIN metrics only.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template

from capstone.alpha_gpt.dsl import DEFAULT_LIMITS, Limits
from capstone.alpha_gpt.operators import OPERATORS
from capstone.alpha_gpt.trials import TrialRecord
from capstone.evaluate import expected_max_sharpe

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PriorRound:
    """What the Quant Developer is told about earlier rounds."""

    round: int
    analyst_summary: str
    evaluated: Sequence[str]


def operator_spec_text(*, groups_available: bool) -> str:
    """One line per usable operator; grouped_* operators only if the panel has groups."""
    lines = []
    for spec in OPERATORS.values():
        if "group" in spec.args and not groups_available:
            continue
        lines.append(f"- {spec.signature}: {spec.doc}")
    return "\n".join(lines)


def target_text(horizon: int) -> str:
    """How the prediction target is described to both roles."""
    if horizon == 1:
        return "return on day t+1"
    return f"compounded return over days t+1 to t+{horizon}"


def render_quant_developer(
    *,
    idea: str,
    fields: Collection[str],
    groups: Collection[str],
    n_alphas: int,
    horizon: int = 1,
    limits: Limits = DEFAULT_LIMITS,
    prior: PriorRound | None = None,
) -> str:
    return _render(
        "quant_developer",
        idea=idea.strip(),
        fields=", ".join(fields),
        groups=", ".join(groups) if groups else "none (grouped_* operators are unavailable)",
        target=target_text(horizon),
        operators=operator_spec_text(groups_available=bool(groups)),
        min_window=limits.min_window,
        max_window=limits.max_window,
        max_lag=limits.max_lag,
        max_depth=limits.max_depth,
        max_nodes=limits.max_nodes,
        n_alphas=n_alphas,
        prior_context=_prior_context(prior),
    )


def render_analyst(
    *,
    idea: str,
    records: Sequence[TrialRecord],
    train_description: str,
    cost_bps: float,
    horizon: int = 1,
) -> str:
    return _render(
        "analyst",
        idea=idea.strip(),
        target=target_text(horizon),
        cost_bps=f"{cost_bps:g}",
        train_description=train_description,
        n_trials=len(records),
        noise_t=f"{noise_tstat(len(records)):.2f}",
        results_table=results_table(records),
    )


def noise_tstat(n_trials: int) -> float:
    """Expected maximum of `n_trials` independent standard-normal t-statistics.

    `expected_max_sharpe` returns that quantile scaled by sqrt(periods/n_obs);
    with n_obs == periods_per_year the scaling is 1, leaving the quantile itself.
    """
    if n_trials < 1:
        return 0.0
    return expected_max_sharpe(n_trials, n_obs=252, periods_per_year=252)


def results_table(records: Sequence[TrialRecord]) -> str:
    header = "| id | round | expression | ic_mean | icir | t | ls_sharpe | turnover | coverage |"
    lines = [header, "|" + "---|" * 9]
    for r in records:
        cells = [
            r.trial_id,
            str(r.round),
            f"`{r.expression}`",
            _num(r.train.ic_mean, 4),
            _num(r.train.icir, 3),
            _num(r.train.ic_tstat, 2),
            _num(r.train.ls_sharpe, 2),
            _num(r.train.turnover, 3),
            _num(r.train.coverage, 2),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _prior_context(prior: PriorRound | None) -> str:
    if prior is None:
        return ""
    evaluated = "\n".join(f"- {expression}" for expression in prior.evaluated)
    return (
        f"\n## Where this idea came from\n"
        f"The Analyst revised the idea above after round {prior.round}. Its reading of the "
        f"evidence so far:\n{prior.analyst_summary.strip()}\n\n"
        f"## Already evaluated (do not resubmit these or trivially equivalent forms)\n"
        f"{evaluated}\n"
    )


def _num(value: float, digits: int) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _render(name: str, **values: object) -> str:
    template = Template((PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8"))
    return template.substitute(values)
