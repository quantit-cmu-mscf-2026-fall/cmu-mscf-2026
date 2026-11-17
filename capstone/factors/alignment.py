"""Does a formula implement its hypothesis? AlphaAgent's second mechanism.

A separate, cheaper model reads the hypothesis and one proposed formula and
answers three yes/no questions: does the formula use the variables the
specification names, point the way it says (higher value = buy), and measure
over its horizon? The score is the share of yeses (0, 1/3, 2/3 or 1), and it
is stored on the proposal with the judge's reason and model.

The judge never sees performance, so a judgement is not a ledger trial. Its
threshold (`FactorConfig.min_alignment`) is fixed from hand-labelled
proposals before any factor is evaluated, never tuned on factor performance:
that would be a search of its own. Until it is calibrated the threshold is 0,
so scores are recorded and nothing is rejected (QUANTIT-83).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from capstone.factors.hypotheses import _FIELDS
from capstone.factors.llm import FactorConfig, call_tool, grammar_help

CHECKS = ("variables", "direction", "horizon")

SYSTEM = (
    "You check whether a cross-sectional equity factor formula implements the hypothesis "
    "it was written for. The formula is evaluated per stock per day; a higher value means "
    "a stronger buy signal. Judge only what the formula computes, not whether the "
    "hypothesis is true or the factor would make money. Answer three questions:\n"
    "- variables: does it use the quantities the specification names (or direct "
    "equivalents), and no unrelated ones that change what it measures?\n"
    "- direction: does a higher value mean what the hypothesis says should outperform?\n"
    "- horizon: do its windows match the horizon the specification states? Answer yes if "
    "the specification states none.\n\n" + grammar_help()
)

TOOL = {
    "name": "record_alignment",
    "description": "Record whether the formula implements the hypothesis.",
    "input_schema": {
        "type": "object",
        "properties": {
            **{check: {"type": "boolean"} for check in CHECKS},
            "reason": {"type": "string"},
        },
        "required": [*CHECKS, "reason"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Alignment:
    score: float
    reason: str
    model: str
    checks: dict[str, bool] = field(default_factory=dict)


def judge(
    client,
    config: FactorConfig,
    hypothesis: dict,
    expression: str,
    rationale: str = "",
    usage: Counter | None = None,
) -> Alignment:
    """Score one formula against its hypothesis with `config.alignment_model`."""
    judge_config = replace(config, model=config.alignment_model)
    lines = [f"{name}: {hypothesis[name]}" for name in _FIELDS]
    user = (
        "Hypothesis\n" + "\n".join(lines) + f"\n\nFormula: {expression}\n"
        f"Author's rationale: {rationale or '(none)'}"
    )
    spent: Counter = Counter()
    answer = call_tool(client, judge_config, system=SYSTEM, user=user, tool=TOOL, usage=spent)
    if usage is not None:  # kept apart from the proposer's tokens: another model's price
        usage.update({f"alignment_{name}": count for name, count in spent.items()})
    checks = {check: bool(answer.get(check)) for check in CHECKS}
    failed = [check for check, passed in checks.items() if not passed]
    reason = str(answer.get("reason", ""))
    if failed:
        reason = f"fails {', '.join(failed)}: {reason}"
    return Alignment(sum(checks.values()) / len(CHECKS), reason, config.alignment_model, checks)
