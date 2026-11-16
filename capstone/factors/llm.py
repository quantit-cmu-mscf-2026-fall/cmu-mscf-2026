"""Calling the model: parameters, the client, and one strict tool per call.

Every call offers exactly one tool marked `strict`, so the answer arrives as
input that validates against its JSON schema, never as free text to be
parsed. The tool is not forced through `tool_choice` (newer models reject
that, and it conflicts with thinking); the system prompt says to answer by
calling it, and a reply without the call raises. The
`anthropic` package is an optional extra (`pip install -e ".[agents]"`) and
is imported only when a real client is made; tests pass a fake client. The
credentials come from the environment or a login profile, never from a file in
this repository.
"""

from __future__ import annotations

import tomllib
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

from capstone.factors.tree import (
    BINARY_PARAMETER_FUNCS,
    BINARY_WINDOW_FUNCS,
    FUNCS,
    MARKET_FIELDS,
    STOCK_FIELDS,
    WINDOW_MAX,
    WINDOW_MIN,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "factors.toml"


@dataclass(frozen=True)
class FactorConfig:
    """The pipeline's parameters; see config/factors.toml for what each one does."""

    model: str = "claude-opus-5"
    max_tokens: int = 16000
    prompt_version: str = "v7"
    hypotheses_per_paper: int = 3
    factors_per_hypothesis: int = 4
    max_repairs: int = 2
    max_zoo_share: float = 0.75
    max_store_share: float = 1.0
    max_nodes: int = 30
    alignment_model: str = ""
    min_alignment: float = 0.0
    avoid_frequent_subtrees: int = 3
    max_bond_leaves: int = 4
    max_member_share: float = 0.75
    max_bond_overlap: float = 0.75
    memory_lambda_max: float = 0.05
    memory_warmup_start: int = 0
    memory_warmup_length: int = 200
    memory_kappa: float = 5.0
    memory_epsilon: float = 1e-6
    memory_veto_confidence: float = 0.3
    memory_veto_failure_rate: float = 0.7
    memory_min_quality: float = 0.10
    memory_high_quality: float = 0.20
    memory_max_correlation: float = 0.70
    memory_pool_capacity: int = 50
    memory_children_per_parent: int = 5

    def __post_init__(self) -> None:
        if self.max_nodes < 3:
            raise ValueError("max_nodes must be at least 3")
        if self.hypotheses_per_paper < 1 or self.factors_per_hypothesis < 1:
            raise ValueError("hypotheses_per_paper and factors_per_hypothesis must be >= 1")
        if self.max_repairs < 0:
            raise ValueError("max_repairs cannot be negative")
        if not 0 <= self.min_alignment <= 1:
            raise ValueError("min_alignment must be in [0, 1]")
        if self.avoid_frequent_subtrees < 0:
            raise ValueError("avoid_frequent_subtrees cannot be negative")
        if self.max_bond_leaves < 2:
            raise ValueError("max_bond_leaves must be at least 2")
        for name in ("max_zoo_share", "max_store_share", "max_member_share", "max_bond_overlap"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.memory_warmup_start < 0 or self.memory_warmup_length < 1:
            raise ValueError("need memory_warmup_start >= 0 and memory_warmup_length >= 1")
        if self.memory_lambda_max < 0 or self.memory_kappa <= 0 or self.memory_epsilon <= 0:
            raise ValueError("need memory_lambda_max >= 0, memory_kappa > 0, memory_epsilon > 0")
        if not 0 <= self.memory_min_quality <= self.memory_high_quality:
            raise ValueError("need 0 <= memory_min_quality <= memory_high_quality")
        if not 0 < self.memory_max_correlation <= 1:
            raise ValueError("memory_max_correlation must be in (0, 1]")
        if self.memory_pool_capacity < 1 or self.memory_children_per_parent < 1:
            raise ValueError("memory_pool_capacity and memory_children_per_parent must be >= 1")
        for name in ("memory_veto_confidence", "memory_veto_failure_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


def load_config(path: str | Path | None = None) -> FactorConfig:
    """Read the parameters file. Unknown keys raise, so a typo is not silently ignored."""
    with open(path or DEFAULT_CONFIG, "rb") as handle:
        values = tomllib.load(handle)
    unknown = set(values) - {f.name for f in fields(FactorConfig)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return FactorConfig(**values)


def make_client():
    """A real Anthropic client; the SDK finds the credentials.

    In order: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, an `ant auth login`
    profile, or workload identity federation (short-lived tokens in CI or a
    cloud). None of them belongs in a file in this repository.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('install the agents extra: pip install -e ".[agents]"') from exc
    return anthropic.Anthropic()


def call_tool(
    client,
    config: FactorConfig,
    *,
    system: str,
    user: str,
    tool: dict,
    usage: Counter | None = None,
) -> dict:
    """One model call that must answer through `tool`; returns the tool's input.

    With `usage`, adds this call's count and input and output tokens to it.
    """
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        system=f"{system}\n\nAnswer only by calling the {tool['name']} tool, exactly once.",
        tools=[{**tool, "strict": True}],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": user}],
    )
    if usage is not None:
        tokens = getattr(response, "usage", None)
        usage["calls"] += 1
        usage["input_tokens"] += getattr(tokens, "input_tokens", 0) or 0
        usage["output_tokens"] += getattr(tokens, "output_tokens", 0) or 0
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
            return dict(block.input)
    raise ValueError(f"the model did not call {tool['name']}")


def grammar_help() -> str:
    """The factor grammar in words, generated from the parser's own tables."""
    windowed = sorted(name for name, takes in FUNCS.items() if takes)
    plain = sorted(name for name, takes in FUNCS.items() if not takes)
    return (
        f"Fields: {', '.join(STOCK_FIELDS)} (per stock), and {', '.join(MARKET_FIELDS)} "
        "(the value-weighted market return, the same for every stock on a date; market "
        "volatility is ts_std(mkt_return, n)). Operators: + - * / and unary minus. "
        f"f(x, window): {', '.join(windowed)}. "
        f"f(x, y, window): {', '.join(sorted(BINARY_WINDOW_FUNCS))}. "
        f"f(x, exponent): {', '.join(sorted(BINARY_PARAMETER_FUNCS))}. "
        f"f(x): {', '.join(plain)}. "
        f"Windows are integer literals in [{WINDOW_MIN}, {WINDOW_MAX}]; exponents are "
        "numeric literals. No other identifiers exist, and every expression must use "
        "at least one per-stock field."
    )
