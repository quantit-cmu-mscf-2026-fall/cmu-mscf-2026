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
    FIELDS,
    FUNCS,
    WINDOW_MAX,
    WINDOW_MIN,
)

EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "factors.toml"


@dataclass(frozen=True)
class FactorConfig:
    """The pipeline's parameters; see config/factors.toml for what each one does."""

    model: str = "claude-opus-5-5"
    effort: str = "high"
    max_tokens: int = 16000
    prompt_version: str = "v4"
    hypotheses_per_paper: int = 3
    factors_per_hypothesis: int = 4
    max_repairs: int = 2
    max_zoo_share: float = 0.75
    max_store_share: float = 1.0
    max_nodes: int = 30

    def __post_init__(self) -> None:
        if self.max_nodes < 3:
            raise ValueError("max_nodes must be at least 3")
        if self.hypotheses_per_paper < 1 or self.factors_per_hypothesis < 1:
            raise ValueError("hypotheses_per_paper and factors_per_hypothesis must be >= 1")
        if self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}")
        if self.max_repairs < 0:
            raise ValueError("max_repairs cannot be negative")
        for name in ("max_zoo_share", "max_store_share"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


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
    Raises on a refusal or an answer from a model other than `config.model`.
    """
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        system=f"{system}\n\nAnswer only by calling the {tool['name']} tool, exactly once.",
        tools=[{**tool, "strict": True}],
        tool_choice={"type": "auto"},
        output_config={"effort": config.effort},
        messages=[{"role": "user", "content": user}],
    )
    # Every stored row records config.model, and the holdout dates depend on
    # that model's training cutoff, so an answer from any other model (or a
    # refusal) must stop the run rather than be stored under the wrong name.
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise ValueError(f"the model declined ({getattr(details, 'category', None)})")
    served = getattr(response, "model", None)
    if served is not None and served != config.model:
        raise ValueError(f"answered by {served}, not the configured {config.model}")
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
        f"Fields: {', '.join(FIELDS)}. Operators: + - * / and unary minus. "
        f"f(x, window): {', '.join(windowed)}. "
        f"f(x, y, window): {', '.join(sorted(BINARY_WINDOW_FUNCS))}. "
        f"f(x, exponent): {', '.join(sorted(BINARY_PARAMETER_FUNCS))}. "
        f"f(x): {', '.join(plain)}. "
        f"Windows are integer literals in [{WINDOW_MIN}, {WINDOW_MAX}]; exponents are "
        "numeric literals. No other identifiers exist, and every expression must use "
        "at least one field."
    )
