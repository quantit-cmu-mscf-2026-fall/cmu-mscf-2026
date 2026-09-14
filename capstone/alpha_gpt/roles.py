"""The two LLM roles of the baseline — Quant Developer and Analyst — and how they are called.

Each role replies with exactly one strict JSON object (`SeedProposal` or
`AnalystReview`), parsed by `capstone.agent.actions.parse_reply` with no repair.
A reply that fails to parse, or that fails the role's `check` (an expression
that is not valid DSL, an alpha id not in the table), is sent back once more
with the problems listed, up to `max_attempts`. Every attempt — prompt, raw
reply, problems — is passed to `record`, so the run folder shows exactly what
the model said, including the replies that were rejected.

Rejected expressions are never evaluated, so they are not trials.

`claude_llm` is the production LLM: `claude -p` started in an empty temporary
directory with every tool disabled and a single turn, so the model can read
neither the repository nor its data, and can only answer from the prompt.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from capstone.agent.actions import ActionParseError, parse_reply
from capstone.agent.claude_cli import CLAUDE_EXECUTABLE, Runner, run_claude
from capstone.alpha_gpt.dsl import DEFAULT_LIMITS, DSLError, Limits, canonical, parse

#: A language model: prompt in, raw reply text out.
LLM = Callable[[str], str]

#: Flags that keep a role call away from the machine: no tools, nothing persisted.
ISOLATION_ARGS = ("--tools", "", "--no-session-persistence")

_MAX_ECHO = 2000


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class AlphaProposal(_Strict):
    expression: str = Field(min_length=1, max_length=512)
    rationale: str = Field(min_length=1, max_length=1000)


class SeedProposal(_Strict):
    type: Literal["seed_proposal"]
    alphas: list[AlphaProposal] = Field(min_length=1, max_length=20)


class AlphaDiagnosis(_Strict):
    alpha_id: str
    note: str


class AnalystReview(_Strict):
    type: Literal["analyst_review"]
    summary: str = Field(min_length=1, max_length=4000)
    diagnoses: list[AlphaDiagnosis]
    revised_idea: str = Field(min_length=1, max_length=2000)


M = TypeVar("M", bound=BaseModel)


class RoleFailedError(RuntimeError):
    """No attempt produced a reply that even parsed."""

    def __init__(self, message: str, *, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class RoleResult(Generic[M]):
    """The accepted reply, and any problems it still had when attempts ran out."""

    reply: M
    attempts: int
    problems: list[str] = field(default_factory=list)


def call_role(
    prompt: str,
    model: type[M],
    *,
    llm: LLM,
    role: str,
    check: Callable[[M], list[str]] | None = None,
    max_attempts: int = 3,
    record: Callable[[dict[str, Any]], None] | None = None,
) -> RoleResult[M]:
    """Ask `llm` for one `model` reply, re-asking with the problems listed.

    Returns as soon as a reply parses and passes `check`. If attempts run out,
    the last reply that parsed is returned with its remaining problems (the
    caller decides what part of it is usable); if none parsed, raises
    `RoleFailedError`. LLM transport errors (e.g. `ClaudeCLIError`) propagate.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    attempts: list[dict[str, Any]] = []
    last_parsed: tuple[M, list[str]] | None = None
    current = prompt
    for attempt in range(1, max_attempts + 1):
        raw = llm(current)
        reply: M | None = None
        try:
            reply = parse_reply(raw, model)
        except ActionParseError as exc:
            problems = [_clip(str(exc))]
        else:
            problems = check(reply) if check else []
            last_parsed = (reply, problems)

        entry = {
            "role": role,
            "attempt": attempt,
            "prompt_sha256": hashlib.sha256(current.encode("utf-8")).hexdigest(),
            "prompt": current,
            "raw": raw,
            "problems": problems,
        }
        attempts.append(entry)
        if record is not None:
            record(entry)

        if reply is not None and not problems:
            return RoleResult(reply=reply, attempts=attempt)
        current = _retry_prompt(prompt, raw, problems)

    if last_parsed is not None:
        reply, problems = last_parsed
        return RoleResult(reply=reply, attempts=max_attempts, problems=problems)
    raise RoleFailedError(
        f"{role}: no parseable reply in {max_attempts} attempts", attempts=attempts
    )


def check_seed_proposal(
    reply: SeedProposal,
    *,
    fields: Collection[str],
    groups: Collection[str],
    n_alphas: int,
    already_evaluated: Collection[str] = (),
    limits: Limits = DEFAULT_LIMITS,
) -> list[str]:
    """Problems with a Quant Developer reply; empty means every alpha is usable."""
    problems = []
    if len(reply.alphas) != n_alphas:
        problems.append(f"expected exactly {n_alphas} alphas, got {len(reply.alphas)}")
    seen: set[str] = set()
    for alpha in reply.alphas:
        try:
            key = canonical(parse(alpha.expression, fields=fields, groups=groups, limits=limits))
        except DSLError as exc:
            problems.append(f"invalid expression {alpha.expression!r}: {exc}")
            continue
        if key in seen:
            problems.append(f"{alpha.expression!r} duplicates another alpha in this reply")
        elif key in already_evaluated:
            problems.append(f"{alpha.expression!r} was already evaluated in an earlier round")
        seen.add(key)
    return problems


def check_analyst_review(reply: AnalystReview, *, known_ids: Collection[str]) -> list[str]:
    return [
        f"alpha_id {d.alpha_id!r} is not in the results table"
        for d in reply.diagnoses
        if d.alpha_id not in known_ids
    ]


def claude_llm(
    *,
    timeout: float | None = 300.0,
    model: str | None = None,
    executable: str = CLAUDE_EXECUTABLE,
    runner: Runner | None = None,
) -> LLM:
    """`claude -p`, isolated: empty temp cwd, no tools, one turn, no session saved.

    A reply that tries to use a tool ends the single turn unsuccessfully, which
    `run_claude` raises as `ClaudeCLIError` rather than returning.
    """
    extra_args = [*ISOLATION_ARGS, *(["--model", model] if model else [])]

    def call(prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="alpha_gpt_role_") as cwd:
            return run_claude(
                prompt,
                max_turns=1,
                timeout=timeout,
                executable=executable,
                extra_args=extra_args,
                cwd=cwd,
                runner=runner,
            )

    return call


def _retry_prompt(prompt: str, raw: str, problems: list[str]) -> str:
    listed = "\n".join(f"- {problem}" for problem in problems)
    return (
        f"{prompt}\n\n# Your previous reply was rejected\n{listed}\n\n"
        f"Previous reply (truncated to {_MAX_ECHO} characters):\n{raw[:_MAX_ECHO]}\n\n"
        "Reply again with only the corrected JSON object."
    )


def _clip(text: str, limit: int = 1000) -> str:
    return text if len(text) <= limit else text[:limit] + " ..."
