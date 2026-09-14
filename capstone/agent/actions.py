"""Structured actions the agent may emit, and the strict parser for Claude's text.

The agent loop asks Claude to reply with exactly one JSON object: either a tool
call or a final answer. This module defines those two shapes and turns the raw
result text from `claude_cli.run_claude` into one of them — or into an
`ActionParseError`.

Nothing is repaired. Markdown fences, prose around the object, trailing commas,
NaN, duplicate keys, extra fields and type coercion ("3" for 3) are all errors,
because a loop that quietly fixes a bad reply is a loop whose trial record no
longer says what the model actually produced. Parsing never executes a tool.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

M = TypeVar("M", bound=BaseModel)


class ActionParseError(ValueError):
    """Claude's text is not exactly one valid agent action.

    Carries the raw text so the caller can log what the model actually said.
    """

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class _Action(BaseModel):
    # strict: no "3" -> 3 coercion; forbid: unknown keys are errors, not dropped.
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ToolCallAction(_Action):
    """Ask the harness to run one named tool with keyword arguments."""

    type: Literal["tool_call"]
    tool: str = Field(min_length=1)
    arguments: dict[str, Any]


class FinalAnswerAction(_Action):
    """End the turn with a final answer."""

    type: Literal["final"]
    content: str


AgentAction = Annotated[ToolCallAction | FinalAnswerAction, Field(discriminator="type")]

#: The `type` values the parser accepts.
ACTION_TYPES = ("tool_call", "final")

_ADAPTER: TypeAdapter[ToolCallAction | FinalAnswerAction] = TypeAdapter(AgentAction)


def parse_action(raw: str) -> ToolCallAction | FinalAnswerAction:
    """Parse Claude's raw result text into exactly one agent action.

    Args:
        raw: the text returned by `run_claude`. It must be a single JSON object;
            whitespace around it is the only thing tolerated.

    Returns:
        A `ToolCallAction` or a `FinalAnswerAction`.

    Raises:
        ActionParseError: if the text is not strict JSON, is not an object, has
            a missing or unsupported `type`, or does not match that type's schema.
    """
    payload = _loads_object(raw, what="agent action")

    action_type = payload.get("type")
    if action_type is None:
        raise ActionParseError("agent action has no 'type' field", raw=raw)
    if action_type not in ACTION_TYPES:
        raise ActionParseError(
            f"unsupported agent action type {action_type!r}; expected one of {ACTION_TYPES}",
            raw=raw,
        )

    try:
        return _ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ActionParseError(f"invalid {action_type!r} action: {exc}", raw=raw) from exc


def parse_reply(raw: str, model: type[M]) -> M:
    """Parse Claude's raw result text into one instance of a structured reply model.

    The same no-repair contract as `parse_action`, for replies whose shape is not
    a tool call or final answer — e.g. a role that must return a list of alpha
    expressions. `model` should itself be strict (`extra="forbid"`, no coercion);
    this function adds the strict JSON layer in front of it.

    Raises:
        ActionParseError: if the text is not strict JSON, is not an object, or
            does not validate against `model`.
    """
    payload = _loads_object(raw, what=f"{model.__name__} reply")
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ActionParseError(f"invalid {model.__name__} reply: {exc}", raw=raw) from exc


def _loads_object(raw: str, *, what: str) -> dict[str, Any]:
    """Strict JSON -> dict, or an `ActionParseError` naming `what` was being parsed."""
    if not isinstance(raw, str):
        raise TypeError(f"raw must be str, got {type(raw).__name__}")

    try:
        payload = json.loads(
            raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except (json.JSONDecodeError, _StrictJSONError) as exc:
        raise ActionParseError(f"{what} is not valid JSON: {exc}", raw=raw) from exc

    if not isinstance(payload, dict):
        raise ActionParseError(
            f"{what} must be a JSON object, got {type(payload).__name__}", raw=raw
        )
    return payload


class _StrictJSONError(ValueError):
    """JSON the stdlib would accept but the action contract does not."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # The stdlib keeps the last duplicate silently; which "type" did the model mean?
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _StrictJSONError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise _StrictJSONError(f"{name} is not valid JSON")
