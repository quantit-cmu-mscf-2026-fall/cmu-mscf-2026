"""The alpha expression language: parse, validate, canonicalise, evaluate.

An alpha is a function-call expression such as::

    normed_rank(neg(ts_delta(close, 5)))

The LLM writes these as text. This module is the only way text becomes a
computation, and it never uses `eval`: the text is parsed by Python's `ast`
module and every node is checked against a closed whitelist — calls to
operators in `OPERATORS`, field and group names the panel actually has, and
numeric literals. Windows and lags must be integer literals within bounds; a
lag cannot be negative, so an expression cannot reference the future.

Parsing returns a small immutable tree. Arguments of commutative operators are
sorted, so two spellings of the same alpha produce equal trees and the same
`canonical` string — the key used to count an alpha as one trial, not two.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd

from capstone.alpha_gpt.operators import OPERATORS, ArgKind
from capstone.alpha_gpt.panel import OHLCVPanel


class DSLError(ValueError):
    """The expression is not a valid alpha. The message says which part and why."""

    def __init__(self, message: str, *, expression: str) -> None:
        super().__init__(message)
        self.expression = expression


@dataclass(frozen=True)
class Limits:
    max_chars: int = 512
    max_depth: int = 8
    max_nodes: int = 40
    min_window: int = 2
    max_window: int = 126
    max_lag: int = 21
    max_abs_const: float = 1e6


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True)
class FieldRef:
    name: str


@dataclass(frozen=True)
class GroupRef:
    name: str


@dataclass(frozen=True)
class Const:
    value: int | float


@dataclass(frozen=True)
class Call:
    op: str
    args: tuple[Node, ...]


Node = FieldRef | GroupRef | Const | Call


def parse(
    expression: str,
    *,
    fields: Collection[str],
    groups: Collection[str] = (),
    limits: Limits = DEFAULT_LIMITS,
) -> Node:
    """Parse and validate `expression` into a canonical tree.

    Args:
        expression: the alpha text.
        fields: field names the expression may reference (e.g. panel.fields).
        groups: group names usable as the group argument of grouped_* operators.
        limits: size and range bounds.

    Raises:
        DSLError: for anything that is not a valid alpha under these fields,
            groups and limits.
    """
    if not isinstance(expression, str):
        raise TypeError(f"expression must be str, got {type(expression).__name__}")
    text = expression.strip()
    if not text:
        raise DSLError("empty expression", expression=expression)
    if len(text) > limits.max_chars:
        raise DSLError(
            f"expression is longer than {limits.max_chars} characters", expression=expression
        )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise DSLError(f"not a valid expression: {exc.msg}", expression=expression) from exc

    converter = _Converter(expression, frozenset(fields), frozenset(groups), limits)
    node = converter.convert(tree.body, "series")

    if depth(node) > limits.max_depth:
        raise DSLError(
            f"expression nesting depth {depth(node)} exceeds {limits.max_depth}",
            expression=expression,
        )
    if size(node) > limits.max_nodes:
        raise DSLError(
            f"expression has {size(node)} nodes, more than {limits.max_nodes}",
            expression=expression,
        )
    return node


def canonical(node: Node) -> str:
    """The canonical text of a tree; `parse(canonical(n))` returns `n` again."""
    if isinstance(node, FieldRef | GroupRef):
        return node.name
    if isinstance(node, Const):
        return repr(node.value)
    return f"{node.op}({', '.join(canonical(arg) for arg in node.args)})"


def depth(node: Node) -> int:
    if isinstance(node, Call):
        return 1 + max(depth(arg) for arg in node.args)
    return 1


def size(node: Node) -> int:
    if isinstance(node, Call):
        return 1 + sum(size(arg) for arg in node.args)
    return 1


def evaluate(node: Node, panel: OHLCVPanel) -> pd.DataFrame:
    """Compute a parsed alpha on `panel`, as a date x asset DataFrame.

    Shared subexpressions are computed once per call.
    """
    cache: dict[str, pd.DataFrame] = {}
    result = _evaluate(node, panel, cache)
    if not isinstance(result, pd.DataFrame):
        raise TypeError("an alpha must evaluate to a DataFrame; parse() guarantees this")
    return result


def _evaluate(node: Node, panel: OHLCVPanel, cache: dict[str, pd.DataFrame]):
    if isinstance(node, FieldRef):
        return panel.fields[node.name]
    if isinstance(node, GroupRef):
        return panel.groups[node.name]
    if isinstance(node, Const):
        return node.value
    key = canonical(node)
    if key not in cache:
        args = [_evaluate(arg, panel, cache) for arg in node.args]
        cache[key] = OPERATORS[node.op](*args)
    return cache[key]


_REJECTED_SYNTAX = {
    ast.BinOp: "infix arithmetic is not allowed; use add, minus, cwise_mul or div",
    ast.Compare: "comparisons are not allowed; use greater or less",
    ast.BoolOp: "and/or are not allowed",
    ast.Attribute: "attribute access is not allowed",
    ast.Subscript: "indexing is not allowed",
}


class _Converter:
    def __init__(
        self, expression: str, fields: frozenset[str], groups: frozenset[str], limits: Limits
    ) -> None:
        self.expression = expression
        self.fields = fields
        self.groups = groups
        self.limits = limits

    def fail(self, message: str) -> DSLError:
        return DSLError(message, expression=self.expression)

    def convert(self, node: ast.expr, kind: ArgKind) -> Node:
        if kind in ("window", "lag"):
            return self._integer(node, kind)
        if kind == "group":
            return self._group(node)
        # series or value
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Name):
            return self._field(node)
        if _is_numeric_literal(node):
            if kind == "series":
                raise self.fail(
                    "a numeric literal is not allowed here; expected a field or operator call"
                )
            return self._const(node)
        raise self.fail(self._describe(node))

    def _call(self, node: ast.Call) -> Call:
        if not isinstance(node.func, ast.Name):
            raise self.fail("only direct operator calls like ts_mean(close, 5) are allowed")
        name = node.func.id
        spec = OPERATORS.get(name)
        if spec is None:
            raise self.fail(f"unknown operator {name!r}")
        if node.keywords:
            raise self.fail(f"keyword arguments are not allowed (in {name})")
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            raise self.fail(f"starred arguments are not allowed (in {name})")
        if len(node.args) != len(spec.args):
            raise self.fail(
                f"{name} takes {len(spec.args)} arguments {spec.signature}, got {len(node.args)}"
            )

        args = tuple(
            self.convert(arg, kind) for arg, kind in zip(node.args, spec.args, strict=True)
        )
        data_args = [a for a, k in zip(args, spec.args, strict=True) if k in ("series", "value")]
        if data_args and all(isinstance(a, Const) for a in data_args):
            raise self.fail(f"{name} needs at least one field or operator call among its inputs")
        if spec.commutative:
            args = tuple(sorted(args, key=canonical))
        return Call(name, args)

    def _field(self, node: ast.Name) -> FieldRef:
        if node.id in self.fields:
            return FieldRef(node.id)
        if node.id in self.groups:
            raise self.fail(
                f"group {node.id!r} can only be used as the group argument of grouped_* operators"
            )
        raise self.fail(f"unknown field {node.id!r}; available: {sorted(self.fields)}")

    def _group(self, node: ast.expr) -> GroupRef:
        if not self.groups:
            raise self.fail("this panel has no groups, so grouped_* operators are unavailable")
        if isinstance(node, ast.Name) and node.id in self.groups:
            return GroupRef(node.id)
        raise self.fail(f"expected a group name, one of {sorted(self.groups)}")

    def _integer(self, node: ast.expr, kind: ArgKind) -> Const:
        low, high = (
            (self.limits.min_window, self.limits.max_window)
            if kind == "window"
            else (1, self.limits.max_lag)
        )
        message = f"{kind} must be an integer literal in [{low}, {high}]"
        value = _literal_value(node) if _is_numeric_literal(node) else None
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            raise self.fail("True/False are not allowed")
        if not isinstance(value, int) or not low <= value <= high:
            if kind == "lag":
                message += "; negative lags would read the future"
            raise self.fail(message)
        return Const(value)

    def _const(self, node: ast.expr) -> Const:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            raise self.fail("True/False are not allowed")
        value = _literal_value(node)
        if not math.isfinite(value):
            raise self.fail("numeric literals must be finite")
        if abs(value) > self.limits.max_abs_const:
            raise self.fail(f"numeric literal magnitude exceeds {self.limits.max_abs_const:g}")
        return Const(value)

    def _describe(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return "True/False are not allowed"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return "strings are not allowed"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return "unary minus is only allowed on numeric literals; use neg(x)"
        for syntax, message in _REJECTED_SYNTAX.items():
            if isinstance(node, syntax):
                return message
        return f"{type(node).__name__} is not allowed"


def _is_numeric_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    )


def _literal_value(node: ast.expr) -> int | float:
    if isinstance(node, ast.UnaryOp):
        return -node.operand.value
    return node.value
