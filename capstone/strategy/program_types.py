"""Versioned unit/entity checks over the existing AST; no evaluation or data qualification.

The schema assumes actual USD and shares, including accounting fields converted from
vendor millions/thousands. A qualified input stage must certify that normalization.
Policy v1 limits signed-power exponents to absolute value 16; dimensionful bases need
integer exponents. Unsupported constructions remain type blockers, not economic verdicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType

from capstone.factors import tree

SCHEMA_VERSION = "normalized-equity-units-v1"
POLICY_VERSION = "equity-unit-check-v1"
MAX_POWER = 16
MAX_UNIT_POWER = 10_000


class UnitTypeError(ValueError):
    """Retain the proposed idea; its construction needs an explicit supported type rule."""


@dataclass(frozen=True)
class Units:
    usd: int = 0
    shares: int = 0
    sessions: int = 0

    def __post_init__(self):
        if any(type(value) is not int or abs(value) > MAX_UNIT_POWER for value in self.powers):
            raise UnitTypeError("unit exponent exceeds the versioned type policy")

    @property
    def powers(self):
        return self.usd, self.shares, self.sessions

    def combine(self, other, sign=1):
        return Units(*(a + sign * b for a, b in zip(self.powers, other.powers, strict=True)))

    def power(self, exponent):
        return Units(*(value * exponent for value in self.powers))

    def __str__(self):
        return (
            " * ".join(
                f"{name}^{power}"
                for name, power in zip(("USD", "shares", "sessions"), self.powers, strict=True)
                if power
            )
            or "dimensionless"
        )


@dataclass(frozen=True)
class ValueType:
    units: Units = Units()
    entity: str = "scalar"

    def __post_init__(self):
        if not isinstance(self.units, Units) or self.entity not in ("scalar", "market", "equity"):
            raise UnitTypeError("unknown unit or entity type")


NORMALIZED_EQUITY_INPUTS = MappingProxyType(
    {
        **{name: ValueType(Units(1, -1), "equity") for name in ("open", "high", "low", "close")},
        **{name: ValueType(Units(shares=1), "equity") for name in ("volume", "shares")},
        **{
            name: ValueType(Units(usd=1), "equity")
            for name in ("cap", "book", "earnings", "sales", "assets")
        },
        "returns": ValueType(Units(), "equity"),
        "mkt_return": ValueType(Units(), "market"),
    }
)


@dataclass(frozen=True)
class TypeEvidence:
    expression_id: str
    output_type: ValueType
    dependencies: tuple[tuple[str, ValueType], ...]
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    policy_version: str = field(default=POLICY_VERSION, init=False)
    normalization_required: bool = field(default=True, init=False)


def _entity(left, right):
    return next(
        entity for entity in ("equity", "market", "scalar") if entity in (left.entity, right.entity)
    )


def _number(node):
    if (
        not isinstance(node, tree.Const)
        or type(node.value) not in (int, float)
        or not math.isfinite(node.value)
    ):
        raise UnitTypeError("finite literal numeric constant required")
    return node.value


def _zero(node):
    return (isinstance(node, tree.Const) and _number(node) == 0) or (
        isinstance(node, tree.Unary) and node.op == "neg" and _zero(node.operand)
    )


def infer_types(node: tree.Node) -> TypeEvidence:
    """Check a closed tree without rewriting it; evidence assumes normalized input units."""
    dependencies = {}
    visited = 0

    def visit(current):
        nonlocal visited
        visited += 1
        if visited > tree.MAX_EXPRESSION_NODES:
            raise UnitTypeError("type input exceeds the existing AST node cap")
        if isinstance(current, tree.Field_):
            if current.name not in NORMALIZED_EQUITY_INPUTS:
                raise UnitTypeError("field has no versioned normalized unit type")
            dependencies[current.name] = NORMALIZED_EQUITY_INPUTS[current.name]
            return dependencies[current.name]
        if isinstance(current, tree.Const):
            _number(current)
            return ValueType()
        if isinstance(current, tree.Unary) and current.op == "neg":
            return visit(current.operand)
        if isinstance(current, tree.BinOp):
            left, right = visit(current.left), visit(current.right)
            if current.op in ("+", "-"):
                if left.units != right.units and not (_zero(current.left) or _zero(current.right)):
                    raise UnitTypeError(
                        f"incompatible additive units: {left.units} and {right.units}"
                    )
                units = right.units if _zero(current.left) else left.units
            elif current.op in ("*", "/"):
                units = left.units.combine(right.units, 1 if current.op == "*" else -1)
            else:
                raise UnitTypeError("unsupported binary AST operator")
            return ValueType(units, _entity(left, right))
        if not isinstance(current, tree.Call):
            raise UnitTypeError("unsupported AST node")
        func = current.func
        if func not in {*tree.FUNCS, *tree.BINARY_WINDOW_FUNCS, *tree.BINARY_PARAMETER_FUNCS}:
            raise UnitTypeError("unsupported function type rule")
        windowed = func in tree.BINARY_WINDOW_FUNCS or tree.FUNCS.get(func, False)
        if windowed and (
            type(current.window) is not int
            or not tree.WINDOW_MIN <= current.window <= tree.WINDOW_MAX
        ):
            raise UnitTypeError("invalid session window")
        if not windowed and current.window is not None:
            raise UnitTypeError("unexpected session window")
        value = visit(current.arg)
        if func in tree.BINARY_WINDOW_FUNCS:
            other = visit(current.arg2)
            if "scalar" in (value.entity, other.entity):
                raise UnitTypeError("rolling pair requires indexed inputs")
            units = value.units.combine(other.units) if func == "ts_cov" else Units()
            return ValueType(units, _entity(value, other))
        if func == "signed_power":
            exponent = _number(current.arg2)
            visit(current.arg2)
            if abs(exponent) > MAX_POWER or (
                value.units != Units() and not float(exponent).is_integer()
            ):
                raise UnitTypeError("exponent is outside the versioned dimensional power policy")
            units = value.units.power(int(exponent)) if value.units != Units() else Units()
            return ValueType(units, value.entity)
        if current.arg2 is not None:
            raise UnitTypeError("unexpected second function argument")
        if value.entity == "scalar" and func != "sign":
            raise UnitTypeError("function requires an indexed input")
        if func in ("rank", "scale", "winsorize") and value.entity != "equity":
            raise UnitTypeError("cross-sectional function requires an equity entity")
        if func == "log" and value.units != Units():
            raise UnitTypeError("log requires a dimensionless input")
        if func in ("rank", "scale", "sign", "zscore", "ts_rank", "log"):
            return ValueType(Units(), value.entity)
        if func in ("ts_argmax", "ts_argmin"):
            return ValueType(Units(sessions=1), value.entity)
        if func == "ts_product":
            return ValueType(value.units.power(current.window), value.entity)
        if func in ("shift", "delta", "abs", "winsorize") or func in (
            "ts_mean",
            "ts_std",
            "ts_min",
            "ts_max",
            "ts_sum",
            "decay_linear",
            "ema",
        ):
            return value
        raise UnitTypeError("function has no versioned unit rule")

    result = visit(node)
    return TypeEvidence(tree.factor_id(node), result, tuple(sorted(dependencies.items())))
