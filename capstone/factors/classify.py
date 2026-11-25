"""Deterministic classification of factor formulas, for distributions over what agents produce.

`classify` reads one tree and returns plain JSON-able values; nothing is fitted or
sampled. Design choices:

- primary_field_group: the group (`FIELD_GROUPS`) with the most field leaves in the
  tree; ties go to the first of `GROUP_PRIORITY` (fundamental, size, volume, returns,
  price, market), so the rarer, more specific input names the factor.
- horizon: `effective_lookback` is the largest, over root-to-leaf paths, of the sum of
  the window lengths on that path, so `shift(ts_sum(returns, 231), 21)` looks back 252
  days and nested windows add. `horizon_bin` is `parameter_bin` of it (0: <=5, 1: 6-21,
  2: 22-63, 3: 64-252, 4: >252); a factor with no window has lookback 0, bin 0.
  `total_lag` is the same sum over `shift` windows alone.
- construction: which operator groups appear (rank, normalization, time_series,
  shift, ratio, regime).
- negated: the sign of the whole expression, from negations and negative constant
  factors met at the root, looking through rank/scale/zscore/winsorize.
- terms and archetype: see `_terms`; each term is matched against the rules in
  `config/factor_archetypes.toml` (DRAFT), first match wins.
"""

from __future__ import annotations

import tomllib
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import pandas as pd

from capstone.factors.memory import (
    FIELD_GROUPS,
    NESTING_FUNCS,
    NORMALIZATION_FUNCS,
    RANK_FUNCS,
    SHIFT_FUNCS,
    parameter_bin,
)
from capstone.factors.tree import (
    REGIME_FUNCS,
    BinOp,
    Call,
    Const,
    Field_,
    Node,
    Unary,
    features_used,
    funcs_used,
    node_count,
    param_count,
    parse,
    unparse,
    uses_regime,
    walk,
    zoo_similarity,
)
from capstone.factors.zoo import default_zoo

ARCHETYPES_PATH = Path(__file__).resolve().parents[2] / "config" / "factor_archetypes.toml"
GROUP_PRIORITY = ("fundamental", "volume", "size", "returns", "price", "market")
_RULE_KEYS = {
    "archetype",
    "primary_groups",
    "groups_all",
    "funcs_any",
    "horizon_bins",
    "negated",
    "has_window",
}
# Wrappers that keep the order of values, so the decomposition looks through them and
# the term is the expression inside (log is taken as monotone increasing).
_SIGN_TRANSPARENT = frozenset(
    {"rank", "scale", "zscore", "winsorize", "log", "volscale", "bear_gate"}
)
# Linear operators with a window: `_negated` reads through them for a term's sign
# (`shift(-x, 5)` is the negation of `shift(x, 5)`), but they stay inside the term.
_SIGN_LINEAR = frozenset({"shift", "delta", "ts_mean", "ts_sum", "ema", "decay_linear"})
_VOLATILITY_FUNCS = frozenset({"ts_std", "ts_cov"})


def load_archetype_rules(path: str | Path = ARCHETYPES_PATH) -> list[dict]:
    """The archetype rules in order. Unknown sections, keys and groups raise."""
    with open(path, "rb") as handle:
        values = tomllib.load(handle)
    if set(values) - {"rule"}:
        raise ValueError(f"unknown sections in {path}: {sorted(set(values) - {'rule'})}")
    rules = values.get("rule", [])
    for rule in rules:
        unknown = set(rule) - _RULE_KEYS
        if unknown:
            raise ValueError(f"unknown keys in [[rule]] of {path}: {sorted(unknown)}")
        if "archetype" not in rule:
            raise ValueError(f"[[rule]] without an archetype in {path}")
        for key in ("primary_groups", "groups_all"):
            bad = set(rule.get(key, [])) - set(GROUP_PRIORITY)
            if bad:
                raise ValueError(f"unknown field groups in {key} of {path}: {sorted(bad)}")
    return rules


@lru_cache(maxsize=1)
def _default_rules() -> tuple[dict, ...]:
    return tuple(load_archetype_rules())


@lru_cache(maxsize=1)
def _zoo():
    return default_zoo()


def _children(node: Node) -> list[Node]:
    if isinstance(node, Unary):
        return [node.operand]
    if isinstance(node, BinOp):
        return [node.left, node.right]
    if isinstance(node, Call):
        return [node.arg] if node.arg2 is None else [node.arg, node.arg2]
    return []


def _path_sum(node: Node, funcs: frozenset[str] | None) -> int:
    """Largest sum of window lengths (of `funcs`, or all) along a root-to-leaf path."""
    own = 0
    if isinstance(node, Call) and node.window and (funcs is None or node.func in funcs):
        own = node.window
    return own + max((_path_sum(k, funcs) for k in _children(node)), default=0)


def _negated(node: Node) -> bool:
    flip = False
    while True:
        if isinstance(node, Unary):
            flip, node = not flip, node.operand
        elif isinstance(node, Call) and node.func in _SIGN_TRANSPARENT | _SIGN_LINEAR:
            node = node.arg
        elif isinstance(node, BinOp) and node.op in "*/":
            if isinstance(node.right, Const):
                const, other = node.right, node.left
            elif node.op == "*" and isinstance(node.left, Const):
                const, other = node.left, node.right
            else:
                return flip
            flip ^= const.value < 0
            node = other
        else:
            return flip


def _primary_group(node: Node) -> str:
    counts = Counter(FIELD_GROUPS[n.name] for n in walk(node) if isinstance(n, Field_))
    return max(counts, key=lambda g: (counts[g], -GROUP_PRIORITY.index(g)))


def _construction(node: Node) -> list[str]:
    funcs = funcs_used(node)
    groups = {
        "rank": funcs & RANK_FUNCS,
        "normalization": funcs & NORMALIZATION_FUNCS,
        "time_series": funcs & NESTING_FUNCS,
        "shift": funcs & SHIFT_FUNCS,
        "ratio": funcs & {"/"},
        "regime": funcs & set(REGIME_FUNCS),
    }
    return sorted(name for name, hit in groups.items() if hit)


def _matches(rule: dict, info: dict) -> bool:
    if "primary_groups" in rule and info["primary_field_group"] not in rule["primary_groups"]:
        return False
    if not set(rule.get("groups_all", [])) <= set(info["field_groups"]):
        return False
    if "funcs_any" in rule and not set(rule["funcs_any"]) & set(info["funcs"]):
        return False
    if "horizon_bins" in rule and info["horizon_bin"] not in rule["horizon_bins"]:
        return False
    if "has_window" in rule and (info["effective_lookback"] > 0) != rule["has_window"]:
        return False
    return not ("negated" in rule and info["negated"] != rule["negated"])


def _has_volatility(node: Node) -> bool:
    return any(isinstance(n, Call) and n.func in _VOLATILITY_FUNCS for n in walk(node))


def _const_value(node: Node) -> float | None:
    """The value of a constant, however its sign is written (`-3` parses as a negation)."""
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Unary) and (inner := _const_value(node.operand)) is not None:
        return -inner
    return None


def _terms(node: Node, sign: int, risk: bool, out: list) -> bool:
    """Split `node` into (term, sign, risk_adjusted) entries appended to `out`.

    Looks through, carrying the sign: `+`; `-` (the right side flips); unary
    negation; multiplication or division by a constant (a negative one flips); `*`
    between two non-constants (an interaction: each factor is a term); the wrappers
    in `_SIGN_TRANSPARENT`; `signed_power` with a positive exponent. A constant is
    not a term, so `rank(x) - 0.5` is `rank(x)` and `0.5 - rank(x)` is its negation.
    A ratio whose denominator contains ts_std or ts_cov is risk adjusted: its
    numerator is split and flagged. Any other ratio and any other call is one term
    (ts_mean and the like are not looked through). Returns whether a multiplicative
    split happened.
    """
    if _const_value(node) is not None:
        return False
    if isinstance(node, Unary):
        return _terms(node.operand, -sign, risk, out)
    if isinstance(node, BinOp):
        if node.op in "+-":
            right = -sign if node.op == "-" else sign
            left_split = _terms(node.left, sign, risk, out)
            return _terms(node.right, right, risk, out) or left_split
        if node.op == "*":
            for const, other in ((node.left, node.right), (node.right, node.left)):
                if (value := _const_value(const)) is not None:
                    return _terms(other, sign * (-1 if value < 0 else 1), risk, out)
            _terms(node.left, sign, risk, out)
            _terms(node.right, sign, risk, out)
            return True
        if (value := _const_value(node.right)) is not None:
            return _terms(node.left, sign * (-1 if value < 0 else 1), risk, out)
        if _has_volatility(node.right):
            return _terms(node.left, sign, True, out)
    elif isinstance(node, Call) and (
        node.func in _SIGN_TRANSPARENT
        or (node.func == "signed_power" and isinstance(node.arg2, Const) and node.arg2.value > 0)
    ):
        return _terms(node.arg, sign, risk, out)
    out.append((node, sign, risk))
    return False


def _describe(node: Node, sign: int, rules) -> dict:
    """Field group, horizon, sign and archetype of one term."""
    lookback = _path_sum(node, None)
    negated = sign * (-1 if _negated(node) else 1) < 0
    info = {
        "field_groups": sorted({FIELD_GROUPS[f] for f in features_used(node)}),
        "primary_field_group": _primary_group(node),
        "effective_lookback": lookback,
        "horizon_bin": parameter_bin(lookback),
        "negated": negated,
        "funcs": sorted(funcs_used(node)),
    }
    archetype = next((r["archetype"] for r in rules if _matches(r, info)), "other")
    return {
        "expr": unparse(node),
        "sign": -1 if negated else 1,
        "primary_field_group": info["primary_field_group"],
        "horizon_bin": info["horizon_bin"],
        "archetype": archetype,
    }


def classify(node: Node, rules: list[dict] | None = None) -> dict:
    """Describe one factor tree with plain JSON-able values.

    The factor is split into terms (`_terms`); each is classified by the rules, and
    `archetype` is their one distinct archetype, or "combination" if they differ.
    """
    rules = rules if rules is not None else _default_rules()
    fields = sorted(features_used(node))
    windows = sorted(n.window for n in walk(node) if isinstance(n, Call) and n.window)
    lookback = _path_sum(node, None)
    shared, share, nearest = zoo_similarity(node, _zoo())
    raw: list = []
    interaction = _terms(node, 1, False, raw)
    terms = [{**_describe(t, s, rules), "risk_adjusted": r} for t, s, r in raw]
    distinct = sorted({t["archetype"] for t in terms})
    return {
        "fields": fields,
        "field_groups": sorted({FIELD_GROUPS[f] for f in fields}),
        "primary_field_group": _primary_group(node),
        "windows": windows,
        "max_window": max(windows, default=0),
        "total_lag": _path_sum(node, frozenset({"shift"})),
        "effective_lookback": lookback,
        "horizon_bin": parameter_bin(lookback),
        "construction": _construction(node),
        "negated": _negated(node),
        "node_count": node_count(node),
        "param_count": param_count(node),
        "uses_regime": uses_regime(node),
        "nearest_zoo": nearest,
        "zoo_shared": shared,
        "zoo_share": share,
        "terms": terms,
        "archetypes": distinct,
        "n_terms": len(terms),
        "interaction": interaction,
        "archetype": distinct[0] if len(distinct) == 1 else "combination",
    }


def classify_expressions(exprs: Mapping[str, str]) -> pd.DataFrame:
    """One row per expression id; regime operators are accepted."""
    rows = {key: classify(parse(text, regime_operators=True)) for key, text in exprs.items()}
    return pd.DataFrame.from_dict(rows, orient="index")
