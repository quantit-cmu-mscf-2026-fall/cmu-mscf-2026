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
- archetype: first matching rule in `config/factor_archetypes.toml` (DRAFT).
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
    uses_regime,
    walk,
    zoo_similarity,
)
from capstone.factors.zoo import default_zoo

ARCHETYPES_PATH = Path(__file__).resolve().parents[2] / "config" / "factor_archetypes.toml"
GROUP_PRIORITY = ("fundamental", "size", "volume", "returns", "price", "market")
_RULE_KEYS = {"archetype", "primary_groups", "groups_all", "funcs_any", "horizon_bins", "negated"}
_SIGN_TRANSPARENT = frozenset({"rank", "scale", "zscore", "winsorize"})


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
        elif isinstance(node, Call) and node.func in _SIGN_TRANSPARENT:
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
    return not ("negated" in rule and info["negated"] != rule["negated"])


def classify(node: Node, rules: list[dict] | None = None) -> dict:
    """Describe one factor tree with plain JSON-able values."""
    fields = sorted(features_used(node))
    windows = sorted(n.window for n in walk(node) if isinstance(n, Call) and n.window)
    lookback = _path_sum(node, None)
    shared, share, nearest = zoo_similarity(node, _zoo())
    out = {
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
    }
    info = {**out, "funcs": sorted(funcs_used(node))}
    out["archetype"] = "other"
    for rule in rules if rules is not None else _default_rules():
        if _matches(rule, info):
            out["archetype"] = rule["archetype"]
            break
    return out


def classify_expressions(exprs: Mapping[str, str]) -> pd.DataFrame:
    """One row per expression id; regime operators are accepted."""
    rows = {key: classify(parse(text, regime_operators=True)) for key, text in exprs.items()}
    return pd.DataFrame.from_dict(rows, orient="index")
