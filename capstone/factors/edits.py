"""A free stand-in for the Opus editor: random, grammar-valid edits of a requested kind.

`RandomEditor.drafts` has the same shape as `deepen.ModelEditor.drafts`, so
the deepen move runs unchanged with it. It makes a random edit of the asked
motif, reads the result back with `memory.edit_motif`, and retries (up to
`max_tries`) until the realized motif is the asked one. It costs nothing,
so checks of the search itself don't spend model calls.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from capstone.factors.deepen import Draft
from capstone.factors.memory import (
    NORMALIZATION_FUNCS,
    RANK_FUNCS,
    SHIFT_FUNCS,
    edit_motif,
    parameter_bin,
)
from capstone.factors.tree import (
    STOCK_FIELDS,
    BinOp,
    Call,
    Field_,
    Node,
    ParseError,
    Unary,
    identity_key,
    parse,
    unparse,
)

WINDOWS = (2, 3, 5, 10, 21, 42, 63, 126, 252)  # every parameter bin
SERIES_FUNCS = ("decay_linear", "ema", "ts_max", "ts_mean", "ts_min", "ts_std", "ts_sum")
NORMALIZERS = tuple(sorted(NORMALIZATION_FUNCS))
OPS = ("+", "-", "*", "/")


def _kids(node: Node) -> list[Node]:
    if isinstance(node, Unary):
        return [node.operand]
    if isinstance(node, BinOp):
        return [node.left, node.right]
    if isinstance(node, Call):
        return [node.arg] if node.arg2 is None else [node.arg, node.arg2]
    return []


def _rebuild(node: Node, kids: list[Node]) -> Node:
    if isinstance(node, Unary):
        return Unary(node.op, kids[0])
    if isinstance(node, BinOp):
        return BinOp(node.op, kids[0], kids[1])
    return Call(node.func, kids[0], node.window, kids[1] if len(kids) > 1 else None)


def _spots(node: Node, path=(), parent=None):
    """Every (path, node, parent), parents first."""
    yield path, node, parent
    for i, kid in enumerate(_kids(node)):
        yield from _spots(kid, (*path, i), node)


def _replace(node: Node, path, new: Node) -> Node:
    if not path:
        return new
    kids = _kids(node)
    kids[path[0]] = _replace(kids[path[0]], path[1:], new)
    return _rebuild(node, kids)


def _is(node, funcs) -> bool:
    return isinstance(node, Call) and node.func in funcs


class RandomEditor:
    """Random edits of the asked motif, reproducible from `seed`."""

    def __init__(self, seed: int = 0, max_tries: int = 20):
        self.rng = random.Random(seed)
        self.max_tries = max_tries

    def drafts(
        self, parent: Node, motif: str, n: int, lineage: Sequence[str], vetoed: Sequence[str]
    ) -> list[Draft]:
        return [self._draft(parent, motif) for _ in range(n)]

    def _draft(self, parent: Node, motif: str) -> Draft:
        fallback = None
        for _ in range(self.max_tries):
            child = self._propose(parent, motif)
            if child is None:
                break
            try:
                child = parse(unparse(child))
            except ParseError:
                continue
            if identity_key(child) == identity_key(parent):
                continue
            if edit_motif(parent, child) == motif:
                return Draft(unparse(child), f"random {motif}", child)
            fallback = child
        if fallback is None:
            return Draft("", f"no {motif} edit found", None, "no edit found")
        return Draft(unparse(fallback), f"random {motif} (realized differently)", fallback)

    # -- proposals ------------------------------------------------------------

    def _pick(self, items):
        return items[self.rng.randrange(len(items))]

    def _window(self, avoid_bin: int | None = None) -> int:
        return self._pick([w for w in WINDOWS if parameter_bin(w) != avoid_bin])

    def _term(self) -> Node:
        field = Field_(self._pick(STOCK_FIELDS))
        return (
            field
            if self.rng.random() < 0.5
            else Call(self._pick(SERIES_FUNCS), field, self._window())
        )

    def _propose(self, tree: Node, motif: str) -> Node | None:
        spots = list(_spots(tree))
        any_spot = self._pick(spots)
        if motif == "rank_switch":
            ranked = [(p, n) for p, n, _ in spots if _is(n, RANK_FUNCS)]
            free = [
                (p, n) for p, n, up in spots if not _is(n, RANK_FUNCS) and not _is(up, RANK_FUNCS)
            ]
            if ranked and (not free or self.rng.random() < 0.5):
                path, node = self._pick(ranked)
                if self.rng.random() < 0.5:
                    return _replace(tree, path, node.arg)  # remove
                if node.func == "rank":
                    return _replace(tree, path, Call("ts_rank", node.arg, self._window()))
                return _replace(tree, path, Call("rank", node.arg))
            if not free:
                return None
            path, node = self._pick(free)
            if self.rng.random() < 0.5:
                return _replace(tree, path, Call("rank", node))
            return _replace(tree, path, Call("ts_rank", node, self._window()))
        if motif == "interaction":
            path, node, _ = any_spot
            pair = (node, self._term()) if self.rng.random() < 0.5 else (self._term(), node)
            return _replace(tree, path, BinOp(self._pick(OPS), *pair))
        if motif == "window_rescale":
            windowed = [(p, n) for p, n, _ in spots if isinstance(n, Call) and n.window]
            if not windowed:
                return None
            path, node = self._pick(windowed)
            new = Call(node.func, node.arg, self._window(parameter_bin(node.window)), node.arg2)
            return _replace(tree, path, new)
        if motif == "operator_sub":
            subs = [
                (p, n)
                for p, n, _ in spots
                if isinstance(n, BinOp) or _is(n, (*SERIES_FUNCS, "ts_corr", "ts_cov"))
            ]
            if not subs:
                return None
            path, node = self._pick(subs)
            if isinstance(node, BinOp):
                op = self._pick([o for o in OPS if o != node.op])
                return _replace(tree, path, BinOp(op, node.left, node.right))
            if node.func in ("ts_corr", "ts_cov"):
                func = "ts_cov" if node.func == "ts_corr" else "ts_corr"
            else:
                func = self._pick([f for f in SERIES_FUNCS if f != node.func])
            return _replace(tree, path, Call(func, node.arg, node.window, node.arg2))
        if motif == "feature_swap":
            fields = [(p, n) for p, n, _ in spots if isinstance(n, Field_)]
            path, node = self._pick(fields)
            return _replace(
                tree, path, Field_(self._pick([f for f in STOCK_FIELDS if f != node.name]))
            )
        if motif == "nesting":
            nested = [(p, n) for p, n, _ in spots if _is(n, SERIES_FUNCS)]
            if nested and self.rng.random() < 0.25:
                path, node = self._pick(nested)
                return _replace(tree, path, node.arg)
            path, node, _ = any_spot
            return _replace(tree, path, Call(self._pick(SERIES_FUNCS), node, self._window()))
        if motif == "temporal_shift":
            shifted = [(p, n) for p, n, _ in spots if _is(n, SHIFT_FUNCS)]
            free = [
                (p, n) for p, n, up in spots if not _is(n, SHIFT_FUNCS) and not _is(up, SHIFT_FUNCS)
            ]
            if shifted and (not free or self.rng.random() < 0.4):
                path, node = self._pick(shifted)
                if self.rng.random() < 0.5:
                    return _replace(tree, path, node.arg)  # remove
                other = "delta" if node.func == "shift" else "shift"
                return _replace(tree, path, Call(other, node.arg, node.window))
            if not free:
                return None
            path, node = self._pick(free)
            func = self._pick(sorted(SHIFT_FUNCS))
            return _replace(tree, path, Call(func, node, self._window()))
        if motif == "normalization":
            normed = [
                (p, n) for p, n, _ in spots if _is(n, NORMALIZATION_FUNCS) or isinstance(n, Unary)
            ]
            if normed and self.rng.random() < 0.3:
                path, node = self._pick(normed)
                return _replace(tree, path, _kids(node)[0])
            path, node, _ = any_spot
            if self.rng.random() < 0.2:
                return _replace(tree, path, Unary("neg", node))
            func = self._pick(NORMALIZERS)
            return _replace(
                tree, path, Call(func, node, self._window() if func == "zscore" else None)
            )
        # other: drop a term, or rewrite a subexpression
        binops = [(p, n) for p, n, _ in spots if isinstance(n, BinOp)]
        if binops and self.rng.random() < 0.5:
            path, node = self._pick(binops)
            return _replace(tree, path, self._pick([node.left, node.right]))
        path, _, _ = any_spot
        return _replace(tree, path, BinOp(self._pick(OPS), self._term(), self._term()))
