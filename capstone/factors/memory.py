"""Learned memory search: what kind of parent and edit, and what each edit did.

Adapted from AlphaMemo (Yu, Zheng, Pan, Liu, Wang & He 2026, arXiv
2606.20625). When the agent deepens a factor (refines it), learned memory
search remembers, for each kind of parent and each kind of edit, whether the
edit did better than expected.

- `parent_context` is the paper's z(p) (Eq. 2): the field groups the parent
  reads, and bins of its quality, size and how often it has been selected.
- `edit_motif` is the paper's m(p, c) (Eq. 9-10): one of the ten `MOTIFS`,
  read off the parent and child trees.
- `record_event` appends each edit to the store's `memory_events` table;
  `MemoryState` reduces the events to the Eq. 6 statistics per (context,
  motif): Welford mean and variance of the residuals, and a Beta posterior
  over failures.

Nothing here reads market data: qualities come from the scorer.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from capstone.factors import store
from capstone.factors.tree import (
    WINDOW_MAX,
    BinOp,
    Call,
    Const,
    Field_,
    Node,
    Unary,
    features_used,
    identity_key,
    node_count,
    structural_key,
)

# ---------------------------------------------------------------------------
# Parent context z(p) = (g, b_q, b_d, b_u)

FIELD_GROUPS = {
    "open": "price",
    "high": "price",
    "low": "price",
    "close": "price",
    "volume": "volume",
    "shares": "volume",
    "returns": "returns",
    "cap": "size",
    "mkt_return": "market",
}
# q0..q3: Q < 0.05, 0.05 <= Q < 0.10, 0.10 <= Q < 0.20, Q >= 0.20.
QUALITY_EDGES = (0.05, 0.10, 0.20)
# d0..d3: node count <= 5, 6-10, 11-20, > 20 (the paper's AST size).
SIZE_EDGES = (5, 10, 20)
# u0..u3: selected 0 times, 1-2, 3-5, 6 or more (the paper's retrieval frequency).
USAGE_EDGES = (0, 2, 5)


def _above(value: float, edges: tuple) -> int:
    return sum(value > edge for edge in edges)


def parent_context(parent: Node, *, quality: float | None, times_selected: int) -> str:
    """The parent's context key, such as ``returns+size|q2|d1|u0``.

    `quality` is the parent's |ICIR| on the search period, or None (or NaN)
    if it has not been scored, which gives the quality bin ``unscored``.
    """
    groups = "+".join(sorted({FIELD_GROUPS[f] for f in features_used(parent)}))
    if quality is None or math.isnan(quality):
        band = "unscored"
    else:
        band = f"q{sum(abs(quality) >= edge for edge in QUALITY_EDGES)}"
    size = f"d{_above(node_count(parent), SIZE_EDGES)}"
    return f"{groups}|{band}|{size}|u{_above(times_selected, USAGE_EDGES)}"


# ---------------------------------------------------------------------------
# Preprocessing before a diff

# Window and constant bins: <= 5, 6-21, 22-63, 64-252, > 252 (|value| for constants).
PARAMETER_EDGES = (5, 21, 63, 252)
# A binned parameter is written as its bin's upper edge, so the tree still unparses.
_BIN_VALUES = (5, 21, 63, 252, WINDOW_MAX)
_COMMUTATIVE_OPS = ("+", "*")
_COMMUTATIVE_FUNCS = ("ts_corr", "ts_cov")


def parameter_bin(value: float) -> int:
    """0-4: which of <= 5, 6-21, 22-63, 64-252, > 252 holds |value|."""
    return _above(abs(value), PARAMETER_EDGES)


def _chain(node: Node, op: str) -> list[Node]:
    if isinstance(node, BinOp) and node.op == op:
        return _chain(node.left, op) + _chain(node.right, op)
    return [node]


def preprocess(node: Node) -> Node:
    """The tree in the form edits are compared in.

    Nested shifts are folded, then windows and constants binned
    (`parameter_bin`, a constant keeps its sign), then redundant wrappers
    removed (``--x`` is ``x``, ``rank(rank(x))`` is ``rank(x)``), then the
    operands of ``+``, ``*``, ts_corr and ts_cov sorted, so the result is
    canonical in the sense of `identity_key`.
    """
    if isinstance(node, Field_):
        return node
    if isinstance(node, Const):
        return Const(math.copysign(_BIN_VALUES[parameter_bin(node.value)], node.value))
    if isinstance(node, Unary):
        operand = preprocess(node.operand)
        return operand.operand if isinstance(operand, Unary) else Unary(node.op, operand)
    if isinstance(node, BinOp):
        left, right = preprocess(node.left), preprocess(node.right)
        if node.op not in _COMMUTATIVE_OPS:
            return BinOp(node.op, left, right)
        terms = sorted(_chain(left, node.op) + _chain(right, node.op), key=identity_key)
        out = terms[0]
        for term in terms[1:]:
            out = BinOp(node.op, out, term)
        return out
    if isinstance(node, Call):
        while node.func == "shift" and isinstance(node.arg, Call) and node.arg.func == "shift":
            node = Call("shift", node.arg.arg, node.window + node.arg.window)
        arg = preprocess(node.arg)
        arg2 = None if node.arg2 is None else preprocess(node.arg2)
        if node.func == "rank" and isinstance(arg, Call) and arg.func == "rank":
            return arg
        if node.func in _COMMUTATIVE_FUNCS:
            arg, arg2 = sorted((arg, arg2), key=identity_key)
        window = None if node.window is None else _BIN_VALUES[parameter_bin(node.window)]
        return Call(node.func, arg, window, arg2)
    raise TypeError(f"not a Node: {node!r}")


# ---------------------------------------------------------------------------
# Diff


def _same(a: Node, b: Node) -> bool:
    return identity_key(a) == identity_key(b)


def _commutative(node: Node) -> bool:
    return (isinstance(node, BinOp) and node.op in _COMMUTATIVE_OPS) or (
        isinstance(node, Call) and node.func in _COMMUTATIVE_FUNCS
    )


def _operands(node: Node) -> list[Node]:
    if isinstance(node, BinOp):
        return _chain(node, node.op) if node.op in _COMMUTATIVE_OPS else [node.left, node.right]
    if isinstance(node, Unary):
        return [node.operand]
    if isinstance(node, Call):
        return [node.arg] if node.arg2 is None else [node.arg, node.arg2]
    return []


def _head(node: Node) -> tuple:
    """The root operation without its operands: same head, same arity."""
    if isinstance(node, Field_):
        return ("field", node.name)
    if isinstance(node, Const):
        return ("const", repr(node.value))
    if isinstance(node, Unary):
        return ("unary", node.op)
    if isinstance(node, BinOp):
        return ("binop", node.op, len(_operands(node)))
    return ("call", node.func, node.window, node.arg2 is None)


def _unmatched(a: list[Node], b: list[Node], ordered: bool) -> tuple[list[Node], list[Node]]:
    """Operands of `a` with no equal in `b`, and the reverse (as multisets if unordered)."""
    if ordered:
        pairs = [(x, y) for x, y in zip(a, b, strict=True) if not _same(x, y)]
        return [x for x, _ in pairs], [y for _, y in pairs]
    only_a = Counter(map(identity_key, a)) - Counter(map(identity_key, b))
    only_b = Counter(map(identity_key, b)) - Counter(map(identity_key, a))
    out_a, out_b = [], []
    for node in a:
        if only_a[identity_key(node)] > 0:
            only_a[identity_key(node)] -= 1
            out_a.append(node)
    for node in b:
        if only_b[identity_key(node)] > 0:
            only_b[identity_key(node)] -= 1
            out_b.append(node)
    return out_a, out_b


def locate_edit(parent: Node, child: Node) -> tuple[Node, Node] | None:
    """The smallest pair of preprocessed subtrees that differ, or None if none do.

    Descends while both nodes have the same head and exactly one operand
    differs (operands of ``+``, ``*``, ts_corr and ts_cov matched as multisets).
    """
    p, c = preprocess(parent), preprocess(child)
    if _same(p, c):
        return None
    while _head(p) == _head(c):
        only_p, only_c = _unmatched(_operands(p), _operands(c), ordered=not _commutative(p))
        if len(only_p) != 1 or len(only_c) != 1:
            break
        p, c = only_p[0], only_c[0]
    return p, c


# ---------------------------------------------------------------------------
# Edit motifs m(p, c)

MOTIFS = (
    "condition_gate",
    "rank_switch",
    "interaction",
    "window_rescale",
    "operator_sub",
    "feature_swap",
    "nesting",
    "temporal_shift",
    "normalization",
    "other",
)
# The grammar has no if / where, so nothing is ever labelled condition_gate.
PRODUCIBLE_MOTIFS = MOTIFS[1:]

RANK_FUNCS = frozenset({"rank", "ts_rank"})
NORMALIZATION_FUNCS = frozenset({"zscore", "scale", "winsorize", "log", "sign", "abs"})
NESTING_FUNCS = frozenset(
    {
        "ts_mean",
        "ts_std",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_argmin",
        "ts_argmax",
        "ts_product",
        "decay_linear",
        "ema",
        "ts_corr",
        "ts_cov",
    }
)
SHIFT_FUNCS = frozenset({"shift", "delta"})


def _wraps(outer: Node, inner: Node, funcs: frozenset[str]) -> bool:
    """`outer` is one of `funcs` applied to `inner` (as either series argument)."""
    if not isinstance(outer, Call) or outer.func not in funcs:
        return False
    return _same(outer.arg, inner) or (outer.arg2 is not None and _same(outer.arg2, inner))


def _func_swap(p: Node, c: Node) -> bool:
    """Two calls of different functions on the same operands; windows may differ."""
    if not (isinstance(p, Call) and isinstance(c, Call)) or p.func == c.func:
        return False
    only_p, only_c = _unmatched(_operands(p), _operands(c), ordered=False)
    return len(_operands(p)) == len(_operands(c)) and not only_p and not only_c


def _classify(p: Node, c: Node) -> str:
    def swap_in(funcs: frozenset[str]) -> bool:
        return _func_swap(p, c) and (p.func in funcs or c.func in funcs)

    if _wraps(c, p, RANK_FUNCS) or _wraps(p, c, RANK_FUNCS) or swap_in(RANK_FUNCS):
        return "rank_switch"
    if isinstance(c, BinOp):
        if c.op in _COMMUTATIVE_OPS:
            new = Counter(map(identity_key, _chain(c, c.op)))
            old = Counter(map(identity_key, _chain(p, c.op)))
            if new - old and not old - new:
                return "interaction"
        elif _same(c.left, p) or _same(c.right, p):
            return "interaction"
    if structural_key(p) == structural_key(c):
        return "window_rescale"
    special = RANK_FUNCS | NORMALIZATION_FUNCS | SHIFT_FUNCS
    if _func_swap(p, c) and p.func not in special and c.func not in special:
        return "operator_sub"
    if isinstance(p, BinOp) and isinstance(c, BinOp) and p.op != c.op:
        only_p, only_c = _unmatched([p.left, p.right], [c.left, c.right], ordered=False)
        if not only_p and not only_c:
            return "operator_sub"
    if isinstance(p, Field_) and isinstance(c, Field_):
        return "feature_swap"
    if _wraps(c, p, NESTING_FUNCS) or _wraps(p, c, NESTING_FUNCS):
        return "nesting"
    if _wraps(c, p, SHIFT_FUNCS) or _wraps(p, c, SHIFT_FUNCS) or swap_in(SHIFT_FUNCS):
        return "temporal_shift"
    if _wraps(c, p, NORMALIZATION_FUNCS) or _wraps(p, c, NORMALIZATION_FUNCS):
        return "normalization"
    if (isinstance(c, Unary) and _same(c.operand, p)) or (
        isinstance(p, Unary) and _same(p.operand, c)
    ):
        return "normalization"
    if _func_swap(p, c) and p.func in NORMALIZATION_FUNCS and c.func in NORMALIZATION_FUNCS:
        return "normalization"
    return "other"


def edit_motif(parent: Node, child: Node) -> str:
    """AlphaMemo's m(p, c): which of the ten `MOTIFS` turned `parent` into `child`.

    Both trees are preprocessed and the smallest differing pair found
    (`locate_edit`); the pair takes the first label that fits, in this order:

    1. ``condition_gate``: never; the grammar has no if / where.
    2. ``rank_switch``: rank or ts_rank added, removed, or swapped with
       another function on the same operands.
    3. ``interaction``: the old subtree combined with a new term by ``*``,
       ``/``, ``+`` or ``-`` (a term added to an existing sum or product counts).
    4. ``window_rescale``: same shape, a window or constant in another bin.
    5. ``operator_sub``: a function or operator replaced, operands kept
       (rank, normalization, shift and delta have their own labels).
    6. ``feature_swap``: one field replaced by another.
    7. ``nesting``: a time-series wrapper (`NESTING_FUNCS`) added or removed.
    8. ``temporal_shift``: a shift or delta added or removed, or swapped with
       another function on the same operands.
    9. ``normalization``: a normalizing function or minus sign added, removed
       or swapped for another.
    10. ``other``: everything else, such as dropping a term or a rewrite; also
        a change that preprocessing makes vanish (a window moved within its bin).

    Departures from the paper: removals of a time-series wrapper, shift or
    delta share the label of adding one (as they already do for rank and
    normalization), so a veto on ``other`` cannot block the edits that
    simplify a parent; constants crossing a bin count as window_rescale.
    """
    pair = locate_edit(parent, child)
    return "other" if pair is None else _classify(*pair)


# ---------------------------------------------------------------------------
# Memory events and statistics (Eq. 4, 6)

# invalid: fails to parse or compute; rejected: fails screening or admission;
# admitted: enters the pool; high_quality: admitted with Q >= 0.20.
STATUSES = ("invalid", "rejected", "admitted", "high_quality")
FAILURES = frozenset({"invalid", "rejected"})


def record_event(
    con: sqlite3.Connection,
    *,
    run_id: str,
    iteration: int,
    parent_id: str,
    context: str,
    motif_intended: str,
    status: str,
    child_expression: str,
    child_id: str | None = None,
    hypothesis_id: str | None = None,
    motif_realized: str | None = None,
    quality: float | None = None,
    baseline: float | None = None,
) -> dict:
    """Append one edit to `memory_events` and return it as `MemoryState.update` takes it.

    A scored child carries its quality Q(c) and its parent's baseline; the
    residual Q(c) - Q-hat(p) (Eq. 4) is computed here, so a row cannot
    disagree with itself.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    for motif in (motif_intended, motif_realized):
        if motif is not None and motif not in MOTIFS:
            raise ValueError(f"unknown motif {motif!r}")
    if (quality is None) != (baseline is None):
        raise ValueError("a scored child needs both its quality and its parent's baseline")
    residual = None if quality is None else quality - baseline
    row = {
        "run_id": run_id,
        "iteration": iteration,
        "parent_id": parent_id,
        "child_id": child_id,
        "child_expression": child_expression,
        "hypothesis_id": hypothesis_id,
        "context": context,
        "motif_intended": motif_intended,
        "motif_realized": motif_realized,
        "status": status,
        "quality": quality,
        "baseline": baseline,
        "residual": residual,
        "created_at": store._now(),
    }
    with con:
        con.execute(
            f"INSERT INTO memory_events ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
            tuple(row.values()),
        )
    return row


def memory_events(con: sqlite3.Connection, run_id: str | None = None) -> list[dict]:
    """Every memory event in the order it was recorded, optionally for one run."""
    if run_id is None:
        rows = con.execute("SELECT * FROM memory_events ORDER BY id")
    else:
        rows = con.execute("SELECT * FROM memory_events WHERE run_id = ? ORDER BY id", (run_id,))
    return [dict(r) for r in rows]


def event_motif(event: dict) -> str:
    """The motif an event counts under: the realized one, else (no child tree) the intended one."""
    return event["motif_realized"] or event["motif_intended"]


@dataclass
class PairStats:
    """Eq. 6 for one (context, motif) pair.

    `n`, `mean`, `m2`: Welford's count, mean and sum of squared deviations of
    the scored children's residuals (mean is 0 while n is 0). `attempts`
    counts every child; `failures` those invalid or rejected. The failure
    posterior is Beta(1 + failures, 1 + attempts - failures).
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    attempts: int = 0
    failures: int = 0

    def add(self, status: str, residual: float | None) -> None:
        self.attempts += 1
        self.failures += status in FAILURES
        if residual is not None:
            self.n += 1
            delta = residual - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (residual - self.mean)

    @property
    def std(self) -> float:
        """Sample standard deviation of the residuals; NaN below two."""
        return math.sqrt(self.m2 / (self.n - 1)) if self.n >= 2 else math.nan

    @property
    def failure_rate(self) -> float:
        """Posterior mean failure rate a- / (a- + b-)."""
        return (1 + self.failures) / (2 + self.attempts)


class MemoryState:
    """The memory, updated one event at a time: (context, motif) -> PairStats."""

    def __init__(self) -> None:
        self.pairs: dict[tuple[str, str], PairStats] = {}

    def update(self, event: dict) -> None:
        key = (event["context"], event_motif(event))
        self.pairs.setdefault(key, PairStats()).add(event["status"], event["residual"])

    def stats(self, context: str, motif: str) -> PairStats:
        """The pair's statistics; an unseen pair has none (n = 0, Beta(1, 1))."""
        return self.pairs.get((context, motif), PairStats())

    @classmethod
    def from_events(cls, events: Iterable[dict]) -> MemoryState:
        state = cls()
        for event in events:
            state.update(event)
        return state


def reduce_events(events: Iterable[dict]) -> dict[tuple[str, str], PairStats]:
    """The same statistics in one batch pass per pair (`statistics`, not Welford),
    so the two paths check each other."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for event in events:
        grouped.setdefault((event["context"], event_motif(event)), []).append(event)
    out = {}
    for key, rows in grouped.items():
        residuals = [r["residual"] for r in rows if r["residual"] is not None]
        n = len(residuals)
        mean = statistics.fmean(residuals) if n else 0.0
        m2 = statistics.variance(residuals) * (n - 1) if n >= 2 else 0.0
        out[key] = PairStats(n, mean, m2, len(rows), sum(r["status"] in FAILURES for r in rows))
    return out
