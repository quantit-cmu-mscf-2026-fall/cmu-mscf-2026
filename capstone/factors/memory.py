"""Learned memory search: what the search remembers about its own edits.

Adapted from AlphaMemo (Yu et al. 2026, arXiv 2606.20625, Sec. 3-4). Each
time the agent turns a parent factor into a child, the step is recorded as a
memory event under two keys:

- the **parent context** (`parent_context`, AlphaMemo's z(p)): which kinds of
  fields the parent reads, how good it is, how deep in its lineage it sits
  and how often it has already been edited. Coarse on purpose, so that
  lessons carry over between similar parents;
- the **edit motif** (`edit_motif`, AlphaMemo's m(p, c)): what kind of change
  turned the parent into the child, read off the two trees.

Each event keeps the child's status and, once it has been scored, its quality,
the quality expected for that kind of parent (the baseline) and the
difference (the residual). `memory_stats` reduces the events for each
(context, motif) pair to the sufficient statistics of AlphaMemo's Eq. 6: how
many residuals, their mean and variance, and a Beta posterior over failures.
Computing the baseline, gating the memory and choosing the next edit come in
the next step (QUANTIT-95).

Events are append-only, like the rest of the store. Nothing here reads market
data; scores come from evaluation, and every scored child is a ledger trial
there, not here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from capstone.factors import store
from capstone.factors.tree import (
    BinOp,
    Call,
    Const,
    Field_,
    Node,
    Unary,
    features_used,
    identity_key,
    structural_key,
)

# Field groups for the parent's category: what kind of information it uses.
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
# Bucket edges for |ICIR| (AlphaMemo's quality threshold is 0.10) and for
# lineage depth and number of children already made from the parent.
QUALITY_EDGES = (0.05, 0.10, 0.20)
DEPTH_EDGES = (1, 2, 3)
USAGE_EDGES = (1, 3, 6)

STATUSES = ("invalid", "rejected", "admitted", "high_quality")
FAILURES = ("invalid", "rejected")


@dataclass(frozen=True)
class ParentContext:
    """AlphaMemo's z(p): category, quality, depth and usage buckets of a parent."""

    category: str
    quality: str
    depth: int
    usage: int

    @property
    def key(self) -> str:
        return f"{self.category}|{self.quality}|d{self.depth}|u{self.usage}"


def _bucket(value: float, edges: tuple) -> int:
    return sum(value >= edge for edge in edges)


def parent_context(
    parent: Node, *, quality: float | None, depth: int, children: int
) -> ParentContext:
    """Bucket a parent: which fields it reads, how good it is, how deep, how used.

    `quality` is the parent's |ICIR| on the search period, or None if it has
    not been scored. `depth` is how many refinements it is from a proposal;
    `children` is how many children have already been made from it.
    """
    groups = sorted({FIELD_GROUPS[f] for f in features_used(parent)})
    band = "unscored" if quality is None else f"q{_bucket(abs(quality), QUALITY_EDGES)}"
    return ParentContext(
        "+".join(groups), band, _bucket(depth, DEPTH_EDGES), _bucket(children, USAGE_EDGES)
    )


# ---------------------------------------------------------------------------
# Edit motifs


def _children(node: Node) -> list[Node]:
    if isinstance(node, Unary):
        return [node.operand]
    if isinstance(node, BinOp):
        return [node.left, node.right]
    if isinstance(node, Call):
        return [node.arg] if node.arg2 is None else [node.arg, node.arg2]
    return []


def _same_head(a: Node, b: Node) -> bool:
    """Same operation at the root, ignoring windows and the children."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Field_):
        return a.name == b.name
    if isinstance(a, BinOp):
        return a.op == b.op
    if isinstance(a, Call):
        return a.func == b.func
    return True  # Unary (only neg) or Const


def _locate(parent: Node, child: Node) -> tuple[Node, Node]:
    """The smallest pair of subtrees where the two trees differ."""
    while _same_head(parent, child):
        pc, cc = _children(parent), _children(child)
        if len(pc) != len(cc):
            break
        differ = [(p, c) for p, c in zip(pc, cc, strict=True) if identity_key(p) != identity_key(c)]
        if len(differ) != 1:
            break
        parent, child = differ[0]
    return parent, child


def _wraps(outer: Node, inner: Node) -> bool:
    return any(identity_key(c) == identity_key(inner) for c in _children(outer))


def edit_motif(parent: Node, child: Node) -> str:
    """AlphaMemo's m(p, c): what kind of edit turned `parent` into `child`.

    One of: `same`; `window` (only windows or constants changed); `wrap:<f>`
    (the old subtree is now inside f, `neg` for a minus sign); `unwrap`;
    `add_term:<op>` (combined with a new term by + - * /); `drop_term`;
    `swap_field`, `swap_func` or `swap_op` (one field, function or operator
    replaced, the rest kept); or `rewrite` for anything larger.
    """
    if identity_key(parent) == identity_key(child):
        return "same"
    p, c = _locate(parent, child)
    if structural_key(p) == structural_key(c):
        return "window"
    if isinstance(c, BinOp) and _wraps(c, p):
        return f"add_term:{c.op}"
    if isinstance(p, BinOp) and _wraps(p, c):
        return "drop_term"
    if isinstance(c, Unary | Call) and _wraps(c, p):
        return f"wrap:{'neg' if isinstance(c, Unary) else c.func}"
    if isinstance(p, Unary | Call) and _wraps(p, c):
        return "unwrap"
    if isinstance(p, Field_) and isinstance(c, Field_):
        return "swap_field"
    same_children = [identity_key(x) for x in _children(p)] == [
        identity_key(x) for x in _children(c)
    ]
    if isinstance(p, Call) and isinstance(c, Call) and same_children:
        return "swap_func"
    if isinstance(p, BinOp) and isinstance(c, BinOp) and same_children:
        return "swap_op"
    if isinstance(p, Const) and isinstance(c, Const):
        return "window"
    return "rewrite"


# ---------------------------------------------------------------------------
# Events and statistics


def record(
    con: sqlite3.Connection,
    *,
    parent_id: str,
    child_expression: str,
    context: ParentContext,
    motif: str,
    status: str,
    child_id: str | None = None,
    hypothesis_id: str | None = None,
    quality: float | None = None,
    baseline: float | None = None,
) -> int:
    """Append one edit's outcome. The residual is quality minus baseline, when both exist."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    residual = quality - baseline if quality is not None and baseline is not None else None
    with con:
        cursor = con.execute(
            "INSERT INTO memory_events (parent_id, child_id, child_expression, hypothesis_id, "
            "context, motif, status, quality, baseline, residual, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                parent_id,
                child_id,
                child_expression,
                hypothesis_id,
                context.key,
                motif,
                status,
                quality,
                baseline,
                residual,
                store._now(),
            ),
        )
    return int(cursor.lastrowid)


@dataclass(frozen=True)
class MotifStats:
    """AlphaMemo Eq. 6 for one (context, motif) pair.

    `n`, `mean` and `variance` describe the residuals of scored children
    (sample variance; NaN below 2). `failures` and `attempts` count every
    child, scored or not; the failure posterior is Beta(1 + failures,
    1 + attempts - failures), starting from a uniform prior. A child that
    failed to parse or was rejected by formula screening is a failure.
    """

    n: int
    mean: float
    variance: float
    failures: int
    attempts: int

    @property
    def failure_alpha(self) -> float:
        return 1.0 + self.failures

    @property
    def failure_beta(self) -> float:
        return 1.0 + self.attempts - self.failures

    @property
    def failure_rate(self) -> float:
        """Posterior mean probability that this edit, here, fails."""
        return self.failure_alpha / (self.failure_alpha + self.failure_beta)


def memory_stats(con: sqlite3.Connection) -> dict[tuple[str, str], MotifStats]:
    """Sufficient statistics for every (context key, motif) pair seen so far."""
    failed = ", ".join(f"'{s}'" for s in FAILURES)
    rows = con.execute(
        "SELECT context, motif, COUNT(*) AS attempts, "
        f"SUM(status IN ({failed})) AS failures, COUNT(residual) AS n, "
        "AVG(residual) AS mean, SUM(residual * residual) AS sq "
        "FROM memory_events GROUP BY context, motif"
    ).fetchall()
    out = {}
    for row in rows:
        n, mean = row["n"], row["mean"] if row["n"] else float("nan")
        variance = (row["sq"] - n * mean * mean) / (n - 1) if n > 1 else float("nan")
        out[(row["context"], row["motif"])] = MotifStats(
            n, mean, max(variance, 0.0) if n > 1 else variance, row["failures"], row["attempts"]
        )
    return out
