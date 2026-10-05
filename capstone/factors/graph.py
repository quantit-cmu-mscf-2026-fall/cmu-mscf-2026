"""The strategy graph: bonds between strategies, and the agent's moves.

A strategy is a stored factor (a leaf, one expression tree) or a **bond**: two
or more strategies joined into one, like atoms in a molecule. A bond records
its members, which may be factors or other bonds, and resolves to the set of
factor trees it ties together (its leaves); everything known about a bond
comes from those trees. `rule` says how the members combine when the bond is
evaluated; `rank_mean` is the equal-weighted mean of the members'
cross-sectional ranks.

Before it is stored, a bond is checked for uniqueness, with the same
largest-shared-subtree measure AlphaAgent (#44) uses for originality, and
nothing here reads market data:

- `member_share`: redundancy inside the bond. The largest share of one leaf
  that is a subtree of another leaf: 1.0 means one member is wholly inside
  another, so the bond adds a factor it already has.
- `bond_overlap`: closeness to the bonds already stored. The Jaccard overlap
  of leaf sets, under the same rule: 1.0 is the same molecule again.
- `leaf_nodes`: complexity, the total node count of the leaves.

Each move the search agent makes is logged in `moves`: `deepen` (refine a
strategy's tree), `bond` (join strategies) or `switch` (move on to another
hypothesis or strategy), with its reason and the policy that chose it. Which
move to make is scored only once factors are evaluated (QUANTIT-81, -84);
until then the log records the moves and the policy that made them, so the
lineage that gives N_searched is complete from the start.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from itertools import permutations

from capstone.factors import store
from capstone.factors.llm import FactorConfig
from capstone.factors.tree import Node, features_used, node_count, parse, subtree_similarity

RULES = ("rank_mean",)
ACTIONS = ("deepen", "bond", "switch")


@dataclass(frozen=True)
class BondOutcome:
    """What happened to one proposed bond, and the uniqueness evidence for it."""

    bond_id: str
    status: str  # stored, duplicate or rejected
    reason: str
    leaves: tuple[str, ...]
    leaf_nodes: int
    member_share: float
    nearest_members: tuple[str, str] | None
    bond_overlap: float
    nearest_bond: str


def bond_id(members: list[str], rule: str) -> str:
    """Stable id of a bond: its rule and its members, in any order."""
    key = json.dumps([rule, sorted(set(members))])
    return "bond_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _trees(con: sqlite3.Connection) -> dict[str, Node]:
    return dict(store.factor_trees(con))


def leaves(con: sqlite3.Connection, strategy_id: str) -> frozenset[str]:
    """The factor ids a strategy ties together: itself, or a bond's leaves."""
    row = con.execute("SELECT leaves FROM bonds WHERE id = ?", (strategy_id,)).fetchone()
    if row is not None:
        return frozenset(json.loads(row["leaves"]))
    if con.execute("SELECT 1 FROM factors WHERE id = ?", (strategy_id,)).fetchone() is None:
        raise KeyError(f"no factor or bond {strategy_id!r} in the store")
    return frozenset([strategy_id])


def _member_share(trees: dict[str, Node], ids: frozenset[str]) -> tuple[float, tuple | None]:
    best, pair = 0.0, None
    for a, b in permutations(sorted(ids), 2):
        share = subtree_similarity(trees[a], trees[b]) / node_count(trees[a])
        if share > best:
            best, pair = share, (a, b)
    return best, pair


def _bond_overlap(con: sqlite3.Connection, ids: frozenset[str], rule: str, own: str):
    best, nearest = 0.0, ""
    for row in con.execute("SELECT id, leaves FROM bonds WHERE rule = ? AND id != ?", (rule, own)):
        other = frozenset(json.loads(row["leaves"]))
        overlap = len(ids & other) / len(ids | other)
        if overlap > best:
            best, nearest = overlap, row["id"]
    return best, nearest


def bond(
    con: sqlite3.Connection,
    members: list[str],
    config: FactorConfig,
    *,
    rule: str = "rank_mean",
    rationale: str = "",
    model: str = "",
) -> BondOutcome:
    """Join strategies into a bond, check its uniqueness, and store it or say why not.

    Every attempt is logged as a `bond` move, so rejected bonds stay on the
    record like rejected proposals do.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    members = sorted(set(members))
    if len(members) < 2:
        raise ValueError("a bond needs at least two distinct members")
    trees = _trees(con)
    ids = frozenset().union(*(leaves(con, m) for m in members))
    bid = bond_id(members, rule)
    size = sum(node_count(trees[i]) for i in ids)
    share, pair = _member_share(trees, ids)
    overlap, nearest = _bond_overlap(con, ids, rule, bid)

    if con.execute("SELECT 1 FROM bonds WHERE id = ?", (bid,)).fetchone():
        status, reason = "duplicate", "already in the store"
    elif len(ids) > config.max_bond_leaves:
        status, reason = "rejected", f"too many factors: {len(ids)}, limit {config.max_bond_leaves}"
    elif share >= config.max_member_share:
        status, reason = "rejected", f"redundant: {share:.0%} of {pair[0]} is in {pair[1]}"
    elif overlap >= config.max_bond_overlap:
        status, reason = "rejected", f"not original: shares {overlap:.0%} of {nearest}'s factors"
    else:
        status, reason = "stored", ""

    with con:
        if status == "stored":
            features = sorted(set().union(*(features_used(trees[i]) for i in ids)))
            con.execute(
                "INSERT INTO bonds (id, rule, leaves, leaf_nodes, member_share, nearest_members, "
                "bond_overlap, nearest_bond, features, rationale, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    bid,
                    rule,
                    json.dumps(sorted(ids)),
                    size,
                    share,
                    json.dumps(pair) if pair else None,
                    overlap,
                    nearest,
                    json.dumps(features),
                    rationale,
                    model,
                    store._now(),
                ),
            )
            con.executemany(
                "INSERT INTO bond_members (bond_id, member_id, member_kind) VALUES (?, ?, ?)",
                [(bid, m, "factor" if m in trees else "bond") for m in members],
            )
        _insert_move(
            con,
            action="bond",
            from_id=json.dumps(members),
            result_id=bid,
            status=status,
            reason=reason or rationale,
            policy=model,
        )
    return BondOutcome(bid, status, reason, tuple(sorted(ids)), size, share, pair, overlap, nearest)


def _insert_move(con: sqlite3.Connection, **row) -> int:
    if row["action"] not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, got {row['action']!r}")
    row.setdefault("created_at", store._now())
    columns = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    return int(
        con.execute(
            f"INSERT INTO moves ({columns}) VALUES ({marks})", tuple(row.values())
        ).lastrowid
    )


def record_move(
    con: sqlite3.Connection,
    action: str,
    *,
    from_id: str | None = None,
    result_id: str | None = None,
    hypothesis_id: str | None = None,
    status: str = "done",
    reason: str = "",
    policy: str = "",
) -> int:
    """Log a `deepen` or `switch` move (bonds log themselves in `bond`)."""
    with con:
        return _insert_move(
            con,
            action=action,
            from_id=from_id,
            result_id=result_id,
            hypothesis_id=hypothesis_id,
            status=status,
            reason=reason,
            policy=policy,
        )


def strategy(con: sqlite3.Connection, strategy_id: str) -> dict:
    """Everything known about a strategy without market data: its trees and uniqueness."""
    ids = leaves(con, strategy_id)
    rows = {r["id"]: r["expression"] for r in store.factors(con)}
    out = {"id": strategy_id, "leaves": {i: rows[i] for i in sorted(ids)}}
    row = con.execute("SELECT * FROM bonds WHERE id = ?", (strategy_id,)).fetchone()
    if row is not None:
        out.update({k: row[k] for k in ("rule", "leaf_nodes", "member_share", "bond_overlap")})
        out["features"] = json.loads(row["features"])
        out["members"] = [
            r["member_id"]
            for r in con.execute(
                "SELECT member_id FROM bond_members WHERE bond_id = ? ORDER BY member_id",
                (strategy_id,),
            )
        ]
    else:
        out["features"] = sorted(features_used(parse(rows[strategy_id])))
    return out
