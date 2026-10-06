"""Hypothesis -> factor trees: propose, parse, check originality, store.

The model proposes formulas for one hypothesis. Each one is parsed; a parse
error goes back to the model to fix, up to `max_repairs` rounds. A formula
that parses is checked for size (AlphaAgent's complexity control, at most
`max_nodes` nodes), for originality against Alpha101 and against the
factors already stored (AlphaAgent Eq. 6), for the store's most overused
structures (frequent subtree avoidance, Alpha Jungle Sec. 3; they are also
named in the prompt), and, if those pass, for alignment
with its hypothesis (`alignment`), then stored or rejected. Every
outcome is written to the store's `proposals` table, so failures stay on
the record. No market data is read and no performance is computed.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from capstone.factors import alignment, store
from capstone.factors.hypotheses import _FIELDS
from capstone.factors.llm import FactorConfig, call_tool, grammar_help
from capstone.factors.tree import (
    Node,
    ParseError,
    factor_id,
    frequent_root_genes,
    node_count,
    parse,
    root_genes,
    zoo_similarity,
)
from capstone.factors.zoo import default_zoo


def system_prompt(config: FactorConfig, avoid: Sequence[str] = ()) -> str:
    avoided = (
        "\n\nThese structures are already overused in the stored factors (t is any "
        "window); do not use them: " + "; ".join(avoid) + "."
        if avoid
        else ""
    )
    return (
        "You write cross-sectional equity factors as formulas in a closed grammar. Each "
        "formula is evaluated per stock per day; a higher value means a stronger buy "
        "signal. Write formulas that measure exactly what the hypothesis specifies, in the "
        f"simplest form that does: at most {config.max_nodes} tree nodes (every field, "
        "constant, operator and function call counts as one), and no redundant terms such "
        "as rank(x) - rank(-x), which carries the same information as rank(x). Do not "
        "reproduce well-known published alphas (such as Alpha101) or copy one with "
        "different windows.\n\n" + grammar_help() + avoided
    )


TOOL = {
    "name": "record_factors",
    "description": "Record factor formulas for the hypothesis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["expression", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["factors"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Outcome:
    """What happened to one proposed formula."""

    expression: str
    status: str  # stored, duplicate or rejected
    reason: str = ""
    factor_id: str | None = None


def _hypothesis_prompt(hypothesis: dict, n: int) -> str:
    lines = [f"{name}: {hypothesis[name]}" for name in _FIELDS]
    return "Hypothesis\n" + "\n".join(lines) + f"\n\nPropose {n} factor formulas."


def _repair_prompt(failures: list[tuple[str, str]]) -> str:
    listed = "\n".join(f"- {expr!r}: {error}" for expr, error in failures)
    return (
        "These formulas did not parse:\n"
        f"{listed}\n\n"
        "Return a corrected version of each, in the same order, using only the grammar."
    )


def _judge(
    con: sqlite3.Connection,
    node: Node,
    expression: str,
    rationale: str,
    hypothesis: dict,
    config: FactorConfig,
    zoo: list[tuple[str, Node]],
    client,
    usage: Counter | None,
    avoid: dict[str, str] | None = None,
    lineage: dict | None = None,
    inherited: frozenset[str] = frozenset(),
) -> Outcome:
    """Screen one parsed formula and store or reject it.

    `lineage` (parent_factor_id, edge_type) records a refinement of an
    existing factor, as the deepen move makes. `inherited` is the parent's
    root genes: a refinement keeps most of its parent by design, so frequent
    subtree avoidance judges only the genes the edit added.
    """
    hypothesis_id = hypothesis["id"]
    meta = dict(rationale=rationale, model=config.model, prompt_version=config.prompt_version)
    meta.update(lineage or {})
    fid = factor_id(node)
    stored = [(name, tree) for name, tree in store.factor_trees(con) if name != fid]
    _, zoo_share, nearest_zoo = zoo_similarity(node, zoo)
    _, store_share, nearest_factor = zoo_similarity(node, stored)
    meta.update(
        zoo_similarity=zoo_share,
        nearest_zoo=nearest_zoo,
        store_similarity=store_share,
        nearest_factor=nearest_factor,
    )

    size = node_count(node)
    if size > config.max_nodes:
        reason = f"too complex: {size} nodes, limit {config.max_nodes}"
    elif zoo_share >= config.max_zoo_share:
        reason = f"not original: {zoo_share:.0%} of it is in {nearest_zoo}"
    elif store_share >= config.max_store_share:
        reason = f"not original: {store_share:.0%} of it is stored factor {nearest_factor}"
    elif avoid and (overused := sorted((root_genes(node) - inherited) & avoid.keys())):
        reason = "frequent subtree: " + "; ".join(avoid[g] for g in overused)
    else:
        # The judge runs last: it costs a model call, the checks above don't.
        if config.alignment_model:
            verdict = alignment.judge(client, config, hypothesis, expression, rationale, usage)
            meta.update(
                alignment=verdict.score,
                alignment_reason=verdict.reason,
                alignment_model=verdict.model,
            )
            if verdict.score < config.min_alignment:
                reason = (
                    f"misaligned: {verdict.score:.2f} < {config.min_alignment}; {verdict.reason}"
                )
                store.reject(con, hypothesis_id, expression, reason, node=node, **meta)
                return Outcome(expression, "rejected", reason)
        fid, is_new = store.add_factor(con, node, hypothesis_id, expression=expression, **meta)
        return Outcome(expression, "stored" if is_new else "duplicate", factor_id=fid)
    store.reject(con, hypothesis_id, expression, reason, node=node, **meta)
    return Outcome(expression, "rejected", reason)


def propose(
    con: sqlite3.Connection,
    hypothesis_id: str,
    client,
    config: FactorConfig,
    zoo: list[tuple[str, Node]] | None = None,
    usage: Counter | None = None,
) -> list[Outcome]:
    """Propose factors for one stored hypothesis and record every outcome."""
    row = con.execute("SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)).fetchone()
    if row is None:
        raise KeyError(f"no hypothesis {hypothesis_id!r} in the store")
    zoo = default_zoo() if zoo is None else zoo
    meta = dict(model=config.model, prompt_version=config.prompt_version)
    frequent = frequent_root_genes(
        [tree for _, tree in store.factor_trees(con)], config.avoid_frequent_subtrees
    )
    avoid = {key: text for key, _, text in frequent}
    system = system_prompt(config, list(avoid.values()))

    user = _hypothesis_prompt(dict(row), config.factors_per_hypothesis)
    outcomes: list[Outcome] = []
    for attempt in range(config.max_repairs + 1):
        answer = call_tool(client, config, system=system, user=user, tool=TOOL, usage=usage)
        failures: list[tuple[str, str]] = []
        for item in answer.get("factors", [])[: config.factors_per_hypothesis]:
            expression = str(item.get("expression", ""))
            rationale = str(item.get("rationale", ""))
            try:
                node = parse(expression)
            except ParseError as exc:
                store.reject(
                    con,
                    hypothesis_id,
                    expression,
                    f"parse error: {exc}",
                    rationale=rationale,
                    **meta,
                )
                outcomes.append(Outcome(expression, "rejected", f"parse error: {exc}"))
                failures.append((expression, str(exc)))
                continue
            outcomes.append(
                _judge(
                    con, node, expression, rationale, dict(row), config, zoo, client, usage, avoid
                )
            )
        if not failures or attempt == config.max_repairs:
            break
        user = _hypothesis_prompt(dict(row), len(failures)) + "\n\n" + _repair_prompt(failures)
    return outcomes
