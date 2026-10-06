"""The agent's deepen move, driven by learned memory search.

One `Deepener.step` is one deepen round:

1. Learned memory search (`capstone.factors.memory`) picks a parent from the
   pool and a kind of edit, from the parent's own merit and what the memory
   says about that edit for that kind of parent; vetoed edits are skipped.
2. Opus writes `memory_children_per_parent` edited formulas in one call,
   given the parent, its lineage, the edit in plain words and the edits
   vetoed for its kind of parent. Formulas that do not parse go back for
   repair, as in the proposer.
3. Each child goes through the proposer's screening (size, originality,
   frequent subtrees, alignment) and is stored, or rejected, as a `refined`
   child of its parent. Frequent subtree avoidance judges only the structure
   the edit added, not what the child inherited from its parent.
4. A child that passed is scored by the `Scorer`; each score is a trial and
   is written to the run ledger at once, before anything uses it. The pool
   then decides whether the child joins it as a future parent.
5. Every child becomes a memory event, and the round a `deepen` move.

The scorer is an interface: on real data it comes from evaluation
(QUANTIT-81); tests pass a fake. It sees only the search period. Memory and
pool learn from search-period quality and from failures, never from the
validation funnel or the holdout.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from capstone.factors import graph, store
from capstone.factors.llm import FactorConfig, call_tool, grammar_help
from capstone.factors.memory import (
    PRODUCIBLE_MOTIFS,
    Baselines,
    MemoryState,
    ParentView,
    edit_motif,
    evidence,
    evidence_table,
    ledger_score,
    parent_context,
    record_event,
    select_action,
    vetoed,
)
from capstone.factors.proposer import _judge
from capstone.factors.tree import (
    BINARY_WINDOW_FUNCS,
    BinOp,
    Call,
    Node,
    ParseError,
    factor_id,
    frequent_root_genes,
    identity_key,
    node_count,
    parse,
    root_genes,
    unparse,
    walk,
)
from capstone.factors.zoo import default_zoo
from capstone.runlog import log_run

LEDGER_NAME = "deepen"
LEDGER_TAGS = ["learned-memory-search", "deepen"]

# ---------------------------------------------------------------------------
# Scoring


@dataclass(frozen=True)
class Scored:
    """A factor's search-period quality and the values its correlations use."""

    quality: float  # Q: |ICIR| on the search period
    values: np.ndarray  # one vector, the same cells for every factor


class Scorer(Protocol):
    def __call__(self, tree: Node) -> Scored | None:
        """The factor's score, or None if it cannot be computed."""


# ---------------------------------------------------------------------------
# The pool of parents (AlphaMemo Sec. 4.5, App. C)


def correlations(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation of `x` with each row of `matrix`, as one array operation.

    Each pair uses the cells where both are finite, centred on those cells; 0
    for a pair with fewer than two such cells or a constant side. float64
    whatever the storage, so large values cannot overflow.
    """
    x = np.ravel(x).astype(np.float64)
    m = np.atleast_2d(matrix).astype(np.float64)
    mask = np.isfinite(x)[None, :] & np.isfinite(m)
    count = mask.sum(axis=1)
    safe = np.maximum(count, 1)
    xs = np.where(mask, x[None, :], 0.0)
    ms = np.where(mask, m, 0.0)
    xc = np.where(mask, xs - (xs.sum(axis=1) / safe)[:, None], 0.0)
    mc = np.where(mask, ms - (ms.sum(axis=1) / safe)[:, None], 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        num = (xc * mc).sum(axis=1)
        den = np.sqrt((xc * xc).sum(axis=1) * (mc * mc).sum(axis=1))
        out = num / den
    return np.where((count >= 2) & (den > 0) & np.isfinite(out), out, 0.0)


@dataclass(frozen=True)
class Decision:
    status: str  # rejected, admitted or high_quality
    reason: str
    replaces: str | None = None
    correlations: dict[str, float] = field(default_factory=dict)


@dataclass
class Pool:
    """The factors the deepen move may refine.

    A child joins with Q >= tau_q (`memory_min_quality`), at most `max_nodes`
    nodes and |corr| <= tau_d (`memory_max_correlation`) with every member.
    A full pool swaps out its lowest-Q member only for a better child;
    otherwise the child is rejected ("pool full"). Pairwise |corr| is kept as
    members join, so a member's largest one is a lookup.
    """

    config: FactorConfig
    quality: dict[str, float] = field(default_factory=dict)
    values: dict[str, np.ndarray] = field(default_factory=dict)
    _corr: dict[frozenset, float] = field(default_factory=dict)

    def _correlations(self, values: np.ndarray) -> dict[str, float]:
        if not self.values:
            return {}
        ids = list(self.values)
        found = np.abs(correlations(values, np.stack([self.values[i] for i in ids])))
        return dict(zip(ids, found.tolist(), strict=True))

    def judge(self, quality: float, nodes: int, values: np.ndarray) -> Decision:
        """The child's status, why, and which member it would replace. Changes nothing."""
        cfg = self.config
        if nodes > cfg.max_nodes:
            return Decision("rejected", f"{nodes} nodes > {cfg.max_nodes}")
        if quality < cfg.memory_min_quality:
            return Decision("rejected", f"Q {quality:.3f} < {cfg.memory_min_quality}")
        corr = self._correlations(values)
        rho = max(corr.values(), default=0.0)
        if rho > cfg.memory_max_correlation:
            return Decision("rejected", f"|corr| {rho:.3f} > {cfg.memory_max_correlation}")
        replaces = None
        if len(self.quality) >= cfg.memory_pool_capacity:
            weakest = min(self.quality, key=lambda f: (self.quality[f], f))
            if quality <= self.quality[weakest]:
                return Decision("rejected", "pool full")
            replaces = weakest
        status = "high_quality" if quality >= cfg.memory_high_quality else "admitted"
        return Decision(status, "", replaces, corr)

    def add(self, fid: str, quality: float, values: np.ndarray, decision: Decision | None = None):
        """Add an admitted child (with its decision) or, without one, a starting parent."""
        if decision is not None and decision.status not in ("admitted", "high_quality"):
            raise ValueError(f"cannot add a {decision.status} child")
        if decision is not None and decision.replaces is not None:
            self.remove(decision.replaces)
        corr = decision.correlations if decision is not None else self._correlations(values)
        for other, value in corr.items():
            if other in self.quality:
                self._corr[frozenset((fid, other))] = value
        self.quality[fid] = quality
        self.values[fid] = values

    def remove(self, fid: str) -> None:
        del self.quality[fid], self.values[fid]
        self._corr = {k: v for k, v in self._corr.items() if fid not in k}

    def rho_max(self, fid: str) -> float:
        """A member's largest |corr| with any other member."""
        return max((v for k, v in self._corr.items() if fid in k), default=0.0)


# ---------------------------------------------------------------------------
# Which edits a parent offers

_SUBSTITUTABLE = frozenset(
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
    }
    | BINARY_WINDOW_FUNCS
)


def applicable(tree: Node, motif: str) -> bool:
    """Whether `tree` offers anything to edit for `motif`: a window rescale needs a
    window, an operator substitution a like-for-like function or operator."""
    if motif == "window_rescale":
        return any(isinstance(n, Call) and n.window for n in walk(tree))
    if motif == "operator_sub":
        return any(
            isinstance(n, BinOp) or (isinstance(n, Call) and n.func in _SUBSTITUTABLE)
            for n in walk(tree)
        )
    return motif in PRODUCIBLE_MOTIFS


# ---------------------------------------------------------------------------
# Opus writes the edits

MOTIF_WORDS = {
    "rank_switch": "add, remove or replace a cross-sectional rank or a time-series rank (ts_rank)",
    "interaction": "combine an existing subexpression with a new term by +, -, * or /",
    "window_rescale": "move one lookback window to a clearly different horizon "
    "(week, month, quarter, year, multi-year), keeping the structure",
    "operator_sub": "replace one function or operator with a like-for-like one, keeping its "
    "arguments (e.g. ts_mean -> ts_std, + -> *); not rank, normalization, shift or delta",
    "feature_swap": "replace one data field with another",
    "nesting": "wrap a subexpression in a time-series function (ts_mean, ts_std, ts_sum, "
    "ts_min, ts_max, ts_argmin, ts_argmax, ts_product, decay_linear, ema, ts_corr, ts_cov), "
    "or remove one such wrapper",
    "temporal_shift": "add or remove a shift or delta, or swap shift and delta",
    "normalization": "add or remove zscore, scale, winsorize, log, sign, abs or a minus sign",
    "other": "any other change, such as dropping a term or rewriting a subexpression",
}

TOOL = {
    "name": "record_edits",
    "description": "Record the edited factor formulas.",
    "input_schema": {
        "type": "object",
        "properties": {
            "edits": {
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
        "required": ["edits"],
        "additionalProperties": False,
    },
}


class BudgetExhausted(RuntimeError):
    """The agreed number of model calls has been used."""


def system_prompt(config: FactorConfig) -> str:
    return (
        "You refine cross-sectional equity factors written in a closed grammar. Each "
        "formula is evaluated per stock per day; a higher value means a stronger buy "
        "signal. You are given a parent formula and one kind of edit. Make exactly that "
        "kind of edit, in exactly one place per formula, each formula different, and "
        f"nothing else. Keep each result within {config.max_nodes} tree nodes.\n\n" + grammar_help()
    )


def edit_prompt(
    parent: Node, motif: str, n: int, lineage: Sequence[str], vetoed_motifs: Sequence[str]
) -> str:
    lines = [f"Parent: {unparse(parent)}"]
    if len(lineage) > 1:
        lines.append("Lineage, oldest first: " + " -> ".join(lineage))
    lines.append(f"Edit to make ({motif}): {MOTIF_WORDS[motif]}.")
    if vetoed_motifs:
        words = "; ".join(f"{m} ({MOTIF_WORDS[m]})" for m in vetoed_motifs)
        lines.append(f"Edits that have mostly failed for parents like this one: {words}.")
    lines.append(f"Write {n} different edited formulas.")
    return "\n".join(lines)


def _repair_prompt(failures: list[tuple[str, str]]) -> str:
    listed = "\n".join(f"- {expr!r}: {error}" for expr, error in failures)
    return (
        "These formulas did not parse:\n"
        f"{listed}\n\n"
        "Return a corrected version of each, in the same order, making the same kind of edit."
    )


@dataclass(frozen=True)
class Draft:
    """One edited formula from the model: parsed, or the last parse error."""

    expression: str
    rationale: str
    tree: Node | None
    error: str = ""


class ModelEditor:
    """Edits written by `config.model`, at most `max_calls` calls in all."""

    def __init__(self, client, config: FactorConfig, *, max_calls: int, usage: Counter):
        self.client = client
        self.config = config
        self.max_calls = max_calls
        self.usage = usage

    def _call(self, user: str, tool: dict = TOOL, system: str | None = None) -> dict:
        if self.usage["calls"] >= self.max_calls:
            raise BudgetExhausted(f"used all {self.max_calls} model calls")
        return call_tool(
            self.client,
            self.config,
            system=system or system_prompt(self.config),
            user=user,
            tool=tool,
            usage=self.usage,
        )

    def _repaired(self, items: list[dict], n: int) -> list[Draft]:
        """Exactly `n` drafts (missing answers are empty), parse failures sent back
        for repair up to `config.max_repairs` rounds."""
        items = list(items)[:n]
        items += [{"expression": "", "rationale": ""}] * (n - len(items))
        drafts = [_draft(i) for i in items]
        for _ in range(self.config.max_repairs):
            broken = [i for i, d in enumerate(drafts) if d.tree is None]
            if not broken:
                break
            answer = self._call(
                _repair_prompt([(drafts[i].expression, drafts[i].error) for i in broken])
            )
            for i, item in zip(broken, answer.get("edits", []), strict=False):
                drafts[i] = _draft(item)
        return drafts

    def drafts(
        self, parent: Node, motif: str, n: int, lineage: Sequence[str], vetoed_motifs: Sequence[str]
    ) -> list[Draft]:
        """`n` edits of kind `motif` to `parent`, parse failures repaired."""
        answer = self._call(edit_prompt(parent, motif, n, lineage, vetoed_motifs))
        return self._repaired(answer.get("edits", []), n)


# ---------------------------------------------------------------------------
# The agent decides; the memory advises


@dataclass(frozen=True)
class Candidate:
    """A parent the agent may deepen, with what it needs to judge it."""

    view: ParentView
    expression: str
    hypothesis: str  # the specification of the hypothesis it implements
    evidence: str  # the memory's table for its kind of parent (`memory.evidence_table`)
    allowed: tuple[str, ...]  # edit kinds it may get: applicable and not vetoed


@dataclass(frozen=True)
class AgentDecision:
    parent_id: str
    motif: str
    reason: str
    evidence_used: str
    drafts: list


DECIDE_TOOL = {
    "name": "record_deepen",
    "description": "Record which parent to deepen, the kind of edit, why, and the edits.",
    "input_schema": {
        "type": "object",
        "properties": {
            "parent_id": {"type": "string"},
            "edit": {"type": "string", "enum": list(PRODUCIBLE_MOTIFS)},
            "reason": {"type": "string"},
            "evidence_used": {"type": "string"},
            "edits": TOOL["input_schema"]["properties"]["edits"],
        },
        "required": ["parent_id", "edit", "reason", "evidence_used", "edits"],
        "additionalProperties": False,
    },
}


def agent_system_prompt(config: FactorConfig) -> str:
    return (
        "You are the research agent refining cross-sectional equity factors written in a "
        "closed grammar; a higher value means a stronger buy signal. You choose one parent "
        "factor to refine and one kind of edit, then write the edited formulas. Use the "
        "memory's evidence: it reports, for each kind of parent, how each kind of edit has "
        "done against what was expected, and how far to trust that. Prefer edits that beat "
        "expectations with good confidence, try untried edits when the evidence is thin, "
        "and stay with the parent's hypothesis. Choose only an edit listed as allowed for "
        "that parent. Each formula makes the chosen kind of edit in exactly one place of the "
        "parent and changes nothing else, so the memory can tell which edit did what. Keep "
        f"each formula within {config.max_nodes} tree nodes.\n\n" + grammar_help()
    )


def decide_prompt(candidates: Sequence[Candidate], n: int) -> str:
    lines = ["Parents you may refine:"]
    for c in candidates:
        v = c.view
        lines += [
            "",
            f"Parent {v.id}: {c.expression}",
            f"  quality {v.quality:.3f}, largest |corr| with the pool {v.rho_max:.2f}, "
            f"refined {v.times_selected} times before; kind {v.context}",
            f"  hypothesis: {c.hypothesis}",
            f"  allowed edits: {', '.join(c.allowed)}",
        ]
    lines += ["", "The memory's evidence, by kind of parent:"]
    for context in sorted({c.view.context for c in candidates}):
        table = next(c.evidence for c in candidates if c.view.context == context)
        lines += ["", f"Kind {context}:", table]
    lines += ["", "What each kind of edit means:"]
    lines += [f"- {m}: {MOTIF_WORDS[m]}" for m in PRODUCIBLE_MOTIFS]
    lines += [
        "",
        f"Choose one parent and one allowed edit, say why and which evidence you used, "
        f"and write {n} different edited formulas of that kind.",
    ]
    return "\n".join(lines)


class DeepenAgent(ModelEditor):
    """The model as the deepen decision-maker: one call chooses the parent and the
    edit and writes the edits. A choice that is not allowed (a vetoed edit, an
    edit the parent does not offer, an unknown parent) is refused and asked
    again once; if it is still not allowed, `decide` returns None and the deepen
    move falls back to the memory's own selection."""

    def decide(self, candidates: Sequence[Candidate], n: int) -> AgentDecision | None:
        by_id = {c.view.id: c for c in candidates}
        user = decide_prompt(candidates, n)
        for _ in range(2):  # the first answer, and one more after a refusal
            answer = self._call(user, DECIDE_TOOL, agent_system_prompt(self.config))
            parent_id, motif = str(answer.get("parent_id", "")), str(answer.get("edit", ""))
            chosen = by_id.get(parent_id)
            if chosen is not None and motif in chosen.allowed:
                return AgentDecision(
                    parent_id,
                    motif,
                    str(answer.get("reason", "")),
                    str(answer.get("evidence_used", "")),
                    self._repaired(answer.get("edits", []), n),
                )
            why = (
                f"there is no parent {parent_id!r}"
                if chosen is None
                else f"{motif!r} is not an allowed edit for parent {parent_id}"
            )
            user = decide_prompt(candidates, n) + f"\n\nYour last choice was refused: {why}."
        return None


def _draft(item: dict) -> Draft:
    expression = str(item.get("expression", ""))
    rationale = str(item.get("rationale", ""))
    try:
        return Draft(expression, rationale, parse(expression))
    except ParseError as exc:
        return Draft(expression, rationale, None, str(exc))


# ---------------------------------------------------------------------------
# The deepen move


@dataclass(frozen=True)
class ChildOutcome:
    expression: str
    motif_intended: str
    motif_realized: str | None
    status: str  # invalid, rejected, admitted or high_quality (memory statuses)
    reason: str
    factor_id: str | None = None
    quality: float | None = None


@dataclass
class Deepener:
    """Learned memory search running the deepen move over one factor store.

    `add_parent` puts stored factors in the pool (scored, admitted whatever
    their quality, so the search has somewhere to start); `step` runs one
    deepen round. `editor` writes the edits (a `ModelEditor`, or a fake in
    tests); `seed` fixes every random choice.
    """

    con: sqlite3.Connection
    config: FactorConfig
    scorer: Scorer
    editor: object
    run_id: str
    seed: int = 0
    client: object = None  # for the alignment judge in screening
    usage: Counter = field(default_factory=Counter)
    zoo: list = field(default_factory=default_zoo)
    ledger_tags: list = field(default_factory=lambda: list(LEDGER_TAGS))
    agent: DeepenAgent | None = None  # if set, the agent decides; else the memory selects
    candidates_shown: int = 8  # parents the agent sees: the best by S_ledger
    refusals: int = 0  # rounds where the agent's choice was refused twice
    pool: Pool = field(init=False)
    memory: MemoryState = field(default_factory=MemoryState)
    baselines: Baselines = field(default_factory=Baselines)
    selected: Counter = field(default_factory=Counter)
    evaluations: int = 0
    fallbacks: int = 0  # rounds where every pair was vetoed and the pick was random

    def __post_init__(self) -> None:
        self.pool = Pool(self.config)
        self.rng = random.Random(self.seed)
        self.trees: dict[str, Node] = {}

    # -- the pool and the store -------------------------------------------

    def add_parent(self, fid: str) -> bool:
        """Score a stored factor and add it to the pool; False if it cannot be scored."""
        row = self.con.execute("SELECT expression FROM factors WHERE id = ?", (fid,)).fetchone()
        if row is None:
            raise KeyError(f"no factor {fid!r} in the store")
        tree = parse(row["expression"])
        scored = self.scorer(tree)
        if scored is None or not math.isfinite(scored.quality):
            return False
        self.pool.add(fid, scored.quality, scored.values)
        self.trees[fid] = tree
        return True

    def _hypothesis(self, fid: str) -> dict:
        row = self.con.execute(
            "SELECT h.* FROM proposals p JOIN hypotheses h ON h.id = p.hypothesis_id "
            "WHERE p.factor_id = ? ORDER BY p.id LIMIT 1",
            (fid,),
        ).fetchone()
        return dict(row)

    def lineage(self, fid: str) -> list[str]:
        """Expressions from the factor's earliest stored ancestor down to it."""
        chain, seen = [fid], {fid}
        while True:
            row = self.con.execute(
                "SELECT parent_factor_id FROM proposals WHERE factor_id = ? "
                "AND edge_type = 'refined' AND status = 'stored' ORDER BY id LIMIT 1",
                (chain[-1],),
            ).fetchone()
            if row is None or row[0] is None or row[0] in seen:
                break
            chain.append(row[0])
            seen.add(row[0])
        out = []
        for f in reversed(chain):
            expr = self.con.execute("SELECT expression FROM factors WHERE id = ?", (f,)).fetchone()
            out.append(expr[0])
        return out

    # -- one round ----------------------------------------------------------

    def _views(self) -> list[ParentView]:
        return [
            ParentView(
                fid,
                parent_context(self.trees[fid], quality=q, times_selected=self.selected[fid]),
                q,
                self.pool.rho_max(fid),
                self.selected[fid],
            )
            for fid, q in sorted(self.pool.quality.items())
        ]

    def _allowed(self, view: ParentView, motif: str) -> bool:
        # With max_store_share at 1.0, screening rejects any child whose whole
        # shape is stored, which is every window rescale of a stored parent:
        # each attempt would only spend a model call to fail.
        if motif == "window_rescale" and self.config.max_store_share >= 1.0:
            return False
        return applicable(self.trees[view.id], motif)

    def choose(self) -> tuple[ParentView, str]:
        """The parent and motif for the next round."""
        views = self._views()
        if not views:
            raise RuntimeError("the pool is empty: add parents first")
        action = select_action(
            views, self.memory, self.evaluations, self.config, self._allowed, self.rng
        )
        if action is None:  # every pair vetoed: explore at random
            self.fallbacks += 1
            parent = self.rng.choice(views)
            motif = self.rng.choice([m for m in PRODUCIBLE_MOTIFS if self._allowed(parent, m)])
            return parent, motif
        return next(v for v in views if v.id == action.parent_id), action.motif

    def _vetoes(self, context: str) -> list[str]:
        return [m for m in PRODUCIBLE_MOTIFS if vetoed(self.memory.stats(context, m), self.config)]

    def candidates(self) -> list[Candidate]:
        """The parents the agent sees, best by S_ledger first, with the memory's evidence."""
        views = sorted(
            self._views(),
            key=lambda v: (-ledger_score(v.quality, v.rho_max, v.times_selected), v.id),
        )[: self.candidates_shown]
        out = []
        for v in views:
            vetoes = set(self._vetoes(v.context))
            allowed = tuple(m for m in PRODUCIBLE_MOTIFS if self._allowed(v, m) and m not in vetoes)
            if not allowed:
                continue
            out.append(
                Candidate(
                    v,
                    unparse(self.trees[v.id]),
                    self._hypothesis(v.id)["specification"],
                    evidence_table(evidence(self.memory, v.context, self.config)),
                    allowed,
                )
            )
        return out

    def step(self) -> list[ChildOutcome]:
        """One deepen round: decide, edit, screen, score, decide admission, remember.

        With an agent, the agent decides the parent and the edit and writes the
        edits in one call; otherwise (or if the agent's choice is refused twice)
        the memory selects and the editor writes. Either way the decision is
        recorded as the deepen move before any child is screened or scored.
        """
        n = self.config.memory_children_per_parent
        decision = None
        if self.agent is not None:
            candidates = self.candidates()
            if candidates:
                decision = self.agent.decide(candidates, n)
                if decision is None:
                    self.refusals += 1
        if decision is not None:
            parent = next(c.view for c in candidates if c.view.id == decision.parent_id)
            motif = decision.motif
            record = {
                "motif": motif,
                "by": "agent",
                "reason": decision.reason,
                "evidence_used": decision.evidence_used,
            }
        else:
            parent, motif = self.choose()
            record = {"motif": motif, "by": "memory"}
        self.selected[parent.id] += 1
        parent_tree = self.trees[parent.id]
        hypothesis = self._hypothesis(parent.id)
        graph.record_move(
            self.con,
            "deepen",
            from_id=parent.id,
            hypothesis_id=hypothesis["id"],
            reason=json.dumps(record),
            policy="learned-memory-search",
        )
        if decision is not None:
            drafts = decision.drafts
        else:
            drafts = self.editor.drafts(
                parent_tree, motif, n, self.lineage(parent.id), self._vetoes(parent.context)
            )
        frequent = frequent_root_genes(
            [tree for _, tree in store.factor_trees(self.con)], self.config.avoid_frequent_subtrees
        )
        avoid = {key: text for key, _, text in frequent}
        lineage_meta = {"parent_factor_id": parent.id, "edge_type": "refined"}
        outcomes = []
        for draft in drafts:
            self.evaluations += 1
            outcomes.append(
                self._child(draft, parent, parent_tree, motif, hypothesis, avoid, lineage_meta)
            )
        return outcomes

    def _child(self, draft, parent, parent_tree, motif, hypothesis, avoid, lineage_meta):
        meta = dict(model=self.config.model, prompt_version=self.config.prompt_version)
        q = baseline = None
        realized = fid = None
        if draft.tree is None:
            reason = f"parse error: {draft.error}" if draft.error else "no edit returned"
            store.reject(
                self.con,
                hypothesis["id"],
                draft.expression,
                reason,
                rationale=draft.rationale,
                **lineage_meta,
                **meta,
            )
            status = "invalid"
        elif identity_key(draft.tree) == identity_key(parent_tree):
            realized = edit_motif(parent_tree, draft.tree)
            store.reject(
                self.con,
                hypothesis["id"],
                draft.expression,
                "the parent itself",
                node=draft.tree,
                rationale=draft.rationale,
                **lineage_meta,
                **meta,
            )
            status, reason = "rejected", "the parent itself"
        else:
            realized = edit_motif(parent_tree, draft.tree)
            screened = _judge(
                self.con,
                draft.tree,
                draft.expression,
                draft.rationale,
                hypothesis,
                self.config,
                self.zoo,
                self.client,
                self.usage,
                avoid,
                lineage=lineage_meta,
                inherited=root_genes(parent_tree),
            )
            fid = factor_id(draft.tree)
            if screened.status != "stored":
                status, reason = "rejected", screened.reason or screened.status
            else:
                status, reason, q = self._score(draft, parent, motif, fid)
                if q is not None:
                    baseline = self.baselines.baseline(parent.context, parent.quality)
                    self.baselines.add(parent.context, q)
        self.memory.update(
            record_event(
                self.con,
                run_id=self.run_id,
                iteration=self.evaluations,
                parent_id=parent.id,
                child_id=fid,
                child_expression=draft.expression,
                hypothesis_id=hypothesis["id"],
                context=parent.context,
                motif_intended=motif,
                motif_realized=realized,
                status=status,
                quality=q,
                baseline=baseline,
            )
        )
        return ChildOutcome(draft.expression, motif, realized, status, reason, fid, q)

    def _score(self, draft: Draft, parent: ParentView, motif: str, fid: str):
        scored = self.scorer(draft.tree)
        if scored is None or not math.isfinite(scored.quality):
            return "invalid", "cannot be scored", None
        # The score is a trial: on the ledger before anything uses it.
        log_run(
            LEDGER_NAME,
            params={
                "run_id": self.run_id,
                "factor_id": fid,
                "expression": unparse(draft.tree),
                "parent_id": parent.id,
                "motif": motif,
                "iteration": self.evaluations,
                "model": self.config.model,
            },
            metrics={"quality": scored.quality},
            seed=self.seed,
            tags=self.ledger_tags,
        )
        decision = self.pool.judge(scored.quality, node_count(draft.tree), scored.values)
        if decision.status in ("admitted", "high_quality"):
            self.pool.add(fid, scored.quality, scored.values, decision)
            self.trees[fid] = draft.tree
        return decision.status, decision.reason, scored.quality
