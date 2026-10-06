"""Pieces of the deepen move: scoring, the pool of parents, and Opus as editor.

The deepen move (the next stage of this stack) refines a stored factor:
learned memory search (`capstone.factors.memory`) picks the parent and the
kind of edit, a model writes the edits, screening and scoring decide each
child's fate. This module holds the parts that do not depend on that loop:

- `Scored`, `Scorer`: what a scorer returns for a factor (search-period
  quality, and its values for correlations); the real scorer comes from
  evaluation (QUANTIT-81), tests pass a fake.
- `Pool`: the parents the deepen move may refine (AlphaMemo Sec. 4.5, App.
  C): admission by quality, size and correlation with every member, and
  replacement of the weakest member when full.
- `applicable`: which kinds of edit a parent offers.
- `ModelEditor`: `config.model` writes edits of a requested kind in one
  call, with parse repairs, under a hard call limit (`BudgetExhausted`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from capstone.factors.llm import FactorConfig, call_tool, grammar_help
from capstone.factors.memory import (
    PRODUCIBLE_MOTIFS,
)
from capstone.factors.tree import (
    BINARY_WINDOW_FUNCS,
    BinOp,
    Call,
    Node,
    ParseError,
    parse,
    unparse,
    walk,
)

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


def _draft(item: dict) -> Draft:
    expression = str(item.get("expression", ""))
    rationale = str(item.get("rationale", ""))
    try:
        return Draft(expression, rationale, parse(expression))
    except ParseError as exc:
        return Draft(expression, rationale, None, str(exc))
