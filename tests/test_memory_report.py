"""Checkpoint reports for learned memory search. Run only with --report.

    pytest -m report --report

Each checkpoint writes one Markdown file to experiments/reports/ (gitignored)
for review. The behaviour itself is tested in test_factor_memory.py.
"""

from __future__ import annotations

import pytest

from capstone.factors.llm import load_config
from capstone.factors.memory import (
    MOTIFS,
    PairStats,
    edit_motif,
    locate_edit,
    parent_context,
    veto_gate,
    vetoed,
)
from capstone.factors.tree import node_count, parse, unparse

pytestmark = pytest.mark.report

MOTIF_EXAMPLES = [
    ("close", "rank(close)"),
    ("rank(close)", "zscore(close, 20)"),
    ("close", "volume / close"),
    ("close + volume", "close + volume + returns"),
    ("ts_mean(close, 5)", "ts_mean(close, 63)"),
    ("ts_mean(close, 10)", "ts_mean(close, 20)"),
    ("ts_mean(close, 5)", "ts_std(close, 252)"),
    ("returns + volume + cap", "returns + volume + close"),
    ("returns", "ts_corr(returns, mkt_return, 63)"),
    ("ts_mean(returns, 5)", "returns"),
    ("shift(close, 5)", "delta(close, 5)"),
    ("shift(returns, 1)", "returns"),
    ("log(volume)", "abs(volume)"),
    ("close + volume + returns", "close + volume"),
    ("ts_mean(close, 5) / ts_mean(volume, 21)", "ts_mean(close, 5) / ts_mean(rank(volume), 21)"),
]
CONTEXT_EXAMPLES = [
    ("-ts_sum(returns, 5)", 0.31, 0),
    ("ts_mean(volume / shares, 21)", 0.08, 2),
    ("rank(close / shift(close, 252)) * cap", 0.12, 4),
    ("ts_cov(returns, mkt_return, 63) / ts_cov(mkt_return, mkt_return, 63)", None, 7),
]


def _cell(text: str) -> str:
    return "`" + text.replace("|", "\\|") + "`"


def test_step1_memory_core(report_dir):
    cfg = load_config()
    lines = ["# Learned memory search: step 1, the memory core", "", "## Label order", ""]
    lines += [f"{i}. `{label}`" for i, label in enumerate(MOTIFS, 1)]
    lines += ["", "## Edit labels on examples", ""]
    lines += ["| Parent | Child | Differing pair | Label |", "|---|---|---|---|"]
    for parent, child in MOTIF_EXAMPLES:
        pair = locate_edit(parse(parent), parse(child))
        where = "none" if pair is None else f"{_cell(unparse(pair[0]))} → {_cell(unparse(pair[1]))}"
        label = edit_motif(parse(parent), parse(child))
        lines.append(f"| {_cell(parent)} | {_cell(child)} | {where} | **{label}** |")
    lines += ["", "## Parent context on examples", ""]
    lines += ["| Parent | Nodes | Q | Times selected | Key |", "|---|---|---|---|---|"]
    for expression, quality, times in CONTEXT_EXAMPLES:
        tree = parse(expression)
        key = parent_context(tree, quality=quality, times_selected=times)
        q = "unscored" if quality is None else f"{quality:.2f}"
        lines.append(f"| {_cell(expression)} | {node_count(tree)} | {q} | {times} | {_cell(key)} |")
    lines += ["", "## When the veto fires (children that all failed to score)", ""]
    lines += [
        f"κ = {cfg.memory_kappa}, τ_c = {cfg.memory_veto_confidence}, "
        f"τ_v = {cfg.memory_veto_failure_rate}. Cell: veto gate / failure rate; ✗ = vetoed.",
        "",
        "| Attempts | " + " | ".join(f"{f} failed" for f in range(9)) + " |",
        "|---" * 10 + "|",
    ]
    for attempts in (1, 2, 3, 4, 5, 8, 12, 20):
        cells = []
        for failures in range(9):
            if failures > attempts:
                cells.append("")
                continue
            s = PairStats(attempts=attempts, failures=failures)
            mark = "✗ " if vetoed(s, cfg) else ""
            g = veto_gate(s, cfg.memory_kappa, cfg.memory_epsilon)
            cells.append(f"{mark}{g:.2f}/{s.failure_rate:.2f}")
        lines.append(f"| {attempts} | " + " | ".join(cells) + " |")
    out = report_dir / "step1_memory_core.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    assert out.stat().st_size > 0


def test_step_evidence_table(report_dir):
    from capstone.factors.memory import MemoryState, evidence, evidence_table

    cfg = load_config()
    ctx = "returns|q2|d0|u1"
    memory = MemoryState()

    def add(motif, status, residual):
        realized = None if status == "invalid" else motif
        memory.update(
            {
                "context": ctx,
                "motif_realized": realized,
                "motif_intended": motif,
                "status": status,
                "residual": residual,
            }
        )

    for r in [0.06, 0.04, 0.08, 0.05, 0.07, 0.03]:
        add("nesting", "admitted", r)
    for r in [-0.04, -0.06, -0.05]:
        add("interaction", "rejected", r)
    add("interaction", "admitted", 0.01)
    for _ in range(5):
        add("rank_switch", "invalid", None)
    add("feature_swap", "admitted", 0.12)
    lines = [
        "# Learned memory search: what the agent reads",
        "",
        f"An example for parents of kind `{ctx}` after 16 children "
        "(made-up outcomes, to show the format).",
        "",
        "```",
        evidence_table(evidence(memory, ctx, cfg)),
        "```",
    ]
    out = report_dir / "evidence_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    assert out.stat().st_size > 0
