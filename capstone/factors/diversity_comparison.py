"""Bounded comparison of actual generated constructions, never family labels alone."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from capstone.factors.research_benchmark import METHOD_FIELDS
from capstone.factors.research_contracts import digest
from capstone.factors.tree import ParseError, parse, structural_key


def construction_inventory(results):
    rows, structures = [], {}
    for arm, cases in sorted(results.items()):
        for packet, artifact in sorted(cases.items()):
            method = artifact.get("proposed_spec")
            expressions = []
            for proposal in artifact.get("proposals", []):
                expression = proposal.get("expression", "")
                try:
                    key = repr(structural_key(parse(expression)))
                except (ParseError, ValueError, TypeError):
                    key = None
                expressions.append(
                    {
                        "expression": expression,
                        "structure": key,
                        "status": proposal.get("status"),
                        "reason": proposal.get("reason", ""),
                    }
                )
                if key:
                    structures.setdefault((arm, key), []).append(packet)
            rows.append(
                {
                    "arm": arm,
                    "packet": packet,
                    "status": artifact["status"],
                    "method_fields": None
                    if method is None
                    else {k: method[k] for k in (*METHOD_FIELDS, "profile")},
                    "formulas": expressions,
                    "executable_verified": False,
                    "formal_admission": False,
                }
            )
    return {
        "cases": rows,
        "structural_repetitions": [
            {"arm": arm, "packets": packets, "structure": structure}
            for (arm, structure), packets in sorted(structures.items())
            if len(packets) > 1
        ],
        "semantic_duplicates": "requires source-linked analyst review; syntax is insufficient",
        "financial_evidence": False,
    }


def blind_packets(benchmark, results):
    packets, mapping = [], {}
    for packet in benchmark["packets"]:
        artifacts = {}
        for arm in sorted(results):
            if packet["id"] in results[arm]:
                label = digest({"packet": packet["id"], "arm": arm, "seed": 404})[:12]
                artifacts[label] = results[arm][packet["id"]]
                mapping[label] = {"packet": packet["id"], "arm": arm}
        packets.append(
            {
                "packet": packet["id"],
                "source_url": packet["source_url"],
                "method_location": packet["method_location"],
                "source_methods": packet["methods"],
                "rubric": packet["expected"]["rubric"],
                "artifacts": artifacts,
            }
        )
    return packets, mapping


def summary(results, accounting):
    return {
        "arms": {
            arm: {
                "completed_attempts": len(cases),
                "statuses": dict(Counter(r["status"] for r in cases.values())),
                "formula_proposals": sum(len(r.get("proposals", [])) for r in cases.values()),
                "documented_specs": sum("proposed_spec" in r for r in cases.values()),
                "verified_executable_strategies": 0,
            }
            for arm, cases in sorted(results.items())
        },
        "outbound_requests": accounting["outbound_reservations"],
        "judgment": "analyst review pending; no automatic winner from labels or syntax",
        "source_reuse": True,
        "autonomous_discovery_tested": False,
        "formal_admission": False,
    }


def specs(data_version, anchor="1992-12-31"):
    from capstone.strategy.diversified import DiversifiedStrategySpec
    from capstone.strategy.reference import ReferenceStrategySpec
    from capstone.strategy.research import contracts

    family = next(
        c["id"]
        for c in contracts()["claims"]
        if c.get("method") == "momentum_6m" and c["status"] == "executable"
    )
    return tuple(
        s
        for side in ("long_only", "long_short")
        for s in (
            ReferenceStrategySpec(family, data_version, anchor, side),
            DiversifiedStrategySpec("weekly_reversal", data_version, anchor, side),
            DiversifiedStrategySpec("monthly_low_volatility", data_version, anchor, side),
        )
    )


def frozen_constructions(data_version):
    return [
        {
            "id": s.id,
            "method": s.method,
            "side": s.side,
            "spec": asdict(s),
            "origin": "prespecified analyst reference adaptation; not an agent discovery",
        }
        for s in specs(data_version)
    ]


def target_relationships(members, books, universe, dates):
    """Compare planned weights; do not confuse these with drifting account holdings."""
    import numpy as np
    import pandas as pd

    rows = []
    for side in ("long_only", "long_short"):
        selected = sorted((s for s in members if s.side == side), key=lambda s: s.id)
        for i, left in enumerate(selected):
            for right in selected[i + 1 :]:
                observations = []
                for at in dates:
                    weights = []
                    for s in (left, right):
                        past = sorted(d for d in books.get(s.id, {}) if d <= at)
                        if not past:
                            break
                        book = books[s.id][past[-1]]
                        book = book.where(universe.loc[at].reindex(book.index).fillna(False), 0)
                        weights.append(book[book != 0])
                    if len(weights) != 2:
                        continue
                    a, b = weights
                    union = set(a.index) | set(b.index)
                    aligned = pd.concat([a, b], axis=1).fillna(0).to_numpy()
                    norm = np.linalg.norm(aligned[:, 0]) * np.linalg.norm(aligned[:, 1])
                    observations.append(
                        {
                            "date": str(at.date()),
                            "jaccard": len(set(a.index) & set(b.index)) / len(union)
                            if union
                            else None,
                            "signed_cosine": float(aligned[:, 0] @ aligned[:, 1] / norm)
                            if norm
                            else None,
                        }
                    )
                valid = [r["jaccard"] for r in observations if r["jaccard"] is not None]
                rows.append(
                    {
                        "layer": "construction",
                        "side": side,
                        "a": left.method,
                        "b": right.method,
                        "mean_target_name_jaccard": float(np.mean(valid)) if valid else None,
                        "observations": observations,
                        "basis": (
                            "planned formation weights after current-universe exits; "
                            "not account holdings"
                        ),
                    }
                )
    return rows
