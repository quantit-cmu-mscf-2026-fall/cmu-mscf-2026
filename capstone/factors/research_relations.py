"""Typed research relationships; conceptual proximity never implies independent returns."""

from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.factors.research_contracts import canonical, digest
from capstone.factors.tree import ParseError, factor_id, parse, structural_key


def research_graph(catalogue, states=None):
    nodes, edges = {}, []
    states = states or {}
    for entry in sorted(catalogue, key=canonical):
        brief, claim = entry["brief"], entry["claim"]
        family, method = brief["family"], brief["method_key"]
        nodes[family] = {"id": family, "kind": "family", "label": family}
        nodes[method] = {
            "id": method,
            "kind": "method",
            "label": str(entry.get("title") or method),
            "empirical_relation": "unknown",
        }
        edges.append(
            {
                "from": method,
                "to": family,
                "type": "classified_as",
                "layer": "conceptual",
                "provenance": "frozen curator registry",
            }
        )
        source = digest({"url": claim["source_url"]})
        nodes[source] = {"id": source, "kind": "source", "label": claim["source_url"]}
        edges.append(
            {
                "from": source,
                "to": method,
                "type": "supports" if claim["full_methods"] else "methods_unavailable",
                "layer": "conceptual",
                "provenance": claim["method_location"],
                "status": states.get(brief["packet_id"], "not_processed"),
            }
        )
        for name in claim["required_inputs"]:
            key = "data:" + name
            nodes[key] = {"id": key, "kind": "dependency", "label": name}
            edges.append(
                {
                    "from": method,
                    "to": key,
                    "type": "requires_data",
                    "layer": "conceptual",
                    "provenance": claim["method_location"],
                }
            )
    return {
        "schema_version": 1,
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "edges": sorted(edges, key=canonical),
        "new_performance_evaluations": 0,
    }


def structural_relation(left_expression, right_expression):
    return {
        "type": "structurally_similar",
        "layer": "construction",
        "same_structure": structural_key(parse(left_expression))
        == structural_key(parse(right_expression)),
        "empirical_relation": "unknown",
    }


def attach_constructions(graph, results, benchmark):
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges, formulas = list(graph["edges"]), {}
    methods = {p["id"]: p["method_key"] for p in benchmark["packets"]}
    for arm, packets in sorted(results.items()):
        for packet, result in sorted(packets.items()):
            for proposal in result.get("proposals", []):
                try:
                    node = parse(proposal["expression"])
                except (ParseError, KeyError):
                    continue
                key = factor_id(node)
                nodes[key] = {
                    "id": key,
                    "kind": "formula_prototype",
                    "label": proposal["expression"],
                    "empirical_relation": "unknown",
                }
                edges.append(
                    {
                        "from": methods[packet],
                        "to": key,
                        "type": "proposes",
                        "layer": "construction",
                        "provenance": f"arm {arm}; unverified prototype",
                    }
                )
                formulas[key] = node
    ids = sorted(formulas)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if structural_key(formulas[left]) == structural_key(formulas[right]):
                edges.append(
                    {
                        "from": left,
                        "to": right,
                        "type": "structurally_similar",
                        "layer": "construction",
                        "provenance": "existing canonical AST",
                    }
                )
    return graph | {
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "edges": sorted(edges, key=canonical),
    }


def conceptual_svg(graph):
    families = sorted(n["id"] for n in graph["nodes"] if n["kind"] == "family")
    grouped = {
        f: sorted(
            {e["from"] for e in graph["edges"] if e["type"] == "classified_as" and e["to"] == f}
        )
        for f in families
    }
    states = {
        e["to"]: e.get("status", "not_processed")
        for e in graph["edges"]
        if e["type"] in ("supports", "methods_unavailable")
    }
    labels = {node["id"]: node["label"] for node in graph["nodes"]}
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="760" viewBox="0 0 960 760" '
        'role="img" aria-label="Research mechanisms and their source methods">',
        '<rect width="960" height="760" fill="white"/>',
        '<text x="30" y="30" font-family="sans-serif" font-size="19">'
        "Conceptual coverage — return relationships remain unknown</text>",
    ]
    for i, family in enumerate(families):
        y = 90 + 105 * i
        pieces.append(f'<rect x="30" y="{y - 21}" width="300" height="42" rx="8" fill="#e8eff8"/>')
        pieces.append(
            f'<text x="44" y="{y + 5}" font-family="sans-serif" font-size="15">'
            + escape(family.replace("_", " "))
            + "</text>"
        )
        for j, method in enumerate(grouped[family]):
            my = y + (j - (len(grouped[family]) - 1) / 2) * 46
            color = {
                "needs_data": "#fff1d4",
                "outside_scope": "#eee5ff",
                "construction_verified": "#dff2e8",
            }.get(states.get(method), "#eef1f4")
            pieces.append(f'<line x1="330" y1="{y}" x2="565" y2="{my}" stroke="#8294aa"/>')
            pieces.append(
                f'<rect x="565" y="{my - 19}" width="360" height="38" rx="8" fill="{color}">'
                "<title>" + escape(states.get(method, "not_processed")) + "</title></rect>"
            )
            pieces.append(
                f'<text x="580" y="{my + 5}" font-family="sans-serif" font-size="15">'
                + escape(labels[method][:43])
                + "</text>"
            )
    pieces.append(
        '<text x="30" y="735" font-family="sans-serif" font-size="14">'
        "Lines show curated classification; distance does not measure correlation."
        "</text></svg>"
    )
    return "".join(pieces)


def return_relation(left, right, *, as_of, basis, costs, data_version, invested=None):
    """Only explicitly supplied returns enter this diagnostic, filtered before calculation."""
    if not all((basis, costs, data_version)):
        raise ValueError("return basis, cost declaration and source version required")
    cutoff = pd.Timestamp(as_of)
    if cutoff > pd.Timestamp("2020-12-31"):
        raise ValueError("discovery cutoff exceeded")
    for series in (left, right):
        if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_unique:
            raise ValueError("unique dated returns required")
    frame = pd.concat([left.loc[left.index <= cutoff], right.loc[right.index <= cutoff]], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if invested is not None:
        frame = frame.loc[
            invested.reindex(frame.index, fill_value=False).fillna(False).astype(bool)
        ]
    valid = len(frame) >= 252 and (frame.std(ddof=1) > 0).all()
    correlation = float(frame.iloc[:, 0].corr(frame.iloc[:, 1])) if valid else None
    if correlation is not None and not np.isfinite(correlation):
        correlation = None
    return {
        "type": "return_correlated",
        "layer": "empirical",
        "basis": basis,
        "costs": costs,
        "data_version": data_version,
        "as_of": cutoff.date().isoformat(),
        "observations": len(frame),
        "correlation": correlation,
        "status": "unknown" if correlation is None else "measured",
        "sample_start": None if frame.empty else frame.index[0].date().isoformat(),
        "sample_end": None if frame.empty else frame.index[-1].date().isoformat(),
        "cash_gap_sensitivity": "both_invested" if invested is not None else "full_series",
        "positive_group_threshold": 0.80,
        "same_positive_group": correlation is not None and correlation >= 0.80,
    }


def coverage_scorecard(catalogue, states):
    counts = Counter(states.values())
    return {
        "packet_states": dict(sorted(states.items())),
        "outcomes": dict(sorted(counts.items())),
        "assigned_families": sorted(
            {e["brief"]["family"] for e in catalogue if e["brief"]["packet_id"] in states}
        ),
        "executable_strategies": 0,
        "financial_admissions": 0,
    }


def write_relationship_report(path, graph, scorecard):
    """Self-contained readable graph inventory; all untrusted labels are escaped."""
    methods = [n for n in graph["nodes"] if n["kind"] == "method"]
    states = {
        e["to"]: e.get("status", "not_processed")
        for e in graph["edges"]
        if e["type"] in ("supports", "methods_unavailable")
    }
    rows = "".join(
        "<tr><td>"
        + escape(n["label"])
        + "</td><td>"
        + escape(states.get(n["id"], "not_processed"))
        + "</td><td>unknown</td></tr>"
        for n in methods
    )
    edges = "".join(
        "<tr>"
        + "".join("<td>" + escape(str(e[k])) + "</td>" for k in ("from", "type", "to", "layer"))
        + "</tr>"
        for e in graph["edges"]
    )
    svg = conceptual_svg(graph)
    html = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Research relationships</title><style>
    body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#20303f}
    table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{padding:.6rem;text-align:left;
    border-bottom:1px solid #ccd5df;overflow-wrap:anywhere}pre{white-space:pre-wrap}
    .scroll{overflow-x:auto}svg{width:100%;height:auto;min-width:720px}
    h1{font-size:1.8rem}</style><h1>Research relationships</h1>
    <p>Conceptual families and construction lineage are distinct from measured return
    relationships. No new strategy returns or financial admissions are produced here.</p>
    """
        + '<div class="scroll">'
        + svg
        + "</div>"
        + """<h2>Methods</h2><table><tr><th>Method</th><th>Research status</th>
        <th>Return relationship</th></tr>"""
        + rows
        + """</table>
    <h2>Typed edges</h2><div class="scroll"><table><tr><th>From</th><th>Relationship</th>
    <th>To</th><th>Layer</th></tr>"""
        + edges
        + "</table></div><h2>Coverage</h2><pre>"
        + escape(canonical(scorecard))
        + "</pre></html>"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
    Path(path).with_suffix(".svg").write_text(svg, encoding="utf-8")
