"""Private, resumable research artifacts and a readable exploratory report."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from capstone import runlog
from capstone.artifacts import atomic_json, canonical, digest
from capstone.release import release_identity, runtime_versions


def manifest(benchmark, configs, checks, policy):
    repo = Path(__file__).resolve().parents[2]
    # Modules that belong to the research controller are hashed once they exist in the tree.
    optional = [
        repo / "capstone/pipeline/research_experiment.py",
        repo / "capstone/pipeline/budget.py",
        repo / "capstone/execution/journal.py",
    ]
    paths = (
        sorted((repo / "capstone/factors").glob("*.py"))
        + [p for p in optional if p.exists()]
        + [repo / "config/factors.toml"]
    )
    runtime = runtime_versions()
    release = release_identity()
    release["merged_main"] = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", release["commit"], "origin/main"],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    try:
        runtime["anthropic"] = version("anthropic")
    except PackageNotFoundError:
        runtime["anthropic"] = None
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "call_limits": benchmark["call_limits"],
        "research_policy": asdict(policy),
        "configs": {name: asdict(config) for name, config in configs.items()},
        "code": {
            p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "release": release,
        "runtime": runtime,
        "seed": 404,
        "integrity_checks": checks,
        "hidden_sdk_retries": 0,
        "max_automatic_transient_retries": 1,
        "financial_trials": 0,
        "calibration_status": "changed generator/policy; formal financial calibration pending",
        "controller_default": False,
        "holdout_access": False,
        "broker_access": False,
    }


def register(root, frozen):
    previous = os.environ.get("CAPSTONE_LEDGER_DIR")
    os.environ["CAPSTONE_LEDGER_DIR"] = str(root / "ledger")
    try:
        key = digest(frozen)
        if not any(e["params"].get("manifest_digest") == key for e in runlog.read_entries()):
            runlog.log_run(
                "research-controller-pilot-registration",
                seed=404,
                params={
                    "manifest_digest": key,
                    "call_limits": frozen["call_limits"],
                    "data_version": "primary-method packet hashes; no market data",
                    "financial_trials": 0,
                },
                tags=["research-controller", "nonfinancial"],
                notes="Registered before agent outputs; separate from financial trial accounting.",
            )
    finally:
        if previous is None:
            os.environ.pop("CAPSTONE_LEDGER_DIR", None)
        else:
            os.environ["CAPSTONE_LEDGER_DIR"] = previous


def checkpoint(path, payload, manifest_id):
    body = {"manifest_id": manifest_id, "payload": payload}
    atomic_json(path, body | {"digest": digest(body)})


def read_checkpoint(path, manifest_id):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: document[k] for k in ("manifest_id", "payload")}
    if document["manifest_id"] != manifest_id or document["digest"] != digest(body):
        raise ValueError("checkpoint identity or contents changed")
    return document["payload"]


def _case_details(root, frozen):
    sections = []
    for packet in frozen["benchmark"]["packets"]:
        sections.append("<h3>" + escape(packet["title"]) + "</h3><ul>")
        sections.append(
            "<li>Requirements unqualified in this pilot: "
            + escape(", ".join(n.replace("_", " ") for n in packet["required_inputs"]))
            + "</li>"
        )
        for arm in ("A", "B", "C"):
            relative = f"cases/{arm}/{packet['id']}.json"
            path = root / relative
            if not path.exists():
                sections.append("<li>" + arm + ": pending</li>")
                continue
            artifact = read_checkpoint(path, digest(frozen))["artifact"]
            sections.append(
                '<li><a href="'
                + relative
                + '">'
                + arm
                + ": "
                + escape(artifact["status"].replace("_", " "))
                + "</a></li>"
            )
            if arm == "B" and artifact.get("interpretation"):
                method = artifact["interpretation"].get("method", {})
                details = "".join(
                    "<dt>" + escape(name) + "</dt><dd>" + escape(str(value)) + "</dd>"
                    for name, value in method.items()
                )
                sections.append(
                    "<li><details><summary>Agent-proposed method; construction unverified"
                    "</summary><dl>" + details + "</dl></details></li>"
                )
        sections.append("</ul>")
    return "".join(sections)


def _coverage_svg(summary):
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="240" '
        'viewBox="0 0 900 240" role="img" aria-label="Review-confirmed mechanism families">',
        '<rect width="900" height="240" fill="white"/>',
        '<text x="20" y="30" font-family="sans-serif" font-size="18">'
        "Review-confirmed mechanism families — exploratory, not trading returns</text>",
    ]
    for i, arm in enumerate(("A", "B", "C")):
        count = len(summary["metrics"][arm]["grounded_families"])
        y = 65 + i * 48
        pieces.append(
            f'<text x="20" y="{y + 18}" font-family="sans-serif" font-size="16">{arm}</text>'
        )
        pieces.append(f'<rect x="60" y="{y}" width="600" height="28" fill="#edf1f6"/>')
        pieces.append(f'<rect x="60" y="{y}" width="{count * 100}" height="28" fill="#286b8e"/>')
        pieces.append(
            f'<text x="680" y="{y + 19}" font-family="sans-serif" font-size="16">{count} / 6</text>'
        )
    return "".join(pieces) + "</svg>"


def write_pilot_report(root, frozen, summary, accounting, stop_reason=None):
    atomic_json(root / "summary.json", summary)
    atomic_json(root / "accounting.json", accounting)
    svg = _coverage_svg(summary)
    (root / "coverage.svg").write_text(svg, encoding="utf-8")
    rows = []
    for arm, values in summary["metrics"].items():
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + escape(str(value)) + "</td>"
                for value in (
                    arm,
                    values["completed_packets"],
                    values["reviewed_packets"],
                    len(values["grounded_families"]),
                    values["verified_method_artifacts"],
                    values["unsupported_claims"],
                    values["outbound_reservations"],
                )
            )
            + "</tr>"
        )
    sources = "".join(
        "<li>"
        + escape(p["title"])
        + ": "
        + ("methods available" if p["full_methods"] else "methods unavailable")
        + ' — <a href="'
        + escape(p["source_url"], quote=True)
        + '">primary source</a></li>'
        for p in frozen["benchmark"]["packets"]
    )
    html = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Research controller pilot</title>
    <style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#20303f}
    table{border-collapse:collapse;width:100%}td,th{padding:.65rem;border-bottom:1px solid #ccd5df;
    text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow:auto}
    .notice{padding:1rem;background:#fff3d6}h1{font-size:1.8rem}</style>
    <h1>Research controller pilot</h1><div class="notice">"""
        + escape(summary["decision"].replace("_", " "))
        + """</div>
    <p>A: existing pipeline. B: coverage controller. C: filter-heavy comparator.</p>
    <p>Automated exploratory review of method packets; reviewer errors may be correlated.
    Verified method artifacts are source interpretations, not executable strategies.
    No historical returns, financial admissions, broker connection or holdout access.</p>"""
    )
    html += "<p>Evidence mode: " + escape(summary["mode"]) + "</p>"
    if summary.get("review_version") == 2:
        dimensions = {
            arm: {
                name: row[name]
                for name in (
                    "complete_specifications",
                    "data_ready",
                    "verified_constructions",
                    "retained_data_blocked_specs",
                    "unsafe_readiness_claims",
                )
            }
            for arm, row in summary["metrics"].items()
        }
        html += (
            "<h2>Evaluator v2 dimensions</h2><p>Source-faithful documentation is separate "
            "from qualified inputs and independent construction. No v1 success criteria "
            "are reused.</p><pre>" + escape(canonical(dimensions)) + "</pre>"
        )
    if stop_reason:
        html += "<p><strong>Stopped:</strong> " + escape(stop_reason) + "</p>"
    html += (
        """<div class="scroll"><table><tr><th>Arm</th><th>Completed / 8</th><th>Reviewed / 8</th>
    <th>Grounded families</th><th>Verified methods</th><th>Unsupported claims</th>
    <th>Calls / 30</th></tr>"""
        + "".join(rows)
        + """</table></div>
    <p>Recorded outbound reservations: """
        + str(accounting["outbound_reservations"])
        + """ / 120.
    Failed or uncertain requests remain counted. Token usage for failed requests may be unavailable;
    this report does not infer zero charges or a dollar balance.</p>
    <p><a href="relationships.html">Research relationship graph</a> ·
    <a href="manifest.json">Frozen study</a> · <a href="summary.json">Machine-readable scores</a> ·
    <a href="accounting.json">Request accounting</a></p>
    <h2>Registered criteria</h2><pre>"""
        + escape(canonical(summary["criteria"]))
        + """</pre>
    <h2>Primary-method inventory</h2><ul>"""
        + sources
        + "</ul>"
    )
    html += '<div class="scroll">' + svg + "</div>"
    html += "<h2>Case evidence</h2>" + _case_details(root, frozen)
    html += (
        "<h2>Unresolved review differences</h2><pre>"
        + escape(canonical(summary["review_disagreements"]))
        + "</pre>"
    )
    if summary.get("review_version") == 2:
        html = html.replace("<th>Verified methods</th>", "<th>Source-faithful complete specs</th>")
    (root / "REPORT.html").write_text(html + "</html>", encoding="utf-8")
