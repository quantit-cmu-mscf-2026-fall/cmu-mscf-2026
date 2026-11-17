"""Offline verification for evaluator v2; this entrypoint cannot issue paid requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from html import escape
from pathlib import Path

from capstone import runlog
from capstone.factors.research_artifacts import atomic_json
from capstone.factors.research_contracts import digest
from capstone.factors.research_evaluator_proof import fixtures, run_proof
from capstone.release import release_identity, runtime_versions


def verify(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__),
        *sorted((repo / "capstone/factors").glob("*.py")),
        *sorted((repo / "capstone/strategy").glob("*.py")),
        repo / "capstone/pipeline/budget.py",
    ]
    manifest = {
        "kind": "synthetic_evaluator_v2_verification",
        "review_version": 2,
        "seed": 404,
        "fixture_hash": digest(fixtures()),
        "paid_request_limit": 0,
        "runtime": runtime_versions(),
        "release": release_identity(),
        "code": {
            p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("verification inputs changed; use a fresh output directory")
    atomic_json(path, manifest)
    atomic_json(root / "answer-key.json", fixtures())
    previous = os.environ.get("CAPSTONE_LEDGER_DIR")
    os.environ["CAPSTONE_LEDGER_DIR"] = str(root / "ledger")
    try:
        if not any(
            e["params"].get("manifest_digest") == digest(manifest) for e in runlog.read_entries()
        ):
            runlog.log_run(
                "evaluator-v2-development-verification",
                seed=404,
                params={
                    "manifest_digest": digest(manifest),
                    "data_version": "self-authored synthetic development fixtures",
                    "financial_trials": 0,
                },
                tags=["nonfinancial", "offline", "evaluator-v2"],
                notes="Registered before fixture checks; no real agent or market data.",
            )
    finally:
        if previous is None:
            os.environ.pop("CAPSTONE_LEDGER_DIR", None)
        else:
            os.environ["CAPSTONE_LEDGER_DIR"] = previous
    proof = run_proof(root)
    assert all(proof["checks"].values())
    atomic_json(root / "proof.json", proof)
    rows = "".join(
        "<tr>"
        + "".join(
            "<td>" + escape(str(v).replace("_", " ")) + "</td>"
            for v in [r["case"], *r["observed"][:-1], "Yes" if r["observed"][-1] else "No"]
        )
        + "</tr>"
        for r in proof["cases"]
    )
    (root / "REPORT.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Evaluator v2 offline proof</title><style>
        body{font:17px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.55}
        table{border-collapse:collapse;width:100%}td,th{padding:.7rem;text-align:left;
        border-bottom:1px solid #ccd5df}</style><h1>Evaluator v2: offline verification</h1>
        <p>All development checks passed. No paid model request or historical performance
        evaluation occurred. The original 62-request pilot and its scores remain unchanged.</p>
        <p>Source fidelity, specification documentation, input readiness and independent
        construction are separate. The table shows fidelity, documentation, data, scope
        and unsafe-claim status. Construction stays unchecked in these cases;
        separate checks exercise passing and failing independent construction.</p>
        <table><tr><th>Synthetic case</th><th>Source fidelity</th><th>Documentation</th>
        <th>Inputs</th><th>Scope</th><th>Unsafe claim?</th></tr>"""
        + rows
        + """</table><p>Known answers are supplied by a scripted reviewer. This establishes
        contracts, rejection/repair limits and replay behavior; real-model semantic reliability
        remains unestablished. Source quotes prove provenance, not entailment.
        No financial admission or automatic controller activation.</p>
        <p><a href="manifest.json">Frozen development manifest</a> ·
        <a href="answer-key.json">Answer key</a> ·
        <a href="proof.json">Machine-readable proof</a></p></html>""",
        encoding="utf-8",
    )
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
