"""Reviewed method requirements, separate from performance and model opinion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[2] / "config" / "research_contracts.json"


def canonical(value) -> str:
    def normalize(item):
        if isinstance(item, dict):
            if len({str(k) for k in item}) != len(item):
                raise ValueError("ambiguous JSON object keys")
            return {str(k): normalize(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        return item

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def implementation_digest(*, validation=False):
    root = Path(__file__).resolve().parents[2]
    paths = [
        "capstone/strategy/research.py",
        "capstone/strategy/recorded.py",
        "capstone/strategy/reference.py",
        "capstone/factors/reference.py",
        "capstone/pipeline/replay.py",
        "capstone/execution/planning.py",
    ]
    if validation:
        paths += [
            "capstone/search_fdr.py",
            "capstone/validation/gate.py",
            "capstone/validation/funnel.py",
            "capstone/validation/multitest.py",
            "capstone/strategy/research_spec.py",
            "capstone/pipeline/actions.py",
            "capstone/pipeline/market_costs.py",
            "capstone/pipeline/historical.py",
            "capstone/pipeline/study.py",
            "capstone/pipeline/study_statistics.py",
        ]
    contents = {}
    for name in paths:
        path = root / name
        contents[name] = (
            hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
            if path.exists()
            else None
        )
    return digest(contents)


def contracts() -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    claims = catalog["claims"]
    if catalog["schema_version"] != 2 or len({c["id"] for c in claims}) != 15:
        raise ValueError("expected the versioned fifteen-claim research audit")
    for claim in claims:
        if claim["status"] not in {"executable", "diagnostic", "blocked"}:
            raise ValueError("unknown claim status")
        if claim["paper"] not in catalog["papers"] or not claim["check"]:
            raise ValueError("every claim requires its source and construction requirement")
        if claim["status"] == "executable" and not claim["method"]:
            raise ValueError("an executable claim needs a reference method")
    return catalog


def claim_for(claim_id: str, *, executable: bool = False) -> dict:
    claim = next((c for c in contracts()["claims"] if c["id"] == claim_id), None)
    if claim is None:
        raise ValueError(f"unknown research claim {claim_id!r}")
    if executable and claim["status"] != "executable":
        raise ValueError(f"claim is {claim['status']}: {claim['check']}")
    return claim


def method_digest(claim_id: str) -> str:
    claim = claim_for(claim_id)
    catalog = contracts()
    return digest(
        {
            "claim": claim,
            "paper": catalog["papers"][claim["paper"]],
            "schema_version": 2,
            "universe": catalog["universe"],
            "execution": catalog["execution"],
            "implementation": implementation_digest(),
        }
    )
