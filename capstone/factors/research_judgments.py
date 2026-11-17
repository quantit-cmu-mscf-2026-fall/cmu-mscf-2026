"""Evaluator v2: grounded documentation, input readiness and construction stay separate."""

from __future__ import annotations

from dataclasses import asdict

from capstone.factors.research_benchmark import METHOD_FIELDS
from capstone.factors.research_contracts import ConstructionEvidence, MethodSpec

VERSION = 2
VERDICTS = ("supported", "disclosed_unspecified", "missing", "underspecified", "contradicted")


def artifact_text(value):
    """Evidence quotes bind to retained string values, independent of JSON row ordering."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(artifact_text(value[k]) for k in sorted(value))
    if isinstance(value, list):
        return "\n".join(artifact_text(v) for v in value)
    return ""


def review_tool(labels):
    """Required object keys enforce one judgment per artifact without an open-length array."""

    def record(verdicts):
        return {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": list(verdicts)},
                "source_quote": {"type": "string"},
                "artifact_quote": {"type": "string"},
            },
            "required": ["verdict", "source_quote", "artifact_quote"],
            "additionalProperties": False,
        }

    fields = {
        "type": "object",
        "properties": {name: record(VERDICTS) for name in METHOD_FIELDS},
        "required": list(METHOD_FIELDS),
        "additionalProperties": False,
    }
    item = {
        "type": "object",
        "properties": {
            "mechanism": record(("supported", "unassessable", "contradicted")),
            "specification": fields,
        },
        "required": ["mechanism", "specification"],
        "additionalProperties": False,
    }
    keys = sorted(labels)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("distinct blinded artifact identities required")
    return {
        "name": "review_methods_v2",
        "description": "Assess source fidelity and documented rules, never execution readiness.",
        "input_schema": {
            "type": "object",
            "properties": {key: item for key in keys},
            "required": keys,
            "additionalProperties": False,
        },
    }


def validate_judgments(answer, packet, artifacts):
    if not isinstance(answer, dict) or set(answer) != set(artifacts):
        raise ValueError("return exactly one keyed judgment for every blinded artifact")
    source = packet["methods"] + "\n" + (packet.get("method_excerpt") or "")
    for key, item in answer.items():
        if not isinstance(item, dict) or set(item) != {"mechanism", "specification"}:
            raise ValueError("review dimensions cannot be added or omitted")
        fields = item["specification"]
        if not isinstance(fields, dict) or set(fields) != set(METHOD_FIELDS):
            raise ValueError("address every specification field separately")
        text = artifact_text(artifacts[key])
        for name, row in {"mechanism": item["mechanism"], **fields}.items():
            allowed = (
                ("supported", "unassessable", "contradicted") if name == "mechanism" else VERDICTS
            )
            if not isinstance(row, dict) or set(row) != {
                "verdict",
                "source_quote",
                "artifact_quote",
            }:
                raise ValueError("use evidence-backed reason codes, not free-form flags")
            if row["verdict"] not in allowed or any(type(v) is not str for v in row.values()):
                raise ValueError("invalid verdict or evidence type")
            sq, aq = row["source_quote"], row["artifact_quote"]
            if (sq and sq not in source) or (aq and aq not in text):
                raise ValueError("evidence quote is absent from the frozen source or artifact")
            if row["verdict"] in ("supported", "contradicted", "underspecified") and not (
                sq.strip() and aq.strip()
            ):
                raise ValueError("substantive judgments require source and artifact evidence")
            if not packet["full_methods"] and row["verdict"] in (
                "supported",
                "contradicted",
                "underspecified",
            ):
                raise ValueError("unavailable primary methods cannot support a method judgment")
            if row["verdict"] == "missing" and aq:
                raise ValueError("an omitted field cannot simultaneously quote a retained rule")
            if row["verdict"] == "missing" and name != "mechanism":
                structured = [
                    artifacts[key].get("proposed_spec", {}),
                    artifacts[key].get("interpretation", {}).get("method", {}),
                ]
                if any(isinstance(v.get(name), str) and v[name].strip() for v in structured):
                    raise ValueError("rule is present; assess fidelity or underspecification")
            if row["verdict"] == "disclosed_unspecified" and (not aq.strip() or sq):
                raise ValueError("an unspecified source rule needs an explicit artifact disclosure")
    return answer


def assess(packet, artifact, judgment, qualified_inputs=(), *, verifier=None):
    """Only the caller's qualified inventory and executable verifier control readiness."""
    validate_judgments({"artifact": judgment}, packet, {"artifact": artifact})
    if isinstance(qualified_inputs, str):
        raise ValueError("qualified inputs must come from a trusted inventory")
    qualified_inputs = tuple(qualified_inputs)
    if any(type(n) is not str or not n for n in qualified_inputs):
        raise ValueError("qualified inputs must come from a trusted inventory")
    interpretation = artifact.get("interpretation", {})
    declared = interpretation.get("required_inputs", [])
    if not isinstance(declared, list) or any(type(n) is not str for n in declared):
        raise ValueError("invalid additional input requirements")
    required = sorted(set(packet["required_inputs"]) | set(declared))
    missing = sorted(set(required) - set(qualified_inputs))
    scope = (
        "in_scope"
        if all(
            u == "sp500"
            for u in (packet["universe"], interpretation.get("universe", packet["universe"]))
        )
        else "outside_scope"
    )
    fields = {n: row["verdict"] for n, row in judgment["specification"].items()}
    source_changed = interpretation.get("source_url", packet["source_url"]) != packet["source_url"]
    unsupported = (
        source_changed
        or artifact.get("status") == "unsupported_claim"
        or any(
            row["verdict"] == "contradicted"
            for row in [judgment["mechanism"], *judgment["specification"].values()]
        )
    )
    grounded = (
        packet["full_methods"]
        and judgment["mechanism"]["verdict"] == "supported"
        and not unsupported
    )
    complete = all(v in ("supported", "disclosed_unspecified") for v in fields.values())
    construction = {"status": "not_checked"}
    if verifier is not None and grounded and complete and not missing and scope == "in_scope":
        spec = MethodSpec(**artifact["proposed_spec"])
        evidence = verifier(spec)
        if (
            not isinstance(evidence, ConstructionEvidence)
            or evidence.spec_id != spec.identity
            or any(
                type(getattr(evidence, n)) is not bool
                for n in ("independent_example", "timing_passed", "construction_passed")
            )
        ):
            raise ValueError("construction requires matching independent typed evidence")
        construction = {
            "status": "passed" if evidence.passed else "failed",
            "evidence": asdict(evidence),
        }
    unsafe = artifact.get("financial_admission", False) is not False or (
        artifact.get("construction_verified", False) is not False
        and construction["status"] != "passed"
    )
    return {
        "review_version": VERSION,
        "source_fidelity": "unsupported"
        if unsupported
        else "supported"
        if grounded
        else "unassessable",
        "specification": {"status": "complete" if complete else "incomplete", "fields": fields},
        "data": {
            "status": "missing_inputs" if missing else "ready",
            "required": required,
            "missing": missing,
        },
        "scope": scope,
        "construction": construction,
        "method_artifact_passed": bool(grounded and complete),
        "unsafe_readiness_claim": unsafe,
        "judgment": judgment,
        "formal_admission": False,
    }
