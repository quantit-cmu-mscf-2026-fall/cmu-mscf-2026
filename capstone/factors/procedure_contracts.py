"""Opt-in full-procedure identities; registration is not financial qualification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from capstone.factors.research_contracts import FAMILIES, canonical, digest

PROCEDURE_KIND = "research_procedure_v1"
KINDS = ("proposal", "refinement", "comparison", "rerun")
SECTIONS = (
    "version generator execution ownership sources qualification allocator feedback arms statistics"
)
# Version 2 is version 1 plus a registered two-tier `evaluation` section. A version-1
# record has no such section, so its bytes and identity are what they were.
TWO_TIER_VERSION = 2
VERSIONS = (1, TWO_TIER_VERSION)
TWO_TIER_RULE = "two-tier-screen-v1"
TWO_TIER_STATISTIC = "search-sharpe-v1"
TWO_TIER_FIELDS = (
    "rule tier1_id statistic survivors_per_family agreement_sample agreement_seed "
    "min_rank_agreement calibration_tier1_error"
)


def fields(row, names):
    if not isinstance(row, dict) or set(row) != set(names.split()):
        raise ValueError("incomplete or unknown procedure fields")


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError("stable logical name required; paths are not scope identities")
    return value


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("content-addressed identity required")
    return value


def day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("canonical decision date required")
    return value


def limit(value):
    if type(value) is not int or value < 0:
        raise ValueError("bounded nonnegative integer required")


def _real(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("a registered two-tier bound must be a number")
    if not low <= value <= high:
        raise ValueError("a registered two-tier bound is out of range")


def two_tier_section(row):
    """Validate a registered two-tier `evaluation` section.

    `tier1_id` names the screen's implementation. The best `survivors_per_family`
    variants of each family by the tier-1 statistic get the exact replay. A sample of
    `agreement_sample` variants, drawn with `agreement_seed` before any result is seen,
    gets the exact replay too; the screen is switched off (every variant replayed) when
    the Spearman rank agreement of tier 1 against tier 2 on it is below
    `min_rank_agreement`. `calibration_tier1_error` is the tier-1 approximation error the
    whole-procedure calibration simulates (noise sd as a multiple of each variant's sd)."""
    fields(row, TWO_TIER_FIELDS)
    if row["rule"] != TWO_TIER_RULE or row["statistic"] != TWO_TIER_STATISTIC:
        raise ValueError("unsupported two-tier evaluation rule")
    identity(row["tier1_id"])
    for name in ("survivors_per_family", "agreement_sample", "agreement_seed"):
        limit(row[name])
    if row["survivors_per_family"] < 1:
        raise ValueError("every family keeps at least one tier-1 survivor")
    if row["agreement_sample"] < 3:
        raise ValueError("a rank-agreement check needs at least three sampled variants")
    _real(row["min_rank_agreement"], -1.0, 1.0)
    _real(row["calibration_tier1_error"], 0.0, 100.0)


def _normalized(row):
    row = json.loads(canonical(row))
    row["ownership"] = sorted(
        ({**item, "aliases": sorted(item["aliases"])} for item in row["ownership"]),
        key=lambda item: item["method_key"],
    )
    row["arms"] = sorted(
        (
            {**arm, "budgets": sorted(arm["budgets"], key=lambda b: b["owner"])}
            for arm in row["arms"]
        ),
        key=lambda arm: arm["name"],
    )
    row["sources"]["prior_study_ids"] = sorted(set(row["sources"]["prior_study_ids"]))
    legacy = row["statistics"]["legacy_scope"]
    legacy["evidence_ids"] = sorted(set(legacy["evidence_ids"]))
    return row


@dataclass(frozen=True)
class ProcedureSpec:
    """Own canonical bytes, so returned nested dictionaries cannot mutate the contract."""

    payload_json: str

    def __post_init__(self):
        row = self.to_record()
        if type(row.get("version")) is not int or row["version"] not in VERSIONS:
            raise ValueError("unsupported procedure version")
        fields(row, SECTIONS + (" evaluation" if row["version"] == TWO_TIER_VERSION else ""))
        if self.payload_json != canonical(row):
            raise ValueError("procedure bytes must be canonical")
        generator = row["generator"]
        fields(generator, "id models prompt_id grammar_id compiler_id code_id runtime_id")
        token(generator["id"])
        if not isinstance(generator["models"], dict) or not generator["models"]:
            raise ValueError("exact configured models and settings required")
        for name in ("prompt_id", "grammar_id", "compiler_id", "code_id", "runtime_id"):
            identity(generator[name])
        execution = row["execution"]
        fields(execution, "implementation_id costs return_basis risk_policy_id")
        identity(execution["implementation_id"])
        identity(execution["risk_policy_id"])
        if execution["return_basis"] != "funded_total_return" or not execution["costs"]:
            raise ValueError("explicit funded return and cost basis required")
        if not isinstance(execution["costs"], dict):
            raise ValueError("exact cost settings required")
        names, aliases, owners, owner_families = set(), set(), set(), {}
        if not isinstance(row["ownership"], list) or not row["ownership"]:
            raise ValueError("canonical family and budget ownership required")
        for owner in row["ownership"]:
            fields(owner, "method_key family budget_owner aliases")
            token(owner["method_key"])
            token(owner["budget_owner"])
            if owner["family"] not in FAMILIES or owner["method_key"] in names:
                raise ValueError("duplicate method or unknown broad coverage family")
            names.add(owner["method_key"])
            owners.add(owner["budget_owner"])
            if owner_families.setdefault(owner["budget_owner"], owner["family"]) != owner["family"]:
                raise ValueError("one budget owner cannot acquire another coverage family")
            if not isinstance(owner["aliases"], list):
                raise ValueError("explicit alias inventory required")
            for alias in owner["aliases"]:
                token(alias)
                if alias in aliases:
                    raise ValueError("an alias has more than one canonical owner")
                aliases.add(alias)
        if names & aliases:
            raise ValueError("alias cannot replace a canonical method")
        sources = row["sources"]
        fields(sources, "knowledge_cutoff knowledge_policy_id reused_discovery prior_study_ids")
        day(sources["knowledge_cutoff"])
        identity(sources["knowledge_policy_id"])
        if type(sources["reused_discovery"]) is not bool:
            raise ValueError("explicit reused-discovery scope required")
        if not isinstance(sources["prior_study_ids"], list):
            raise ValueError("prior discovery inventory required")
        for key in sources["prior_study_ids"]:
            identity(key)
        qualification = row["qualification"]
        fields(qualification, "receipt_policy_id membership_clock decision_clock availability_rule")
        identity(qualification["receipt_policy_id"])
        if (
            qualification["membership_clock"]
            not in {"prior_qualified_prefix", "retrospective_fixed_pool"}
            or qualification["availability_rule"] != "strictly_before_decision"
        ):
            raise ValueError("explicit qualification and membership timing required")
        token(qualification["decision_clock"])
        allocator = row["allocator"]
        fields(allocator, "implementation_id membership_policy_id policy")
        identity(allocator["implementation_id"])
        identity(allocator["membership_policy_id"])
        if not isinstance(allocator["policy"], dict) or not allocator["policy"]:
            raise ValueError("exact allocator policy required")
        feedback = row["feedback"]
        fields(
            feedback,
            "policy_id schema_repairs transient_retries max_refinements_per_parent "
            "max_refinement_depth max_feedback_rounds stopping_rule_id",
        )
        identity(feedback["policy_id"])
        identity(feedback["stopping_rule_id"])
        for name, value in feedback.items():
            if name not in {"policy_id", "stopping_rule_id"}:
                limit(value)
        if feedback["schema_repairs"] > 1 or feedback["transient_retries"] > 1:
            raise ValueError("repair/retry bounds cannot exceed the existing controller")
        if not isinstance(row["arms"], list) or not row["arms"]:
            raise ValueError("all comparison arms and slot ceilings required")
        arms = set()
        for arm in row["arms"]:
            fields(arm, "name classification budgets model_call_limit")
            token(arm["name"])
            if arm["name"] in arms or arm["classification"] not in {"research", "control"}:
                raise ValueError("duplicate or unclassified arm")
            arms.add(arm["name"])
            limit(arm["model_call_limit"])
            seen = set()
            for budget in arm["budgets"]:
                fields(budget, "owner proposal refinement comparison rerun")
                if budget["owner"] not in owners or budget["owner"] in seen:
                    raise ValueError("unowned or duplicate arm budget")
                seen.add(budget["owner"])
                for kind in KINDS:
                    limit(budget[kind])
            if seen != owners:
                raise ValueError("arm must explicitly bound every canonical budget owner")
        statistics = row["statistics"]
        fields(
            statistics,
            "counting_rule bootstrap_rule compatibility_rule final_test_rule_id legacy_scope",
        )
        if (
            statistics["counting_rule"] != "classified-logical-opportunities-v1"
            or statistics["bootstrap_rule"] != "verified-return-columns-v1"
            or statistics["compatibility_rule"] != "exact-procedure-and-scope-v1"
        ):
            raise ValueError("unsupported statistical accounting rule")
        identity(statistics["final_test_rule_id"])
        legacy = statistics["legacy_scope"]
        fields(
            legacy,
            "state reported_counts inherited_classification_id resolution_rule_id "
            "justification evidence_ids effective_scope",
        )
        identity(legacy["inherited_classification_id"])
        counts = legacy["reported_counts"]
        if (
            not isinstance(counts, list)
            or not counts
            or any(type(n) is not int or n < 1 for n in counts)
        ):
            raise ValueError("original reported scope must remain visible")
        if legacy["state"] == "unresolved":
            if any(
                legacy[k]
                for k in ("resolution_rule_id", "justification", "evidence_ids", "effective_scope")
            ):
                raise ValueError("unresolved scope cannot imply a resolution")
        elif legacy["state"] == "registered_resolution":
            identity(legacy["resolution_rule_id"])
            if not isinstance(legacy["justification"], str) or not legacy["justification"].strip():
                raise ValueError("legacy scope resolution needs a scientific justification")
            if not isinstance(legacy["evidence_ids"], list) or not legacy["evidence_ids"]:
                raise ValueError("legacy scope resolution needs saved evidence")
            for key in legacy["evidence_ids"]:
                identity(key)
            scope = legacy["effective_scope"]
            fields(
                scope,
                "combination_rule prior_effective_trials calibration_rule_id scope_description",
            )
            if (
                not isinstance(scope["scope_description"], str)
                or not scope["scope_description"].strip()
            ):
                raise ValueError("inherited effective scope needs a concrete definition")
            if scope["combination_rule"] == "add_registered_prior_count":
                limit(scope["prior_effective_trials"])
                if scope["prior_effective_trials"] < 1 or scope["calibration_rule_id"] is not None:
                    raise ValueError("explicit positive inherited count required")
            elif scope["combination_rule"] == "whole_procedure_calibration":
                identity(scope["calibration_rule_id"])
                if scope["prior_effective_trials"] is not None:
                    raise ValueError("a non-scalar procedure rule cannot invent a scalar count")
            else:
                raise ValueError("unknown inherited/current scope combination rule")
            rule = {
                name: legacy[name]
                for name in (
                    "reported_counts",
                    "inherited_classification_id",
                    "justification",
                    "effective_scope",
                )
            }
            if digest(rule) != legacy["resolution_rule_id"]:
                raise ValueError("inherited effective-scope rule identity changed")
        else:
            raise ValueError("unknown original trial scope cannot be discarded")
        if row["version"] == TWO_TIER_VERSION:
            two_tier_section(row["evaluation"])
        if row != _normalized(row):
            raise ValueError("procedure inventories must use canonical ordering")

    @classmethod
    def freeze(cls, **sections):
        """Version 1 without an `evaluation` section (every existing identity, unchanged);
        version 2 when the procedure opts in to the registered two-tier evaluation."""
        version = TWO_TIER_VERSION if "evaluation" in sections else 1
        try:
            return cls(canonical(_normalized({"version": version, **sections})))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("invalid procedure structure") from exc

    @classmethod
    def from_record(cls, row):
        if row.get("version") not in VERSIONS:
            raise ValueError("unsupported procedure version")
        if (row["version"] == TWO_TIER_VERSION) != ("evaluation" in row):
            raise ValueError("a two-tier evaluation section needs procedure version 2, and only it")
        return cls.freeze(**{key: value for key, value in row.items() if key != "version"})

    @property
    def two_tier(self):
        """The registered two-tier evaluation section, or None (exact replay of every variant)."""
        return self.to_record().get("evaluation")

    def to_record(self):
        return json.loads(self.payload_json)

    @property
    def identity(self):
        return digest({"kind": PROCEDURE_KIND, "payload": self.to_record()})

    def owner(self, method):
        for row in self.to_record()["ownership"]:
            if method == row["method_key"] or method in row["aliases"]:
                return row
        raise ValueError("method or alias has no registered canonical owner")

    def arm(self, name):
        for row in self.to_record()["arms"]:
            if row["name"] == name:
                return row
        raise ValueError("arm was not registered")


SCOPE_RESOLUTION = "procedure_scope_resolution_v1"


def scope_evidence(legacy):
    """The saved `procedure_scope_resolution_v1` payload a registered resolution must cite."""
    return {
        "rule_id": legacy["resolution_rule_id"],
        "inherited_classification_id": legacy["inherited_classification_id"],
        "reported_counts": legacy["reported_counts"],
        "unresolved_scope_items": [],
        "justification": legacy["justification"],
        "effective_scope": legacy["effective_scope"],
    }


def whole_procedure_resolution(legacy, *, calibration_rule_id, justification, scope_description):
    """Resolve the original reported scope by whole-procedure calibration.

    No scalar inherited count is invented (`prior_effective_trials` stays None); the
    procedure's error rate is instead measured end to end under `calibration_rule_id`.
    Returns (resolved legacy_scope, the evidence payload to save before registration).
    """
    if legacy.get("state") != "unresolved":
        raise ValueError("only an unresolved legacy scope can be resolved")
    scope = {
        "combination_rule": "whole_procedure_calibration",
        "prior_effective_trials": None,
        "calibration_rule_id": identity(calibration_rule_id),
        "scope_description": scope_description,
    }
    rule = {
        "reported_counts": legacy["reported_counts"],
        "inherited_classification_id": legacy["inherited_classification_id"],
        "justification": justification,
        "effective_scope": scope,
    }
    resolved = {
        **legacy,
        "state": "registered_resolution",
        "resolution_rule_id": digest(rule),
        "justification": justification,
        "effective_scope": scope,
    }
    evidence = scope_evidence(resolved)
    resolved["evidence_ids"] = [digest({"kind": SCOPE_RESOLUTION, "payload": evidence})]
    return resolved, evidence
