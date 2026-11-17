"""Prespecified historical controls, separate from generated candidate trials."""

from __future__ import annotations

import pandas as pd

from capstone.execution.benchmark_runtime import benchmark_due
from capstone.execution.contracts import execution_contract
from capstone.execution.prefix_protocol import (
    BENCHMARK_EXECUTION_VERSION,
    FALLBACK_BENCHMARK_EXECUTION_VERSION,
    HELD_BENCHMARK_EXECUTION_VERSION,
)
from capstone.factors.reference import sessions
from capstone.strategy.historical_benchmark import (
    ELIGIBILITY_RULES,
    EXECUTION_VERSIONS,
    HistoricalBenchmarkSpec,
)
from capstone.strategy.research import digest

V5, V6, V7, V8 = EXECUTION_VERSIONS
# Controls frozen into a prefix run: v2 specs on the completed-prefix route.
PREFIX_POLICY = "observed-prefix-comparisons-v1"
# New prefix registrations replay controls under the v7 benchmark arithmetic.
# Capped fallback exits (v8 benchmark arithmetic) for the latest registrations.
LATEST_PREFIX = "observed-prefix-comparisons-v3"
# Registered control policy -> (benchmark spec version, execution version). New freezes
# use the last; older registrations keep verifying (and running) under their own pair.
POLICIES = {
    "observed-historical-comparisons-v1": (1, V5),
    "observed-historical-comparisons-v2": (2, V5),
    "observed-historical-comparisons-v3": (2, V6),
    "observed-historical-comparisons-v4": (2, V7),
    "observed-historical-comparisons-v5": (2, V8),
    PREFIX_POLICY: (2, BENCHMARK_EXECUTION_VERSION),
    "observed-prefix-comparisons-v2": (2, HELD_BENCHMARK_EXECUTION_VERSION),
    LATEST_PREFIX: (2, FALLBACK_BENCHMARK_EXECUTION_VERSION),
}
LATEST = "observed-historical-comparisons-v5"
ELIGIBILITY = {
    1: "required formation history; audit exclusions; no NaN-to-zero admission",
    2: "required formation history, qualified same-day quote and positive sizing price; "
    "audit exclusions; no NaN-to-zero admission",
}
CONTROLS = (
    ("cash", "long_only"),
    ("cash", "long_short"),
    ("cap_weight_universe_proxy", "long_only"),
    ("equal_weight_universe_proxy", "long_only"),
    ("known_momentum_12_1", "long_only"),
    ("known_momentum_12_1", "long_short"),
)


def freeze_controls(metadata, start, end, *, policy=LATEST):
    """Only calendar/source metadata is examined before results are registered."""
    version, execution_version = POLICIES[policy]
    blockers, anchor = [], None
    try:
        calendar = sessions(metadata["calendar"])
        evaluation = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
        if evaluation.empty or calendar.get_loc(evaluation[0]) == 0:
            raise ValueError("benchmark evaluation requires an actual prior session")
        anchor = str(calendar[calendar.get_loc(evaluation[0]) - 1].date())
    except (KeyError, ValueError, TypeError) as error:
        blockers.append(str(error))
    records = []
    for kind, profile in CONTROLS:
        spec = (
            None
            if anchor is None
            else HistoricalBenchmarkSpec(
                kind,
                metadata["data_version"],
                anchor,
                profile,
                version=version,
                eligibility_rule=ELIGIBILITY_RULES[version],
            )
        )
        record = {
            "kind": kind,
            "profile": profile,
            "specification": None if spec is None else spec.to_record(),
            "execution_contract": None
            if spec is None
            else execution_contract(spec, version=execution_version),
            "registration_blockers": list(blockers),
            "label": kind.replace("_", " "),
            "proxy": kind.endswith("universe_proxy"),
            "official_index_replication": False,
            "momentum_method": "existing sum231 shifted21; demeaned proportional weights",
            "momentum_phase": "21-session clock anchored at actual evaluation prior session",
            "eligibility": ELIGIBILITY[version],
            "initial_account": "fresh declared evaluation capital; no preexisting shares",
            "source_model_slot": False,
            "formal_admission": False,
        }
        record["id"] = digest(record)
        records.append(record)
    return {
        "policy": policy,
        "execution_version": execution_version,
        "controls": records,
        "comparison_selection": "all_prespecified; no winner selection",
    }


def verify_controls(manifest, metadata):
    registered = (manifest.get("benchmark_controls") or {}).get("policy")
    policy = registered if registered in POLICIES else LATEST
    expected = freeze_controls(metadata, manifest["start"], manifest["end"], policy=policy)
    if manifest.get("benchmark_controls") != expected:
        raise ValueError("historical benchmark registration changed or is incompatible")


def benchmark_preflight(spec, inputs, policy):
    qualification = inputs.qualification
    blockers = list(qualification.get("global_blockers", []))
    if not qualification.get("source_hashes", {}).get("daily"):
        blockers.append("daily_source_identity_missing")
    for role in ("calendar", "events"):
        if not qualification.get("sources", {}).get(role, {}).get("qualified"):
            blockers.append(role + "_source_unqualified")
    if not qualification.get("event_return_identities", {}).get("verified"):
        blockers.append("corporate_action_return_identity_unverified")
    fields = (
        {"cap"}
        if spec.kind == "cap_weight_universe_proxy"
        else ({"returns"} if spec.kind == "known_momentum_12_1" else set())
    )
    for name in fields:
        if not isinstance(inputs.panel.fields.get(name), pd.DataFrame):
            blockers.append("feature_missing:" + name)
        if not isinstance((inputs.panel.availability or {}).get(name), pd.Series):
            blockers.append("availability_missing:" + name)
    return {
        "policy": policy,
        "source_preflight_passed": not blockers,
        "source_hashes": qualification.get("source_hashes", {}),
        "blockers": sorted(set(blockers)),
        "formal_admission": False,
    }


def evaluate_controls(root, inputs, manifest, scenario_runner):
    rows = []
    for record in manifest["benchmark_controls"]["controls"]:
        row = {**record, "scenarios": {}, "formal_admission": False}
        blockers = list(record["registration_blockers"])
        spec = (
            None
            if record["specification"] is None
            else HistoricalBenchmarkSpec.from_record(record["specification"])
        )
        if spec is not None:
            row["strategy_id"] = spec.id
            preflight = benchmark_preflight(spec, inputs, manifest["benchmark_controls"]["policy"])
            row["input_preflight"] = preflight
            blockers.extend(preflight["blockers"])
            try:
                if not benchmark_due(spec, inputs.panel, spec.anchor):
                    blockers.append("benchmark_prior_session_is_not_scheduled_formation")
            except ValueError as error:
                blockers.append(str(error))
        if blockers:
            row.update(status="data_or_initialization_blocked", blockers=sorted(set(blockers)))
            for name in manifest["scenarios"]:
                row["scenarios"][name] = {
                    "status": row["status"],
                    "completed": False,
                    "blocker": "; ".join(row["blockers"]),
                }
        else:
            for name, multiplier in manifest["scenarios"].items():
                scenario, _ = scenario_runner(
                    root,
                    spec,
                    inputs,
                    manifest,
                    name,
                    multiplier,
                    execution_version=manifest["benchmark_controls"]["execution_version"],
                )
                row["scenarios"][name] = scenario
            base = row["scenarios"]["base"]
            row["status"] = (
                "evaluated"
                if base["completed"]
                else base["status"]
                if base["status"] == "unqualified"
                else "blocked"
            )
        rows.append(row)
    return rows
