"""Prespecified portfolio comparisons and prior-only saved allocation schedules."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from capstone.execution.replay_identity import frame_identity
from capstone.strategy.allocation_state import BASIS, CLOCK, make_decision, pool_at
from capstone.strategy.program_schedule import ProgramSchedule, session_date
from capstone.strategy.research import digest
from capstone.strategy.robust_allocation import AllocationPolicyV2
from capstone.strategy.saved_portfolio import SavedProgramPortfolioSpec

MODES = ("equal_weight", "median_window_sharpe", "shrunk_max_sharpe")
PROFILES = ("long_only", "long_short")
SCENARIOS = {"base": 1, "doubled_cost": 2, "zero_cost_counterfactual": 0}
POOL_SCOPE = "retrospective_full_window_completion_conditioned_pool"
# v1: full pool, zero cash and leave-one-family-out. v2 appends exposure-matched controls,
# so every v1 row (and its id) is unchanged; manifests without a set name are v1.
LEGACY_COMPARISON_SET = "pooled-portfolio-comparisons-v1"
COMPARISON_SET = "pooled-portfolio-comparisons-v2"
# v3: the v2 slots over a chronological prefix-run source. Each slot's pool_ids are
# candidates; the pool at each decision is the receipts available strictly before it.
CHRONOLOGICAL_COMPARISON_SET = "pooled-portfolio-comparisons-v3"
CHRONOLOGICAL_SCOPE = "chronological_prior_qualified_prefix_pool"
# A prefix run registers its controls on `benchmark-observed-prefix-v1`, and the
# chronological control is built from their receipts. A prefix run registered before
# that has no control to build from and keeps this blocker.
PREFIX_CONTROL_BLOCKER = "benchmark_control_unavailable_on_prefix_route"
CONTROL_KINDS = ("cap_weight_universe_proxy", "equal_weight_universe_proxy")
EXPOSURE_RULE = (
    "universe-proxy net return x (portfolio gross / proxy gross) at the latest execution-se"
    "ssion close strictly before each session; zero-return cash through the first execution"
)


def _clock(specs, source):
    calendar = pd.DatetimeIndex(source["inputs"].panel.calendar)
    dates = source["matrix"].index
    before = calendar[calendar < dates[0]]
    if before.empty:
        return None, [], ["prior_completed_session_missing"]
    prior = before[-1]
    anchors = {spec.anchor for spec in specs}
    clock = ProgramSchedule(kind="month_end", interval=1)
    if len(anchors) > 1 or any(spec.schedule != clock for spec in specs):
        return None, [], ["only_synchronized_monthly_programs_are_supported"]
    anchor = next(iter(anchors)) if anchors else str(prior.date())
    try:
        if not clock.decision_due(calendar, anchor, prior):
            return (
                anchor,
                [],
                ["evaluation_origin_is_not_after_scheduled_monthly_formation"],
            )
        formations = clock.formations(calendar, anchor, dates[-1])
    except ValueError as error:
        return anchor, [], [str(error)]
    return anchor, [str(day.date()) for day in formations], []


def _chronological(source):
    return source.get("membership_clock") == CLOCK


def comparisons(source, version=None):
    chronological = _chronological(source)
    if version is None:
        version = CHRONOLOGICAL_COMPARISON_SET if chronological else COMPARISON_SET
    allowed = (
        {CHRONOLOGICAL_COMPARISON_SET} if chronological else {LEGACY_COMPARISON_SET, COMPARISON_SET}
    )
    if version not in allowed:
        raise ValueError("unknown portfolio comparison set " + repr(version))
    result = []
    for profile in PROFILES:
        full = tuple(spec for spec in source["specs"] if spec.side == profile)
        families = sorted(
            {row["family"] for row in source["inclusions"] if row["profile"] == profile}
        )
        groups = [("full_pool", None, full), ("zero_cash", None, ())]
        groups.extend(
            (
                "leave_one_family_out",
                family,
                tuple(spec for spec in full if spec.mechanism_id != family),
            )
            for family in families
        )
        for mode in MODES:
            for kind, omitted, members in groups:
                anchor, decisions, blockers = _clock(members, source)
                row = {
                    "profile": profile,
                    "mode": mode,
                    "kind": kind,
                    "omitted_family": omitted,
                    "pool_ids": sorted(spec.id for spec in members),
                    "qualified_full_pool_ids": sorted(spec.id for spec in full),
                    "families": sorted({spec.mechanism_id for spec in members}),
                    "pool_selection_scope": POOL_SCOPE,
                    "comparison_scope": (
                        "identical_pool_control_no_qualified_family_member"
                        if kind == "leave_one_family_out"
                        and omitted not in {spec.mechanism_id for spec in full}
                        else "empty_pool_cash_control"
                        if not members
                        else "pooled_stock_comparison"
                    ),
                    "anchor": anchor,
                    "decision_sessions": decisions,
                    "construction_blockers": blockers,
                    "coefficient_basis": BASIS,
                    "source_return_basis": "funded_total_return",
                    "cash_reference": "zero_return_cash",
                    "formal_admission": False,
                }
                if chronological:
                    row.update(pool_selection_scope=CHRONOLOGICAL_SCOPE, membership_clock=CLOCK)
                row["id"] = digest(row)
                result.append(row)
    if version == LEGACY_COMPARISON_SET:
        return result
    # Universe proxies are long-only funded controls (HistoricalBenchmarkSpec), so only the
    # long-only full pools get an exposure-matched control.
    registered = (
        (source.get("benchmark_controls") or {}).get("controls", [])
        if chronological
        else source["report"].get("benchmarks", [])
    )
    proxies = {row["kind"]: row["id"] for row in registered if row["profile"] == "long_only"}
    for full in [row for row in result if row["kind"] == "full_pool"]:
        if full["profile"] != "long_only":
            continue
        for kind in CONTROL_KINDS:
            row = {
                "profile": full["profile"],
                "mode": full["mode"],
                "kind": "exposure_matched_control",
                "omitted_family": None,
                "control_kind": kind,
                "benchmark_control_id": proxies.get(kind),
                "matched_comparison_id": full["id"],
                "pool_ids": [],
                "comparison_scope": "exposure_matched_universe_proxy_control",
                "exposure_rule": EXPOSURE_RULE,
                "construction_blockers": []
                if kind in proxies
                else [PREFIX_CONTROL_BLOCKER if chronological else "benchmark_control_missing"],
                "source_return_basis": "funded_total_return",
                "cash_reference": "zero_return_cash",
                "formal_admission": False,
            }
            row["id"] = digest(row)
            result.append(row)
    return result


def prefix_evidence(source, identifiers, frame, decision):
    """Qualify upstream first; a full future report ID is not a decision input.

    This binds actual earlier funded observations and static construction/cost
    contracts, conditional on the retrospectively qualified fixed pool.
    """
    specs = {spec.id: spec for spec in source["specs"]}
    if _chronological(source):
        return _receipt_evidence(source, identifiers, frame, decision, specs)
    contracts = {
        row["strategy_id"]: row["scenarios"]["base"]["execution_contract"]["id"]
        for row in source["report"]["candidates"]
        if row["status"] == "evaluated"
    }
    return digest(
        {
            "version": "qualified-funded-return-prefix-v1",
            "return_prefix": frame_identity(frame.loc[frame.index < decision], decision),
            "leaves": {
                name: {
                    "claim_id": specs[name].claim_id,
                    "method_spec_id": specs[name].method_spec_id,
                    "construction_id": specs[name].construction_id,
                    "source_ids": list(specs[name].source_ids),
                    "base_execution_contract": contracts[name],
                }
                for name in identifiers
            },
            "data_version": source["inputs"].panel.data_version,
            "return_basis": "funded_total_return",
            "cash_reference": "zero_return_cash",
            "costs": source["manifest"]["costs"],
            "source_scenario": "base",
            "pool_selection_scope": POOL_SCOPE,
        }
    )


def _receipt_evidence(source, identifiers, frame, decision, specs):
    """Bind only receipts available before the decision; later receipts cannot enter."""
    events = [row for row in source["membership"] if row["strategy_id"] in identifiers]
    latest, departed = pool_at(events, decision)
    end = max((row["available_at"] for row in latest.values()), default=None)
    known = frame.loc[frame.index < decision, sorted(latest)]
    return digest(
        {
            "version": "qualified-funded-return-prefix-v2",
            "return_prefix": frame_identity(known.loc[:end] if end else known.iloc[:0], decision),
            "receipts": {name: row["receipt_id"] for name, row in sorted(latest.items())},
            "departed": dict(sorted(departed.items())),
            "leaves": {
                name: {
                    "claim_id": specs[name].claim_id,
                    "method_spec_id": specs[name].method_spec_id,
                    "construction_id": specs[name].construction_id,
                    "source_ids": list(specs[name].source_ids),
                }
                for name in sorted(latest)
            },
            "prefix_execution_contract": source["prefix_contract"],
            "data_version": source["inputs"].panel.data_version,
            "return_basis": "funded_total_return",
            "cash_reference": "zero_return_cash",
            "costs": source["manifest"]["costs"],
            "source_scenario": "base",
            "pool_selection_scope": CHRONOLOGICAL_SCOPE,
        }
    )


def saved_portfolio(comparison, source, policy):
    if comparison["construction_blockers"]:
        raise ValueError("; ".join(comparison["construction_blockers"]))
    by_id = {spec.id: spec for spec in source["specs"]}
    identifiers = comparison["pool_ids"]
    frame = source["matrix"].reindex(columns=identifiers)
    calendar = pd.DatetimeIndex(source["inputs"].panel.calendar)
    chronological = _chronological(source)
    events = tuple(row for row in source.get("membership", ()) if row["strategy_id"] in identifiers)
    pool = {"membership": events} if chronological else {"pool_ids": identifiers}
    rows = []
    for value in comparison["decision_sessions"]:
        day = session_date(value)
        following = calendar[calendar.get_loc(day) + 1]
        rows.append(
            make_decision(
                frame,
                as_of=day,
                execution_session=following,
                **pool,
                profile=comparison["profile"],
                input_evidence_id=prefix_evidence(source, identifiers, frame, day),
                previous=rows[-1] if rows else None,
                mode=comparison["mode"],
                policy=policy,
            )
        )
    return SavedProgramPortfolioSpec(
        specs=tuple(by_id[key] for key in identifiers),
        returns=frame,
        profile=comparison["profile"],
        data_version=source["inputs"].panel.data_version,
        anchor=comparison["anchor"],
        policy=policy,
        allocation_mode=comparison["mode"],
        schedule=tuple(rows),
        **({"membership_clock": CLOCK, "membership": events} if chronological else {}),
    )


def allocation_summary(saved):
    rows = saved.schedule
    invested = [row for row in rows if any(value > 0 for value in row["weights"].values())]
    return {
        "definition_id": saved.id,
        "decision_count": len(rows),
        "first_positive_allocation": invested[0]["decision_session"] if invested else None,
        "insufficient_history_decisions": sum(
            row["status"] == "insufficient_history_cash" for row in rows
        ),
        "all_cash_decisions": sum(
            not any(value > 0 for value in row["weights"].values()) for row in rows
        ),
        "policy": asdict(saved.policy),
        "cold_start": (
            "cash until the frozen policy has sufficient earlier completed leaf-r"
            "eturn history; no invented warmup returns"
        ),
        "coefficient_convention": "fraction_of_funded_capital; positive cash earns zero",
        "training": (
            "source completed base scenario only; prior observations strictly before each decision"
        ),
        "performance_basis": "pooled stock replay, never a weighted standalone-return stream",
    }


def default_policy():
    return AllocationPolicyV2()
