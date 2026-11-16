"""Explicit timing contracts; software construction is separate from admission.

This module is the registry of which strategy kinds may be executed and under which
timing rules. Several strategy kinds live in modules that are not part of every build
(program, portfolio, saved portfolio, reference, research and benchmark specifications).
Each is registered lazily: when its module imports, the kind is accepted exactly as the
registry below says; when the module does not exist yet, a spec of that kind cannot be
constructed, and a route that names the kind refuses with `ExecutionContractBlocked`.
`unavailable_kinds()` lists what the current build lacks. Nothing here substitutes a
different behaviour for a missing kind.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import tomllib
from copy import deepcopy
from functools import cache
from numbers import Integral
from pathlib import Path

from capstone.execution.prefix_protocol import BENCHMARK_EXECUTION_VERSION as BENCHMARK_PREFIX
from capstone.execution.prefix_protocol import (
    BENCHMARK_PREFIX_VERSIONS,
    FALLBACK_BENCHMARK_EXECUTION_VERSION,
    FALLBACK_EXECUTION_VERSION,
    HELD_BENCHMARK_EXECUTION_VERSION,
    HELD_EXECUTION_VERSION,
)
from capstone.execution.prefix_protocol import PROGRAM_VERSIONS as PREFIX_PROGRAM_VERSIONS
from capstone.factors.research_contracts import canonical, digest
from capstone.strategy.compiled import CompiledStrategySpec
from capstone.strategy.contract import StrategySpec
from capstone.strategy.diversified import DiversifiedStrategySpec
from capstone.strategy.recorded import require_current_implementation


class _Absent:
    """Stands in for a spec class whose module is not in this build: nothing is an instance."""


def _module(name):
    """The module, or None when that module itself does not exist in this build.

    A module that exists but fails to import (a missing dependency of its own) raises.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (name == exc.name or name.startswith(exc.name + ".")):
            return None
        raise


def _attr(module, name, default):
    found = _module(module)
    return default if found is None else getattr(found, name)


# Strategy kinds registered lazily: class name -> the module that defines it.
_OPTIONAL_KINDS = {
    "HistoricalBenchmarkSpec": "capstone.strategy.historical_benchmark",
    "PortfolioSpec": "capstone.strategy.portfolio",
    "ProgramStrategySpec": "capstone.strategy.program",
    "BenchmarkSpec": "capstone.strategy.reference",
    "ReferenceStrategySpec": "capstone.strategy.reference",
    "ResearchStrategySpec": "capstone.strategy.research_spec",
    "SavedProgramPortfolioSpec": "capstone.strategy.saved_portfolio",
    "SelectedPortfolioSpec": "capstone.strategy.selected_portfolio",
}


@cache
def _kind(name):
    return _attr(_OPTIONAL_KINDS[name], name, _Absent)


def unavailable_kinds():
    """Names of registered strategy kinds whose module is not in this build."""
    return tuple(sorted(name for name in _OPTIONAL_KINDS if _kind(name) is _Absent))


# Full-window v5/v6/v7 plus the completed-prefix benchmark routes (v6 / v7 arithmetic).
_BENCHMARK = "capstone.strategy.historical_benchmark"
BENCHMARK_ROUTE = _attr(_BENCHMARK, "ROUTE", None)
FULL_WINDOW_BENCHMARK_VERSIONS = _attr(_BENCHMARK, "EXECUTION_VERSIONS", ())
BENCHMARK_VERSIONS = (*FULL_WINDOW_BENCHMARK_VERSIONS, *BENCHMARK_PREFIX_VERSIONS)
PREFIX_VERSIONS = (*PREFIX_PROGRAM_VERSIONS, *BENCHMARK_PREFIX_VERSIONS)
FIXED_SHARE_VERSIONS = ("fixed-share-next-open-v2", "fixed-share-next-open-v3")
PROGRAM_EVENTS_VERSIONS = (
    "program-events-v3",
    "program-events-v4",
    "program-events-v5",
    "program-events-v6",
    "program-events-v7",
)
# Saved-portfolio replay versions come from their own module (absent: none are registered).
_SAVED = "capstone.strategy.saved_portfolio"
SAVED_VERSIONS = _attr(_SAVED, "EXECUTION_VERSIONS", ())
CHRONOLOGICAL_VERSIONS = _attr(_SAVED, "CHRONOLOGICAL_VERSIONS", ())
HELD_EXECUTION_VERSIONS = _attr(_SAVED, "HELD_EXECUTION_VERSIONS", {})
FALLBACK_EXECUTION_VERSIONS = _attr(_SAVED, "FALLBACK_EXECUTION_VERSIONS", {})
# Each route's version plus capped fallback exits for held events that stay
# unqualifiable, held-action classification from the event session only, and the
# registered cap above which a scenario is `unqualified` (evidence.fallback_report).
FALLBACK_VERSIONS = (
    "program-events-v7",
    "observed-benchmark-next-open-v8",
    FALLBACK_EXECUTION_VERSION,
    FALLBACK_BENCHMARK_EXECUTION_VERSION,
    *FALLBACK_EXECUTION_VERSIONS.values(),
)
FALLBACK_RULES = {
    "held_unqualifiable": "exit_at_last_observed_close_capped_cost_v1",
    "held_action_classification": "as_of_event_session_v1",
    # A name without a valid open drops only its own order (no whole-book expiry); a held
    # name exits by the fallback once its open has been missing this many sessions.
    "held_without_open": "drop_order_carry_then_fallback_exit_v1",
    "held_open_gap_exit_sessions": 5,
    # Exceeding either share makes the scenario `unqualified`. These two numbers are a
    # ported policy of the original project, not a team decision.
    "fallback_cap": {"position_day_share": 0.01, "absolute_pnl_share": 0.01},
}
# Held `unsupported` CRSP actions valued from the CRSP total return
# (crsp_actions.valued_events), deferred formation entries re-qualified at the replan and
# close cash-in-lieu; the saved-portfolio and prefix routes adopt them too.
HELD_ACTION_VERSIONS = (
    "program-events-v5",
    "observed-benchmark-next-open-v7",
    "program-events-v6",
    HELD_EXECUTION_VERSION,
    HELD_BENCHMARK_EXECUTION_VERSION,
    *HELD_EXECUTION_VERSIONS.values(),
    *FALLBACK_VERSIONS,
)
# v3/v4+/benchmark v6+: a held name without a sizing price is carried at its last valid
# mark; a held name with a price but no qualified quote exits at the capped half-spread.
HELD_CARRY_VERSIONS = (
    "fixed-share-next-open-v3",
    "program-events-v4",
    "observed-benchmark-next-open-v6",
    BENCHMARK_PREFIX,
    *HELD_ACTION_VERSIONS,
)
# Declared cash-in-lieu default (actions.CASH_IN_LIEU_DEFAULT).
CASH_IN_LIEU_VERSIONS = (
    "program-events-v4",
    "observed-benchmark-next-open-v6",
    BENCHMARK_PREFIX,
    *HELD_ACTION_VERSIONS,
)
# A fresh calendar-window account whose prior session is not a window decision starts
# in cash and first trades at the next scheduled window decision (no look-ahead: the
# calendar alone says when, and no book formed before the start is entered late).
WINDOW_START_VERSIONS = ("program-events-v6", "program-events-v7")
WINDOW_START = "cash_until_next_window_entry_v1"
WINDOW_START_MODE = "new_account_cash_until_window_entry"


class ExecutionContractBlocked(ValueError):
    """An unsupported execution route or state transition, retained as a blocker."""

    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(reason + (": " + detail if detail else ""))


# Version families defined by a module this build may lack: a version of such a family
# that is not registered above belongs to that missing module and is refused by name.
_MODULE_VERSION_FAMILIES = ("saved-program-portfolio-", "observed-benchmark-next-open-")


def refuse_unbuilt_version(version):
    if (
        isinstance(version, str)
        and version.startswith(_MODULE_VERSION_FAMILIES)
        and version not in {*SAVED_VERSIONS, *FULL_WINDOW_BENCHMARK_VERSIONS}
    ):
        raise ExecutionContractBlocked("execution_route_not_in_this_build", version)


def execution_leaves(specs):
    if isinstance(specs, (list, tuple)):
        roots = tuple(specs)
    else:
        roots = (specs,)
    saved_kind = _kind("SavedProgramPortfolioSpec")
    leaf_kinds = (
        _kind("ProgramStrategySpec"),
        CompiledStrategySpec,
        StrategySpec,
        DiversifiedStrategySpec,
        _kind("ReferenceStrategySpec"),
        _kind("ResearchStrategySpec"),
        _kind("BenchmarkSpec"),
    )
    leaves = []
    for spec in roots:
        require_current_implementation(spec)
        if isinstance(spec, _kind("HistoricalBenchmarkSpec")):
            raise ExecutionContractBlocked("historical_benchmark_requires_execution_v5")
        if isinstance(spec, saved_kind) and not spec.specs:
            continue
        if isinstance(spec, _kind("PortfolioSpec")):
            leaves.extend(execution_leaves(spec.specs))
        elif isinstance(spec, leaf_kinds):
            leaves.append(spec)
        else:
            raise ExecutionContractBlocked(
                "unsupported_execution_specification", type(spec).__name__
            )
    if not leaves and not (len(roots) == 1 and isinstance(roots[0], saved_kind)):
        raise ExecutionContractBlocked("empty_execution_profile")
    return tuple(leaves)


def execution_policy(specs, *, purpose="replay"):
    """Pure route validation; no source reads on daily planning calls."""
    if purpose not in {"replay", "frozen"}:
        raise ExecutionContractBlocked("unsupported_execution_purpose")
    program_kind = _kind("ProgramStrategySpec")
    portfolio_kind = _kind("PortfolioSpec")
    selected_kind = _kind("SelectedPortfolioSpec")
    roots = specs if isinstance(specs, (tuple, list)) else (specs,)
    if any(isinstance(spec, _kind("HistoricalBenchmarkSpec")) for spec in roots):
        raise ExecutionContractBlocked("historical_benchmark_requires_execution_v5")
    saved = [spec for spec in roots if isinstance(spec, _kind("SavedProgramPortfolioSpec"))]
    if saved:
        if len(roots) != 1:
            raise ExecutionContractBlocked("saved_allocation_requires_one_root")
        # The new root validates its complete pool/profile/monthly clock at construction.
        execution_leaves(saved[0])
        return {"route": "program-v1", "profile": saved[0].side}
    leaves = execution_leaves(specs)
    if any(isinstance(spec, CompiledStrategySpec) for spec in leaves):
        raise ExecutionContractBlocked("following_close_program_in_next_open_route")
    programs = [isinstance(spec, program_kind) for spec in leaves]
    if any(programs) and not all(programs):
        raise ExecutionContractBlocked("mixed_program_legacy_execution_unqualified")
    profiles = {getattr(spec, "side", "long_short") for spec in leaves}
    if len(profiles) != 1:
        raise ExecutionContractBlocked("mixed_execution_profiles")
    given = specs if isinstance(specs, (list, tuple)) else (specs,)
    if purpose == "frozen" and any(isinstance(spec, selected_kind) for spec in given):
        raise ExecutionContractBlocked("selected_portfolio_retired_use_saved_portfolio")
    if purpose == "frozen" and any(
        isinstance(spec, (portfolio_kind, _kind("BenchmarkSpec"), StrategySpec)) for spec in given
    ):
        raise ExecutionContractBlocked("unsupported_frozen_specification_interface")
    if any(
        isinstance(spec, program_kind) and spec.execution != "following_session_open"
        for spec in leaves
    ):
        raise ExecutionContractBlocked("program_execution_timing_incompatible")
    if all(programs) and len({(spec.anchor, spec.schedule.id) for spec in leaves}) != 1:
        raise ExecutionContractBlocked("asynchronous_program_formations_unqualified")

    def check_allocations(spec):
        if isinstance(spec, selected_kind):
            # Retired for program execution; the class stays for old studies.
            raise ExecutionContractBlocked("selected_portfolio_retired_use_saved_portfolio")
        if isinstance(spec, portfolio_kind):
            if spec.mode != "equal_family":
                raise ExecutionContractBlocked("variable_program_allocations_unqualified")
            for member in spec.specs:
                check_allocations(member)

    if all(programs):
        for spec in roots:
            check_allocations(spec)
    return {
        "route": "program-v1" if all(programs) else "legacy-compatible-v1",
        "profile": profiles.pop(),
    }


def execution_contract(specs, *, purpose="replay", version="fixed-share-next-open-v2"):
    refuse_unbuilt_version(version)
    if version in BENCHMARK_VERSIONS:
        if purpose != "replay" or type(specs) is not _kind("HistoricalBenchmarkSpec"):
            raise ExecutionContractBlocked("benchmark_v5_requires_one_historical_only_root")
        from capstone.strategy.historical_benchmark import implementation_digest

        if specs.implementation != implementation_digest():
            raise ExecutionContractBlocked("benchmark_implementation_incompatible")
        return _record(BENCHMARK_ROUTE, specs.side, version=version)
    if version in PREFIX_PROGRAM_VERSIONS and (
        purpose != "replay" or type(specs) is not _kind("ProgramStrategySpec")
    ):
        raise ExecutionContractBlocked("prefix_v6_requires_one_historical_program_leaf")
    policy = execution_policy(specs, purpose=purpose)
    roots = specs if isinstance(specs, (tuple, list)) else (specs,)
    saved = len(roots) == 1 and isinstance(roots[0], _kind("SavedProgramPortfolioSpec"))
    if saved != (version in SAVED_VERSIONS) or saved and version not in roots[0].replay_versions:
        # v4 = fixed pool, v5 = chronological membership; never interchangeable.
        expected = roots[0].execution_version if saved else SAVED_VERSIONS[0]
        raise ExecutionContractBlocked(
            "saved_allocation_requires_explicit_execution_" + expected.rsplit("-", 1)[1]
        )
    if version in PROGRAM_EVENTS_VERSIONS and (
        policy["route"] != "program-v1" or purpose != "replay"
    ):
        raise ExecutionContractBlocked("event_protocol_requires_program_historical_replay")
    if version in HELD_CARRY_VERSIONS and purpose != "replay":
        # Carry needs a replay's last valid marks; frozen order plans stay on v2.
        raise ExecutionContractBlocked("held_carry_contract_requires_historical_replay")
    return _record(policy["route"], policy["profile"], version=version)


def verify_plan_specifications(plan):
    """Opaque identifiers cannot independently establish an execution route."""
    specifications = plan.get("specifications")
    identifiers = plan.get("strategy_ids")
    if (
        not isinstance(specifications, dict)
        or not specifications
        or not isinstance(identifiers, list)
        or any(not isinstance(value, str) for value in identifiers)
        or len(set(identifiers)) != len(identifiers)
        or set(specifications) != set(identifiers)
    ):
        raise ExecutionContractBlocked("plan_specifications_missing_or_incompatible")
    # Schema -> constructor. A schema whose kind is not in this build is unsupported.
    constructors = {4: DiversifiedStrategySpec}
    for schema, name in {
        2: "ReferenceStrategySpec",
        3: "ResearchStrategySpec",
        6: "ProgramStrategySpec",
        7: "SavedProgramPortfolioSpec",
    }.items():
        if _kind(name) is not _Absent:
            constructors[schema] = _kind(name)
    specs = []
    for identifier, text in sorted(specifications.items()):
        try:
            if not isinstance(text, str):
                raise ValueError("specification must be text")
            schema = tomllib.loads(text).get("schema_version")
            if type(schema) is not int or schema not in constructors:
                raise ValueError("unsupported frozen specification schema")
            spec = constructors[schema].from_toml(text)
            if spec.id != identifier or spec.to_toml() != text:
                raise ValueError("noncanonical frozen specification identity")
            specs.append(spec)
        except (ValueError, TypeError, KeyError) as exc:
            raise ExecutionContractBlocked("plan_specification_incompatible", str(exc)) from exc
    policy = execution_policy(specs, purpose="frozen")
    record = plan["execution_contract"]
    roots = tuple(specs)
    saved = len(roots) == 1 and isinstance(roots[0], _kind("SavedProgramPortfolioSpec"))
    if saved != (record["version"] in SAVED_VERSIONS) or (
        saved and record["version"] != roots[0].execution_version
    ):
        raise ExecutionContractBlocked("plan_execution_route_incompatible")
    if record["version"] in {*PROGRAM_EVENTS_VERSIONS, *BENCHMARK_VERSIONS, *PREFIX_VERSIONS}:
        raise ExecutionContractBlocked("historical_event_contract_in_order_plan_unqualified")
    if any(policy[key] != record[key] for key in ("route", "profile")):
        raise ExecutionContractBlocked("plan_execution_route_incompatible")
    return tuple(specs)


# Every module whose bytes a contract's implementation digest covers: name -> module.
# A module that is not in this build contributes nothing (there is no behaviour to
# identify); the digest therefore changes when one lands, and records made before it
# fail verification as implementation-incompatible, which is the intended refusal.
IMPLEMENTATION_MODULES = {
    "planning": "capstone.execution.planning",
    "exit_planning": "capstone.execution.exit_planning",
    "schedule_dispatch": "capstone.strategy.schedule_dispatch",
    "daily_planning": "capstone.execution.daily_planning",
    "admission": "capstone.execution.evidence",
    "bundle": "capstone.execution.bundle",
    "frozen_plan": "capstone.execution.frozen",
    "replay": "capstone.pipeline.replay",
    "replay_identity": "capstone.execution.replay_identity",
    "actions": "capstone.pipeline.actions",
    "action_qualification": "capstone.pipeline.historical",
    "cost_schedule": "capstone.pipeline.market_costs",
    "target_dispatch": "capstone.strategy.contract",
    "program": "capstone.strategy.program",
    "recorded_specification": "capstone.strategy.recorded",
    "allocation": "capstone.strategy.portfolio",
    "broker_preflight": "capstone.execution.instructions",
    "replay_accounting": "capstone.execution.replay_accounting",
    "checkpoint_state": "capstone.execution.replay_state",
    "checkpoint_integrity": "capstone.execution.replay_checkpoint",
    "saved_portfolio": "capstone.strategy.saved_portfolio",
    "allocation_state": "capstone.strategy.allocation_state",
}
# Modules a route cannot be identified without; absent, the route is refused by name.
BENCHMARK_MODULES = {
    "benchmark_runtime": "capstone.execution.benchmark_runtime",
    "historical_benchmark": "capstone.strategy.historical_benchmark",
}
PREFIX_MODULES = {
    "prefix_protocol": "capstone.execution.prefix_protocol",
    "observed_prefix": "capstone.pipeline.prefix_inputs",
}


def _source(module):
    loaded = sys.modules.get(module)
    if loaded is not None:
        file = getattr(loaded, "__file__", None)
        return None if file is None else Path(file)
    try:
        found = importlib.util.find_spec(module)
    except (ModuleNotFoundError, ValueError):
        return None
    return None if found is None or found.origin is None else Path(found.origin)


def implementation_paths(version):
    """The source files a contract of this version identifies, by name."""
    paths = {"contracts": Path(__file__)}
    for name, module in IMPLEMENTATION_MODULES.items():
        path = _source(module)
        if path is not None:
            paths[name] = path
    required = {}
    if version in BENCHMARK_VERSIONS:
        required |= BENCHMARK_MODULES
    if version in PREFIX_VERSIONS:
        required |= PREFIX_MODULES
    for name, module in required.items():
        path = _source(module)
        if path is None:
            raise ExecutionContractBlocked("execution_route_not_in_this_build", module)
        paths[name] = path
    return paths


def _route_constant(module, name):
    found = _module(module)
    if found is None:
        raise ExecutionContractBlocked("execution_route_not_in_this_build", module)
    return getattr(found, name)


def _record(route, profile, *, version="fixed-share-next-open-v2"):
    refuse_unbuilt_version(version)
    if not isinstance(version, str) or version not in {
        *FIXED_SHARE_VERSIONS,
        *PROGRAM_EVENTS_VERSIONS,
        *PREFIX_PROGRAM_VERSIONS,
        *SAVED_VERSIONS,
        *BENCHMARK_VERSIONS,
    }:
        raise ExecutionContractBlocked("execution_contract_version_incompatible")
    if (
        not isinstance(route, str)
        or route not in {"program-v1", "legacy-compatible-v1", BENCHMARK_ROUTE}
        or not isinstance(profile, str)
        or profile not in {"long_only", "long_short"}
    ):
        raise ExecutionContractBlocked("execution_contract_policy_incompatible")
    if (
        version in {*PROGRAM_EVENTS_VERSIONS, *SAVED_VERSIONS, *PREFIX_PROGRAM_VERSIONS}
        and route != "program-v1"
    ):
        raise ExecutionContractBlocked("event_protocol_requires_program_historical_replay")
    if (version in BENCHMARK_VERSIONS) != (route == BENCHMARK_ROUTE):
        raise ExecutionContractBlocked("benchmark_execution_route_incompatible")
    program_route = route == "program-v1"
    paths = implementation_paths(version)
    record = {
        "schema_version": 1,
        "version": version,
        "route": route,
        "profile": profile,
        "decision": "completed_session",
        "execution": "following_session_open",
        "price_model": "observed_next_session_open_approximation",
        "order_type": "MKT",
        "tif": "DAY",
        "outside_rth": False,
        "quantities": "fixed_from_completed_session_snapshot",
        "between_formations": "sticky_membership_exit_only"
        if program_route
        else "legacy_fixed_share_hold",
        "corporate_action_intents": "blocked_pending_qualification"
        if program_route
        else "legacy_replay_rules",
        "implementation": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
    }
    if version in {
        *PROGRAM_EVENTS_VERSIONS,
        *SAVED_VERSIONS,
        *BENCHMARK_VERSIONS,
        *PREFIX_VERSIONS,
    }:
        record.update(
            corporate_action_intents="observed_events_expire_then_explicitly_replan_v1",
            checkpoint="completed_session_account_v1",
            scope="historical_replay_only",
        )
    if version in HELD_CARRY_VERSIONS:
        record.update(
            held_without_price="carry_at_last_valid_mark_v1",
            held_without_quote="exit_at_capped_half_spread_v1",
        )
    if version in CASH_IN_LIEU_VERSIONS:
        record["fractional_corporate_action"] = _route_constant(
            "capstone.pipeline.actions", "CASH_IN_LIEU_DEFAULT"
        )
    if version in HELD_ACTION_VERSIONS:
        record.update(
            held_unsupported_action=_route_constant(
                "capstone.pipeline.crsp_actions", "HELD_VALUE_RULE"
            ),
            deferred_entry="requalify_quote_and_price_at_replan_v1",
            fractional_without_open="cash_in_lieu_effective_close_v1",
        )
    if version in WINDOW_START_VERSIONS:
        record["calendar_window_start"] = WINDOW_START
    if version in FALLBACK_VERSIONS:
        record.update(deepcopy(FALLBACK_RULES))
    if version in SAVED_VERSIONS:
        record.update(
            scope="historical_replay_and_offline_plan_only",
            allocations="saved_prior_funded_coefficients_v1",
            allocation_validation="supplied_funded_return_derivation_at_registration",
        )
    if version in CHRONOLOGICAL_VERSIONS:
        record.update(
            allocations="saved_prior_funded_coefficients_v2",
            membership_clock="prior_qualified_prefix",
            membership_exit="explicit_capital_transition_then_shared_planner_close_v1",
        )
    if version in BENCHMARK_VERSIONS:
        record.update(
            scope="historical_benchmark_only",
            between_formations="sticky_membership_exit_only",
            benchmark_eligibility="formation_required_history_v1",
            benchmark_phase="evaluation_prior_session_v1",
        )
        if version != "observed-benchmark-next-open-v5":
            # v6+ pairs with eligibility rule v1 or v2; the rule is bound by the spec id,
            # so the contract says where it lives instead of naming v1 (v5's label stays pinned).
            record["benchmark_eligibility"] = "registered_spec_eligibility_rule"
    if version in PREFIX_PROGRAM_VERSIONS:
        record.update(
            scope="historical_completed_prefix_only",
            input_identity="observed_members_and_reachable_conversions_v1",
            source_qualification="separate_authenticated_prefix_receipt_required",
        )
    if version in BENCHMARK_PREFIX_VERSIONS:
        record.update(
            scope="historical_benchmark_completed_prefix_only",
            input_identity="observed_members_and_reachable_conversions_v1",
            source_qualification="separate_authenticated_prefix_receipt_required",
        )
    record["id"] = digest(record)
    return record


def verify_execution_contract(record, *, inspection=False):
    if not isinstance(record, dict) or record.get("id") != digest(
        {k: v for k, v in record.items() if k != "id"}
    ):
        raise ExecutionContractBlocked("execution_contract_integrity_failure")
    expected = _record(record.get("route"), record.get("profile"), version=record.get("version"))

    def policy(row):
        return {k: v for k, v in row.items() if k not in {"id", "implementation"}}

    if canonical(policy(record)) != canonical(policy(expected)) or set(record) != set(expected):
        raise ExecutionContractBlocked("execution_contract_policy_incompatible")
    implementation = record["implementation"]
    if not isinstance(implementation, dict) or set(implementation) != set(
        expected["implementation"]
    ):
        raise ExecutionContractBlocked("execution_contract_implementation_incompatible")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for value in implementation.values()
    ):
        raise ExecutionContractBlocked("execution_contract_implementation_incompatible")
    if not inspection and implementation != expected["implementation"]:
        raise ExecutionContractBlocked("execution_contract_implementation_incompatible")
    return record


def validate_program_instruments(specs, panel):
    if type(specs) is _kind("HistoricalBenchmarkSpec"):
        from capstone.execution.benchmark_runtime import validate_benchmark

        validate_benchmark(specs, panel)
        return
    if execution_policy(specs)["route"] != "program-v1":
        return
    if panel.universe.columns.has_duplicates or any(
        isinstance(name, bool) or not isinstance(name, Integral) or name <= 0
        for name in panel.universe.columns
    ):
        raise ExecutionContractBlocked("program_instrument_identity_unqualified")


def program_positions(current):
    positions = {}
    for name, quantity in current.items():
        valid_name = not isinstance(name, bool) and (
            (isinstance(name, Integral) and name > 0)
            or (
                isinstance(name, str)
                and name.isascii()
                and name.isdecimal()
                and str(int(name)) == name
                and int(name) > 0
            )
        )
        if (
            not valid_name
            or int(name) in positions
            or isinstance(quantity, bool)
            or not isinstance(quantity, Integral)
        ):
            raise ExecutionContractBlocked("program_account_position_identity_unqualified")
        positions[int(name)] = int(quantity)
    return positions
