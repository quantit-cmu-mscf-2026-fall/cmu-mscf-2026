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

import importlib
from functools import cache
from numbers import Integral

from capstone.execution.prefix_protocol import BENCHMARK_EXECUTION_VERSION as BENCHMARK_PREFIX
from capstone.execution.prefix_protocol import (
    BENCHMARK_PREFIX_VERSIONS,
    FALLBACK_BENCHMARK_EXECUTION_VERSION,
    FALLBACK_EXECUTION_VERSION,
    HELD_BENCHMARK_EXECUTION_VERSION,
    HELD_EXECUTION_VERSION,
)
from capstone.execution.prefix_protocol import PROGRAM_VERSIONS as PREFIX_PROGRAM_VERSIONS
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
