"""Route validation: which strategy kinds may execute, and the refusals for the rest.

The registry accepts a strategy kind lazily. A module that is not in this build leaves its
kind unavailable and any route that names it refuses; a module that is present is used as
the registry says. The tests simulate both by controlling `sys.modules`.
"""

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from test_recorded_specs import FakeSpec, record_text

from capstone.execution import contracts, prefix_protocol
from capstone.execution.contracts import (
    ExecutionContractBlocked,
    execution_leaves,
    execution_policy,
    program_positions,
    unavailable_kinds,
    validate_program_instruments,
)
from capstone.factors.evaluation import TradingSpec
from capstone.factors.research_contracts import canonical
from capstone.strategy.compiled import COMMON, RULES, CompiledStrategySpec
from capstone.strategy.contract import StrategySpec
from capstone.strategy.diversified import DiversifiedStrategySpec
from capstone.strategy.recorded import (
    RecordedExecutionBlocked,
    RecordedStrategySpec,
    load_compatible_spec,
)

EXPRESSION = "rank(ts_mean(volume, 5)) - rank(delta(close, 3))"
LAZY_MODULES = sorted(set(contracts._OPTIONAL_KINDS.values()))


@pytest.fixture(autouse=True)
def fresh_registry():
    contracts._kind.cache_clear()
    yield
    contracts._kind.cache_clear()


@pytest.fixture
def absent(monkeypatch):
    """Every lazily registered module is missing, whatever the build contains."""
    for name in LAZY_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    # The module-level version tables were read at import; a build without the benchmark
    # module has no full-window benchmark versions.
    monkeypatch.setattr(contracts, "FULL_WINDOW_BENCHMARK_VERSIONS", ())
    monkeypatch.setattr(
        contracts, "BENCHMARK_VERSIONS", tuple(prefix_protocol.BENCHMARK_PREFIX_VERSIONS)
    )
    contracts._kind.cache_clear()


def diversified(side="long_only", method="weekly_reversal"):
    return DiversifiedStrategySpec(method, "synthetic-v1", "2000-12-29", side)


def compiled(side="long_only"):
    program = RULES["weekly_reversal"] | COMMON | {"profile": side}
    return CompiledStrategySpec(
        "weekly_reversal",
        canonical(program),
        "a" * 64,
        "b" * 64,
        "synthetic-reference-v2",
        "2000-12-29",
        side,
    )


def factor_spec():
    return StrategySpec(EXPRESSION, TradingSpec(1, "none", 0.0), "synthetic-v1", "2000-12-29")


# ---------------------------------------------------------------- built kinds


def test_a_diversified_leaf_is_the_legacy_route_with_its_own_profile():
    assert execution_policy(diversified()) == {
        "route": "legacy-compatible-v1",
        "profile": "long_only",
    }
    assert execution_policy([diversified("long_short")])["profile"] == "long_short"


def test_a_factor_spec_replays_on_the_legacy_route_but_never_freezes():
    leaf = factor_spec()
    assert execution_leaves(leaf) == (leaf,)
    assert execution_policy(leaf)["route"] == "legacy-compatible-v1"
    with pytest.raises(ExecutionContractBlocked, match="unsupported_frozen_specification"):
        execution_policy(leaf, purpose="frozen")


def test_a_following_close_program_is_refused_in_the_next_open_route():
    with pytest.raises(ExecutionContractBlocked, match="following_close_program"):
        execution_policy(compiled())


def test_mixed_profiles_empty_roots_and_unknown_objects_are_refused():
    with pytest.raises(ExecutionContractBlocked, match="mixed_execution_profiles"):
        execution_policy([diversified("long_only"), diversified("long_short")])
    with pytest.raises(ExecutionContractBlocked, match="empty_execution_profile"):
        execution_leaves(())
    with pytest.raises(ExecutionContractBlocked, match="unsupported_execution_specification"):
        execution_leaves(object())
    with pytest.raises(ExecutionContractBlocked, match="unsupported_execution_purpose"):
        execution_policy(diversified(), purpose="live")


def test_an_inspection_only_record_cannot_authorize_execution():
    record = load_compatible_spec(FakeSpec, record_text())
    assert isinstance(record, RecordedStrategySpec)
    with pytest.raises(RecordedExecutionBlocked, match="recorded implementation unavailable"):
        execution_leaves(record)
    with pytest.raises(RecordedExecutionBlocked):
        execution_policy([diversified(), record])


# ------------------------------------------------------ lazily registered kinds


def test_kinds_whose_modules_are_missing_are_listed_and_refuse_by_name(absent):
    assert unavailable_kinds() == tuple(sorted(contracts._OPTIONAL_KINDS))
    assert contracts.BENCHMARK_VERSIONS == prefix_protocol.BENCHMARK_PREFIX_VERSIONS
    # A spec of a kind nobody can construct yet is not an execution specification.
    with pytest.raises(ExecutionContractBlocked, match="unsupported_execution_specification"):
        execution_policy(SimpleNamespace(side="long_only", specs=()))


def test_a_module_that_exists_but_cannot_import_is_loud_not_absent(monkeypatch):
    def failing(name):
        raise ModuleNotFoundError("no module named some_dependency", name="some_dependency")

    monkeypatch.setattr(contracts, "importlib", SimpleNamespace(import_module=failing))
    with pytest.raises(ModuleNotFoundError, match="some_dependency"):
        contracts._module("capstone.strategy.program")


def test_a_module_that_is_itself_missing_reads_as_absent():
    assert contracts._module("capstone.strategy.no_such_strategy_kind") is None


@dataclass(frozen=True)
class FakeProgram:
    anchor: str = "2020-01-02"
    side: str = "long_short"
    execution: str = "following_session_open"
    schedule: object = field(default_factory=lambda: SimpleNamespace(id="s1"))


@dataclass(frozen=True)
class FakePortfolio:
    specs: tuple
    mode: str = "equal_family"


@pytest.fixture
def program_modules(monkeypatch):
    """Present modules for the program and portfolio kinds, as registered lazily."""
    program = ModuleType("capstone.strategy.program")
    program.ProgramStrategySpec = FakeProgram
    portfolio = ModuleType("capstone.strategy.portfolio")
    portfolio.PortfolioSpec = FakePortfolio
    monkeypatch.setitem(sys.modules, "capstone.strategy.program", program)
    monkeypatch.setitem(sys.modules, "capstone.strategy.portfolio", portfolio)
    contracts._kind.cache_clear()


def test_a_registered_program_kind_takes_the_program_route(program_modules):
    assert "ProgramStrategySpec" not in unavailable_kinds()
    assert execution_policy(FakeProgram()) == {"route": "program-v1", "profile": "long_short"}
    assert execution_policy(FakePortfolio((FakeProgram(), FakeProgram())))["route"] == "program-v1"


def test_program_routes_refuse_mixing_timing_asynchrony_and_variable_allocations(program_modules):
    with pytest.raises(ExecutionContractBlocked, match="mixed_program_legacy"):
        execution_policy([FakeProgram(), diversified("long_short")])
    with pytest.raises(ExecutionContractBlocked, match="program_execution_timing"):
        execution_policy(FakeProgram(execution="same_close"))
    other = FakeProgram(anchor="2020-02-03")
    with pytest.raises(ExecutionContractBlocked, match="asynchronous_program_formations"):
        execution_policy([FakeProgram(), other])
    variable = FakePortfolio((FakeProgram(),), mode="risk_weighted")
    with pytest.raises(ExecutionContractBlocked, match="variable_program_allocations"):
        execution_policy(variable)


# ------------------------------------------------------------ instrument checks


def panel_with(columns):
    return SimpleNamespace(universe=pd.DataFrame(columns=columns))


def test_program_instruments_must_be_positive_integer_permnos(program_modules):
    validate_program_instruments(FakeProgram(), panel_with([101, 102]))
    for bad in ([101, 101], [101, 0], [101, "102"], [True, 5]):
        with pytest.raises(ExecutionContractBlocked, match="instrument_identity"):
            validate_program_instruments(FakeProgram(), panel_with(bad))
    # The legacy route does not constrain instrument names.
    validate_program_instruments(diversified(), panel_with(["a", "a"]))


def test_program_positions_accept_only_positive_integer_names_and_share_counts():
    assert program_positions({101: 5, "102": -3}) == {101: 5, 102: -3}
    for bad in ({0: 1}, {"01": 1}, {True: 1}, {101: 1.5}, {101: True}, {101: 1, "101": 2}):
        with pytest.raises(ExecutionContractBlocked, match="position_identity"):
            program_positions(bad)


# ----------------------------------------------------------- version registries


def test_version_registries_nest_the_way_each_rule_extends_the_previous():
    assert set(contracts.FALLBACK_VERSIONS) <= set(contracts.HELD_ACTION_VERSIONS)
    assert set(contracts.HELD_ACTION_VERSIONS) <= set(contracts.HELD_CARRY_VERSIONS)
    assert set(contracts.WINDOW_START_VERSIONS) <= set(contracts.PROGRAM_EVENTS_VERSIONS)
    assert set(prefix_protocol.PROGRAM_VERSIONS) <= set(contracts.PREFIX_VERSIONS)
    assert set(prefix_protocol.BENCHMARK_PREFIX_VERSIONS) <= set(contracts.BENCHMARK_VERSIONS)
    assert contracts.FALLBACK_RULES["fallback_cap"] == {
        "position_day_share": 0.01,
        "absolute_pnl_share": 0.01,
    }
