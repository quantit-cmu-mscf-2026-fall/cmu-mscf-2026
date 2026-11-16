"""Execution contracts: a recorded route, its implementation identity and its refusals."""

# ruff: noqa: F811  (fixtures imported from test_execution_routes are used by name)

import sys
from copy import deepcopy
from types import ModuleType

import pytest
from test_execution_routes import (  # noqa: F401  (fixtures)
    FakeProgram,
    absent,
    diversified,
    fresh_registry,
    program_modules,
)

from capstone.execution import contracts
from capstone.execution.contracts import (
    ExecutionContractBlocked,
    execution_contract,
    implementation_paths,
    verify_execution_contract,
    verify_plan_specifications,
)
from capstone.factors.research_contracts import digest


def rehash(record):
    return record | {"id": digest({k: v for k, v in record.items() if k != "id"})}


def fake_module(monkeypatch, tmp_path, name, **attributes):
    """A module that is present in this build, backed by a real file for its digest."""
    module = ModuleType(name)
    module.__file__ = str(tmp_path / (name.rsplit(".", 1)[1] + ".py"))
    (tmp_path / (name.rsplit(".", 1)[1] + ".py")).write_text(f"# {name}\n")
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    contracts._kind.cache_clear()
    return module


# ------------------------------------------------------------------ the record


def test_a_legacy_contract_records_the_timing_policy_and_its_own_identity():
    record = execution_contract(diversified("long_short"))
    assert record["version"] == "fixed-share-next-open-v2"
    assert (record["route"], record["profile"]) == ("legacy-compatible-v1", "long_short")
    assert record["execution"] == "following_session_open"
    assert (record["order_type"], record["tif"], record["outside_rth"]) == ("MKT", "DAY", False)
    assert record["between_formations"] == "legacy_fixed_share_hold"
    assert record["id"] == digest({k: v for k, v in record.items() if k != "id"})
    assert verify_execution_contract(record) is record
    assert execution_contract(diversified("long_short")) == record
    assert execution_contract(diversified("long_only"))["id"] != record["id"]


def test_the_implementation_identity_covers_exactly_the_modules_present():
    record = execution_contract(diversified())
    assert set(record["implementation"]) == set(implementation_paths(record["version"]))
    assert {"contracts", "planning", "target_dispatch"} <= set(record["implementation"])
    assert all(len(value) == 64 for value in record["implementation"].values())


def test_a_held_carry_version_adds_its_rules_and_only_replays_may_use_it():
    carry = execution_contract(diversified(), version="fixed-share-next-open-v3")
    assert carry["held_without_price"] == "carry_at_last_valid_mark_v1"
    assert "held_without_price" not in execution_contract(diversified())
    with pytest.raises(ExecutionContractBlocked, match="held_carry_contract_requires"):
        execution_contract(diversified(), purpose="frozen", version="fixed-share-next-open-v3")


@pytest.mark.parametrize("change", ["extra", "profile", "exit", "hash", "hash_key"])
def test_self_consistent_contract_mutations_do_not_authorize_changed_policy(change):
    record = deepcopy(execution_contract(diversified()))
    if change == "extra":
        record["optional_escape"] = True
    elif change == "profile":
        record["profile"] = "unrestricted"
    elif change == "exit":
        record["between_formations"] = "daily_resize"
    elif change == "hash":
        record["implementation"]["planning"] = "1" * 64
    else:
        record["implementation"].pop("planning")
    with pytest.raises(ExecutionContractBlocked):
        verify_execution_contract(rehash(record))


def test_a_changed_record_without_its_identity_fails_the_integrity_check():
    record = deepcopy(execution_contract(diversified()))
    record["tif"] = "GTC"
    with pytest.raises(ExecutionContractBlocked, match="integrity_failure"):
        verify_execution_contract(record)
    for bad in (None, [], {}):
        with pytest.raises(ExecutionContractBlocked, match="integrity_failure"):
            verify_execution_contract(bad)


def test_old_implementation_hashes_are_inspectable_but_never_executable():
    record = deepcopy(execution_contract(diversified()))
    record["implementation"]["planning"] = "2" * 64
    old = rehash(record)
    assert verify_execution_contract(old, inspection=True) is old
    with pytest.raises(ExecutionContractBlocked, match="implementation_incompatible"):
        verify_execution_contract(old)
    record["implementation"]["planning"] = "XYZ"
    with pytest.raises(ExecutionContractBlocked, match="implementation_incompatible"):
        verify_execution_contract(rehash(record), inspection=True)


def test_a_module_that_lands_later_changes_what_old_records_verify_against(
    monkeypatch, tmp_path, fresh_registry
):
    record = execution_contract(diversified())
    monkeypatch.setitem(sys.modules, "capstone.execution.frozen", None)
    assert verify_execution_contract(record) is record
    fake_module(monkeypatch, tmp_path, "capstone.execution.frozen")
    assert "frozen_plan" in implementation_paths(record["version"])
    with pytest.raises(ExecutionContractBlocked, match="implementation_incompatible"):
        verify_execution_contract(record)
    # The module set is part of the identity: inspection does not waive it.
    with pytest.raises(ExecutionContractBlocked, match="implementation_incompatible"):
        verify_execution_contract(record, inspection=True)


# --------------------------------------------------------------------- refusals


def test_unknown_versions_routes_and_profiles_are_refused():
    with pytest.raises(ExecutionContractBlocked, match="version_incompatible"):
        execution_contract(diversified(), version="fixed-share-next-open-v9")
    for version in (None, 2, ["fixed-share-next-open-v2"]):
        with pytest.raises(ExecutionContractBlocked, match="version_incompatible"):
            contracts._record("legacy-compatible-v1", "long_only", version=version)
    with pytest.raises(ExecutionContractBlocked, match="policy_incompatible"):
        contracts._record("somewhere-else", "long_only")
    with pytest.raises(ExecutionContractBlocked, match="policy_incompatible"):
        contracts._record("legacy-compatible-v1", "unrestricted")


def test_event_protocols_need_a_program_route_and_a_replay():
    with pytest.raises(ExecutionContractBlocked, match="event_protocol_requires_program"):
        execution_contract(diversified(), version="program-events-v3")


def test_benchmark_and_prefix_routes_need_their_own_root_types(absent):
    for version in ("benchmark-observed-prefix-v1", "benchmark-observed-prefix-v3"):
        with pytest.raises(ExecutionContractBlocked, match="benchmark_v5_requires_one"):
            execution_contract(diversified(), version=version)
    for version in ("program-observed-prefix-v6", "program-observed-prefix-v8"):
        with pytest.raises(ExecutionContractBlocked, match="prefix_v6_requires_one"):
            execution_contract(diversified(), version=version)


def test_versions_owned_by_a_missing_module_are_refused_by_name(absent):
    for version in (
        "saved-program-portfolio-v4",
        "saved-program-portfolio-v9",
        "observed-benchmark-next-open-v5",
        "observed-benchmark-next-open-v8",
    ):
        with pytest.raises(ExecutionContractBlocked, match="execution_route_not_in_this_build"):
            execution_contract(diversified(), version=version)
        with pytest.raises(ExecutionContractBlocked, match="execution_route_not_in_this_build"):
            contracts._record("program-v1", "long_only", version=version)


def test_a_route_needing_a_missing_input_module_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "capstone.pipeline.prefix_inputs", None)
    with pytest.raises(ExecutionContractBlocked, match="prefix_inputs"):
        implementation_paths("program-observed-prefix-v6")
    monkeypatch.setitem(sys.modules, "capstone.execution.benchmark_runtime", None)
    with pytest.raises(ExecutionContractBlocked, match="benchmark_runtime"):
        implementation_paths("benchmark-observed-prefix-v1")
    assert "prefix_protocol" not in implementation_paths("fixed-share-next-open-v2")


# ------------------------------------------------------- program route versions


def test_a_program_event_contract_records_replay_scope_and_expiry_rules(program_modules):
    record = execution_contract(FakeProgram(), version="program-events-v3")
    assert (record["route"], record["scope"]) == ("program-v1", "historical_replay_only")
    assert record["corporate_action_intents"] == "observed_events_expire_then_explicitly_replan_v1"
    assert record["between_formations"] == "sticky_membership_exit_only"
    assert "fractional_corporate_action" not in record
    assert verify_execution_contract(record) is record
    with pytest.raises(ExecutionContractBlocked, match="event_protocol_requires_program"):
        execution_contract(FakeProgram(), purpose="frozen", version="program-events-v3")


def test_an_action_rule_version_names_the_qualification_it_needs(program_modules, monkeypatch):
    monkeypatch.setitem(sys.modules, "capstone.pipeline.actions", None)
    with pytest.raises(ExecutionContractBlocked, match="execution_route_not_in_this_build"):
        execution_contract(FakeProgram(), version="program-events-v4")


def test_fallback_versions_record_the_cap_window_start_and_valued_actions(
    program_modules, monkeypatch, tmp_path
):
    fake_module(monkeypatch, tmp_path, "capstone.pipeline.actions", CASH_IN_LIEU_DEFAULT="cash-v1")
    fake_module(monkeypatch, tmp_path, "capstone.pipeline.crsp_actions", HELD_VALUE_RULE="held-v1")
    record = execution_contract(FakeProgram(), version="program-events-v7")
    assert record["fractional_corporate_action"] == "cash-v1"
    assert record["held_unsupported_action"] == "held-v1"
    assert record["calendar_window_start"] == contracts.WINDOW_START
    assert record["fallback_cap"] == {"position_day_share": 0.01, "absolute_pnl_share": 0.01}
    assert record["held_open_gap_exit_sessions"] == 5
    assert {"actions", "action_qualification"} <= set(record["implementation"])
    assert verify_execution_contract(record) is record
    v5 = execution_contract(FakeProgram(), version="program-events-v5")
    assert "fallback_cap" not in v5 and "calendar_window_start" not in v5
    assert v5["held_without_price"] == "carry_at_last_valid_mark_v1"


# -------------------------------------------------------------- frozen plans


def plan_for(spec, **change):
    return {
        "strategy_ids": [spec.id],
        "specifications": {spec.id: spec.to_toml()},
        "execution_contract": execution_contract(spec, purpose="frozen"),
    } | change


def test_a_frozen_plans_specifications_must_rebuild_its_route():
    spec = diversified("long_short")
    assert verify_plan_specifications(plan_for(spec)) == (spec,)


@pytest.mark.parametrize(
    "change",
    [
        {"specifications": {}},
        {"specifications": None},
        {"strategy_ids": ["other"]},
        {"strategy_ids": [1]},
        {"strategy_ids": "all"},
    ],
)
def test_missing_or_unmatched_specifications_are_refused(change):
    with pytest.raises(ExecutionContractBlocked, match="specifications_missing"):
        verify_plan_specifications(plan_for(diversified()) | change)


def test_a_specification_that_is_not_its_own_identity_is_refused():
    spec = diversified()
    with pytest.raises(ExecutionContractBlocked, match="plan_specification_incompatible"):
        verify_plan_specifications(
            plan_for(spec, strategy_ids=["x" * 16], specifications={"x" * 16: spec.to_toml()})
        )
    with pytest.raises(ExecutionContractBlocked, match="plan_specification_incompatible"):
        verify_plan_specifications(plan_for(spec, specifications={spec.id: spec.to_toml() + "\n"}))
    with pytest.raises(ExecutionContractBlocked, match="plan_specification_incompatible"):
        verify_plan_specifications(plan_for(spec, specifications={spec.id: 7}))


def test_a_schema_whose_kind_is_missing_cannot_be_frozen(absent):
    text = 'schema_version = 6\nid = "abc"\n'
    plan = {"strategy_ids": ["abc"], "specifications": {"abc": text}, "execution_contract": {}}
    with pytest.raises(ExecutionContractBlocked, match="unsupported frozen specification schema"):
        verify_plan_specifications(plan)


def test_a_plan_cannot_swap_in_another_route_or_a_historical_contract():
    spec = diversified("long_only")
    other = execution_contract(diversified("long_short"), purpose="frozen")
    with pytest.raises(ExecutionContractBlocked, match="plan_execution_route_incompatible"):
        verify_plan_specifications(plan_for(spec, execution_contract=other))
    historical = other | {"version": "program-events-v3", "profile": "long_only"}
    with pytest.raises(ExecutionContractBlocked, match="historical_event_contract"):
        verify_plan_specifications(plan_for(spec, execution_contract=historical))
    saved = other | {"version": "saved-program-portfolio-v4"}
    with pytest.raises(ExecutionContractBlocked, match="plan_execution_route_incompatible"):
        verify_plan_specifications(plan_for(spec, execution_contract=saved))
