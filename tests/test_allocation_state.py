import numpy as np
import pandas as pd
import pytest

from capstone.factors.research_contracts import digest
from capstone.strategy.allocation_state import make_decision, verify_decision
from capstone.strategy.robust_allocation import AllocationPolicyV2


def inputs(columns=("a", "b")):
    rng = np.random.default_rng(404)
    dates = pd.bdate_range("2000-01-03", periods=60)
    return pd.DataFrame(rng.normal(0.001, 0.005, (60, len(columns))), index=dates, columns=columns)


def decision(frame, *, at="2000-03-01", execution="2000-03-02", previous=None):
    return make_decision(
        frame,
        as_of=at,
        execution_session=execution,
        pool_ids=list(frame),
        profile="long_only",
        input_evidence_id="1" * 64,
        previous=previous,
        policy=AllocationPolicyV2(lookback=30, minimum=20, window=10, step=5),
    )


def test_future_rows_and_permutations_cannot_change_saved_coefficients():
    frame = inputs()
    expected = decision(frame)
    changed = frame.copy()
    changed.loc[changed.index >= "2000-03-01"] *= -1000
    assert decision(changed.sample(frac=1, random_state=9).loc[:, ["b", "a"]]) == expected
    assert not expected["standalone_costs_deducted_again"]
    assert sum(expected["weights"].values()) + expected["cash_weight"] == pytest.approx(1)


def test_zero_and_one_member_pools_are_explicit_valid_cases():
    cash = decision(inputs(()))
    assert cash["status"] == "empty_pool_cash" and cash["cash_weight"] == 1
    single = decision(inputs(("a",)))
    assert set(single["weights"]) == {"a"}
    assert verify_decision(single) == single


def test_previous_target_state_is_bound_and_cannot_reset():
    frame = inputs()
    earlier = decision(frame)
    later = decision(frame, at="2000-03-15", execution="2000-03-16", previous=earlier)
    assert later["previous_state_id"] == earlier["id"]
    with pytest.raises(ValueError, match="earlier"):
        decision(frame, previous=later)
    changed = dict(earlier)
    changed["weights"] = {"a": 100, "b": 0}
    with pytest.raises(ValueError, match="integrity"):
        verify_decision(changed)


def test_missing_prior_return_never_becomes_cash_or_zero_imputation():
    frame = inputs()
    frame.iloc[20, 0] = np.nan
    with pytest.raises(ValueError, match="missing or nonfinite"):
        decision(frame)


def test_empty_pool_still_requires_a_known_objective():
    with pytest.raises(ValueError, match="known allocation mode"):
        make_decision(
            inputs(()),
            as_of="2000-03-01",
            execution_session="2000-03-02",
            pool_ids=[],
            profile="long_only",
            input_evidence_id="1" * 64,
            mode="unknown",
        )


# --- Chronological membership: funded-program-allocation-state-v2 ---------------

POLICY = AllocationPolicyV2(lookback=30, minimum=20, window=10, step=5)
# Generated on main 79dd6bf (fixed-pool route); old identities must keep verifying.
MAIN_V1_RECORD = {
    "as_of": "2000-03-01",
    "cash_comparison": {
        "amortized_liquidation_cost": 0.0,
        "sharpe": None,
        "status": "undefined_zero_volatility",
    },
    "cash_prefix_hash": "77b940cfc747fdbd65e1a63f93e3faab558de14bfad16af1c54c86c9ed037690",
    "cash_reference": "zero_return_cash",
    "cash_reference_subtractions": 1,
    "cash_weight": 1.0,
    "coefficient_units": "fraction_of_funded_capital",
    "decision_session": "2000-03-01",
    "execution_session": "2000-03-02",
    "formal_admission": False,
    "history_end": "2000-02-29",
    "id": "8ef1ffc876fb920f5a713fc81bc1ae305d1db4f858b2043b449da4aac82095f0",
    "input_evidence_id": "1" * 64,
    "mode": "median_window_sharpe",
    "observations": 42,
    "optimizer_record_id": "ea8770c017d76064bb863bfa502c5c9342387684599918312ae3867309d7232d",
    "policy": {
        "annual_risk_limit": 0.1,
        "holding_periods": 63,
        "lookback": 500,
        "mean_shrink": 0.5,
        "minimum": 400,
        "ongoing_bps": 0,
        "periods": 252,
        "policy_version": 2,
        "step": 5,
        "switch_bps": 0,
        "window": 10,
    },
    "pool_ids": ["a"],
    "previous_coefficient_convention": "last_saved_targets; not reconstructed_sleeve_wealth",
    "previous_state_id": None,
    "profile": "long_only",
    "return_basis": "funded_incremental_over_zero_cash_v1",
    "return_prefix_hash": "347fdf89f40eff2f39255092aac2b9d57389978453540028d3c250543eec7c99",
    "risk_free_subtractions": 0,
    "schema_version": 4,
    "solver": {
        "cash_edge_rule": "positive_feasible_objective_required_except_fixed_equal_weight",
        "cost_model": "invested_ongoing_plus_actual_coefficient_switch_amortization",
        "global_optimum_guaranteed": False,
        "lattice_points_evaluated": 0,
        "lattice_refinement_starts": 4,
        "lattice_status": "not_run",
        "lattice_step": 0.1,
        "maximum_lattice_points": 10000,
        "scope": "risk_cap_surface_and_feasible_prior",
    },
    "source_return_basis": "funded_total_return",
    "standalone_costs_deducted_again": False,
    "status": "insufficient_history_cash",
    "training_objective": None,
    "version": "funded-program-allocation-state-v1",
    "weights": {"a": 0.0},
}


def receipt(name, available, start="2000-01-03"):
    return {
        "strategy_id": name,
        "kind": "receipt",
        "receipt_id": digest({"receipt": name, "available": available}),
        "available_at": available,
        "history_start": start,
    }


def departure(name, available, reason="prefix_execution_failed"):
    return {"strategy_id": name, "kind": "departure", "available_at": available, "reason": reason}


def chrono(frame, membership, *, at, execution, previous=None, mode="equal_weight"):
    return make_decision(
        frame,
        as_of=at,
        execution_session=execution,
        membership=membership,
        profile="long_only",
        input_evidence_id="1" * 64,
        previous=previous,
        mode=mode,
        policy=POLICY,
    )


def test_v1_fixed_pool_identity_from_main_still_verifies_and_chains():
    assert verify_decision(dict(MAIN_V1_RECORD)) == MAIN_V1_RECORD
    later = make_decision(
        inputs(("a",)),
        as_of="2000-03-15",
        execution_session="2000-03-16",
        pool_ids=["a"],
        profile="long_only",
        input_evidence_id="1" * 64,
        previous=MAIN_V1_RECORD,
        policy=AllocationPolicyV2(lookback=500, minimum=400, window=10, step=5),
    )
    assert later["version"] == "funded-program-allocation-state-v1"
    assert set(later) == set(MAIN_V1_RECORD)
    with pytest.raises(ValueError, match="membership clock"):
        chrono(
            inputs(),
            [receipt("a", "2000-03-14")],
            at="2000-03-15",
            execution="2000-03-16",
            previous=MAIN_V1_RECORD,
        )


def test_v2_pool_is_receipts_before_decision_and_later_receipts_do_not_change_ids():
    frame = inputs(("a", "b", "c"))
    known = [receipt("a", "2000-02-14"), receipt("b", "2000-02-14")]
    first = chrono(frame, known, at="2000-02-15", execution="2000-02-16")
    assert first["version"] == "funded-program-allocation-state-v2"
    assert first["membership_clock"] == "prior_qualified_prefix"
    assert first["pool_ids"] == ["a", "b"] and first["history_end"] == "2000-02-14"
    assert first["membership"]["a"]["receipt_id"] == known[0]["receipt_id"]
    # Receipts available on/after the decision, a future member, a future departure
    # and perturbed later rows cannot change the earlier decision's id or weights.
    later = [
        *known,
        receipt("a", "2000-02-29"),
        receipt("b", "2000-02-29"),
        receipt("c", "2000-02-29"),
    ]
    changed = frame.copy()
    changed.loc[changed.index > "2000-02-14"] *= -1000
    again = chrono(
        changed.iloc[::-1, ::-1],
        [*later, departure("b", "2000-03-15")],
        at="2000-02-15",
        execution="2000-02-16",
    )
    assert again == first
    following = chrono(frame, later, at="2000-03-01", execution="2000-03-02", previous=first)
    assert following["pool_ids"] == ["a", "b", "c"] and following["entered"] == ["c"]
    assert following["previous_state_id"] == first["id"]
    assert following["capital_transition"] is None
    with pytest.raises(ValueError, match="pool_ids or membership"):
        make_decision(
            frame,
            as_of="2000-02-15",
            execution_session="2000-02-16",
            pool_ids=["a"],
            membership=known,
            profile="long_only",
            input_evidence_id="1" * 64,
            policy=POLICY,
        )


def test_departed_member_requires_and_records_transition():
    frame = inputs()
    first = chrono(
        frame,
        [receipt("a", "2000-02-14"), receipt("b", "2000-02-14")],
        at="2000-02-15",
        execution="2000-02-16",
    )
    assert first["weights"]["b"] > 0
    stale = [receipt("a", "2000-02-14"), receipt("b", "2000-02-14"), receipt("a", "2000-02-29")]
    with pytest.raises(ValueError, match="explicit capital transition"):
        chrono(frame, stale, at="2000-03-01", execution="2000-03-02", previous=first)
    with pytest.raises(ValueError, match="explicit capital transition"):
        chrono(
            frame,
            [receipt("a", "2000-02-29")],
            at="2000-03-01",
            execution="2000-03-02",
            previous=first,
        )
    # A departure that is only available at/after the decision cannot be used yet.
    with pytest.raises(ValueError, match="explicit capital transition"):
        chrono(
            frame,
            [*stale, departure("b", "2000-03-01")],
            at="2000-03-01",
            execution="2000-03-02",
            previous=first,
        )
    moved = chrono(
        frame,
        [*stale, departure("b", "2000-02-29")],
        at="2000-03-01",
        execution="2000-03-02",
        previous=first,
    )
    assert moved["pool_ids"] == ["a"] and "b" not in moved["weights"]
    assert moved["capital_transition"] == {
        "retired": {"b": first["weights"]["b"]},
        "to_cash": first["weights"]["b"],
        "reason": {"b": "prefix_execution_failed"},
    }
    assert verify_decision(moved) == moved
    forged = dict(moved, capital_transition=dict(moved["capital_transition"], to_cash=0.0))
    forged["id"] = digest({k: v for k, v in forged.items() if k != "id"})
    with pytest.raises(ValueError, match="capital transition"):
        verify_decision(forged)


def test_last_member_exit_is_a_valid_all_cash_decision():
    frame = inputs(("a",))
    first = chrono(frame, [receipt("a", "2000-02-14")], at="2000-02-15", execution="2000-02-16")
    assert first["weights"]["a"] > 0
    cash = chrono(
        frame,
        [receipt("a", "2000-02-14"), departure("a", "2000-02-29", "registry_retired")],
        at="2000-03-01",
        execution="2000-03-02",
        previous=first,
    )
    assert cash["pool_ids"] == [] and cash["weights"] == {} and cash["cash_weight"] == 1
    assert cash["status"] == "empty_pool_cash"
    assert cash["capital_transition"]["retired"] == {"a": first["weights"]["a"]}
    assert verify_decision(cash) == cash


def test_entering_member_uses_prefix_history_only():
    frame = inputs(("a", "c"))
    frame.loc[frame.index < "2000-02-14", "c"] = np.nan
    events = [receipt("a", "2000-02-29"), receipt("c", "2000-02-29", start="2000-02-14")]
    young = chrono(frame, events, at="2000-03-01", execution="2000-03-02")
    # Twelve prefix sessions are fewer than the policy minimum: weight zero, explicit.
    assert young["insufficient_history_ids"] == ["c"] and young["weights"]["c"] == 0
    assert young["weights"]["a"] > 0
    grown = [*events, receipt("a", "2000-03-14"), receipt("c", "2000-03-14", start="2000-02-14")]
    mature = chrono(frame, grown, at="2000-03-15", execution="2000-03-16", previous=young)
    assert mature["insufficient_history_ids"] == [] and mature["weights"]["c"] > 0
    assert mature["observations"] == 22  # the common window starts at c's first prefix session
    # Values before c's prefix start or after the latest available receipt are never inputs.
    for end, decision in (("2000-02-29", young), ("2000-03-14", mature)):
        changed = frame.copy()
        changed.loc[changed.index < "2000-02-14", "c"] = 1000.0
        changed.loc[changed.index > end] *= -1000
        if decision is young:
            assert chrono(changed, events, at="2000-03-01", execution="2000-03-02") == young
        else:
            again = chrono(changed, grown, at="2000-03-15", execution="2000-03-16", previous=young)
            assert again == mature


def test_decisions_after_the_discovery_period_are_refused_without_the_current_route():
    dates = pd.bdate_range("2020-11-02", periods=60)
    frame = pd.DataFrame(np.random.default_rng(1).normal(0, 0.005, (60, 2)), dates, ["a", "b"])
    with pytest.raises(ValueError, match="current-feature route"):
        decision(frame, at="2021-01-20", execution="2021-01-21")


def test_frame_identity_ignores_rows_after_its_end_and_row_order():
    from capstone.strategy.allocation_state import frame_identity

    frame = inputs()
    end = frame.index[30]
    changed = frame.copy()
    changed.iloc[40:] += 1
    assert frame_identity(frame, end) == frame_identity(changed.iloc[::-1], end)
    assert frame_identity(frame, end) != frame_identity(frame * 2, end)
    assert frame_identity(None, end) is None
