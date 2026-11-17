"""Known book arithmetic and saved coefficients; no market-performance claims."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from capstone.factors.evaluation import Panel
from capstone.strategy.allocation_state import make_decision
from capstone.strategy.contract import target_weights
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.research import digest
from capstone.strategy.robust_allocation import AllocationPolicyV2
from capstone.strategy.saved_portfolio import SavedProgramPortfolioSpec


@pytest.fixture
def case():
    calendar = pd.bdate_range("1999-12-01", "2000-04-05")
    signal = pd.DataFrame(
        np.tile([10.0, 20.0, 30.0, 40.0], (len(calendar), 1)),
        index=calendar,
        columns=pd.Index([101, 102, 103, 104], name="permno"),
    )
    panel = Panel(
        {"close": signal},
        signal.gt(0),
        signal * 0,
        data_version="saved-fixture-v1",
        calendar=calendar,
        availability={"close": pd.Series(calendar, index=calendar)},
    )
    high = ProgramStrategySpec(
        expression="close",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="high",
        data_version=panel.data_version,
        anchor="2000-01-31",
        side="long_short",
        tail=0.5,
        schedule=ProgramSchedule(kind="month_end", interval=1),
    )
    low = replace(high, direction="low", mechanism_id="low")
    returns = pd.DataFrame(
        {
            spec.id: np.tile([0.002, 0.004], len(calendar) // 2 + 1)[: len(calendar)]
            for spec in (high, low)
        },
        index=calendar,
    )
    return panel, (high, low), returns


def portfolio(
    case,
    *,
    members=None,
    through="2000-03-31",
    returns=None,
    mode="equal_weight",
    side="long_short",
    policy=None,
):
    panel, specs, history = case
    members = specs if members is None else members
    history = history if returns is None else returns
    policy = policy or AllocationPolicyV2(
        lookback=30, minimum=20, window=10, step=5, annual_risk_limit=1
    )
    rows = []
    for day in ("2000-01-31", "2000-02-29", "2000-03-31"):
        if day > through:
            break
        next_session = panel.calendar[panel.calendar.get_loc(pd.Timestamp(day)) + 1]
        rows.append(
            make_decision(
                history,
                as_of=day,
                execution_session=next_session,
                pool_ids=[s.id for s in members],
                profile=side,
                input_evidence_id="4" * 64,
                previous=rows[-1] if rows else None,
                mode=mode,
                policy=policy,
            )
        )
    return SavedProgramPortfolioSpec(
        specs=members,
        returns=history,
        schedule=tuple(rows),
        profile=side,
        data_version=panel.data_version,
        anchor="2000-01-31",
        policy=policy,
        allocation_mode=mode,
    )


DECISIONS = ("2000-01-31", "2000-02-29", "2000-03-31")


def receipt(spec, available):
    return {
        "strategy_id": spec.id,
        "kind": "receipt",
        "receipt_id": digest({"receipt": spec.id, "available": available}),
        "available_at": available,
        "history_start": "1999-12-01",
    }


def departure(spec, available, reason="prefix_execution_failed"):
    return {
        "strategy_id": spec.id,
        "kind": "departure",
        "available_at": available,
        "reason": reason,
    }


def chronological(case, members, events, *, through="2000-03-31", mode="equal_weight"):
    """A v5 saved portfolio whose pool is decided by prior receipts at every decision."""
    panel, _, history = case
    policy = AllocationPolicyV2(lookback=30, minimum=20, window=10, step=5, annual_risk_limit=1)
    returns = pd.DataFrame({spec.id: history.iloc[:, 0] for spec in members})
    rows = []
    for day in DECISIONS:
        if day > through:
            break
        next_session = panel.calendar[panel.calendar.get_loc(pd.Timestamp(day)) + 1]
        rows.append(
            make_decision(
                returns,
                as_of=day,
                execution_session=next_session,
                membership=events,
                profile=members[0].side,
                input_evidence_id="4" * 64,
                previous=rows[-1] if rows else None,
                mode=mode,
                policy=policy,
            )
        )
    return SavedProgramPortfolioSpec(
        specs=tuple(members),
        returns=returns,
        schedule=tuple(rows),
        profile=members[0].side,
        data_version=panel.data_version,
        anchor="2000-01-31",
        policy=policy,
        allocation_mode=mode,
        membership_clock="prior_qualified_prefix",
        membership=tuple(events),
    )


def test_v5_members_and_history_recompute_over_the_chronological_pool(case):
    panel, (high, low), _ = case
    events = [
        receipt(high, "2000-01-28"),
        receipt(high, "2000-02-28"),
        receipt(low, "2000-02-28"),
        receipt(high, "2000-03-30"),
        departure(low, "2000-03-30"),
    ]
    saved = chronological(case, (high, low), events)
    assert saved.execution_version == "saved-program-portfolio-v5"
    assert [row["pool_ids"] for row in saved.schedule] == [
        [high.id],
        sorted([high.id, low.id]),
        [high.id],
    ]
    assert saved.members("2000-01-31") == (high,)
    assert saved.schedule[2]["capital_transition"]["retired"] == {
        low.id: saved.schedule[1]["weights"][low.id]
    }
    assert saved.verify_history(panel, "2000-03-31") == saved.prefix_id("2000-03-31")
    frozen = saved.frozen("2000-02-29")
    assert "2000-03-30" not in frozen.to_toml()
    loaded = SavedProgramPortfolioSpec.from_toml(frozen.to_toml(), returns=frozen.returns)
    assert loaded.to_toml() == frozen.to_toml() and loaded.id == saved.id
    assert loaded.verify_history(panel, "2000-02-29") == saved.prefix_id("2000-02-29")
    # Rehashed manual pools or a dropped departure cannot verify.
    with pytest.raises(ValueError, match="explicit capital transition|actual prior funded"):
        replace(saved, membership=tuple(events[:-1])).verify_history(panel, "2000-03-31")
    with pytest.raises(ValueError, match="clock"):
        replace(saved, membership_clock="retrospective_fixed_pool")


def test_v4_fixed_pool_identity_is_pinned(case):
    saved = portfolio(case)
    assert saved.execution_version == "saved-program-portfolio-v4"
    # Definition keys and record versions the fixed-pool route has always generated.
    assert sorted(saved.definition) == [
        "allocation_mode",
        "anchor",
        "cash_reference",
        "clock",
        "data_version",
        "implementation",
        "mode",
        "policy",
        "profile",
        "return_basis",
        "source_return_basis",
        "specifications",
        "version",
    ]
    assert saved.definition["version"] == 1
    assert {row["version"] for row in saved.schedule} == {"funded-program-allocation-state-v1"}
    assert "membership_clock" not in saved.to_toml()


def test_opposing_books_net_in_existing_combiner_without_refitting(case, monkeypatch):
    saved = portfolio(case)
    assert saved.verify_history(case[0], "2000-03-31") == saved.prefix_id("2000-03-31")
    monkeypatch.setattr(
        "capstone.strategy.robust_allocation.optimize_allocation",
        lambda *a, **k: pytest.fail("planner refit"),
    )
    assert target_weights(saved, case[0], "2000-01-31").empty
    assert {s.id for s in saved.members("2000-01-31")} == {s.id for s in case[1]}


def test_empty_pool_and_single_positive_sleeve_are_explicit(case):
    empty = portfolio(case, members=(), returns=case[2].iloc[:, :0])
    assert empty.side == "long_short" and empty.verify_history(case[0], "2000-03-31")
    assert empty.weights(case[0], "2000-01-31").empty
    one = portfolio(case, members=case[1][:1], returns=case[2].iloc[:, :1])
    book = one.weights(case[0], "2000-01-31")
    assert book.to_dict() == {101: -0.25, 102: -0.25, 103: 0.25, 104: 0.25}


def test_future_saved_decisions_and_returns_do_not_change_frozen_prefix(case):
    full = portfolio(case)
    earlier = portfolio(case, through="2000-01-31")
    assert full.id == earlier.id
    assert full.frozen("2000-02-10").to_toml() == earlier.frozen("2000-02-10").to_toml()
    changed = case[2].copy()
    changed.loc["2000-02-01":] *= -1000
    altered = portfolio(case, returns=changed)
    assert altered.frozen("2000-01-31").to_toml() == full.frozen("2000-01-31").to_toml()
    assert [s.id for s in full.specs] == [s.id for s in earlier.specs]


def test_rehashed_manual_allocation_cannot_claim_optimizer_evidence(case):
    saved = portfolio(case)
    rows = deepcopy(saved.schedule)
    first = rows[0]
    first["weights"] = dict.fromkeys(first["pool_ids"], 0.0)
    first["cash_weight"] = 1.0
    first["id"] = digest({k: v for k, v in first.items() if k != "id"})
    for previous, row in zip(rows, rows[1:], strict=False):
        row["previous_state_id"] = previous["id"]
        row["id"] = digest({k: v for k, v in row.items() if k != "id"})
    forged = replace(saved, schedule=rows)
    with pytest.raises(ValueError, match="actual prior funded"):
        forged.verify_history(case[0], "2000-03-31")


def test_frozen_toml_roundtrip_preserves_definition_and_exact_states(case):
    saved = portfolio(case).frozen("2000-02-29")
    loaded = SavedProgramPortfolioSpec.from_toml(saved.to_toml(), returns=saved.returns)
    assert loaded.to_toml() == saved.to_toml()
    assert loaded.verify_history(case[0], "2000-02-29") == saved.prefix_id("2000-02-29")


def test_cash_coefficient_is_retained_without_renormalizing_stock_weights(case):
    policy = AllocationPolicyV2(lookback=30, minimum=20, window=10, step=5, annual_risk_limit=0.01)
    saved = portfolio(case, members=case[1][:1], returns=case[2].iloc[:, :1], policy=policy)
    row = saved.decision("2000-01-31")
    assert 0 < row["cash_weight"] < 1
    book = saved.weights(case[0], "2000-01-31")
    assert book.abs().sum() == pytest.approx(sum(row["weights"].values()))
    assert book.abs().sum() + row["cash_weight"] == pytest.approx(1)


def test_zero_coefficients_do_not_construct_unselected_unavailable_signals(case):
    history = case[2] * -1
    saved = portfolio(case, returns=history, mode="median_window_sharpe")
    assert saved.members("2000-01-31") == ()
    panel = deepcopy(case[0])
    panel.fields.clear()
    assert saved.weights(panel, "2000-01-31").empty


@pytest.mark.parametrize(
    "failure", ["missing_month", "return_gap", "return_precision", "profile", "clock"]
)
def test_missing_or_incompatible_inputs_cannot_become_a_verified_allocation(case, failure):
    saved = portfolio(case)
    if failure == "missing_month":
        saved = replace(saved, schedule=saved.schedule[:1])
    elif failure == "return_gap":
        saved = replace(saved, returns=saved.returns.drop(saved.returns.index[10]))
    elif failure == "return_precision":
        frame = saved.returns.copy()
        frame.index = pd.DatetimeIndex(
            [day.strftime("%Y-%m-%d") + " 16:00:00" for day in frame.index]
        )
        saved = replace(saved, returns=frame)
    elif failure == "profile":
        with pytest.raises(ValueError, match="profile"):
            replace(saved, profile="long_only")
        return
    else:
        member = replace(saved.specs[0], schedule=ProgramSchedule(interval=20))
        with pytest.raises(ValueError, match="monthly"):
            replace(saved, specs=(member,))
        return
    with pytest.raises(ValueError, match="missing|completed-session|explicit saved"):
        saved.verify_history(case[0], "2000-03-31")


def test_return_rows_and_member_order_do_not_change_saved_derivation(case):
    baseline = portfolio(case)
    shuffled = portfolio(
        case, members=case[1][::-1], returns=case[2].sample(frac=1, random_state=7).iloc[:, ::-1]
    )
    assert baseline.to_toml() == shuffled.to_toml()
    assert baseline.verify_history(case[0], "2000-03-31") == shuffled.verify_history(
        case[0], "2000-03-31"
    )


def test_dates_after_the_discovery_period_and_foreign_implementations_are_refused(case):
    saved = portfolio(case)
    with pytest.raises(ValueError, match="sealed holdout"):
        saved.validate_clock(case[0], "2021-01-29")
    assert not saved.current_bound("2021-01-29")
    with pytest.raises(ValueError, match="implementation is incompatible"):
        replace(saved, implementation="0" * 64)
    flagged = deepcopy(dict(saved.schedule[-1]))
    flagged["current_provenance"] = {"kind": "unsupported"}
    with pytest.raises(ValueError):
        replace(saved, schedule=(*saved.schedule[:-1], flagged))
