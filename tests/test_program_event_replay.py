"""Independent arithmetic and restart fixtures; no historical performance evidence."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from capstone.execution.contracts import ExecutionContractBlocked
from capstone.execution.replay_state import read_checkpoint, write_checkpoint
from capstone.factors.evaluation import Panel
from capstone.pipeline.replay import ReplayCosts, replay
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.research import digest


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture
def case():
    dates = pd.bdate_range("2020-06-22", "2020-07-10")
    names = pd.Index([101, 102, 103, 104, 105], name="permno")
    signal = pd.DataFrame(
        np.tile([10.0, 20.0, 30.0, 40.0, 50.0], (len(dates), 1)), index=dates, columns=names
    )
    universe = signal.gt(0)
    universe[105] = False
    panel = Panel(
        {"close": signal},
        universe,
        signal * 0,
        data_version="event-fixture-v1",
        calendar=dates,
        availability={"close": pd.Series(dates, index=dates)},
    )
    spec = ProgramStrategySpec(
        expression="close",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="fixture",
        data_version=panel.data_version,
        anchor="2020-06-23",
        tail=0.5,
        schedule=ProgramSchedule(interval=20),
    )
    prices = signal * 0 + 10
    return panel, spec, prices


def events(*rows):
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["event_id", "event_date", "permno", "kind"])
    )


def run(
    case,
    *,
    start="2020-06-24",
    end="2020-06-30",
    event_table=None,
    execution_version="program-events-v3",
    **kwargs,
):
    panel, spec, prices = case
    return replay(
        spec,
        panel,
        prices,
        prices,
        start=start,
        end=end,
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        execution_version=execution_version,
        events=events() if event_table is None else event_table,
        **kwargs,
    )


def test_actual_prior_formation_initializes_new_account_and_midcycle_remains_diagnostic(case):
    result = run(case)
    assert result["positions"] == {103: 50, 104: 50}
    assert result["cash"] == 0
    assert result["initialization"] == "new_account_at_scheduled_formation"
    assert result["initialization_qualifying"] and not result["formal_admission"]
    midcycle = run(case, start="2020-06-25")
    assert midcycle["positions"] == {} and midcycle["cash"] == 1000
    assert not midcycle["initialization_qualifying"]


@pytest.mark.parametrize(
    "side, instrument, expected_claim", [("long_only", 103, 50), ("long_short", 101, -25)]
)
def test_dividends_entitlements_payments_and_short_obligations_reconcile(
    case, side, instrument, expected_claim
):
    panel, spec, prices = case
    case = panel, replace(spec, side=side), prices
    prices.loc["2020-06-25":, instrument] = 9
    table = events(
        dict(
            event_id="d",
            event_date="2020-06-25",
            permno=instrument,
            kind="dividend",
            amount=1,
            pay_date="2020-06-29",
        )
    )
    result = run(case, event_table=table)
    assert result["history"].loc["2020-06-25", "receivables"] == expected_claim
    assert result["history"].loc["2020-06-25", "distribution_pnl"] == expected_claim
    assert result["history"].loc["2020-06-26", "receivables"] == expected_claim
    assert result["history"].loc["2020-06-29", "receivables"] == 0
    assert (result["history"].equity == 1000).all()
    assert len(result["orders"]) == 2 if side == "long_only" else len(result["orders"]) == 4


def test_split_adjusts_only_held_instrument_and_exit_closes_preserve_survivors(case):
    panel, _, prices = case
    prices.loc["2020-06-25":, 103] = 5
    panel.universe.loc["2020-06-26":, 104] = False
    table = events(dict(event_id="s", event_date="2020-06-25", permno=103, kind="split", ratio=2))
    result = run(case, event_table=table)
    assert result["positions"] == {103: 100}
    assert result["cash"] == 500
    assert result["history"].loc["2020-06-25", "positions"] == {103: 100, 104: 50}
    assert [(r["permno"], r["shares"]) for r in result["orders"] if "shares" in r] == [
        (103, 50),
        (104, 50),
        (104, -50),
    ]
    assert (result["history"].equity == 1000).all()


@pytest.mark.parametrize("kind", ["settlement", "conversion"])
def test_authoritative_terminal_and_conversion_events_preserve_value(case, kind):
    panel, _, prices = case
    panel.universe.loc["2020-06-25":, 103] = False
    prices.loc["2020-06-25":, 103] = np.nan
    event = dict(event_id="t", event_date="2020-06-25", permno=103, kind=kind, amount=10)
    if kind == "conversion":
        event.update(amount=2, ratio=1, new_permno=105)
        prices.loc[:, 105] = 8
        panel.universe.loc["2020-06-25":, 105] = True
    result = run(case, event_table=events(event))
    assert result["positions"] == {104: 50} and result["cash"] == 500
    assert (result["history"].equity == 1000).all()


def test_missing_open_is_blocked_then_original_frozen_formation_recovers(case):
    panel, spec, prices = case
    opens = prices.copy()
    opens.loc["2020-06-24", 103] = np.nan
    panel.fields["close"].loc["2020-06-24":, [101, 102]] = 1e9
    result = replay(
        spec,
        panel,
        opens,
        prices,
        start="2020-06-24",
        end="2020-06-26",
        capital=1000,
        costs=ReplayCosts(0, 0, 0, 0, 0),
        execution_version="program-events-v3",
        events=events(),
    )
    assert result["orders"][0]["status"] == "blocked_missing_execution_price"
    fills = [r for r in result["orders"] if "shares" in r]
    assert [(r["date"], r["permno"], r["shares"]) for r in fills] == [
        ("2020-06-25", 103, 50),
        ("2020-06-25", 104, 50),
    ]
    assert all(
        r["source_intent_id"] == result["orders"][0]["source_intent"]["source_intent_id"]
        for r in fills
    )


def test_pending_split_expires_original_quantity_and_replans_from_observed_close(case):
    _, _, prices = case
    prices.loc["2020-06-24":, 103] = 5
    table = events(dict(event_id="s", event_date="2020-06-24", permno=103, kind="split", ratio=2))
    result = run(case, event_table=table)
    assert result["orders"][0]["status"] == "expired_corporate_action"
    fills = [r for r in result["orders"] if "shares" in r]
    assert [(r["date"], r["permno"], r["shares"]) for r in fills] == [
        ("2020-06-25", 103, 100),
        ("2020-06-25", 104, 50),
    ]


def test_restart_preserves_receivables_pending_exits_and_exact_economic_path(case, tmp_path):
    panel, _, prices = case
    prices.loc["2020-06-25":, 103] = 9
    panel.universe.loc["2020-06-25":, 104] = False
    table = events(
        dict(
            event_id="d",
            event_date="2020-06-25",
            permno=103,
            kind="dividend",
            amount=1,
            pay_date="2020-06-29",
        )
    )
    full = run(case, event_table=table)
    first = run(case, end="2020-06-25", event_table=table)
    path = write_checkpoint(first["checkpoint"], tmp_path / "checkpoint.json")
    seed = read_checkpoint(path)
    second = run(case, start="2020-06-26", event_table=table, checkpoint=seed)
    assert_frame_equal(pd.concat([first["history"], second["history"]]), full["history"])
    assert first["orders"] + second["orders"] == full["orders"]
    assert second["checkpoint"] == full["checkpoint"]
    assert second["positions"] == {103: 50} and second["cash"] == 550
    assert second["initialization"] == "restored_replay_checkpoint"


@pytest.mark.parametrize("change", ["hash", "valuation", "prefix", "gap", "capital"])
def test_incompatible_checkpoint_cannot_produce_a_completed_result(case, change):
    first = run(case, end="2020-06-25")
    seed = deepcopy(first["checkpoint"])
    start = "2020-06-26"
    if change in {"hash", "valuation"}:
        seed["cash"] += 1
        if change == "valuation":
            seed["checkpoint_id"] = digest({k: v for k, v in seed.items() if k != "checkpoint_id"})
    elif change == "prefix":
        case[2].loc["2020-06-24", 103] = 9
    elif change == "gap":
        start = "2020-06-29"
    else:
        seed["config"]["origin_capital"] = 2000
        seed["checkpoint_id"] = digest({k: v for k, v in seed.items() if k != "checkpoint_id"})
    completed = []
    with pytest.raises(ExecutionContractBlocked, match="checkpoint"):
        run(case, start=start, checkpoint=seed, on_completed_session=completed.append)
    assert completed == []


def test_unsupported_held_event_and_missing_valuation_remain_hard_blockers(case):
    with pytest.raises(ExecutionContractBlocked, match="corporate_action_transition"):
        run(
            case,
            event_table=events(
                dict(event_id="u", event_date="2020-06-25", permno=103, kind="unsupported")
            ),
        )
    case[2].loc["2020-06-25", 103] = np.nan
    with pytest.raises(ExecutionContractBlocked, match="held_valuation_missing"):
        run(case)


def _held_price_gap(case):
    """Slice-01 shape: a held name loses its price between formations, then recovers."""
    prices = case[2]
    prices.loc["2020-06-25":, 104] = 12.0
    prices.loc["2020-06-26":"2020-06-29", 104] = np.nan


@pytest.mark.parametrize(
    "version, completes",
    [
        ("fixed-share-next-open-v2", False),
        ("fixed-share-next-open-v3", True),
        ("program-events-v3", False),
        ("program-events-v4", True),
    ],
)
def test_held_name_losing_its_price_is_carried_only_under_new_versions(case, version, completes):
    _held_price_gap(case)
    if not completes:
        with pytest.raises(
            ValueError,
            match="valid raw closing prices|held_valuation_missing|pending_execution_price",
        ):
            run(case, execution_version=version)
        return
    result = run(case, execution_version=version)
    assert result["positions"] == {103: 50, 104: 50}
    # Marked at the last valid price (12) through the gap; no order touches 104.
    assert result["history"].loc["2020-06-26", "equity"] == 50 * 10 + 50 * 12
    assert result["history"].loc["2020-06-29", "market_values"][104] == 50 * 12
    carried = [r for r in result["orders"] if r.get("status") == "carried_at_last_valid_mark"]
    assert [(r["date"], r["permnos"]) for r in carried] == [
        ("2020-06-26", [104]),
        ("2020-06-29", [104]),
    ]
    assert not any(r.get("permno") == 104 and r["date"] > "2020-06-25" for r in result["orders"])


def test_carried_mark_survives_a_checkpoint_restart(case, tmp_path):
    _held_price_gap(case)
    full = run(case, execution_version="program-events-v4")
    first = run(case, end="2020-06-26", execution_version="program-events-v4")
    assert first["checkpoint"]["previous_close"]["104"] == 12.0
    seed = read_checkpoint(write_checkpoint(first["checkpoint"], tmp_path / "checkpoint.json"))
    second = run(case, start="2020-06-29", execution_version="program-events-v4", checkpoint=seed)
    assert_frame_equal(pd.concat([first["history"], second["history"]]), full["history"])
    assert second["checkpoint"] == full["checkpoint"]


@pytest.mark.parametrize("version", ["program-events-v3", "program-events-v4"])
def test_fractional_split_without_cash_in_lieu_completes_only_under_v4(case, version):
    """Slice-01 shape: a held fractional split with no observed cash-in-lieu price."""
    table = events(
        dict(event_id="f", event_date="2020-06-26", permno=103, kind="split", ratio=1.25)
    )
    if version == "program-events-v3":
        with pytest.raises(ExecutionContractBlocked, match="needs observed cash-in-lieu"):
            run(case, event_table=table, execution_version=version)
        return
    result = run(case, event_table=table, execution_version=version)
    split = next(r for r in result["corporate_actions"] if r.get("event_id") == "f")
    assert split["shares"] == 62
    assert split["cash_in_lieu"] == {
        "rule": "cash_in_lieu_following_open_v1",
        "permno": 103,
        "fraction": 0.5,
        "price": 10.0,
        "amount": 5.0,
    }
    assert result["execution_contract"]["fractional_corporate_action"] == (
        "cash_in_lieu_following_open_v1"
    )


SUCCESSOR = {
    "fixed-share-next-open-v2": "fixed-share-next-open-v3",
    "program-events-v3": "program-events-v4",
}
OLD_POLICIES = {
    "fixed-share-next-open-v2": "a93dd8fcffc66f405f23f4aa74834cb135e6a546adbf6a7637a1a08b0174b026",
    "program-events-v3": "c916f56a900534a6c894435464672166bbb8cd9d657d6fab54d11664ef57de14",
}


@pytest.mark.parametrize("version", sorted(OLD_POLICIES))
def test_old_execution_contract_policies_are_pinned_against_main(case, version):
    """Pinned on main 529ab09: v2/v3 contract policy (all fields except code digests)."""
    from capstone.execution.contracts import execution_contract, verify_execution_contract

    record = execution_contract(case[1], version=version)
    policy = {k: v for k, v in record.items() if k not in {"id", "implementation"}}
    assert digest(policy) == OLD_POLICIES[version]
    assert verify_execution_contract(record) == record
    new = execution_contract(case[1], version=SUCCESSOR[version])
    assert new["held_without_price"] == "carry_at_last_valid_mark_v1"
    assert new["held_without_quote"] == "exit_at_capped_half_spread_v1"
    assert (
        digest({k: v for k, v in new.items() if k not in {"id", "implementation"}})
        != (OLD_POLICIES[version])
    )
