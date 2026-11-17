import numpy as np
import pandas as pd
import pytest

from capstone.factors.evaluation import Panel
from capstone.factors.research_contracts import canonical
from capstone.pipeline.return_units import simulate
from capstone.strategy.compiled import (
    COMMON,
    RULES,
    CompiledStrategySpec,
    compiled_weights,
    decision_due,
)


def frame(days=6, names=(1, 2), value=0.01):
    index = pd.bdate_range("2020-03-02", periods=days)
    returns = pd.DataFrame(value, index=index, columns=pd.Index(names, name="permno"))
    return returns, returns.notna(), returns.isna()


def test_next_session_fill_earns_from_the_following_return_and_costs_reconcile():
    returns, universe, terminal = frame()
    target = pd.Series({1: 0.5, 2: 0.5})
    history, used = simulate(
        returns, universe, {returns.index[0]: target}, terminal, side="long_only", multiplier=1
    )
    # Decided on day 0, filled on day 1, so day 1 itself earns nothing.
    assert history["return"].iloc[0] == 0 and history["gross"].iloc[0] == 0
    assert history["gross"].iloc[1] == pytest.approx(1.0, abs=1e-3)
    assert history["return"].iloc[2] == pytest.approx(0.01, abs=1e-3)
    fee = history["transaction_cost"].iloc[1]
    assert fee == pytest.approx(1e5 * 6 / 10000, rel=1e-3)
    assert used == []
    free, _ = simulate(
        returns, universe, {returns.index[0]: target}, terminal, side="long_only", multiplier=0
    )
    assert (free["transaction_cost"] == 0).all()
    assert free["equity"].iloc[-1] > history["equity"].iloc[-1]


def test_a_name_leaving_the_universe_is_sold_and_terminal_flags_zero_it():
    returns, universe, terminal = frame(days=5)
    universe.loc[returns.index[3:], 2] = False
    target = pd.Series({1: 0.5, 2: 0.5})
    history, _ = simulate(
        returns, universe, {returns.index[0]: target}, terminal, side="long_only", multiplier=1
    )
    assert history["names"].iloc[2] == 2 and history["names"].iloc[3] == 1
    assert history["turnover"].iloc[3] > 0


def test_invalid_inputs_are_refused():
    returns, universe, terminal = frame()
    target = pd.Series({1: 0.5, 2: 0.5})
    with pytest.raises(ValueError, match="unknown diagnostic"):
        simulate(returns, universe, {}, terminal, side="sideways")
    with pytest.raises(ValueError, match="unknown diagnostic"):
        simulate(returns, universe, {}, terminal, side="long_only", multiplier=3)
    with pytest.raises(ValueError, match="dates must be unique"):
        simulate(returns.iloc[::-1], universe, {}, terminal, side="long_only")
    late = returns.copy()
    late.index = pd.bdate_range("2020-12-28", periods=len(late))
    with pytest.raises(ValueError, match="sealed holdout"):
        simulate(
            late, universe.set_axis(late.index), {}, terminal.set_axis(late.index), side="long_only"
        )
    with pytest.raises(ValueError, match="cannot short"):
        simulate(
            returns,
            universe,
            {returns.index[0]: pd.Series({1: -0.5, 2: 0.5})},
            terminal,
            side="long_only",
        )
    with pytest.raises(ValueError, match="invalid diagnostic target exposure"):
        simulate(returns, universe, {returns.index[0]: target * 3}, terminal, side="long_only")
    with pytest.raises(ValueError, match="unmapped"):
        simulate(
            returns, universe, {returns.index[0]: pd.Series({9: 1.0})}, terminal, side="long_only"
        )


def test_an_unresolved_held_return_stops_the_simulation():
    returns, universe, terminal = frame()
    returns.loc[returns.index[3], 1] = np.nan
    with pytest.raises(ValueError, match="unresolved/invalid held return"):
        simulate(
            returns,
            universe,
            {returns.index[0]: pd.Series({1: 1.0})},
            terminal,
            side="long_only",
        )


def test_calendar_program_replay_earns_exactly_the_four_source_days():
    calendar = pd.bdate_range("2020-10-01", "2021-01-12").difference(
        pd.to_datetime(["2020-11-26", "2020-12-25", "2021-01-01"])
    )
    dates = calendar[calendar <= "2020-12-31"]
    names = pd.Index([1, 2], name="permno")
    returns = pd.DataFrame(-0.005, index=dates, columns=names)
    wanted_days = pd.to_datetime(["2020-11-02", "2020-11-03", "2020-11-04", "2020-11-30"])
    returns.loc[wanted_days] = 0.01
    universe = returns.notna()
    panel = Panel(
        {"returns": returns},
        universe,
        returns * 0,
        data_version="fixture",
        calendar=calendar,
        availability={"returns": pd.Series(dates, index=dates)},
    )
    candidate = CompiledStrategySpec(
        "turn_of_month",
        canonical(RULES["turn_of_month"] | COMMON | {"profile": "long_only"}),
        "a" * 64,
        "b" * 64,
        "fixture",
        "2020-10-01",
        "long_only",
    )
    targets = {
        at: compiled_weights(candidate, panel, at)
        for at in dates
        if decision_due(candidate, panel, at)
    }
    history, _ = simulate(returns, universe, targets, universe * False, side="long_only")
    november = history.loc["2020-11-01":"2020-11-30", "return"]
    assert list(november[november.abs() > 1e-10].index) == list(wanted_days)
    assert (1 + november).prod() - 1 == pytest.approx(1.01**4 - 1)
