"""Historical per-instrument cost rates for a decision date (`capstone.pipeline.market_costs`).

Synthetic spreads and cost assumptions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from capstone import shared_data as sd
from capstone.pipeline.market_costs import HistoricalCostSchedule


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    short_borrow_bps_annual: float = 200.0
    financing_bps_annual: float = 500.0


DATES = pd.to_datetime(["2000-01-03", "2000-01-04"])


def schedule(spreads=None, *, multiplier=1.0, costs=None):
    spreads = (
        pd.DataFrame({1: [4.0, 999.0], 2: [8.0, 999.0]}, index=DATES)
        if spreads is None
        else spreads
    )
    return HistoricalCostSchedule(spreads, costs or Costs(), multiplier=multiplier)


def test_a_rate_is_commission_plus_the_stocks_half_spread_plus_slippage():
    s = schedule()
    assert s.rates(DATES[0]) == {1: 10.0, 2: 14.0}
    assert s.components(DATES[0])[2] == {"commission": 1.0, "spread": 8.0, "slippage": 5.0}


def test_a_future_quote_cannot_change_the_rate_of_an_earlier_date():
    spreads = pd.DataFrame({1: [4.0, 999.0], 2: [8.0, 999.0]}, index=DATES)
    s = schedule(spreads)
    before = s.rates(DATES[0])
    spreads.loc[DATES[1]] = 10000  # the schedule owns a copy of what it was given
    assert s.rates(DATES[0]) == before == {1: 10.0, 2: 14.0}


def test_a_date_without_an_observation_is_refused_not_guessed():
    with pytest.raises(ValueError, match="cost observation is missing"):
        schedule().rates(pd.Timestamp("2000-01-05"))


def test_a_stock_without_a_good_quote_has_no_rate():
    spreads = pd.DataFrame(
        {1: [4.0], 2: [np.nan], 3: [-1.0], 4: [np.inf]}, index=DATES[:1], columns=[1, 2, 3, 4]
    )
    assert schedule(spreads).rates(DATES[0]) == {1: 10.0}


@pytest.mark.parametrize("multiplier", [0, 1, 2])
def test_the_cost_scenarios_scale_every_part(multiplier):
    s = schedule(multiplier=multiplier)
    assert s.rates(DATES[0]) == {1: 10.0 * multiplier, 2: 14.0 * multiplier}
    assert s.carry_rates() == (200.0 * multiplier, 500.0 * multiplier)
    assert s.fallback_bps() == pytest.approx((1 + sd.MAX_SPREAD / 2 * 1e4 + 5) * multiplier)


def test_a_scenario_outside_the_frozen_study_is_refused():
    with pytest.raises(ValueError, match="outside the frozen study"):
        schedule(multiplier=3)


def test_the_zero_cost_scenario_needs_no_spread_at_all():
    spreads = pd.DataFrame({1: [np.nan, 2.0]}, index=DATES)
    zero = schedule(spreads, multiplier=0)
    assert zero.rates(DATES[0]) == {1: 0.0}
    assert schedule(spreads, multiplier=1).rates(DATES[0]) == {}


def test_duplicate_observations_are_refused():
    twice = pd.DataFrame({1: [4.0, 5.0]}, index=[DATES[0], DATES[0]])
    with pytest.raises(ValueError, match="unique"):
        schedule(twice)
    columns = pd.DataFrame([[4.0, 5.0]], index=DATES[:1], columns=[1, 1])
    with pytest.raises(ValueError, match="unique"):
        schedule(columns)


def test_the_id_follows_every_input_and_not_the_row_order():
    base = schedule()
    assert schedule().id == base.id
    plain = pd.DataFrame({1: [4.0, 999.0], 2: [8.0, 999.0]}, index=DATES)
    assert schedule(plain.iloc[::-1][[2, 1]]).id == base.id
    changed = [
        schedule(multiplier=2),
        schedule(costs=Costs(commission_bps=2.0)),
        schedule(costs=Costs(slippage_bps=6.0)),
        schedule(costs=Costs(short_borrow_bps_annual=1.0)),
        schedule(costs=Costs(financing_bps_annual=1.0)),
        schedule(pd.DataFrame({1: [4.0, 999.0], 2: [9.0, 999.0]}, index=DATES)),
    ]
    assert len({s.id for s in changed} | {base.id}) == len(changed) + 1


def test_the_returned_tables_are_copies():
    s = schedule()
    s.components(DATES[0])[1]["spread"] = 0.0
    s.rates(DATES[0])[1] = 0.0
    assert s.rates(DATES[0])[1] == 10.0
