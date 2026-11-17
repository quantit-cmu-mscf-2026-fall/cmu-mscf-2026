"""Portfolio specifications reach the strategy contract and keep the frozen panel's calendar."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from capstone.execution.data import FrozenSnapshotSource
from capstone.factors.evaluation import Panel
from capstone.strategy.contract import target_weights
from capstone.strategy.portfolio import PortfolioSpec
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.selected_portfolio import SelectedPortfolioSpec


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
        data_version="portfolio-fixture-v1",
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
    return panel, (high, low), pd.DataFrame(index=calendar, columns=[high.id, low.id], dtype=float)


def test_snapshot_source_keeps_calendar_availability_and_cuts_by_date(case):
    panel, _, _ = case
    cut = FrozenSnapshotSource(panel).panel(("close",), pd.Timestamp("2000-02-29"))
    assert cut.calendar.equals(panel.calendar)
    assert cut.data_version == panel.data_version
    assert cut.availability["close"].index.max() == pd.Timestamp("2000-02-29")
    assert cut.universe.index.max() == pd.Timestamp("2000-02-29")


def test_contract_builds_a_portfolio_book_from_prior_returns_only(case):
    panel, (high, low), returns = case
    pool = PortfolioSpec((high, low), returns)
    book = target_weights(pool, panel, "2000-02-29")
    members = [target_weights(spec, panel, "2000-02-29") for spec in (high, low)]
    expected = (members[0] * 0.5).add(members[1] * 0.5, fill_value=0)
    pd.testing.assert_series_equal(book, expected[expected != 0], check_names=False)
    assert book.empty  # opposing long-short books net to nothing
    alone = target_weights(PortfolioSpec((high,), returns), panel, "2000-02-29")
    pd.testing.assert_series_equal(alone, members[0])
    assert not alone.empty


def test_contract_routes_selected_portfolios_through_their_schedule(case):
    panel, (high, low), returns = case
    schedule = ({"decision_session": "2000-02-29", "start": "2000-03-01", "selected": [high.id]},)
    selected = SelectedPortfolioSpec((high, low), returns, schedule=schedule)
    assert target_weights(selected, panel, "2000-02-28").empty
    chosen = target_weights(selected, panel, "2000-03-31")
    pd.testing.assert_series_equal(chosen, target_weights(high, panel, "2000-03-31"))


def test_contract_refuses_an_unsupported_portfolio_like_object(case):
    panel, (high, _), _ = case

    class Pool:
        specs = (high,)
        id = "x"

    with pytest.raises(ValueError, match="unsupported portfolio"):
        target_weights(Pool(), panel, "2000-02-29")
