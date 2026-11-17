"""Between-formation exit planning on synthetic fixtures; not financial evidence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from capstone.execution.exit_planning import program_exit_plan
from capstone.factors.evaluation import Panel
from capstone.strategy.program import ProgramStrategySpec
from capstone.strategy.program_schedule import ProgramSchedule

OFF_DAY = "2020-06-24"  # anchor 2020-06-23, interval 3: not a formation session
PRICES = {101: 10.0, 102: 20.0, 103: 30.0, 104: 40.0}


@pytest.fixture
def panel():
    dates = pd.bdate_range("2020-06-22", periods=12)
    names = pd.Index([101, 102, 103, 104], name="permno")
    close = pd.DataFrame(
        np.arange(12)[:, None] * np.array([1.0, 2.0, 3.0, 4.0]) + 10,
        index=dates,
        columns=names,
    )
    fields = {"close": close, "returns": close.pct_change(fill_method=None)}
    return Panel(
        fields,
        pd.DataFrame(True, index=dates, columns=names),
        close * 0,
        data_version="fixture",
        calendar=dates,
        availability={name: pd.Series(dates, index=dates) for name in fields},
    )


@pytest.fixture
def spec():
    return ProgramStrategySpec(
        expression="delta(close, 1)",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="fixture/mechanism",
        data_version="fixture",
        anchor="2020-06-23",
        tail=0.5,
        schedule=ProgramSchedule(interval=3),
    )


def plan(spec, panel, current, **kwargs):
    return program_exit_plan([spec], panel, OFF_DAY, current, PRICES, 1000.0, **kwargs)


def test_names_outside_every_program_close_and_survivors_keep_their_shares(spec, panel):
    out = plan(spec, panel, {101: 5, 103: 2})
    assert out["targets"] == {101: 0, 103: 2}
    assert out["projected"] == {101: 0, 103: 2}
    assert [(o["permno"], o["shares"]) for o in out["orders"]] == [(101, -5)]


def test_nothing_to_exit_leaves_the_book_untouched(spec, panel):
    out = plan(spec, panel, {103: 2, 104: 1})
    assert out["targets"] == out["projected"] == {103: 2, 104: 1}
    assert out["orders"] == []


def test_zero_positions_are_ignored(spec, panel):
    out = plan(spec, panel, {101: 0, 103: 2})
    assert out["projected"] == {103: 2}


def test_formation_session_is_refused(spec, panel):
    with pytest.raises(ValueError, match="ordinary rebalance"):
        program_exit_plan([spec], panel, "2020-06-26", {101: 5}, PRICES, 1000.0)


def test_specification_set_must_be_nonempty_program_specs_of_one_profile(spec, panel):
    for bad in ([], [object()]):
        with pytest.raises(ValueError, match="nonempty all-schema6"):
            program_exit_plan(bad, panel, OFF_DAY, {}, PRICES, 1000.0)
    short = replace(spec, side="long_short")
    with pytest.raises(ValueError, match="one portfolio profile"):
        program_exit_plan([spec, short], panel, OFF_DAY, {}, PRICES, 1000.0)
    with pytest.raises(ValueError, match="distinct"):
        program_exit_plan([spec, spec], panel, OFF_DAY, {}, PRICES, 1000.0)


@pytest.mark.parametrize(
    "current",
    [[(101, 1)], {True: 1}, {-1: 1}, {"101": 1}, {101: 1.5}, {101: True}],
)
def test_malformed_positions_are_refused(spec, panel, current):
    with pytest.raises(ValueError, match="current positions|position identifiers"):
        plan(spec, panel, current)
