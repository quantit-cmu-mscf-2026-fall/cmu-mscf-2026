"""Independent arithmetic fixtures for the strict historical benchmark inputs.

Synthetic panels only; these are never financial performance evidence.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from capstone.execution.benchmark_runtime import (
    benchmark_due,
    benchmark_plan,
    benchmark_weights,
    execution_target,
    validate_benchmark,
)
from capstone.execution.contracts import BENCHMARK_VERSIONS
from capstone.factors.evaluation import Panel
from capstone.strategy.historical_benchmark import EXECUTION_VERSION, HistoricalBenchmarkSpec
from capstone.strategy.program import FormationEligibilityError


@pytest.fixture
def case():
    dates = pd.bdate_range("1994-11-01", "1996-03-05").difference(pd.DatetimeIndex(["1996-01-01"]))
    values = pd.DataFrame(
        np.tile([0.01, 0.02, 0.03, 0.04], (len(dates), 1)),
        index=dates,
        columns=[101, 102, 103, 104],
    )
    panel = Panel(
        {"returns": values, "cap": values * 1e10, "close": values * 0 + 10},
        values.gt(0),
        values * 0,
        data_version="synthetic-benchmark-v1",
        calendar=dates,
        availability={name: pd.Series(dates, index=dates) for name in ("returns", "cap", "close")},
    )
    return panel, values * 0 + 10


def specification(panel, kind="equal_weight_universe_proxy", side="long_only"):
    return HistoricalBenchmarkSpec(kind, panel.data_version, "1995-12-29", side)


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("cash", {}),
        ("equal_weight_universe_proxy", {101: 0.25, 102: 0.25, 103: 0.25, 104: 0.25}),
        ("cap_weight_universe_proxy", {101: 0.1, 102: 0.2, 103: 0.3, 104: 0.4}),
        ("known_momentum_12_1", {103: 0.25, 104: 0.75}),
    ],
)
def test_existing_algorithms_have_independent_reference_weights(case, kind, expected):
    panel, _ = case
    spec = specification(panel, kind)
    book = benchmark_weights(spec, panel, "1995-12-29")
    assert book.to_dict() == pytest.approx(expected)
    assert not book.attrs["formation_eligibility"][spec.id]["excluded"]


def test_long_short_momentum_and_formation_history_exclusions_are_explicit(case):
    panel, _ = case
    spec = specification(panel, "known_momentum_12_1", "long_short")
    assert benchmark_weights(spec, panel, spec.anchor).to_dict() == pytest.approx(
        {101: -0.375, 102: -0.125, 103: 0.125, 104: 0.375}
    )
    span = panel.calendar[panel.calendar.get_loc(pd.Timestamp(spec.anchor)) - 251 :]
    panel.fields["returns"].loc[span[0], 101] = np.nan
    book = benchmark_weights(spec, panel, spec.anchor)
    assert book.to_dict() == pytest.approx({102: -0.5, 104: 0.5})
    audit = book.attrs["formation_eligibility"][spec.id]
    assert audit["excluded"] == {"101": "required_formation_return_history_incomplete"}
    assert audit["eligible"] == [102, 103, 104]
    assert audit["formal_admission"] is False


@pytest.mark.parametrize(
    "failure", ["field", "availability", "intraday", "global_session", "few_members"]
)
def test_missing_global_dependencies_cannot_become_eligibility_or_fills(case, failure):
    panel, _ = case
    spec = specification(panel, "known_momentum_12_1")
    if failure == "field":
        panel.fields.pop("returns")
    elif failure == "availability":
        panel.availability.pop("returns")
    elif failure == "intraday":
        original = panel.availability["returns"]
        panel.availability["returns"] = pd.to_datetime(
            original.dt.strftime("%Y-%m-%d") + " 16:00:00"
        )
    elif failure == "global_session":
        missing = panel.calendar[panel.calendar.get_loc(pd.Timestamp(spec.anchor)) - 251]
        panel.fields["returns"] = panel.fields["returns"].drop(missing)
    else:
        panel.fields["returns"].loc[:, [101, 102, 103]] = np.nan
    with pytest.raises(ValueError):
        benchmark_weights(spec, panel, spec.anchor)
    if failure == "few_members":
        with pytest.raises(FormationEligibilityError):
            benchmark_weights(spec, panel, spec.anchor)


def test_future_values_and_row_column_permutations_preserve_earlier_targets(case):
    panel, _ = case
    spec = specification(panel, "known_momentum_12_1")
    expected = benchmark_weights(spec, panel, spec.anchor)
    changed = deepcopy(panel)
    changed.fields["returns"].loc["1996-01-02":] = -1
    changed.fields = {name: frame.iloc[::-1, ::-1] for name, frame in changed.fields.items()}
    changed.universe = changed.universe.iloc[::-1, ::-1]
    assert_series_equal(expected, benchmark_weights(spec, changed, spec.anchor))


def test_overflow_cannot_be_masked_as_a_valid_empty_momentum_book(case):
    panel, _ = case
    panel.fields["returns"].loc[:, :] = 1e308
    spec = specification(panel, "known_momentum_12_1")
    with pytest.raises(ValueError, match="undefined sum or weight-conversion domain") as caught:
        benchmark_weights(spec, panel, spec.anchor)
    assert caught.value.formation_eligibility["eligible"] == [101, 102, 103, 104]


def test_the_spec_and_the_frozen_data_version_must_match_exactly(case):
    panel, _ = case
    spec = specification(panel)
    validate_benchmark(spec, panel)
    other = HistoricalBenchmarkSpec(spec.kind, "another-version", spec.anchor, spec.side)
    with pytest.raises(ValueError, match="exact type and frozen data version"):
        validate_benchmark(other, panel)
    with pytest.raises(ValueError, match="exact type"):
        validate_benchmark(spec.legacy, panel)
    panel.universe.columns = [101, 101, 103, 104]
    with pytest.raises(ValueError, match="positive integer identities"):
        validate_benchmark(spec, panel)


def test_due_dates_follow_the_schedule_and_refuse_a_gappy_calendar(case):
    panel, _ = case
    momentum = specification(panel, "known_momentum_12_1")
    offset = panel.calendar.get_loc(pd.Timestamp(momentum.anchor))
    assert benchmark_due(momentum, panel, panel.calendar[offset + 21])
    assert not benchmark_due(momentum, panel, panel.calendar[offset + 20])
    gappy = Panel(
        panel.fields,
        panel.universe.drop(panel.calendar[offset + 1]),
        panel.half_spread,
        data_version=panel.data_version,
        calendar=panel.calendar,
        availability=panel.availability,
    )
    with pytest.raises(ValueError, match="differ from the authoritative calendar"):
        benchmark_due(momentum, gappy, panel.calendar[offset + 5])
    with pytest.raises(ValueError, match="actual scheduled formation"):
        benchmark_weights(momentum, panel, panel.calendar[offset + 3])


def test_execution_targets_need_an_explicit_benchmark_version(case):
    panel, _ = case
    spec = specification(panel)
    assert EXECUTION_VERSION in BENCHMARK_VERSIONS
    book = execution_target(spec, panel, spec.anchor, EXECUTION_VERSION)
    assert_series_equal(book, benchmark_weights(spec, panel, spec.anchor))
    with pytest.raises(ValueError, match="explicit execution v5"):
        execution_target(spec, panel, spec.anchor, "fixed-share-next-open-v2")


def test_a_formation_plan_sizes_whole_shares_from_the_target_weights(case):
    panel, prices = case
    spec = specification(panel)
    weights = benchmark_weights(spec, panel, spec.anchor)
    plan = benchmark_plan(
        spec,
        panel,
        spec.anchor,
        {},
        prices.loc[spec.anchor].to_dict(),
        1000,
        weights=weights,
        min_notional=100,
        fee_bps=10,
    )
    assert plan["intent"] == "formation_rebalance"
    assert plan["targets"] == dict.fromkeys([101, 102, 103, 104], 24)


def test_between_formations_only_members_that_left_are_sold(case):
    panel, prices = case
    spec = specification(panel)
    day = panel.calendar[panel.calendar.get_loc(pd.Timestamp(spec.anchor)) + 2]
    panel.universe.loc[day:, 104] = False
    plan = benchmark_plan(
        spec,
        panel,
        day,
        {101: 24, 104: 24},
        prices.loc[day].to_dict(),
        1000,
        weights=None,
        min_notional=100,
        fee_bps=0,
    )
    assert plan["intent"] == "membership_exit_only"
    assert plan["targets"] == {101: 24, 104: 0}
