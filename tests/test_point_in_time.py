"""Point-in-time level fields (`capstone.factors.point_in_time`): no later split
reaches an earlier date, and the anchor of CRSP's cumulative factors is irrelevant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_factor_evaluation import fake_rows

from capstone.factors import point_in_time as pit
from capstone.factors.evaluation import build_panel, factor_signal, factor_weights
from capstone.factors.interpret import evaluate
from capstone.factors.reference import field
from capstone.factors.tree import parse

PRICE_COLUMNS = ["dlyopen", "dlyhigh", "dlylow", "dlyprc", "dlybid", "dlyask"]

EXPRESSIONS = [
    "close",
    "-rank(close)",
    "close / ts_max(close, 10)",
    "rank(close * shares)",
    "rank(close + volume)",
    "ts_corr(close, volume, 10)",
    "rank(ema(close, 5)) - rank(volume / shares)",
    "ts_mean(rank(close * shares), 3)",
    "rank(close) * returns + scale(volume)",
]


def split(rows: pd.DataFrame, permno: int, day: pd.Timestamp, ratio: float) -> pd.DataFrame:
    """`rows` with a `ratio`-for-1 split of `permno` on `day`, as CRSP records it:
    nominal prices fall and share counts rise from `day` on, and the cumulative
    factors (anchored to the last date) rise by `ratio` before it."""
    rows = rows.copy()
    after = (rows.permno == permno) & (rows.date >= day)
    before = (rows.permno == permno) & (rows.date < day)
    rows.loc[after, PRICE_COLUMNS] /= ratio
    rows.loc[after, ["dlyvol", "shrout"]] *= ratio
    rows.loc[before, ["dlycumfacpr", "dlycumfacshr"]] *= ratio
    return rows


def split_rows(seed=0):
    rows = fake_rows(n_stocks=8, n_days=60, seed=seed)
    days = rows.date.unique()
    for permno, at, ratio in [
        (10000, 15, 2.0),
        (10001, 30, 3.0),
        (10000, 40, 4.0),
        (10005, 22, 0.5),
    ]:
        rows = split(rows, permno, days[at], ratio)
    return rows


def signal(expression, panel):
    return factor_signal(parse(expression), panel)


def assert_same(a: pd.DataFrame, b: pd.DataFrame, rtol=1e-9):
    assert a.index.equals(b.index) and a.columns.equals(b.columns)
    np.testing.assert_array_equal(a.isna().to_numpy(), b.isna().to_numpy())
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=rtol, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_a_split_appended_after_the_history_leaves_every_past_level_signal_and_weight(expression):
    rows = split_rows(seed=2)
    day = rows.date.unique()[50]
    base, later = build_panel(rows), build_panel(split(rows, 10002, day, 8.0))
    early = base.universe.index < day
    for name in pit.LEVEL_FIELDS:
        pd.testing.assert_frame_equal(base.fields[name][early], later.fields[name][early])
    assert_same(signal(expression, base)[early], signal(expression, later)[early])
    tree = parse(expression)
    assert_same(factor_weights(tree, base)[early], factor_weights(tree, later)[early])


def test_a_levels_rank_does_not_depend_on_where_the_loaded_window_starts():
    # Rebased to each stock's first loaded row, every close is 1 on that row and
    # rank(close) ranks growth since the window began. Point in time it ranks the
    # nominal prices of the day, whatever the window.
    rows = split_rows(seed=3)
    days = rows.date.unique()
    long, short = build_panel(rows), build_panel(rows[rows.date >= days[20]])
    both = long.universe.index >= days[20]
    assert_same(signal("rank(close)", long)[both], signal("rank(close)", short))
    nominal = rows.assign(price=rows.dlyprc.abs()).pivot(
        index="date", columns="permno", values="price"
    )
    want = nominal.loc[days[20:]].rank(axis=1, pct=True)
    assert_same(signal("rank(close)", short), want.rename_axis(columns="permno"), rtol=1e-12)


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_rescaling_each_stocks_cumulative_factors_changes_nothing(expression):
    rows = split_rows()
    rng = np.random.default_rng(1)
    rescaled = rows.copy()
    for column in ("dlycumfacpr", "dlycumfacshr"):
        k = {p: float(np.exp(rng.uniform(-5, 5))) for p in rows.permno.unique()}
        rescaled[column] = rows[column] * rows.permno.map(k)
    assert_same(signal(expression, build_panel(rows)), signal(expression, build_panel(rescaled)))


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_each_date_is_the_tree_on_the_panel_re_anchored_at_that_date(expression):
    panel = build_panel(split_rows(seed=4))
    tree = parse(expression)
    got = factor_signal(tree, panel)
    dates = panel.universe.index
    for at in (dates[14], dates[15], dates[29], dates[30], dates[41], dates[-1]):
        fields = pit.anchored_at(panel.fields, panel.split_factors, at)
        want = evaluate(tree, fields).loc[at].where(panel.universe.loc[at])
        np.testing.assert_allclose(got.loc[at], want, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_a_level_on_its_own_date_is_the_nominal_level():
    rows = split_rows(seed=6)
    nominal = rows.assign(price=rows.dlyprc.abs()).pivot(index="date", columns="permno")
    got = signal("close", build_panel(rows))
    pd.testing.assert_frame_equal(
        got, nominal["price"].rename_axis(columns="permno"), check_names=False
    )


def test_a_panel_without_split_factors_is_evaluated_as_given():
    panel = build_panel(split_rows(seed=7))
    plain = type(panel)(dict(panel.fields), panel.universe, panel.half_spread)
    want = evaluate(
        parse("-rank(close)"),
        panel.fields,
        strict_domains=True,
        cross_sectional_universe=panel.universe,
    )
    assert_same(signal("-rank(close)", plain), want.where(panel.universe))


def test_reference_features_read_levels_on_one_basis_across_a_split():
    panel = build_panel(split_rows(seed=8))
    dates = panel.universe.index
    panel.calendar = dates
    panel.availability = {"close": pd.Series(dates, index=dates)}
    close = field(panel, "close", dates[-1])
    ratio = close / close.shift(1) - 1
    np.testing.assert_allclose(
        ratio.iloc[1:], panel.fields["returns"].iloc[1:], rtol=1e-9, atol=1e-12
    )
