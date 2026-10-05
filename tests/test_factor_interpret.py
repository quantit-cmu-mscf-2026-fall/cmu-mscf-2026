"""The factor interpreter: every operator defined, causal, and checked by hand.

All on small synthetic panels; no market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.factors.interpret import EvaluationError, evaluate
from capstone.factors.tree import (
    BINARY_PARAMETER_FUNCS,
    BINARY_WINDOW_FUNCS,
    FUNCS,
    STOCK_FIELDS,
    parse,
)
from capstone.factors.zoo import default_zoo


def _panel(n_dates: int = 80, n_stocks: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    cols = [f"S{i}" for i in range(n_stocks)]
    returns = pd.DataFrame(rng.normal(0, 0.02, (n_dates, n_stocks)), index=dates, columns=cols)
    close = 50 * (1 + returns).cumprod()
    open_ = close.shift(1).fillna(close)
    volume = pd.DataFrame(rng.lognormal(10, 0.5, (n_dates, n_stocks)), index=dates, columns=cols)
    shares = pd.DataFrame(1e6, index=dates, columns=cols)
    return {
        "open": open_,
        "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99,
        "close": close,
        "volume": volume,
        "returns": returns,
        "shares": shares,
        "cap": close * shares,
        "mkt_return": returns.mean(axis=1),
    }


def _one(values: list[float], expression: str, field: str = "close") -> pd.Series:
    """Evaluate on one stock whose `field` takes `values`; returns that stock's series."""
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    frame = pd.DataFrame({"A": values}, index=dates, dtype=float)
    return evaluate(parse(expression), {field: frame})["A"]


# --- every operator ---------------------------------------------------------

ALL_OPERATORS = [
    *(f"{f}(close, 5)" for f, takes_window in FUNCS.items() if takes_window),
    *(f"{f}(close)" for f, takes_window in FUNCS.items() if not takes_window),
    *(f"{f}(close, volume, 5)" for f in BINARY_WINDOW_FUNCS),
    *(f"{f}(close, 2)" for f in BINARY_PARAMETER_FUNCS),
]


@pytest.mark.parametrize("expression", ALL_OPERATORS)
def test_every_operator_is_causal(expression):
    # Changing the data after day t must not change any value up to day t.
    panel = _panel()
    cut = 50
    later = {k: v.copy() for k, v in panel.items()}
    for name in ("close", "volume"):
        later[name].iloc[cut + 1 :] *= 3.0
    before = evaluate(parse(expression), panel).iloc[: cut + 1]
    after = evaluate(parse(expression), later).iloc[: cut + 1]
    pd.testing.assert_frame_equal(before, after)
    assert np.isfinite(before.to_numpy()).any(), expression


def test_every_alpha101_member_evaluates():
    panel = _panel(n_dates=300)
    for name, tree in default_zoo():
        values = evaluate(tree, panel)
        assert values.shape == panel["close"].shape, name
        assert np.isfinite(values.to_numpy()).any(), name


# --- definitions, by hand ---------------------------------------------------


def test_time_series_operators_match_hand_computation():
    x = [1.0, 3.0, 2.0, 5.0, 4.0]
    assert _one(x, "shift(close, 2)").tolist()[2:] == [1.0, 3.0, 2.0]
    assert _one(x, "delta(close, 1)").tolist()[1:] == [2.0, -1.0, 3.0, -1.0]
    assert np.isnan(_one(x, "ts_mean(close, 3)").iloc[1])  # window not full yet
    assert _one(x, "ts_mean(close, 3)").iloc[-1] == pytest.approx(11 / 3)
    # Today's 4 is the second largest of (2, 5, 4): rank 2 of 3.
    assert _one(x, "ts_rank(close, 3)").iloc[-1] == pytest.approx(2 / 3)
    # Max of (2, 5, 4) is on the 2nd of the 3 days, counting the oldest as 1.
    assert _one(x, "ts_argmax(close, 3)").iloc[-1] == 2.0
    assert _one(x, "ts_argmin(close, 3)").iloc[-1] == 1.0
    # Weights 1, 2, 3 from the oldest, over (2, 5, 4): (2 + 10 + 12) / 6.
    assert _one(x, "decay_linear(close, 3)").iloc[-1] == pytest.approx(4.0)
    assert _one(x, "ts_product(close, 2)").iloc[-1] == 20.0
    window = np.array([2.0, 5.0, 4.0])
    expected = (4.0 - window.mean()) / window.std(ddof=1)
    assert _one(x, "zscore(close, 3)").iloc[-1] == pytest.approx(expected)
    assert np.isnan(_one(x, "ema(close, 3)").iloc[1])
    assert _one(x, "ema(close, 3)").iloc[-1] == pytest.approx(
        pd.Series(x).ewm(span=3, adjust=False).mean().iloc[-1]
    )


def test_cross_sectional_operators_match_hand_computation():
    dates = pd.bdate_range("2020-01-01", periods=1)
    close = pd.DataFrame([[1.0, 4.0, 2.0, 2.0]], index=dates, columns=list("ABCD"))
    rank = evaluate(parse("rank(close)"), {"close": close}).iloc[0]
    assert rank.tolist() == [0.25, 1.0, 0.625, 0.625]  # ties averaged
    scaled = evaluate(parse("scale(-close)"), {"close": close}).iloc[0]
    assert scaled.abs().sum() == pytest.approx(1.0) and scaled["B"] == pytest.approx(-4 / 9)
    wide = pd.DataFrame([[0.0] * 20 + [100.0]], index=dates)
    clipped = evaluate(parse("winsorize(close)"), {"close": wide}).iloc[0]
    mean, std = wide.iloc[0].mean(), wide.iloc[0].std()
    assert clipped.iloc[-1] == pytest.approx(mean + 3 * std)


def test_undefined_results_are_nan_not_inf():
    assert np.isnan(_one([0.0, 2.0], "log(close)").iloc[0])
    assert np.isnan(_one([-1.0, 2.0], "log(close)").iloc[0])
    values = evaluate(
        parse("close / volume"),
        {
            "close": pd.DataFrame({"A": [1.0, 2.0]}),
            "volume": pd.DataFrame({"A": [0.0, 1.0]}),
        },
    )["A"]
    assert np.isnan(values.iloc[0]) and values.iloc[1] == 2.0
    assert _one([-4.0, 9.0], "signed_power(close, 0.5)").tolist() == [-2.0, 3.0]


def test_a_market_field_is_the_same_for_every_stock():
    panel = _panel()
    # A formula must read a stock field, so close * 0 is there only to parse.
    values = evaluate(parse("close * 0 + mkt_return"), panel)
    for column in values:
        pd.testing.assert_series_equal(
            values[column], panel["mkt_return"].rename(column), check_freq=False
        )


# --- errors -----------------------------------------------------------------


def test_panel_problems_are_named():
    panel = _panel()
    with pytest.raises(EvaluationError, match="no field 'volume'"):
        evaluate(parse("close / volume"), {"close": panel["close"]})
    with pytest.raises(EvaluationError, match="share one index"):
        evaluate(parse("close / volume"), {**panel, "volume": panel["volume"].iloc[1:]})
    with pytest.raises(EvaluationError, match="per-stock field"):
        evaluate(parse("rank(close)"), {"close": panel["close"].iloc[:, 0]})
    assert set(STOCK_FIELDS) <= set(_panel())
