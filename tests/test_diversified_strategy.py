from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from reference_fakes import reference_panel

from capstone.strategy.diversified import (
    DiversifiedStrategySpec,
    decision_due,
)
from capstone.strategy.diversified import (
    diversified_weights as target_weights,
)


def make_panel():
    panel = reference_panel()
    panel.universe.loc[:, :] = True
    dates = panel.universe.index
    # Alternating monthly returns with independently known rank of monthly volatility.
    sign = np.where(dates.month % 2, 1.0, -1.0)
    for i, n in enumerate(panel.universe.columns):
        panel.fields["returns"][n] = sign * (i + 1) / 10000
        panel.fields["cap"][n] = float(i + 1)
    return panel


@pytest.mark.parametrize("method", ["weekly_reversal", "monthly_low_volatility"])
@pytest.mark.parametrize("side", ["long_only", "long_short"])
def test_specs_roundtrip_and_construction_obeys_profiles(method, side):
    panel = make_panel()
    spec = DiversifiedStrategySpec(method, panel.data_version, "2000-12-29", side)
    assert DiversifiedStrategySpec.from_toml(spec.to_toml()) == spec
    result = target_weights(spec, panel, "2005-11-30")
    assert result.abs().sum() == pytest.approx(1)
    assert result.sum() == pytest.approx(1 if side == "long_only" else 0)
    assert (result >= 0).all() if side == "long_only" else (result < 0).any()


def test_weekly_reference_is_return_deviation_weighted_and_holiday_aware():
    panel = make_panel()
    formation = pd.Timestamp("2005-11-25")
    n = len(panel.universe.columns)
    dates = panel.universe.index
    week = dates[dates.to_period("W-FRI") == formation.to_period("W-FRI")]
    for i, name in enumerate(panel.universe.columns):
        panel.fields["returns"].loc[week, name] = (1 + i / 100) ** (1 / len(week)) - 1
    spec = DiversifiedStrategySpec(
        "weekly_reversal", panel.data_version, "2000-12-29", "long_short"
    )
    actual = target_weights(spec, panel, formation)
    raw = np.mean(np.arange(n) / 100) - np.arange(n) / 100
    np.testing.assert_allclose(
        actual.reindex(panel.universe.columns, fill_value=0), raw / abs(raw).sum(), atol=1e-12
    )
    assert decision_due(spec, panel, formation)
    assert not decision_due(spec, panel, "2005-11-28")
    # An observed exchange holiday changes the completed-week boundary, not its row count.
    panel.calendar = panel.calendar[panel.calendar != formation]
    panel.universe = panel.universe.drop(index=formation)
    panel.fields = {k: v.drop(index=formation) for k, v in panel.fields.items()}
    assert decision_due(spec, panel, "2005-11-24")


def test_monthly_risk_quintile_uses_formation_caps_and_requires_real_history():
    panel = make_panel()
    spec = DiversifiedStrategySpec("monthly_low_volatility", panel.data_version, "2000-12-29")
    actual = target_weights(spec, panel, "2005-11-30")
    names = list(panel.universe.columns[: len(panel.universe.columns) // 5])
    assert set(actual.index) == set(names)
    expected = panel.fields["cap"].loc["2005-11-30", names]
    np.testing.assert_allclose(actual.loc[names], expected / expected.sum())
    panel.fields["returns"].loc["2005-11-01", names] = np.nan
    assert not set(target_weights(spec, panel, "2005-11-30").index) & set(names)
    panel.fields["returns"].loc[:"2004-12-31"] = np.nan
    with pytest.raises(ValueError, match="five qualified"):
        target_weights(spec, panel, "2005-11-30")


@pytest.mark.parametrize("method", ["weekly_reversal", "monthly_low_volatility"])
def test_future_data_and_input_column_order_cannot_change_prior_books(method):
    panel = make_panel()
    spec = DiversifiedStrategySpec(method, panel.data_version, "2000-12-29")
    expected = target_weights(spec, panel, "2005-11-30")
    changed = deepcopy(panel)
    for key, frame in changed.fields.items():
        frame.loc[frame.index > "2005-11-30"] = 1000
        changed.fields[key] = frame[frame.columns[::-1]]
    changed.universe = changed.universe[changed.universe.columns[::-1]]
    pd.testing.assert_series_equal(expected, target_weights(spec, changed, "2005-11-30"))
    panel.fields["returns"].index = panel.fields["returns"].index[::-1]
    with pytest.raises(ValueError, match="dates must increase"):
        target_weights(spec, panel, "2005-11-30")


def test_monthly_risk_cannot_be_replaced_by_a_daily_volatility_proxy():
    panel = make_panel()
    # Stock 1 has violent offsetting daily moves but a zero monthly return.
    # It must beat stocks with smooth daily moves and variable monthly returns.
    returns = panel.fields["returns"]
    returns.loc[:, 1] = 0.0
    for _, block in returns.groupby(returns.index.to_period("M")):
        if len(block) >= 2:
            returns.loc[block.index[0], 1] = 0.5
            returns.loc[block.index[1], 1] = -1 / 3
    panel.fields["cap"].loc[:, 1] = 1e10
    spec = DiversifiedStrategySpec("monthly_low_volatility", panel.data_version, "2000-12-29")
    actual = target_weights(spec, panel, "2005-11-30")
    assert 1 in actual and actual.loc[1] > 0.99
    daily_risk = returns.loc["2001-01-01":"2005-11-30"].std()
    assert daily_risk.loc[1] > daily_risk.drop(1).max()


def test_changed_specification_or_implementation_is_refused():
    spec = DiversifiedStrategySpec("weekly_reversal", "v1", "2000-12-29")
    text = spec.to_toml()
    with pytest.raises(ValueError, match="changed"):
        DiversifiedStrategySpec.from_toml(text.replace("long_only", "long_short"))
    with pytest.raises(ValueError, match="implementation changed"):
        DiversifiedStrategySpec("weekly_reversal", "v1", "2000-12-29", method_hash="0" * 64)
    with pytest.raises(ValueError, match="unknown"):
        DiversifiedStrategySpec("momentum", "v1", "2000-12-29")
