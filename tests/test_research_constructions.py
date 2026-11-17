from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from reference_fakes import reference_panel

from capstone.pipeline.study import definition
from capstone.strategy.contract import load_spec, target_weights
from capstone.strategy.research_spec import ResearchStrategySpec, candidate_spec


def small_panel(n=20):
    source = reference_panel()
    names = list(source.universe.columns[:n])
    panel = replace(
        source,
        fields={k: f.loc[:, names].copy() for k, f in source.fields.items()},
        universe=source.universe.loc[:, names].copy(),
        half_spread=source.half_spread.loc[:, names].copy(),
    )
    panel.universe.loc[:, :] = True
    for i, name in enumerate(names):
        panel.fields["returns"][name] = (i + 1) * 0.0001
        panel.fields["cap"][name] = (i + 1) * 1000.0
    return panel


def fundamental_panel():
    panel = small_panel(12)
    panel.fields["cap"].loc[:, :] = 1_000_000.0
    panel.fields["cap"].loc[:, 1] = 3_000_000.0
    panel.company = pd.DataFrame({n: n for n in panel.universe.columns}, index=panel.universe.index)
    panel.company.loc[:, 2] = 1  # two classes of the same company
    panel.sic = pd.DataFrame(5000, index=panel.universe.index, columns=panel.universe.columns)
    panel.sic.loc[:, 12] = 6000  # financial company excluded even with the highest score
    panel.fundamentals = pd.DataFrame(
        [
            {
                "permco": n,
                "datadate": pd.Timestamp("2004-12-31"),
                "published_at": pd.Timestamp("2005-03-15"),
                "vintage_at": pd.Timestamp("2005-03-15"),
                "book": n * 1_000_000.0,
                "revenue": n * 100.0,
                "cogs": 0.0,
                "assets": 10000.0,
            }
            for n in [1, *range(3, 13)]
        ]
    )
    return panel


def test_all_frozen_profiles_have_distinct_roundtripping_identities(tmp_path):
    specs = [candidate_spec(c, "test-version", "2000-12-29") for c in definition()["candidates"]]
    assert len({s.id for s in specs}) == 36
    for i, spec in enumerate(specs):
        path = tmp_path / f"s-{i}.toml"
        path.write_text(spec.to_toml())
        assert load_spec(path) == spec


def test_weighting_refinement_uses_known_formation_caps_without_changing_selections():
    panel = small_panel()
    spec = ResearchStrategySpec(
        "716bdaf7915c65be",
        "momentum_6m",
        "alternate_weighting",
        panel.data_version,
        "2000-12-29",
        "long_only",
        weighting="capitalization",
    )
    result = target_weights(spec, panel, "2005-11-30")
    assert set(result.index) == {19, 20}
    np.testing.assert_allclose(result.reindex([19, 20]), [19 / 39, 20 / 39])
    original = result.copy()
    for frame in panel.fields.values():
        frame.loc[frame.index > "2005-11-30"] *= 3
    pd.testing.assert_series_equal(original, target_weights(spec, panel, "2005-11-30"))


@pytest.mark.parametrize("method", ["value", "profitability"])
@pytest.mark.parametrize("side", ["long_only", "long_short"])
def test_fundamental_examples_are_company_level_and_use_prior_june_accounts(method, side):
    panel = fundamental_panel()
    spec = ResearchStrategySpec(
        f"paper-{method}", method, "fundamental", panel.data_version, "2000-12-29", side
    )
    result = target_weights(spec, panel, "2005-12-30")
    assert result.loc[11] == (1 if side == "long_only" else 0.5)
    if side == "long_short":
        assert result.loc[1] == -0.5
    assert 2 not in result and 12 not in result
    future = panel.fundamentals.copy()
    future["datadate"] = pd.Timestamp("2005-12-31")
    future["published_at"] = future["vintage_at"] = pd.Timestamp("2006-03-01")
    future["book"] *= 1000
    panel.fundamentals = pd.concat([future, panel.fundamentals]).sample(frac=1, random_state=4)
    pd.testing.assert_series_equal(result, target_weights(spec, panel, "2005-12-30"))
    panel.universe.loc["2005-12-30", 11] = False
    assert 11 not in target_weights(spec, panel, "2005-12-30")


def test_unpublished_or_unqualified_accounts_cannot_be_traded():
    panel = fundamental_panel()
    spec = ResearchStrategySpec(
        "paper-value", "value", "fundamental", panel.data_version, "2000-12-29"
    )
    panel.fundamentals["published_at"] = pd.Timestamp("2005-07-01")
    with pytest.raises(ValueError, match="insufficient"):
        target_weights(spec, panel, "2005-12-30")
    panel.fundamentals = None
    with pytest.raises(ValueError, match="qualified"):
        target_weights(spec, panel, "2005-12-30")


@pytest.mark.parametrize(
    "tail,weighting,expected",
    [
        (0.3, "capitalization", {9: 0.2, 10: 0.2, 11: 0.6}),
        (0.2, "capitalization", {10: 0.25, 11: 0.75}),
        (0.3, "equal", {9: 1 / 3, 10: 1 / 3, 11: 1 / 3}),
    ],
)
def test_profitability_tail_and_weighting_examples_have_known_answers(tail, weighting, expected):
    panel = fundamental_panel()
    panel.fields["cap"].loc[:, 11] = 3_000_000.0
    spec = ResearchStrategySpec(
        "paper-profitability",
        "profitability",
        "fundamental",
        panel.data_version,
        "2000-12-29",
        "long_only",
        tail,
        weighting,
    )
    result = target_weights(spec, panel, "2005-12-30")
    assert set(result.index) == set(expected)
    np.testing.assert_allclose(result, [expected[n] for n in result.index])
