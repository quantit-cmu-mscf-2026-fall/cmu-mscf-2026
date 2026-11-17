"""Compiled and diversified specs load and construct through the one strategy contract."""

import pandas as pd
import pytest
from reference_fakes import reference_panel

from capstone.factors.evaluation import DISCOVERY_END, HoldoutError
from capstone.factors.research_contracts import canonical
from capstone.strategy.compiled import COMMON, RULES, CompiledStrategySpec, compiled_weights
from capstone.strategy.contract import load_spec, save_spec, target_weights
from capstone.strategy.diversified import DiversifiedStrategySpec, diversified_weights


def compiled(method="weekly_reversal", side="long_only"):
    program = RULES[method] | COMMON | {"profile": side}
    return CompiledStrategySpec(
        method, canonical(program), "a" * 64, "b" * 64, "synthetic-reference-v2", "2000-12-29", side
    )


@pytest.mark.parametrize("method", ["weekly_reversal", "monthly_low_volatility"])
@pytest.mark.parametrize("side", ["long_only", "long_short"])
def test_diversified_file_round_trips_and_builds_through_the_contract(tmp_path, method, side):
    panel = reference_panel()
    spec = DiversifiedStrategySpec(method, panel.data_version, "2000-12-29", side)
    path = tmp_path / "spec.toml"
    save_spec(spec, path)
    assert load_spec(path) == spec
    at = "2005-11-30" if method == "monthly_low_volatility" else "2005-11-25"
    pd.testing.assert_series_equal(
        target_weights(spec, panel, at), diversified_weights(spec, panel, at)
    )


@pytest.mark.parametrize("method", ["weekly_reversal", "monthly_low_volatility"])
def test_compiled_file_round_trips_and_builds_through_the_contract(tmp_path, method):
    panel = reference_panel()
    spec = compiled(method)
    path = tmp_path / "spec.toml"
    save_spec(spec, path)
    assert load_spec(path) == spec
    at = "2005-11-30" if method == "monthly_low_volatility" else "2005-11-25"
    pd.testing.assert_series_equal(
        target_weights(spec, panel, at), compiled_weights(spec, panel, at)
    )


def test_an_altered_spec_file_is_refused_by_the_contract_loader(tmp_path):
    spec = DiversifiedStrategySpec("weekly_reversal", "v1", "2000-12-29")
    path = tmp_path / "spec.toml"
    path.write_bytes(spec.to_toml().replace("long_only", "long_short").encode("utf-8"))
    with pytest.raises(ValueError, match="changed"):
        load_spec(path)


def test_the_contract_never_reads_the_holdout_for_these_specs():
    panel = reference_panel()
    late = pd.Timestamp(DISCOVERY_END) + pd.Timedelta(days=30)
    panel.universe.loc[late] = True
    spec = DiversifiedStrategySpec("weekly_reversal", panel.data_version, "2000-12-29")
    with pytest.raises(HoldoutError):
        target_weights(spec, panel, "2005-11-25")
