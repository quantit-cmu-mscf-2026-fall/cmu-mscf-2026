from dataclasses import replace

import pandas as pd
import pytest

from capstone.factors.evaluation import HoldoutError
from capstone.factors.research_contracts import canonical
from capstone.pipeline.reference_fixture import reference_panel
from capstone.strategy.compiled import COMMON, RULES, CompiledStrategySpec, compiled_weights
from capstone.strategy.contract import load_spec, target_weights


@pytest.mark.parametrize(
    "method,side",
    [
        (method, side)
        for method in RULES
        for side in ("long_only", "long_short")
        if not (method == "turn_of_month" and side == "long_short")
    ],
)
def test_compiled_spec_loads_and_dispatches_to_its_construction(tmp_path, method, side):
    panel = reference_panel()
    spec = CompiledStrategySpec(
        method,
        canonical(RULES[method] | COMMON | {"profile": side}),
        "a" * 64,
        "b" * 64,
        panel.data_version,
        "2000-12-29",
        side,
    )
    path = tmp_path / "compiled.toml"
    path.write_text(spec.to_toml())
    assert load_spec(path) == spec
    pd.testing.assert_series_equal(
        target_weights(spec, panel, "2005-11-30"),
        compiled_weights(spec, panel, "2005-11-30"),
    )
    path.write_text(spec.to_toml().replace(spec.id, "x" * 16))
    with pytest.raises(ValueError, match="changed"):
        load_spec(path)


def test_compiled_dispatch_keeps_the_whole_panel_holdout_refusal():
    panel = reference_panel()
    spec = CompiledStrategySpec(
        "weekly_reversal",
        canonical(RULES["weekly_reversal"] | COMMON | {"profile": "long_only"}),
        "a" * 64,
        "b" * 64,
        panel.data_version,
        "2000-12-29",
        "long_only",
    )
    panel.universe.loc[pd.Timestamp("2021-01-04")] = True
    with pytest.raises(HoldoutError):
        target_weights(spec, panel, "2005-11-30")
    with pytest.raises(HoldoutError):
        target_weights(replace(spec, anchor="2001-01-02"), panel, "2005-11-30")
