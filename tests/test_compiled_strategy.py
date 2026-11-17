import pandas as pd
import pytest
from reference_fakes import reference_panel

from capstone.factors.research_contracts import canonical
from capstone.strategy.compiled import (
    COMMON,
    RULES,
    CompiledStrategySpec,
    _window,
    compiled_weights,
    decision_due,
    validate_program,
)
from capstone.strategy.diversified import DiversifiedStrategySpec, diversified_weights


def spec(method="weekly_reversal", side="long_only"):
    program = RULES[method] | COMMON | {"profile": side}
    return CompiledStrategySpec(
        method, canonical(program), "a" * 64, "b" * 64, "synthetic-reference-v2", "2000-12-29", side
    )


@pytest.mark.parametrize(
    "method,side",
    [
        (m, s)
        for m in RULES
        for s in ("long_only", "long_short")
        if not (m == "turn_of_month" and s == "long_short")
    ],
)
def test_compiled_specs_roundtrip_and_reuse_the_native_constructions(method, side):
    candidate = spec(method, side)
    assert CompiledStrategySpec.from_toml(candidate.to_toml()) == candidate
    assert candidate.hypothesis_id == "compiled-mechanism/" + method
    if method != "turn_of_month":
        panel = reference_panel()
        native = DiversifiedStrategySpec(method, candidate.data_version, "2000-12-29", side)
        at = "2005-11-30" if method == "monthly_low_volatility" else "2005-11-25"
        pd.testing.assert_series_equal(
            compiled_weights(candidate, panel, at), diversified_weights(native, panel, at)
        )


def test_calendar_next_close_timing_holiday_and_no_unscheduled_bets():
    calendar = pd.bdate_range("2020-11-02", "2021-02-01").difference(
        pd.to_datetime(["2020-11-26", "2020-12-25", "2021-01-01", "2021-01-18"])
    )
    # Nov25 decision -> Nov27 close fill -> Nov30 (last session) earns return.
    assert _window(calendar, "2020-11-25")
    # Dec2 decision -> Dec3 fill -> Dec4 is the fourth, excluded, session.
    assert not _window(calendar, "2020-12-02")
    assert not _window(calendar, "2020-11-20")
    with pytest.raises(ValueError, match="three future"):
        _window(calendar, "2021-02-01")


def test_wrong_horizon_proxy_profile_and_source_identity_cannot_compile():
    wrong = RULES["monthly_low_volatility"] | COMMON | {"profile": "long_only"}
    wrong["signal"] = "daily_volatility_1260"
    with pytest.raises(ValueError, match="mismatch"):
        validate_program("monthly_low_volatility", wrong, "long_only")
    with pytest.raises(ValueError, match="does not authorize"):
        spec("turn_of_month", "long_short")
    candidate = spec()
    with pytest.raises(ValueError, match="implementation changed"):
        CompiledStrategySpec(**(candidate.__dict__ | {"implementation": "c" * 64}))
    with pytest.raises(ValueError, match="compiled strategy changed"):
        CompiledStrategySpec.from_toml(candidate.to_toml().replace("long_only", "long_short"))


def test_calendar_formation_names_are_frozen_at_entry_and_equal_weighted():
    panel = reference_panel()
    candidate = spec("turn_of_month")
    baseline = compiled_weights(candidate, panel, "2006-01-30")
    assert baseline.sum() == pytest.approx(1)
    assert baseline.nunique() == 1
    panel.universe.loc["2006-01-30", 1] = False
    actual = compiled_weights(candidate, panel, "2006-01-30")
    assert 1 not in actual
    assert actual.sum() < baseline.sum()
    # A session outside every window holds nothing.
    assert compiled_weights(candidate, reference_panel(), "2006-01-18").empty


def test_data_version_and_anchor_must_match_the_panel():
    panel = reference_panel()
    with pytest.raises(ValueError, match="differs"):
        compiled_weights(spec(), panel, "2000-01-03")
    other = CompiledStrategySpec(
        "weekly_reversal",
        canonical(RULES["weekly_reversal"] | COMMON | {"profile": "long_only"}),
        "a" * 64,
        "b" * 64,
        "other-version",
        "2000-12-29",
        "long_only",
    )
    with pytest.raises(ValueError, match="differs"):
        compiled_weights(other, panel, "2005-11-25")


def test_decisions_are_due_only_at_window_boundaries_for_calendar_programs():
    panel = reference_panel()
    candidate = spec("turn_of_month")
    calendar = panel.calendar
    sessions_in_year = calendar[(calendar >= "2005-01-03") & (calendar <= "2005-12-30")]
    due = [d for d in sessions_in_year if decision_due(candidate, panel, d)]
    # Entry and exit boundary of each monthly window only: far fewer than the sessions.
    assert 12 <= len(due) <= 36 < len(sessions_in_year)
    assert not decision_due(candidate, panel, "2005-12-14")
