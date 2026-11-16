"""Independent construction arithmetic and causal timing examples, not financial evidence."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from capstone.factors import interpret
from capstone.factors.evaluation import Panel
from capstone.factors.tree import parse
from capstone.strategy.contract import InsufficientHistoryError
from capstone.strategy.program import ProgramStrategySpec, decision_due, program_weights
from capstone.strategy.program_schedule import ProgramSchedule
from capstone.strategy.program_types import (
    POLICY_VERSION,
    SCHEMA_VERSION,
    Units,
    UnitTypeError,
)


@pytest.fixture
def panel():
    dates = pd.bdate_range("2020-06-22", periods=12)
    names = pd.Index([101, 102, 103, 104], name="permno")
    close = pd.DataFrame(
        np.arange(12)[:, None] * np.array([1.0, 2.0, 3.0, 4.0]) + 10,
        index=dates,
        columns=names,
    )
    cap = pd.DataFrame(np.tile([10.0, 20.0, 30.0, 70.0], (12, 1)), index=dates, columns=names)
    returns = close.pct_change(fill_method=None)
    fields = {"close": close, "cap": cap, "returns": returns}
    return Panel(
        fields,
        pd.DataFrame(True, index=dates, columns=names),
        cap * 0,
        data_version="qualified-fixture-v1",
        calendar=dates.append(pd.DatetimeIndex(["2020-07-08"])),
        availability={name: pd.Series(dates, index=dates) for name in fields},
    )


@pytest.fixture
def spec():
    return ProgramStrategySpec(
        expression="delta(close, 1)",
        source_ids=("1" * 64,),
        method_spec_id="2" * 64,
        claim_id="3" * 64,
        mechanism_id="unlisted/arbitrary-mechanism",
        data_version="qualified-fixture-v1",
        anchor="2020-06-23",
        tail=0.5,
        schedule=ProgramSchedule(interval=3),
    )


def test_independent_equal_and_cap_tails(spec, panel):
    at = "2020-06-23"
    equal = program_weights(spec, panel, at)
    assert equal.to_dict() == {103: 0.5, 104: 0.5}
    cap = program_weights(replace(spec, weighting="capitalization"), panel, at)
    assert cap.to_dict() == {103: 0.3, 104: 0.7}
    low = program_weights(replace(spec, direction="low"), panel, at)
    assert low.to_dict() == {101: 0.5, 102: 0.5}


def test_public_contract_loads_and_constructs_general_programs(spec, panel, tmp_path):
    from capstone.strategy.contract import load_spec, save_spec, target_weights

    path = tmp_path / "unfamiliar-source.toml"
    save_spec(spec, path)
    loaded = load_spec(path)
    assert isinstance(loaded, ProgramStrategySpec) and loaded == spec
    assert_series_equal(
        target_weights(loaded, panel, spec.anchor), program_weights(spec, panel, spec.anchor)
    )


def test_public_contract_refuses_a_program_panel_past_the_discovery_period(spec, panel):
    from capstone.factors.evaluation import HoldoutError
    from capstone.strategy.contract import target_weights

    future = pd.Timestamp("2026-10-29")
    panel.universe.loc[future] = True
    with pytest.raises(HoldoutError, match="current-feature route"):
        target_weights(spec, panel, future)


def test_company_accounting_is_refused_until_a_company_aware_route_exists(spec, monkeypatch):
    # The public grammar does not parse accounting fields yet; the guard still has to
    # refuse them, so it is exercised by naming one and making the feature set report it.
    from capstone.factors import tree
    from capstone.strategy.contract import target_weights
    from capstone.strategy.program import require_runtime_capabilities

    monkeypatch.setattr(tree, "FUNDAMENTAL_FIELDS", ("book",), raising=False)
    monkeypatch.setattr(tree, "features_used", lambda node: {"book", "cap"})
    with pytest.raises(ValueError, match="company/share-class alignment"):
        require_runtime_capabilities(spec)
    with pytest.raises(ValueError, match="company/share-class alignment"):
        target_weights(spec, None, spec.anchor)


def test_independent_long_short_gross_and_net(spec, panel):
    book = program_weights(
        replace(spec, side="long_short", weighting="capitalization"),
        panel,
        "2020-06-23",
    )
    assert book.to_dict() == pytest.approx({101: -1 / 6, 102: -1 / 3, 103: 0.15, 104: 0.35})
    assert book.abs().sum() == pytest.approx(1)
    assert book.sum() == pytest.approx(0)


def test_frozen_book_and_sticky_membership_exits(spec, panel):
    first = program_weights(spec, panel, "2020-06-23")
    panel.fields["cap"].loc["2020-06-24"] *= 1000
    panel.fields["close"].loc["2020-06-24", 101] = 100000
    assert_series_equal(program_weights(spec, panel, "2020-06-24"), first)
    panel.universe.loc["2020-06-24", 104] = False
    assert program_weights(spec, panel, "2020-06-24").to_dict() == {103: 0.5}
    assert program_weights(spec, panel, "2020-06-25").to_dict() == {103: 0.5}


def test_formation_membership_does_not_peek_at_future(spec, panel):
    panel.universe.loc["2020-06-23", 104] = False
    formed = program_weights(spec, panel, "2020-06-23")
    assert formed.to_dict() == {103: 1.0}
    panel.universe.loc["2020-06-24":] = False
    assert_series_equal(program_weights(spec, panel, "2020-06-23"), formed)


def test_ties_are_stable_by_security_identifier(spec, panel):
    panel.fields["close"].loc["2020-06-23", [103, 104]] = 14
    book = program_weights(replace(spec, tail=0.25), panel, "2020-06-23")
    assert book.to_dict() == {104: 1.0}


def test_input_row_and_column_permutations(spec, panel):
    expected = program_weights(spec, panel, "2020-06-25")
    for name, frame in panel.fields.items():
        panel.fields[name] = frame.iloc[::-1, ::-1]
        panel.availability[name] = panel.availability[name].iloc[::-1]
    panel.universe = panel.universe.iloc[::-1, ::-1]
    panel.calendar = panel.calendar[::-1]
    assert_series_equal(program_weights(spec, panel, "2020-06-25"), expected)


def test_prefix_and_future_perturbations(spec, panel):
    at = pd.Timestamp("2020-06-25")
    next_day = pd.Timestamp("2020-06-26")
    expected = program_weights(spec, panel, at)
    changed = deepcopy(panel)
    for name in changed.fields:
        changed.fields[name].loc[changed.fields[name].index > at] = np.nan
        changed.availability[name].loc[changed.availability[name].index > at] = pd.NaT
    changed.universe.loc[changed.universe.index > at] = False
    assert_series_equal(program_weights(spec, changed, at), expected)
    changed.fields = {name: frame.loc[:at] for name, frame in changed.fields.items()}
    changed.availability = {name: values.loc[:at] for name, values in changed.availability.items()}
    changed.universe = changed.universe.loc[:at]
    changed.calendar = changed.calendar[changed.calendar <= next_day]
    assert_series_equal(program_weights(spec, changed, at), expected)


def test_fixed_history_has_no_loaded_prefix_dependency(spec, panel):
    expected = program_weights(spec, panel, "2020-06-29")
    # The formation's two required close rows are retained; older rows may differ.
    panel.fields["close"].loc[:"2020-06-24"] = -999
    assert_series_equal(program_weights(spec, panel, "2020-06-29"), expected)


@pytest.mark.parametrize(
    "problem",
    [
        "field",
        "observation",
        "availability",
        "late",
        "score",
        "duplicate",
        "held_membership",
    ],
)
def test_missing_inputs_block_explicitly(spec, panel, problem):
    if problem == "field":
        del panel.fields["close"]
    elif problem == "observation":
        panel.fields["close"] = panel.fields["close"].drop(pd.Timestamp("2020-06-22"))
    elif problem == "availability":
        panel.availability["close"] = panel.availability["close"].drop(pd.Timestamp("2020-06-22"))
    elif problem == "late":
        panel.availability["close"].loc["2020-06-22"] = pd.Timestamp("2020-06-24")
    elif problem == "score":
        panel.fields["close"].loc["2020-06-22", 104] = np.nan
    elif problem == "duplicate":
        panel.fields["close"] = pd.concat([panel.fields["close"], panel.fields["close"].iloc[:1]])
    else:
        panel.universe = panel.universe.drop(pd.Timestamp("2020-06-24"))
    with pytest.raises(ValueError):
        program_weights(spec, panel, "2020-06-25")


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_selected_cap_blocks(spec, panel, value):
    panel.fields["cap"].loc["2020-06-23", 104] = value
    with pytest.raises(ValueError, match="capitalization"):
        program_weights(replace(spec, weighting="capitalization"), panel, "2020-06-23")


def test_cap_weighting_uses_only_formation_cap(spec, panel):
    panel.fields["cap"] = panel.fields["cap"].loc[[pd.Timestamp("2020-06-23")]]
    panel.availability["cap"] = panel.availability["cap"].loc[[pd.Timestamp("2020-06-23")]]
    assert program_weights(
        replace(spec, weighting="capitalization"), panel, "2020-06-25"
    ).to_dict() == {103: 0.3, 104: 0.7}


def test_extreme_finite_caps_do_not_overflow_to_an_empty_book(spec, panel):
    panel.fields["cap"].loc["2020-06-23", [103, 104]] = 1e308
    book = program_weights(replace(spec, weighting="capitalization"), panel, "2020-06-23")
    assert book.to_dict() == {103: 0.5, 104: 0.5}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_outer_transform_cannot_hide_invalid_raw_inputs(spec, panel, value):
    panel.fields["close"].loc["2020-06-23", 104] = value
    with pytest.raises(ValueError, match="missing or nonfinite"):
        program_weights(replace(spec, expression="sign(close)"), panel, "2020-06-23")


@pytest.mark.parametrize(
    "expression",
    [
        "sign((close - 25) / (close - 11))",
        "rank((close - 25) / (close - 11))",
        "scale((close - 25) / (close - 11))",
        "1 / (1 / (close - 11))",
        "sign(close * (1e308 * 1e308))",
        "close + (1 / 0)",
    ],
)
def test_undefined_intermediate_cannot_become_a_valid_score(panel, expression):
    row = interpret.evaluate(parse(expression), panel.fields, strict_domains=True).loc["2020-06-23"]
    assert pd.isna(row.loc[101])


def test_undefined_domains_are_nan_in_the_default_and_the_strict_interpreter():
    # The interpreter turns an undefined result into NaN in its default mode as well, so
    # an outer sign cannot hide an inner infinity in either mode.
    fields = {"close": pd.DataFrame([[11.0, 21.0, 31.0, 41.0]])}
    node = parse("sign((close - 25) / (close - 11))")
    for strict in (False, True):
        result = interpret.evaluate(node, fields, strict_domains=strict).iloc[0]
        assert pd.isna(result.iloc[0])
        assert result.iloc[1:].tolist() == [-1.0, 1.0, 1.0]
        assert (
            interpret.evaluate(parse("close + (1 / 0)"), fields, strict_domains=strict)
            .isna()
            .all()
            .all()
        )


def test_strict_scale_uses_finite_normalization_before_outer_signal():
    fields = {
        "close": pd.DataFrame([[1e308, 1e308]]),
        "cap": pd.DataFrame([[1.0, 2.0]]),
    }
    node = parse("scale(close) + cap")
    assert interpret.evaluate(node, fields).iloc[0].tolist() == [1.0, 2.0]
    assert interpret.evaluate(node, fields, strict_domains=True).iloc[0].tolist() == [
        1.5,
        2.5,
    ]


@pytest.mark.parametrize(
    "expression", ["close + log(2)", "close + abs(2)", "close + ts_mean(2, 2)"]
)
def test_strict_function_constant_arguments_broadcast_explicitly(panel, expression):
    node = parse(expression)
    with pytest.raises(AttributeError):
        interpret.evaluate(node, panel.fields)
    result = interpret.evaluate(node, panel.fields, strict_domains=True)
    expected = panel.fields["close"].copy()
    expected += np.log(2) if "log" in expression else 2
    if "ts_mean" in expression:
        expected.iloc[0] = np.nan
    pd.testing.assert_frame_equal(result, expected)


def test_strict_correlation_avoids_internal_variance_product_overflow():
    close = pd.DataFrame([[1.0e150, 2.0e150], [1.1e150, 2.1e150]])
    fields = {"close": close}
    node = parse("ts_corr(close, close, 2)")
    assert interpret.evaluate(node, fields).iloc[-1].tolist() == [0.0, 0.0]
    assert interpret.evaluate(node, fields, strict_domains=True).iloc[-1].tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_cross_sections_use_dated_members_and_keep_historical_exits(spec, panel):
    at = pd.Timestamp("2020-06-24")
    previous = pd.Timestamp("2020-06-23")
    panel.fields["close"].loc[previous] = [1.0, 3.0, 2.0, 1e9]
    panel.fields["close"].loc[at] = [1.0, 7.0, np.nan, 1e9]
    panel.fields["open"] = panel.fields["close"] * 0 + 10
    panel.availability["open"] = panel.availability["close"].copy()
    panel.universe.loc[previous] = [True, True, True, False]
    panel.universe.loc[at] = [True, True, False, False]
    built = replace(spec, expression="shift(rank(close), 1) - close / open", anchor=str(at.date()))
    # Prior ranks are (1/3, 1, 2/3), including the later-exited security 103.
    # Scores for today's 101/102 are 1/3-.1 and 1-.7; hence 102 is selected.
    assert program_weights(built, panel, at).to_dict() == {102: 1.0}
    panel.fields["close"].loc[:, 104] = -1e20
    panel.universe.loc["2020-06-25", 104] = True
    assert program_weights(built, panel, at).to_dict() == {102: 1.0}
    # The same dated rank helper also supports rolling compositions.
    fields = {"close": panel.fields["close"].loc[[previous, at], [101, 102, 103, 104]]}
    mask = panel.universe.loc[[previous, at], fields["close"].columns]
    values = interpret.evaluate(
        parse("ts_mean(rank(close), 2)"),
        fields,
        strict_domains=True,
        cross_sectional_universe=mask,
    )
    assert values.loc[at, [101, 102]].tolist() == pytest.approx([5 / 12, 1])


def test_missing_historical_cross_section_member_blocks(spec, panel):
    panel.universe.loc["2020-06-24", 103] = False
    panel.fields["close"].loc["2020-06-23", 103] = np.nan
    built = replace(spec, expression="shift(rank(close), 1)", anchor="2020-06-24")
    with pytest.raises(ValueError, match="scores"):
        program_weights(built, panel, "2020-06-24")


def test_price_history_is_not_removed_for_pre_entry_dates(spec, panel):
    panel.universe.loc["2020-06-22", 104] = False
    assert program_weights(spec, panel, "2020-06-23").to_dict() == {103: 0.5, 104: 0.5}


def test_later_publication_cannot_rewrite_an_earlier_feature_row(spec, panel):
    panel.availability["close"].loc["2020-06-22"] = pd.Timestamp("2020-06-23")
    with pytest.raises(ValueError, match="observation session"):
        program_weights(spec, panel, "2020-06-23")


def test_intraday_publication_gets_a_precision_blocker(spec, panel):
    panel.availability["close"].loc["2020-06-23"] = pd.Timestamp("2020-06-23 16:00")
    with pytest.raises(ValueError, match="availability precision"):
        program_weights(spec, panel, "2020-06-23")


@pytest.mark.parametrize(
    "expression",
    ["close + volume", "close - 25", "log(close)", "close + ts_mean(2, 2)"],
)
def test_constructor_enforces_independent_units_without_rewriting_proposal(spec, expression):
    before = parse(expression)
    with pytest.raises(UnitTypeError):
        replace(spec, expression=expression)
    assert parse(expression) == before


def test_constructor_carries_actual_type_evidence_and_keeps_qualification_separate(
    spec,
):
    typed = replace(spec, expression="returns / cap")
    evidence = typed.type_evidence
    assert evidence.output_type.units == Units(usd=-1)
    assert evidence.output_type.entity == "equity"
    assert evidence.schema_version == SCHEMA_VERSION
    assert evidence.policy_version == POLICY_VERSION
    assert evidence.normalization_required is True
    assert typed.gate_report == ""
    assert ProgramStrategySpec.from_toml(typed.to_toml()).type_evidence == evidence
    with pytest.raises((TypeError, ValueError)):
        replace(typed, type_evidence_json="pretend-qualified")


def test_serialized_type_claim_cannot_override_ast_inference(spec):
    import json
    import tomllib

    row = tomllib.loads(spec.to_toml())
    payload = json.loads(row["payload"])
    claimed = json.loads(payload["type_evidence_json"])
    claimed["normalization_required"] = False
    payload["type_evidence_json"] = json.dumps(claimed)
    changed = f'schema_version = 6\nid = "{spec.id}"\npayload = {json.dumps(json.dumps(payload))}\n'
    with pytest.raises(ValueError, match="changed"):
        ProgramStrategySpec.from_toml(changed)


def test_unit_checker_implementation_is_bound_to_new_program_identity(spec, monkeypatch, tmp_path):
    from capstone.strategy import program

    replacement = tmp_path / "program_types.py"
    replacement.write_text("# changed unit implementation\n", encoding="utf-8")
    monkeypatch.setattr(program.program_types, "__file__", str(replacement))
    with pytest.raises(ValueError, match="implementation changed"):
        replace(spec)


@pytest.mark.parametrize(
    "expression",
    [
        "sign((close / open - 2.5) / (close / open - 1.1))",
        "rank((close / open - 2.5) / (close / open - 1.1))",
        "scale((close / open - 2.5) / (close / open - 1.1))",
        "1 / (1 / (close / open - 1.1))",
        "sign(close * (1e308 * 1e308))",
        "close * (1 / 0)",
    ],
)
def test_type_valid_constructor_still_blocks_undefined_numeric_domains(spec, panel, expression):
    panel.fields["open"] = panel.fields["close"] * 0 + 10
    panel.availability["open"] = panel.availability["close"].copy()
    typed = replace(spec, expression=expression)
    assert typed.type_evidence.normalization_required
    with pytest.raises(ValueError, match="scores"):
        program_weights(typed, panel, "2020-06-23")


def test_typed_ratio_and_strict_pit_interpreter_compose_in_target_construction(spec, panel):
    panel.fields["open"] = panel.fields["close"] * 0 + 10
    panel.availability["open"] = panel.availability["close"].copy()
    typed = replace(spec, expression="rank(log(close / open))")
    assert typed.type_evidence.output_type.units == Units()
    assert program_weights(typed, panel, "2020-06-23").to_dict() == {103: 0.5, 104: 0.5}


@pytest.mark.parametrize(
    "expression",
    [
        "close + cap",
        "close / cap",
        "sign(close - 15)",
        "ts_mean(close, 2)",
        "ts_std(close, 2)",
        "rank(close)",
        "ts_corr(close, cap, 2)",
        "signed_power(close, 2)",
    ],
)
def test_strict_mode_retains_finite_supported_numerics(panel, expression):
    node = parse(expression)
    default = interpret.evaluate(node, panel.fields)
    strict = interpret.evaluate(node, panel.fields, strict_domains=True)
    pd.testing.assert_frame_equal(strict, default)


def test_insufficient_history_and_flat_scores(spec, panel):
    with pytest.raises(InsufficientHistoryError):
        program_weights(replace(spec, expression="ts_mean(close, 10)"), panel, "2020-06-23")
    panel.fields["close"].loc["2020-06-23"] = 10
    with pytest.raises(ValueError, match="distinction"):
        program_weights(replace(spec, expression="close"), panel, "2020-06-23")


def test_serialization_identity_and_nonadmission(spec):
    assert ProgramStrategySpec.from_toml(spec.to_toml()) == spec
    assert replace(spec, expression=" delta( close , 1 ) ").id == spec.id
    changed = replace(spec, source_ids=("3" * 64,))
    assert changed.id != spec.id
    assert changed.construction_id == spec.construction_id
    assert changed.factor_id == spec.factor_id
    assert replace(spec, mechanism_id="renamed").construction_id == spec.construction_id
    assert replace(spec, direction="low").id != spec.id
    assert replace(spec, schedule=ProgramSchedule(interval=4)).id != spec.id
    multi = replace(spec, source_ids=("1" * 64, "3" * 64))
    assert replace(multi, source_ids=multi.source_ids[::-1]).id == multi.id
    assert spec.fields == ("close",)
    assert replace(spec, weighting="capitalization").fields == ("cap", "close")
    assert spec.gate_report == ""
    with pytest.raises(ValueError):
        ProgramStrategySpec.from_toml(spec.to_toml().replace(spec.id, "0" * 16, 1))
    with pytest.raises(ValueError, match="implementation"):
        replace(spec, implementation="0" * 64)


def test_duplicate_key_resists_equivalent_expression_and_provenance_renaming(spec):
    first = replace(spec, expression="close + low")
    changed = replace(
        first,
        expression="low + close",
        source_ids=("9" * 64,),
        mechanism_id="new-label",
    )
    assert first.factor_id == changed.factor_id
    assert first.construction_id == changed.construction_id
    assert first.id != changed.id


def test_claim_and_method_revisions_preserve_controller_allocation_family(spec):
    changed_claim = replace(spec, claim_id="4" * 64)
    changed_method = replace(spec, method_spec_id="5" * 64)
    source_alias = replace(spec, source_ids=("6" * 64,))
    changed_program = replace(spec, expression="close")
    assert spec.claim_id != spec.method_spec_id
    for revised in (changed_claim, changed_method, source_alias, changed_program):
        assert revised.hypothesis_id == spec.hypothesis_id == spec.mechanism_id
        assert revised.id != spec.id
    assert (
        changed_claim.construction_id
        == changed_method.construction_id
        == source_alias.construction_id
        == spec.construction_id
    )
    assert replace(spec, mechanism_id="other-controller-family").hypothesis_id != spec.hypothesis_id
    assert ProgramStrategySpec.from_toml(changed_claim.to_toml()).claim_id == "4" * 64


def test_claim_binding_is_required_and_hash_typed(spec):
    import json
    import tomllib

    with pytest.raises(ValueError, match="claim"):
        replace(spec, claim_id="blog-alias")
    row = tomllib.loads(spec.to_toml())
    payload = json.loads(row["payload"])
    del payload["claim_id"]
    text = f'schema_version = 6\nid = "{spec.id}"\npayload = {json.dumps(json.dumps(payload))}\n'
    with pytest.raises(ValueError, match="malformed"):
        ProgramStrategySpec.from_toml(text)


@pytest.mark.parametrize(
    "change",
    [
        {"expression": "ema(close, 5)"},
        {"expression": "python(close)"},
        {"tail": True},
        {"tail": 0},
        {"tail": np.nan},
        {"tail": 0.6},
        {"gross": True},
        {"gross": 2},
        {"side": "levered"},
        {"construction": "overlapping_cohorts"},
        {"exit_rule": "calendar_event"},
        {"universe_rule": "future_survivors"},
        {"execution": "same_close"},
        {"availability": "future_revision"},
        {"weighting": "optimized"},
        {"source_ids": ()},
        {"source_ids": ("blog",)},
        {"anchor": "2020-06-23T16:00:00"},
        {"anchor": "NaT"},
    ],
)
def test_unsupported_semantics_fail_without_substitution(spec, change):
    with pytest.raises(ValueError):
        replace(spec, **change)


def test_schedule_actual_holiday_boundaries_and_interval_anchor():
    calendar = pd.DatetimeIndex(
        [
            "2020-06-29",
            "2020-06-30",
            "2020-07-01",
            "2020-07-02",
            "2020-07-06",
            "2020-07-07",
        ]
    )
    weekly = ProgramSchedule("week_end", 1)
    monthly = ProgramSchedule("month_end", 1)
    assert weekly.formations(calendar, "2020-06-29", "2020-07-06").tolist() == [
        pd.Timestamp("2020-07-02")
    ]
    assert monthly.formations(calendar, "2020-06-29", "2020-07-06").tolist() == [
        pd.Timestamp("2020-06-30")
    ]
    interval = ProgramSchedule(interval=2)
    assert interval.formations(calendar, "2020-06-30", "2020-07-06").tolist() == [
        pd.Timestamp("2020-06-30"),
        pd.Timestamp("2020-07-02"),
    ]
    assert weekly.decision_due(calendar, "2020-06-29", "2020-07-02")
    assert not weekly.decision_due(calendar, "2020-06-29", "2020-07-01")


def test_schedule_and_target_share_formation(spec, panel):
    assert decision_due(spec, panel, "2020-06-23")
    assert not decision_due(spec, panel, "2020-06-24")
    assert decision_due(spec, panel, "2020-06-26")
    weekly = replace(spec, schedule=ProgramSchedule("week_end", 1))
    with pytest.raises(InsufficientHistoryError, match="no completed"):
        program_weights(weekly, panel, "2020-06-24")


@pytest.mark.parametrize("problem", ["absent", "anchor", "next", "duplicates"])
def test_authoritative_calendar_required(spec, panel, problem):
    if problem == "absent":
        panel.calendar = None
    elif problem == "anchor":
        panel.calendar = panel.calendar[panel.calendar != pd.Timestamp(spec.anchor)]
    elif problem == "next":
        panel.calendar = panel.calendar[panel.calendar <= pd.Timestamp("2020-06-25")]
    else:
        panel.calendar = panel.calendar.append(panel.calendar[:1])
    with pytest.raises(ValueError):
        program_weights(spec, panel, "2020-06-25")


def test_existing_binary_rolling_formula_and_market_series_are_reused(spec, panel):
    market = pd.Series(np.arange(len(panel.universe)) / 100, index=panel.universe.index)
    panel.fields["mkt_return"] = market
    panel.availability["mkt_return"] = pd.Series(market.index, index=market.index)
    expression = "ts_cov(returns, mkt_return, 2) / ts_cov(mkt_return, mkt_return, 2)"
    built = replace(spec, expression=expression, anchor="2020-06-24")
    # Each stock's simple-return change declines with its larger first-day return.
    book = program_weights(built, panel, "2020-06-24")
    assert book.to_dict() == {101: 0.5, 102: 0.5}


def test_membership_reentry_only_at_new_formation(spec, panel):
    panel.universe.loc["2020-06-24", 104] = False
    assert 104 not in program_weights(spec, panel, "2020-06-25")
    assert 104 in program_weights(spec, panel, "2020-06-26")


@pytest.mark.parametrize(
    "changes",
    [
        {"side": []},
        {"direction": {}},
        {"weighting": []},
        {"tail": 10**400},
    ],
)
def test_malformed_public_program_contract_is_a_value_error(spec, changes):
    with pytest.raises(ValueError):
        replace(spec, **changes)


def test_malformed_public_schedule_kind_is_a_value_error():
    with pytest.raises(ValueError):
        ProgramSchedule(kind=[])
