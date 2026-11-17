from dataclasses import FrozenInstanceError, asdict

import pytest

from capstone.factors import tree
from capstone.strategy.program_types import (
    MAX_POWER,
    NORMALIZED_EQUITY_INPUTS,
    POLICY_VERSION,
    SCHEMA_VERSION,
    Units,
    UnitTypeError,
    infer_types,
)

FUNDAMENTALS = ("book", "earnings", "sales", "assets")
needs_fundamentals = pytest.mark.skipif(
    not set(FUNDAMENTALS) <= set(tree.FIELDS),
    reason="the annual fundamental fields enter the grammar with the fundamentals change",
)


def typed(expression):
    return infer_types(tree.parse(expression))


def test_normalized_schema_covers_all_legacy_fields_with_explicit_entities_and_units():
    assert set(tree.FIELDS) <= set(NORMALIZED_EQUITY_INPUTS)
    assert set(NORMALIZED_EQUITY_INPUTS) - set(tree.FIELDS) <= set(FUNDAMENTALS)
    assert NORMALIZED_EQUITY_INPUTS["close"].units == Units(usd=1, shares=-1)
    assert NORMALIZED_EQUITY_INPUTS["volume"].units == Units(shares=1)
    assert NORMALIZED_EQUITY_INPUTS["shares"].units == Units(shares=1)
    assert all(
        NORMALIZED_EQUITY_INPUTS[name].units == Units(usd=1) for name in ("cap", *FUNDAMENTALS)
    )
    assert NORMALIZED_EQUITY_INPUTS["mkt_return"].entity == "market"
    with pytest.raises(TypeError):
        NORMALIZED_EQUITY_INPUTS["close"] = NORMALIZED_EQUITY_INPUTS["volume"]


@pytest.mark.parametrize(
    "expression,dependencies",
    [
        pytest.param("book / cap", {"book", "cap"}, marks=needs_fundamentals),
        ("volume / shares", {"volume", "shares"}),
        ("close * shares / cap", {"close", "shares", "cap"}),
        pytest.param("earnings / assets", {"earnings", "assets"}, marks=needs_fundamentals),
        ("log(close / shift(close, 1))", {"close"}),
    ],
)
def test_dimensionless_equity_ratios_have_closed_dependency_evidence(expression, dependencies):
    evidence = typed(expression)
    assert evidence.output_type.units == Units() and evidence.output_type.entity == "equity"
    assert {name for name, _ in evidence.dependencies} == dependencies
    assert evidence.schema_version == SCHEMA_VERSION and evidence.policy_version == POLICY_VERSION
    assert evidence.normalization_required is True
    assert "qualification" not in asdict(evidence)
    with pytest.raises(FrozenInstanceError):
        evidence.normalization_required = False


def test_beta_and_residual_broadcast_market_to_equity_without_changing_units():
    beta = "ts_cov(returns, mkt_return, 60) / ts_cov(mkt_return, mkt_return, 60)"
    evidence = typed(f"returns - ({beta}) * mkt_return")
    assert evidence.output_type.units == Units() and evidence.output_type.entity == "equity"
    assert [name for name, _ in evidence.dependencies] == ["mkt_return", "returns"]


@pytest.mark.parametrize(
    "expression",
    [
        "close + volume",
        pytest.param("book + close", marks=needs_fundamentals),
        "close + 1",
        "1 - close",
        "close - 0.001",
        "log(close)",
    ],
)
def test_unsupported_dimensions_are_type_blockers_without_rewriting_expression(expression):
    node = tree.parse(expression)
    before = tree.identity_key(node)
    with pytest.raises(UnitTypeError):
        infer_types(node)
    assert tree.identity_key(node) == before


@pytest.mark.parametrize("expression", ["close + 0", "0 - close", "close - -0", "close + low"])
def test_compatible_price_addition_and_only_literal_scalar_zero_are_supported(expression):
    assert typed(expression).output_type.units == Units(1, -1)


@pytest.mark.parametrize(
    "function",
    ["shift", "delta", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum", "decay_linear", "ema"],
)
def test_rolling_linear_operators_preserve_units(function):
    assert typed(f"{function}(close, 5)").output_type.units == Units(1, -1)


@pytest.mark.parametrize(
    "expression",
    [
        "rank(close)",
        "scale(close)",
        "sign(close)",
        "zscore(close, 5)",
        "ts_rank(volume, 5)",
        "ts_corr(close, volume, 5)",
    ],
)
def test_rank_scale_standardization_and_correlation_are_dimensionless(expression):
    assert typed(expression).output_type.units == Units()


def test_covariance_product_argextrema_and_signed_power_have_correct_dimensions():
    assert typed("ts_cov(close, volume, 5)").output_type.units == Units(usd=1)
    assert typed("ts_product(close, 5)").output_type.units == Units(usd=5, shares=-5)
    assert typed("ts_argmax(close, 5)").output_type.units == Units(sessions=1)
    assert typed("ts_argmin(close, 5)").output_type.units == Units(sessions=1)
    assert typed("signed_power(close, 2)").output_type.units == Units(usd=2, shares=-2)
    assert typed("signed_power(returns, 0.5)").output_type.units == Units()
    assert typed("abs(close)").output_type.units == Units(usd=1, shares=-1)
    assert typed("winsorize(close)").output_type.units == Units(usd=1, shares=-1)


@pytest.mark.parametrize(
    "expression",
    [
        "signed_power(close, 0.5)",
        f"signed_power(returns, {MAX_POWER + 1})",
        "ts_product(ts_product(close, 1000), 1000)",
    ],
)
def test_dimensional_or_excessive_exponents_remain_explicit_capability_blockers(expression):
    with pytest.raises(UnitTypeError, match="exponent|policy"):
        typed(expression)


@pytest.mark.parametrize("exponent", [float("nan"), float("inf"), -float("inf")])
def test_handcrafted_nonfinite_exponent_cannot_bypass_the_closed_numeric_boundary(exponent):
    with pytest.raises(UnitTypeError, match="finite"):
        infer_types(tree.Call("signed_power", tree.Field_("close"), arg2=tree.Const(exponent)))


@pytest.mark.parametrize(
    "expression", ["close * ts_mean(1, 5)", "close * rank(mkt_return)", "close * scale(mkt_return)"]
)
def test_entity_mismatch_and_unindexed_rolling_inputs_are_rejected(expression):
    with pytest.raises(UnitTypeError, match="indexed|equity entity"):
        typed(expression)


def test_legacy_parser_identity_remains_available_for_preexisting_untyped_expression():
    expression = "ts_mean(close + volume, 5)"
    node = tree.parse(expression)
    identity = tree.factor_id(node)
    assert tree.parse(tree.unparse(node)) == node
    with pytest.raises(UnitTypeError):
        infer_types(node)
    assert tree.factor_id(tree.parse(expression)) == identity


def test_future_grammar_addition_cannot_inherit_an_unreviewed_preserve_units_rule(monkeypatch):
    monkeypatch.setitem(tree.FUNCS, "future_function", False)
    with pytest.raises(UnitTypeError, match="no versioned unit rule"):
        infer_types(tree.Call("future_function", tree.Field_("close")))
