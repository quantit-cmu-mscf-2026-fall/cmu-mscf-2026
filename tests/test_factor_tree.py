"""The factor grammar, its two canonical keys, and originality against Alpha101."""

from __future__ import annotations

import pytest

from capstone.factors.tree import (
    FIELDS,
    OHLCV,
    ParseError,
    factor_id,
    features_used,
    identity_key,
    node_count,
    param_count,
    parse,
    structural_key,
    subtree_similarity,
    unparse,
    zoo_similarity,
)
from capstone.factors.zoo import ALPHA101_EXPRESSIONS, STARTER_ZOO_EXPRESSIONS, default_zoo

# ---------------------------------------------------------------------------
# Grammar


@pytest.mark.parametrize(
    "expression",
    [
        "__import__(os)",
        "close.__class__",
        "close[0]",
        "ts_mean(close, 2.5)",
        "ts_mean(close, 0)",
        "ts_mean(close, 1e1)",
        "unknown(close)",
        "1 + 2",
        "",
        "ts_corr(close, volume.__class__, 5)",
        "ts_cov(close, volume, -2)",
        "ts_argmax(close, 5.5)",
        "decay_linear(close, 0)",
        "ts_product(close, 1261)",
    ],
)
def test_parser_rejects_unsafe_or_out_of_grammar_expressions(expression):
    with pytest.raises(ParseError):
        parse(expression)


def test_functions_are_not_fields():
    for name in ("sign", "scale", "ts_argmax", "ts_argmin", "decay_linear", "ts_product"):
        with pytest.raises(ParseError):
            parse(name)


def test_parameters_are_counted():
    node = parse("ts_mean(close, 5) / ts_std(close, 20) + 0.5")
    assert param_count(node) == 3
    assert features_used(node) == {"close"}


# ---------------------------------------------------------------------------
# Round trip


@pytest.mark.parametrize(
    "expression",
    [
        *ALPHA101_EXPRESSIONS.values(),
        "close * 0.00001",
        "signed_power(close - open, 0.5)",
        "-(-close)",
        "ts_corr(rank(high), rank(volume), 3) * 1e6",
    ],
)
def test_unparse_reads_back_as_the_same_factor(expression):
    node = parse(expression)
    assert parse(unparse(node)) == node
    assert factor_id(parse(unparse(node))) == factor_id(node)


# ---------------------------------------------------------------------------
# Identity vs shape


def test_operand_order_under_plus_and_times_is_the_same_factor():
    assert factor_id(parse("ts_mean(close + volume, 5)")) == factor_id(
        parse("ts_mean(volume + close, 5)")
    )
    assert factor_id(parse("rank(close) * volume")) == factor_id(parse("volume * rank(close)"))


def test_order_still_matters_for_minus_and_divide():
    assert factor_id(parse("close - open")) != factor_id(parse("open - close"))
    assert factor_id(parse("close / open")) != factor_id(parse("open / close"))


def test_windows_and_constants_change_identity_but_not_shape():
    short, long = parse("ts_mean(close, 5) + 1"), parse("ts_mean(close, 20) + 2")
    assert identity_key(short) != identity_key(long)
    assert factor_id(short) != factor_id(long)
    assert structural_key(short) == structural_key(long)


def test_factor_id_is_stable_hex():
    fid = factor_id(parse("close / shift(close, 20) - 1"))
    assert len(fid) == 16
    int(fid, 16)
    assert fid == factor_id(parse("close/shift(close,20)-1"))


# ---------------------------------------------------------------------------
# Originality


def test_similarity_ignores_parameters_and_commutative_order():
    first = parse("ts_mean(close + volume, 5)")
    second = parse("ts_mean(volume + close, 20)")
    assert subtree_similarity(first, second) == node_count(first)


def test_unrelated_trees_share_nothing_bigger_than_a_field():
    assert subtree_similarity(parse("ts_std(high, 10)"), parse("rank(volume)")) == 0
    assert subtree_similarity(parse("ts_std(close, 10)"), parse("rank(close)")) == 1


def test_zoo_is_alpha101_in_ohlcv_plus_returns_spellings():
    zoo = dict(default_zoo())
    assert set(ALPHA101_EXPRESSIONS) <= set(zoo)
    assert len(ALPHA101_EXPRESSIONS) >= 25
    for name in ALPHA101_EXPRESSIONS:
        assert features_used(zoo[name]) <= set(OHLCV), name
    respelled = [name for name in zoo if name.endswith("_returns")]
    assert "alpha_014_returns" in respelled
    for name in respelled:
        assert "returns" in features_used(zoo[name]), name


def test_new_fields_parse_and_turnover_is_expressible():
    node = parse("rank(ts_mean(volume / shares, 21)) * -ts_sum(returns, 5) / log(cap)")
    assert features_used(node) == {"volume", "shares", "returns", "cap"}
    assert set(FIELDS) == {*OHLCV, "returns", "shares", "cap"}


def test_a_published_alpha_is_not_original_with_the_returns_field_either():
    copied = parse("-1 * rank(delta(returns, 3)) * ts_corr(open, volume, 10)")
    _, share, nearest = zoo_similarity(copied, default_zoo())
    assert share == pytest.approx(1.0)
    assert nearest.startswith("alpha_014")


def test_a_published_alpha_is_not_original_even_with_new_windows():
    copied = parse(ALPHA101_EXPRESSIONS["alpha_101"])
    retuned = parse("(close - open) / ((high - low) + 0.01)")

    for node in (copied, retuned):
        _, share, nearest = zoo_similarity(node, default_zoo())
        assert share == pytest.approx(1.0)
        assert nearest == "alpha_101"

    starter = [(name, parse(expr)) for name, expr in STARTER_ZOO_EXPRESSIONS.items()]
    assert zoo_similarity(copied, starter)[1] < 0.5


def test_empty_zoo_finds_nothing():
    assert zoo_similarity(parse("rank(close)"), []) == (0, 0.0, "")


def test_windows_reach_five_trading_years():
    assert parse("close / shift(close, 1260) - 1")
    with pytest.raises(ParseError, match="outside"):
        parse("shift(close, 1261)")


def test_nested_shifts_are_the_same_factor_as_one_shift():
    nested = parse("shift(close, 252) / shift(shift(shift(close, 252), 252), 252)")
    single = parse("shift(close, 252) / shift(close, 756)")
    assert factor_id(nested) == factor_id(single)
    assert structural_key(nested) == structural_key(single)
    assert factor_id(single) != factor_id(parse("shift(close, 252) / shift(close, 504)"))


# ---------------------------------------------------------------------------
# Regressions found in review


def test_a_constant_that_overflows_is_rejected_not_stored_as_inf():
    with pytest.raises(ParseError, match="finite"):
        parse("ts_rank(volume, 20) * 1e999")
    with pytest.raises(ParseError, match="finite"):
        parse("signed_power(close, 1e999)")


def test_deep_nesting_is_a_parse_error_not_a_crash():
    with pytest.raises(ParseError):
        parse("-" * 3000 + "close")
    with pytest.raises(ParseError):
        parse("(" * 600 + "close" + ")" * 600)


def test_sums_and_products_of_three_terms_are_one_factor_in_any_order():
    assert factor_id(parse("close + volume + open")) == factor_id(parse("open + volume + close"))
    assert factor_id(parse("close * (volume * open)")) == factor_id(
        parse("(open * close) * volume")
    )
    assert factor_id(parse("close - volume - open")) != factor_id(parse("open - volume - close"))


def test_correlation_and_covariance_are_symmetric():
    assert factor_id(parse("ts_corr(close, volume, 10)")) == factor_id(
        parse("ts_corr(volume, close, 10)")
    )
    assert factor_id(parse("ts_cov(high, low, 5)")) == factor_id(parse("ts_cov(low, high, 5)"))


def test_a_folded_shift_is_entirely_similar_to_its_single_shift():
    nested, single = parse("shift(shift(close, 1), 1)"), parse("shift(close, 2)")
    assert subtree_similarity(nested, single) == node_count(nested)


def test_every_capstone_subpackage_is_listed_for_installation():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    listed = set(
        tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"]
    )
    found = {
        ".".join(init.parent.relative_to(root).parts)
        for init in (root / "capstone").rglob("__init__.py")
    }
    assert found <= listed, f"add to [tool.setuptools] packages: {sorted(found - listed)}"
