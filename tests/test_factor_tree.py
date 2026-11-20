"""The factor grammar, its two canonical keys, and originality against Alpha101."""

from __future__ import annotations

import pytest

from capstone.factors.tree import (
    FIELDS,
    FUNDAMENTAL_FIELDS,
    OHLCV,
    BinOp,
    Const,
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


ZOO_MEMBERS = {
    2, 3, 4, 6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 26, 28, 29, 30, 31,
    33, 34, 35, 37, 38, 39, 40, 43, 44, 45, 52, 53, 54, 55, 56, 60, 101,
}  # fmt: skip


def test_zoo_is_alpha101_in_the_grammar_plus_returns_spellings():
    zoo = dict(default_zoo())
    assert set(ALPHA101_EXPRESSIONS) == {f"alpha_{n:03d}" for n in ZOO_MEMBERS}
    assert set(ALPHA101_EXPRESSIONS) <= set(zoo)
    for name in ALPHA101_EXPRESSIONS:
        assert features_used(zoo[name]) <= set(FIELDS), name
    respelled = [name for name in zoo if name.endswith("_returns")]
    assert "alpha_014_returns" in respelled
    for name in respelled:
        assert "returns" in features_used(zoo[name]), name


@pytest.mark.parametrize(
    "verbatim",
    [
        # Alpha#56 as printed in Kakushadze (2016), with the grammar's `returns` and `cap`.
        "0 - (1 * (rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3))"
        " * rank(returns * cap)))",
        "rank(ts_corr(shift(open - close, 1), close, 200)) + rank(open - close)",
        "(1 - rank(sign(close - shift(close, 1)) + sign(shift(close, 1) - shift(close, 2))"
        " + sign(shift(close, 2) - shift(close, 3)))) * ts_sum(volume, 5) / ts_sum(volume, 20)",
        "(-1 * ts_min(low, 5) + shift(ts_min(low, 5), 5))"
        " * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5)",
        "0 - (1 * ((2 * scale(rank((((close - low) - (high - close)) / (high - low)) * volume)))"
        " - scale(rank(ts_argmax(close, 10)))))",
    ],
)
def test_a_verbatim_alpha101_member_is_not_original(verbatim):
    assert zoo_similarity(parse(verbatim), default_zoo())[1] == 1.0


def test_new_fields_parse_and_turnover_is_expressible():
    node = parse("rank(ts_mean(volume / shares, 21)) * -ts_sum(returns, 5) / log(cap)")
    assert features_used(node) == {"volume", "shares", "returns", "cap"}
    assert set(FIELDS) == {*OHLCV, "returns", "shares", "cap", "mkt_return", *FUNDAMENTAL_FIELDS}


def test_market_return_writes_beta_and_idiosyncratic_volatility():
    # QUANTIT-82: Ang et al. (2006)'s idiosyncratic volatility, the residual of
    # returns on the market, is now expressible without a new operator, and
    # stays inside the proposer's node limit.
    beta = "ts_cov(returns, mkt_return, 63) / ts_cov(mkt_return, mkt_return, 63)"
    idio = parse(f"-ts_std(returns - {beta} * mkt_return, 21)")
    assert features_used(idio) == {"returns", "mkt_return"}
    assert node_count(idio) <= 30
    assert "mkt_return" in features_used(parse("ts_std(mkt_return, 21) * rank(cap)"))


def test_a_factor_from_market_fields_alone_is_rejected():
    # Same value for every stock on a date: it cannot rank stocks.
    with pytest.raises(ParseError, match="no stock field"):
        parse("ts_std(mkt_return, 21)")


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


@pytest.mark.parametrize("n", [500, 1000, 2000])
@pytest.mark.parametrize("op", ["+", "*", "-"])
def test_a_long_flat_expression_is_a_parse_error_not_a_recursion_error(n, op):
    # A left-leaning chain is not nested in the source, but the tree is n deep.
    with pytest.raises(ParseError):
        parse(f"close{op}" * n + "close")


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


# ---------------------------------------------------------------------------
# Frequent subtree avoidance (Alpha Jungle Sec. 3)


def test_root_genes_have_only_field_leaves_and_ignore_windows():
    from capstone.factors.tree import gene_text, root_genes

    genes = root_genes(parse("rank(ts_mean(close, 5) / volume) * signed_power(returns, 2)"))
    # signed_power(returns, 2) has a constant leaf; bare fields are not genes.
    assert genes == root_genes(parse("rank(ts_mean(close, 20) / volume)"))
    assert "(signed_power F:returns C)" not in genes and "F:close" not in genes
    assert root_genes(parse("ts_mean(close, 5)")) == root_genes(parse("ts_mean(close, 63)"))
    assert gene_text(parse("ts_corr(close, volume, 21)")) == "ts_corr(close, volume, t)"


def test_frequent_genes_are_closed_counted_once_per_factor_and_capped_at_k():
    from capstone.factors.tree import frequent_root_genes

    trees = [
        parse("rank(close / shift(close, 5))"),
        parse("rank(close / shift(close, 20))"),
        parse("close / shift(close, 10) * volume"),
        parse("ts_std(returns, 21) + ts_std(returns, 63)"),
    ]
    found = frequent_root_genes(trees, k=3)
    texts = [text for _, _, text in found]
    # shift(close, t) is in three factors, but so is its parent close / shift(close, t):
    # only the larger, closed structure is named.
    assert texts[0] == "(close / shift(close, t))" and found[0][1] == 0.75
    assert "shift(close, t)" not in texts
    assert texts[1] == "rank((close / shift(close, t)))"
    # ts_std(returns, t) twice in one factor still counts once, and once is not frequent.
    assert all("ts_std" not in t for t in texts)
    assert len(frequent_root_genes(trees, k=1)) == 1
    assert frequent_root_genes([], k=3) == []


NEGATED_CORR = [
    "ts_corr(open, volume, 10) * -1",
    "-1 * ts_corr(open, volume, 10)",
    "-(ts_corr(open, volume, 10))",
    "0 - ts_corr(open, volume, 10)",
    "-(-(-(ts_corr(open, volume, 10))))",
    "(-1 * -1) * -(ts_corr(volume, open, 10))",
]


def test_every_spelling_of_a_negation_is_one_factor():
    ids = {factor_id(parse(e)) for e in NEGATED_CORR}
    assert len(ids) == 1
    assert factor_id(parse("-ts_corr(open, volume, 10)")) in ids


def test_a_double_negation_cancels_and_is_not_the_negation():
    plain = factor_id(parse("ts_corr(open, volume, 10)"))
    assert factor_id(parse("-(-(ts_corr(open, volume, 10)))")) == plain
    assert factor_id(parse("-1 * -1 * ts_corr(open, volume, 10)")) == plain
    assert plain != factor_id(parse("-ts_corr(open, volume, 10)"))


def test_a_negated_product_keeps_its_other_factors_and_constants():
    assert factor_id(parse("-(close * volume)")) == factor_id(parse("close * -volume"))
    assert factor_id(parse("-2 * close")) == factor_id(parse("close * -2"))
    assert factor_id(parse("-2 * close")) != factor_id(parse("-3 * close"))
    assert factor_id(parse("-2 * close")) != factor_id(parse("2 * close"))
    assert factor_id(parse("-close")) != factor_id(parse("close"))


def test_a_negative_constant_built_in_code_keys_like_the_parsed_expression():
    built = BinOp("*", Const(-1.0), parse("close"))
    assert identity_key(built) == identity_key(parse(unparse(built)))
    assert factor_id(built) == factor_id(parse("-close"))


@pytest.mark.parametrize("expression", NEGATED_CORR)
def test_a_negated_copy_of_a_zoo_member_is_not_partly_original(expression):
    zoo = default_zoo()
    _, share, closest = zoo_similarity(parse(expression), zoo)
    assert share == 1.0
    assert closest == "alpha_006"


def test_negative_constants_built_in_code_round_trip_to_the_same_factor():
    from capstone.factors.tree import Call, Unary

    close = parse("close")
    built = [
        BinOp("*", Const(-1.0), close),
        BinOp("*", close, Const(-2.5)),
        BinOp("+", close, Const(-1.0)),
        BinOp("*", Const(-0.0), close),
        Call("signed_power", close, None, Const(-2.0)),
    ]
    for node in built:
        again = parse(unparse(node))
        assert identity_key(again) == identity_key(node)
        assert factor_id(again) == factor_id(node)
    assert factor_id(built[0]) == factor_id(BinOp("*", Unary("neg", Const(1.0)), close))
    assert factor_id(built[3]) == factor_id(parse("-0.0 * close"))


def test_a_negative_constant_does_not_change_zoo_similarity():
    zoo = default_zoo()
    built = BinOp("*", parse("ts_corr(open, volume, 10)"), Const(-1.0))
    assert zoo_similarity(built, zoo) == zoo_similarity(parse(unparse(built)), zoo)
    assert zoo_similarity(built, zoo)[2] == "alpha_006"


SCALED_CORR = [
    "2 * ts_corr(open, volume, 10)",
    "ts_corr(open, volume, 10) * 0.5",
    "ts_corr(open, volume, 10) / 3",
    "-1 * ts_corr(open, volume, 10)",
    "-2 * ts_corr(open, volume, 10)",
    "ts_corr(open, volume, 10) * -2",
    "ts_corr(open, volume, 10) / -3",
    "-(ts_corr(open, volume, 10)) / 4",
    "-2 * ts_corr(open, volume, 10) / 5",
]


@pytest.mark.parametrize("expression", SCALED_CORR)
def test_a_rescaled_copy_of_a_zoo_member_is_not_partly_original(expression):
    _, share, closest = zoo_similarity(parse(expression), default_zoo())
    assert share == 1.0
    assert closest == "alpha_006"


def test_rescaling_keeps_the_sign_in_the_shape_key():
    base = structural_key(parse("ts_corr(open, volume, 10)"))
    assert structural_key(parse("3 * ts_corr(open, volume, 10)")) == base
    assert structural_key(parse("ts_corr(open, volume, 10) / 3")) == base
    negated = structural_key(parse("-ts_corr(open, volume, 10)"))
    assert negated != base
    for e in ("-2 * ts_corr(open, volume, 10)", "ts_corr(open, volume, 10) / -3"):
        assert structural_key(parse(e)) == negated
    assert structural_key(parse("-2 * ts_corr(open, volume, 10) / -3")) == base


def test_rescaling_changes_the_id_but_not_the_shape():
    assert factor_id(parse("2 * close")) != factor_id(parse("3 * close"))
    assert structural_key(parse("2 * close")) == structural_key(parse("3 * close"))
    assert factor_id(parse("close / 2")) != factor_id(parse("close"))
    # a constant inside a sum is not a rescaling
    assert structural_key(parse("close + 2")) != structural_key(parse("close"))
    assert structural_key(parse("2 / close")) != structural_key(parse("close"))


def test_subtracting_is_adding_a_negation():
    assert factor_id(parse("close - open")) == factor_id(parse("close + -open"))
    assert factor_id(parse("close - open")) == factor_id(parse("-open + close"))
    assert factor_id(parse("close - open")) != factor_id(parse("open - close"))
    assert factor_id(parse("close - (open + volume)")) == factor_id(parse("close - open - volume"))
    assert factor_id(parse("close - 0 - open")) != factor_id(parse("close - open"))
    assert factor_id(parse("close + 0 - open")) != factor_id(parse("close - open"))


@pytest.mark.parametrize(
    ("spelling", "gene"),
    [
        ("shift({}, {})", "shift(returns, t)"),
        ("-(-({}))", "(close / open)"),
        ("-(-(close) * volume)", None),
    ],
)
def test_a_gene_that_canonicalises_into_its_own_parent_is_still_frequent(spelling, gene):
    # Review of #61: `shift(shift(x, a), b)` and `-(-K)` share a key with their own
    # child, so the gene was its own closer and the most frequent structure dropped out.
    from capstone.factors.tree import frequent_root_genes

    if spelling.startswith("-(-(close) *"):
        # Each spelling of the product is the other's parent in one tree.
        trees = [
            parse("-(-(close) * volume) + returns"),
            parse("-(close * volume) + high"),
            parse("-(-(close) * volume) / open"),
        ]
        supports = [s for _, s, text in frequent_root_genes(trees, k=5) if "volume" in text]
        assert supports == [1.0]
        return
    if spelling.startswith("shift"):
        trees = [
            parse("shift(shift(returns, 1), 5) * volume"),
            parse("shift(returns, 3) / close"),
            parse("shift(returns, 10) * volume"),
        ]
    else:
        trees = [
            parse("-(-(close / open)) * volume"),
            parse("-(-(close / open)) + returns"),
            parse("-(-(close / open)) - high"),
        ]
    found = {text: support for _, support, text in frequent_root_genes(trees, k=5)}
    assert found.get(gene) == 1.0
