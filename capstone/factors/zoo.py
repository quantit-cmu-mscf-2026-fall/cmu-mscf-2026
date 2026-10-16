"""The reference zoo for originality: Alpha101 in the factor grammar.

AlphaAgent measures a proposal's originality as the largest common subtree
it shares with any member of an existing alpha zoo, "such as Alpha101"
(Kakushadze 2016, arXiv:1601.00991). The zoo is therefore not decoration:
it is the entire empirical content of `S(f)`. A small zoo of obvious
shapes makes every proposal look original and turns the paper's central
regularizer into a constant.

This module carries the subset of Alpha101 expressible in this repo's
closed OHLCV grammar, transcribed member by member. Excluded, and why:

* anything using `vwap`, `cap`, or `IndClass.*` — the CRSP daily pull
  carries no volume-weighted average price, market capitalisation, or
  industry classification, and inventing a proxy would put a made-up
  series into the originality baseline;
* anything using a ternary or a comparison (`(x < 0) ? a : b`) — the
  grammar has no conditional, by design: every operator must be
  differentiable-in-spirit and cheap to canonicalise for Eq. 5.

Transcription notes. Alpha101's `delay` is `shift`, `correlation` is
`ts_corr`, `covariance` is `ts_cov`, `stddev` is `ts_std`, `sum` is
`ts_sum`, `Ts_Rank` is `ts_rank`, and `adv{d}` is `ts_mean(volume, d)`.
`returns` is written out as `close / shift(close, 1) - 1`, which keeps
the transcription over OHLCV and lets a proposal that rebuilds daily returns
match the zoo on that subtree. Since the grammar also has a `returns` field,
`default_zoo` holds every member that uses returns twice, once in each
spelling, so neither way of writing returns makes a copy look original.
Alpha101's `x^n` becomes
`signed_power(x, n)`; the two agree wherever the base is positive, which
is every price field.
"""

from __future__ import annotations

from capstone.factors.tree import Node, parse

RETURNS = "close / shift(close, 1) - 1"

# Kakushadze (2016), transcribed into the repo grammar. Keys keep the
# paper's numbering so a reader can check any line against the source.
ALPHA101_EXPRESSIONS: dict[str, str] = {
    "alpha_002": ("-1 * ts_corr(rank(delta(log(volume), 2)), rank((close - open) / open), 6)"),
    "alpha_003": "-1 * ts_corr(rank(open), rank(volume), 10)",
    "alpha_004": "-1 * ts_rank(rank(low), 9)",
    "alpha_006": "-1 * ts_corr(open, volume, 10)",
    "alpha_008": (
        f"-1 * rank(ts_sum(open, 5) * ts_sum({RETURNS}, 5) "
        f"- shift(ts_sum(open, 5) * ts_sum({RETURNS}, 5), 10))"
    ),
    "alpha_012": "sign(delta(volume, 1)) * (-1 * delta(close, 1))",
    "alpha_013": "-1 * rank(ts_cov(rank(close), rank(volume), 5))",
    "alpha_014": f"-1 * rank(delta({RETURNS}, 3)) * ts_corr(open, volume, 10)",
    "alpha_015": "-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3)",
    "alpha_016": "-1 * rank(ts_cov(rank(high), rank(volume), 5))",
    "alpha_017": (
        "-1 * rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) "
        "* rank(ts_rank(volume / ts_mean(volume, 20), 5))"
    ),
    "alpha_018": (
        "-1 * rank(ts_std(abs(close - open), 5) + (close - open) + ts_corr(close, open, 10))"
    ),
    "alpha_019": (
        "-1 * sign((close - shift(close, 7)) + delta(close, 7)) "
        f"* (1 + rank(1 + ts_sum({RETURNS}, 250)))"
    ),
    "alpha_020": (
        "-1 * rank(open - shift(high, 1)) * rank(open - shift(close, 1)) "
        "* rank(open - shift(low, 1))"
    ),
    "alpha_022": "-1 * delta(ts_corr(high, volume, 5), 5) * rank(ts_std(close, 20))",
    "alpha_026": "-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
    "alpha_028": ("scale(ts_corr(ts_mean(volume, 20), low, 5) + (high + low) / 2 - close)"),
    "alpha_033": "rank(-1 * (1 - open / close))",
    "alpha_034": (
        f"rank((1 - rank(ts_std({RETURNS}, 2) / ts_std({RETURNS}, 5))) "
        "+ (1 - rank(delta(close, 1))))"
    ),
    "alpha_035": (
        "ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) "
        f"* (1 - ts_rank({RETURNS}, 32))"
    ),
    "alpha_038": "-1 * rank(ts_rank(close, 10)) * rank(close / open)",
    "alpha_039": (
        "-1 * rank(delta(close, 7) "
        "* (1 - rank(decay_linear(volume / ts_mean(volume, 20), 9)))) "
        f"* (1 + rank(ts_sum({RETURNS}, 250)))"
    ),
    "alpha_040": "-1 * rank(ts_std(high, 10)) * ts_corr(high, volume, 10)",
    "alpha_043": ("ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * delta(close, 7), 8)"),
    "alpha_044": "-1 * ts_corr(high, rank(volume), 5)",
    "alpha_045": (
        "-1 * rank(ts_sum(shift(close, 5), 20) / 20) * ts_corr(close, volume, 2) "
        "* rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))"
    ),
    "alpha_053": "-1 * delta(((close - low) - (high - close)) / (close - low), 9)",
    "alpha_054": (
        "-1 * ((low - close) * signed_power(open, 5)) / ((low - high) * signed_power(close, 5))"
    ),
    "alpha_055": (
        "-1 * ts_corr(rank((close - ts_min(low, 12)) "
        "/ (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)"
    ),
    "alpha_101": "(close - open) / ((high - low) + 0.001)",
}

# The pre-Alpha101 hand-written set. Kept because several tests want a
# small zoo whose contents they control; it is no longer the default.
STARTER_ZOO_EXPRESSIONS: dict[str, str] = {
    "zoo_momentum_20": "close / shift(close, 20) - 1",
    "zoo_reversal_5": "-(close / shift(close, 5) - 1)",
    "zoo_volatility_20": "ts_std(close / shift(close, 1) - 1, 20)",
    "zoo_volume_ratio_20": "volume / ts_mean(volume, 20)",
    "zoo_daily_range": "(high - low) / close",
    "zoo_intraday_return": "close / open - 1",
    "zoo_overnight_gap": "open / shift(close, 1) - 1",
    "zoo_volume_change": "volume / shift(volume, 5) - 1",
    "zoo_close_location": "(close - low) / (high - low)",
    "zoo_range_volatility": "ts_std((high - low) / close, 20)",
    "zoo_price_trend_gap": "close / ts_mean(close, 20) - 1",
    "zoo_volume_volatility": "ts_std(volume, 20) / ts_mean(volume, 20)",
}


def default_zoo() -> list[tuple[str, Node]]:
    """Alpha101 members as (name, tree), the reference set for originality.

    A member written with daily returns appears a second time, as
    `<name>_returns`, with the `returns` field in place of the written-out
    `close / shift(close, 1) - 1`.
    """
    zoo = [(name, parse(expression)) for name, expression in ALPHA101_EXPRESSIONS.items()]
    zoo += [
        (f"{name}_returns", parse(expression.replace(RETURNS, "returns")))
        for name, expression in ALPHA101_EXPRESSIONS.items()
        if RETURNS in expression
    ]
    return zoo
