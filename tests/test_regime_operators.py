"""The `volscale` and `bear_gate` grammar operators, off by default.

- off (the default), the grammar and the prompt text are what they were: the operators
  are unknown identifiers;
- on (`parse(..., regime_operators=True)`), they parse, round-trip and evaluate
  from `mkt_return` through t-1 only, NaN in warm-up;
- a regime variant is counted as one trial in its base formula's family, and names
  its base candidate for stage 3's spanning set (fake proposals, no model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone import regimes
from capstone.factors import interpret, tree
from capstone.factors.interpret import VOLSCALE_TARGET, evaluate
from capstone.factors.llm import grammar_help
from capstone.factors.regime_variants import regime_bases, regime_families, trials_per_family
from capstone.factors.tree import ParseError, parse, regime_base, unparse, uses_regime

# --- off by default -------------------------------------------------------------------


@pytest.mark.parametrize("expression", ["volscale(close)", "bear_gate(rank(returns))"])
def test_off_the_operators_are_unknown(expression):
    with pytest.raises(ParseError, match="unknown identifier"):
        parse(expression)
    assert uses_regime(parse(expression, regime_operators=True))


def test_off_the_grammar_tables_and_prompt_are_unchanged():
    assert not set(tree.REGIME_FUNCS) & set(tree.FUNCS)
    text = grammar_help()
    assert text == grammar_help(regime_operators=False)
    assert "volscale" not in text and "bear_gate" not in text
    on = grammar_help(regime_operators=True)
    assert on.startswith(text) and "volscale" in on and "bear_gate" in on


# --- evaluation -----------------------------------------------------------------------


def _panel(n=900, stocks=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-01", periods=n)
    cols = pd.Index(range(stocks), name="permno")
    returns = pd.DataFrame(rng.normal(0, 0.02, (n, stocks)), index=idx, columns=cols)
    vol = np.where(np.arange(n) < n // 2, 0.008, 0.03)
    drift = np.where(np.arange(n) < n // 2, 0.0008, -0.002)
    mkt = pd.Series(drift + vol * rng.standard_normal(n), index=idx, name="mkt_return")
    return {"returns": returns, "close": (1 + returns).cumprod(), "mkt_return": mkt}


def test_volscale_and_bear_gate_values():
    panel = _panel()
    x = panel["returns"]
    rv = regimes.regime_series(panel["mkt_return"])["rv126"]
    got = evaluate(parse("volscale(returns)", regime_operators=True), panel)
    pd.testing.assert_frame_equal(got, x.mul(VOLSCALE_TARGET / rv, axis=0))
    assert got.iloc[: regimes.RV_WINDOW].isna().all().all()  # warm-up, never 0

    gated = evaluate(parse("bear_gate(returns)", regime_operators=True), panel)
    down = regimes.regime_series(panel["mkt_return"])["bear"]
    assert (down == 1).any() and (down == 0).any()
    assert (gated[down == 1] == 0).all().all()
    pd.testing.assert_frame_equal(gated[down == 0], x[down == 0])
    assert gated[down.isna()].isna().all().all()


@pytest.mark.parametrize("expression", ["volscale(returns)", "bear_gate(rank(returns))"])
def test_no_look_ahead_in_the_market(expression):
    """Perturbing the market from day t on leaves every value up to t unchanged."""
    panel = _panel()
    node = parse(expression, regime_operators=True)
    base = evaluate(node, panel)
    rng = np.random.default_rng(1)
    for t in (600, 700, 850):
        moved = dict(panel)
        mkt = panel["mkt_return"].copy()
        mkt.iloc[t:] = rng.normal(0, 0.1, len(mkt) - t)
        moved["mkt_return"] = mkt
        pd.testing.assert_frame_equal(evaluate(node, moved).iloc[:t], base.iloc[:t])


def test_needs_the_market_field():
    panel = _panel()
    del panel["mkt_return"]
    with pytest.raises(interpret.EvaluationError, match="mkt_return"):
        evaluate(parse("volscale(returns)", regime_operators=True), panel)


def test_round_trip_and_keys():
    node = parse("volscale(ts_mean(returns, 21)) + bear_gate(rank(close))", regime_operators=True)
    assert parse(unparse(node), regime_operators=True) == node
    base = regime_base(node)
    assert not uses_regime(base)
    assert unparse(base) == unparse(parse("ts_mean(returns, 21) + rank(close)"))
    assert tree.factor_id(node) != tree.factor_id(base)


# --- trial counting and family (fake proposals, no model) -----------------------------

# What a fake model proposed: candidate -> (hypothesis, expression).
PROPOSED = {
    "c1": ("h_mom", "ts_mean(returns, 126)"),
    "c2": ("h_mom", "volscale(ts_mean(returns, 126))"),
    "c3": ("h_vol", "bear_gate(ts_mean(returns, 126))"),  # another hypothesis wrapped c1
    "c4": ("h_vol", "ts_std(returns, 63)"),
    "c5": ("h_vol", "volscale(ts_std(close, 21))"),  # its base is no candidate
}


def _fake():
    expressions = pd.Series({c: e for c, (_, e) in PROPOSED.items()})
    family = pd.Series({c: h for c, (h, _) in PROPOSED.items()})
    return expressions, family


def test_family_assignment_and_trial_count():
    expressions, family = _fake()
    assert regime_bases(expressions) == {"c2": "c1", "c3": "c1", "c5": None}
    moved = regime_families(expressions, family)
    assert moved.to_dict() == {
        "c1": "h_mom",
        "c2": "h_mom",
        "c3": "h_mom",  # counted in its base formula's family
        "c4": "h_vol",
        "c5": "h_vol",  # no base candidate: its own
    }
    counts = trials_per_family(expressions, family)
    assert counts == {"h_mom": 3, "h_vol": 2}
    assert sum(counts.values()) == len(PROPOSED)  # every variant is one trial
    # Without regime variants nothing moves.
    plain = expressions[["c1", "c4"]]
    pd.testing.assert_series_equal(
        regime_families(plain, family[["c1", "c4"]]), family[["c1", "c4"]]
    )
