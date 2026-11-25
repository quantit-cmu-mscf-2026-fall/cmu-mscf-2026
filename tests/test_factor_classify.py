"""Factor classification: fields, horizon, construction, archetype, zoo nearness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from capstone.factors.classify import classify, classify_expressions, load_archetype_rules
from capstone.factors.tree import parse
from capstone.factors.zoo import ALPHA101_EXPRESSIONS

SUMMARY = Path("G:/My Drive/mscf-capstone-data/factors/crsp-v1_summary.parquet")


def c(text: str) -> dict:
    return classify(parse(text, regime_operators=True))


def test_turnover_liquidity_long_horizon():
    out = c("-(rank(ts_mean((volume / shares), 252)))")
    assert out["archetype"] == "liquidity"
    assert out["primary_field_group"] == "volume"
    assert out["horizon_bin"] == 3
    assert out["negated"] is True
    assert out["construction"] == ["rank", "ratio", "time_series"]


def test_momentum_12_1_lag_and_horizon():
    out = c("shift(ts_sum(returns, 231), 21)")
    assert out["archetype"] == "momentum"
    assert out["total_lag"] == 21
    assert out["max_window"] == 231
    assert out["effective_lookback"] == 252
    assert out["horizon_bin"] == 3


def test_short_reversal():
    out = c("-(rank(ts_sum(returns, 5)))")
    assert out["archetype"] == "reversal"
    assert out["horizon_bin"] == 0


def test_volatility():
    assert c("ts_std(returns, 60)")["archetype"] == "volatility"


def test_fundamental_and_value():
    assert c("rank(earnings / assets)")["archetype"] == "fundamental_quality"
    assert c("rank(book / cap)")["archetype"] == "value"


def test_size_and_primary_group_tie_break():
    assert c("-(log(cap))")["archetype"] == "size"
    assert c("returns * cap")["primary_field_group"] == "size"


def test_regime_flagged():
    assert c("volscale(ts_sum(returns, 100))")["uses_regime"] is True
    assert c("ts_sum(returns, 100)")["uses_regime"] is False


def test_zoo_member_nearest_itself():
    out = c(ALPHA101_EXPRESSIONS["alpha_101"])
    assert out["zoo_share"] == 1.0
    assert out["nearest_zoo"] == "alpha_101"


def test_deterministic_and_json_able():
    a, b = c("rank(ts_corr(close, volume, 10))"), c("rank(ts_corr(close, volume, 10))")
    assert a == b
    assert json.loads(json.dumps(a)) == a


def test_classify_expressions_frame():
    df = classify_expressions({"a": "ts_std(returns, 20)", "b": "book / cap"})
    assert list(df.index) == ["a", "b"]
    assert df.loc["a", "archetype"] == "volatility"


def test_unknown_key_raises(tmp_path):
    bad = tmp_path / "a.toml"
    bad.write_text('[[rule]]\narchetype = "x"\nbogus = 1\n')
    with pytest.raises(ValueError, match="unknown keys"):
        load_archetype_rules(bad)


def test_shipped_rules_load():
    assert load_archetype_rules()


def test_crsp_v1_expressions_classify(capsys):
    if not SUMMARY.exists():
        pytest.skip("crsp-v1 summary not available")
    summary = pd.read_parquet(SUMMARY).drop_duplicates("factor_id")
    df = classify_expressions(dict(zip(summary["factor_id"], summary["expression"], strict=True)))
    assert len(df) == len(summary)
    with capsys.disabled():
        print("\narchetype counts:", df["archetype"].value_counts().to_dict())
