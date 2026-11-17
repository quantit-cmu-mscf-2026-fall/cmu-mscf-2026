from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest

from capstone.factors.research_relations import (
    attach_constructions,
    research_graph,
    return_relation,
    structural_relation,
    write_relationship_report,
)


def relation(left, right, **kwargs):
    return return_relation(
        left,
        right,
        as_of="2020-12-31",
        basis="gross_fractional_illustration",
        costs="zero; diagnostic only",
        data_version="fixture-v1",
        **kwargs,
    )


def test_future_observations_and_row_order_cannot_change_correlation():
    dates = pd.bdate_range("2019-01-01", periods=800)
    rng = np.random.default_rng(404)
    one = pd.Series(rng.normal(size=len(dates)), index=dates)
    two = one * 0.7 + pd.Series(rng.normal(size=len(dates)), index=dates)
    original = relation(one, two)
    one.loc[one.index > "2020-12-31"] *= 1000
    assert relation(one.sample(frac=1, random_state=3), two.iloc[::-1]) == original
    assert original["sample_end"] == "2020-12-31"


def test_unknown_is_not_zero_and_cash_gap_views_are_separate():
    dates = pd.bdate_range("2019-01-01", periods=300)
    one = pd.Series(np.arange(300) % 17, index=dates, dtype=float)
    assert relation(one.iloc[:251], one)["correlation"] is None
    assert relation(one * 0, one)["status"] == "unknown"
    invested = pd.Series(True, index=dates)
    invested.iloc[:80] = False
    assert relation(one, one)["same_positive_group"]
    sensitivity = relation(one, one, invested=invested)
    assert sensitivity["status"] == "unknown" and sensitivity["observations"] == 220
    assert sensitivity["cash_gap_sensitivity"] == "both_invested"
    with pytest.raises(ValueError, match="unique"):
        relation(pd.concat([one, one]), one)


def test_renamed_window_is_structural_similarity_not_proven_return_similarity():
    result = structural_relation("ts_mean(returns, 5)", "ts_mean(returns, 21)")
    assert result["same_structure"] and result["empirical_relation"] == "unknown"


def test_graph_preserves_layers_and_emits_safe_readable_svg(tmp_path):
    def entry(key):
        return {
            "title": "<script>alert(1)</script>",
            "brief": {"packet_id": key, "family": "liquidity_supply", "method_key": key},
            "claim": {
                "source_url": "https://example.org/primary",
                "full_methods": True,
                "method_location": "synthetic fixture",
                "required_inputs": ["dated_flows"],
            },
        }

    entries = [entry("one"), entry("two")]
    graph = research_graph(entries, {"one": "needs_data"})
    assert graph == research_graph(list(reversed(entries)), {"one": "needs_data"})
    results = {
        "A": {
            "one": {"proposals": [{"expression": "ts_mean(returns, 5)"}]},
            "two": {"proposals": [{"expression": "ts_mean(returns, 21)"}]},
        }
    }
    graph = attach_constructions(
        graph, results, {"packets": [{"id": key, "method_key": key} for key in ("one", "two")]}
    )
    assert any(e["type"] == "structurally_similar" for e in graph["edges"])
    assert not any(e["layer"] == "empirical" for e in graph["edges"])
    path = tmp_path / "relationships.html"
    write_relationship_report(path, graph, {})
    assert "<script>" not in path.read_text(encoding="utf-8")
    assert "needs_data" in path.read_text(encoding="utf-8")
    ElementTree.fromstring(path.with_suffix(".svg").read_text(encoding="utf-8"))
