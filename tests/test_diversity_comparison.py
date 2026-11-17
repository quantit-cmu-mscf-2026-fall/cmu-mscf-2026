"""Comparing generated constructions: structure and target overlap, never return evidence."""

from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

from capstone.factors.diversity_comparison import (
    blind_packets,
    construction_inventory,
    frozen_constructions,
    specs,
    summary,
    target_relationships,
)
from capstone.factors.research_benchmark import METHOD_FIELDS, load_benchmark

BENCHMARK = Path(__file__).resolve().parents[1] / "config/research_benchmark.json"


def test_structural_similarity_is_not_counted_as_return_diversification():
    output = construction_inventory(
        {
            "A": {
                "first": {
                    "status": "prototype",
                    "proposals": [{"expression": "ts_sum(returns, 5)", "status": "stored"}],
                },
                "renamed": {
                    "status": "prototype",
                    "proposals": [{"expression": "ts_sum(returns, 20)", "status": "stored"}],
                },
            }
        }
    )
    assert output["structural_repetitions"]
    assert not output["financial_evidence"]
    assert all(not c["executable_verified"] for c in output["cases"])


def test_an_unparseable_formula_has_no_structure_and_a_spec_keeps_its_method_fields():
    method = {**dict.fromkeys(METHOD_FIELDS, "x"), "profile": "long_only"}
    output = construction_inventory(
        {
            "A": {
                "bad": {
                    "status": "prototype",
                    "proposed_spec": method,
                    "proposals": [{"expression": "not a formula (", "status": "rejected"}],
                },
                "none": {"status": "needs_data"},
            }
        }
    )
    by_packet = {c["packet"]: c for c in output["cases"]}
    assert by_packet["bad"]["formulas"][0]["structure"] is None
    assert by_packet["bad"]["method_fields"]["profile"] == "long_only"
    assert by_packet["none"]["method_fields"] is None
    assert not output["structural_repetitions"]


def test_blinded_labels_hide_the_arm_and_are_stable_for_a_packet_and_arm():
    data = load_benchmark(BENCHMARK)
    first = data["packets"][0]["id"]
    results = {"A": {first: {"status": "one"}}, "B": {first: {"status": "two"}}}
    packets, mapping = blind_packets(data, results)
    assert len(packets) == len(data["packets"])
    labelled = packets[0]["artifacts"]
    assert len(labelled) == 2 and not set(labelled) & {"A", "B"}
    assert {m["arm"] for m in mapping.values()} == {"A", "B"}
    assert blind_packets(data, results)[1] == mapping


def test_summary_counts_attempts_and_never_names_a_winner():
    results = {
        "A": {"p1": {"status": "prototype", "proposals": [{}, {}], "proposed_spec": {}}},
        "B": {"p1": {"status": "needs_data"}, "p2": {"status": "needs_data"}},
    }
    out = summary(results, {"outbound_reservations": 7})
    assert out["arms"]["A"]["formula_proposals"] == 2
    assert out["arms"]["A"]["documented_specs"] == 1
    assert out["arms"]["B"]["statuses"] == {"needs_data": 2}
    assert out["outbound_requests"] == 7
    assert not out["formal_admission"] and not out["autonomous_discovery_tested"]
    assert all(a["verified_executable_strategies"] == 0 for a in out["arms"].values())


def test_frozen_constructions_are_six_prespecified_adaptations_not_discoveries():
    chosen = specs("synthetic-v1")
    assert len(chosen) == 6
    assert {s.side for s in chosen} == {"long_only", "long_short"}
    rows = frozen_constructions("synthetic-v1")
    assert [r["id"] for r in rows] == [s.id for s in chosen]
    assert len({r["id"] for r in rows}) == 6
    assert all("not an agent discovery" in r["origin"] for r in rows)


def test_target_overlap_is_separate_from_returns_and_uses_only_available_books():
    dates = pd.bdate_range("2000-01-03", periods=3)
    members = [NS(id="a", method="A", side="long_only"), NS(id="b", method="B", side="long_only")]
    universe = pd.DataFrame(True, index=dates, columns=[1, 2])
    books = {
        "a": {dates[0]: pd.Series({1: 1.0}), dates[-1]: pd.Series({2: 1.0})},
        "b": {dates[0]: pd.Series({1: 1.0})},
    }
    result = target_relationships(members, books, universe, dates)
    assert result[0]["observations"][0]["jaccard"] == 1
    assert result[0]["observations"][-1]["jaccard"] == 0
    assert result[0]["layer"] == "construction" and "correlation" not in result[0]
    books["a"][dates[-1]] = pd.Series({1: 1.0})
    changed = target_relationships(members, books, universe, dates)
    assert changed[0]["observations"][:-1] == result[0]["observations"][:-1]


def test_names_outside_the_current_universe_do_not_count_as_overlap():
    dates = pd.bdate_range("2000-01-03", periods=1)
    members = [NS(id="a", method="A", side="long_short"), NS(id="b", method="B", side="long_short")]
    universe = pd.DataFrame({1: [True], 2: [False]}, index=dates)
    books = {"a": {dates[0]: pd.Series({1: 1.0, 2: -1.0})}, "b": {dates[0]: pd.Series({1: 1.0})}}
    row = target_relationships(members, books, universe, dates)[0]["observations"][0]
    assert row["jaccard"] == 1 and row["signed_cosine"] == 1
