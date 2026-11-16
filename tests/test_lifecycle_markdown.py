"""LIFECYCLE.md renders the same content as LIFECYCLE.json for a synthetic report."""

from __future__ import annotations

import copy

from capstone.pipeline.lifecycle_markdown import markdown

SHA = "a" * 64


def document():
    return {
        "schema": "lifecycle-report-v1",
        "run": "run-1",
        "stages": [
            {
                "stage": "research",
                "artifact": "research/manifest.json",
                "sha256": SHA,
                "blockers": ["missing_input", "data_stale"],
                "cleared_by": ["provide the input", "refresh the data"],
            },
            {"stage": "empty", "artifact": "", "sha256": "", "blockers": [], "cleared_by": []},
        ],
        "recommendations": [
            {
                "recommendation": "retirement recommended",
                "strategy_id": "s-1",
                "action": "lifecycle retire --id s-1 --reason <text>",
            }
        ],
        "conceptual": [
            {
                "stage": "research",
                "status": "ok",
                "blockers": ["claim_pending"],
                "claims": [
                    {
                        "packet_id": "idea-0",
                        "status": "accepted",
                        "claim_id": "c" * 64,
                        "claim": {"statement": "turnover | reverts", "horizon": ["1m"]},
                        "equation": "r = a + b x",
                        "declared_defaults": [["window", 20]],
                        "blockers": [],
                    }
                ],
            }
        ],
        "lineage": {
            "chains": [
                {
                    "source_revisions": ["b" * 64],
                    "claim_id": "c" * 64,
                    "method_record_id": "d" * 64,
                    "program_id": "program-1",
                    "portfolios": ["e" * 64],
                    "plans": ["plan-1"],
                    "journals": [],
                    "monitors": ["monitor-1"],
                }
            ]
        },
        "empirical": [
            {
                "kind": "monitor",
                "stage": "monitor-1",
                "strategies": {
                    "s-1": {
                        "session": "2020-01-02",
                        "e_value": 1.5,
                        "n_obs": 40,
                        "decision": "hold",
                    }
                },
            },
            {
                "kind": "study",
                "stage": "evaluate",
                "id": "study-1",
                "blockers": ["thin_sample"],
                "scenarios": {"base": {"sharpe": 0.5, "blockers": ["wide_band"]}},
            },
            {"kind": "plan", "stage": "plan-1", "orders": 3},
        ],
    }


def test_every_section_is_rendered_from_the_document():
    text = markdown(document())
    assert text.startswith("# Lifecycle: run-1\n")
    for heading in (
        "## Stages",
        "## Recommendations",
        "## Conceptual",
        "## Lineage",
        "## Empirical",
    ):
        assert heading in text
    assert "`missing_input` -> provide the input; `data_stale` -> refresh the data" in text
    assert "- retirement recommended: s-1 -> `lifecycle retire --id s-1 --reason <text>`" in text
    assert "### research (ok)" in text and "- equation: `r = a + b x`" in text
    assert "e_value 1.5, n_obs 40, hold" in text
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_ids_are_shortened_to_twelve_characters_and_empty_cells_are_dashes():
    text = markdown(document())
    assert "a" * 12 in text and "a" * 13 not in text
    assert "| empty | - | - | - |" in text  # no artifact, no hash, nothing to clear


def test_a_pipe_in_a_cell_cannot_break_the_table_and_no_recommendation_says_none():
    base = document()
    assert "turnover \\| reverts" in markdown(base)
    quiet = copy.deepcopy(base)
    quiet["recommendations"] = []
    assert "- none" in markdown(quiet)


def test_scenario_blockers_join_the_rows_blockers_and_the_rendering_is_deterministic():
    text = markdown(document())
    assert "| study | evaluate | study-1 | base |" in text
    assert "thin_sample, wide_band" in text
    assert markdown(document()) == text
