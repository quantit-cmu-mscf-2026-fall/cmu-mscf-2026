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


def test_scenario_blockers_join_the_rows_blockers():
    text = markdown(document())
    assert "| study | evaluate | study-1 | base |" in text
    assert "thin_sample, wide_band" in text


def test_output_does_not_depend_on_insertion_order():
    forward = document()
    backward = copy.deepcopy(forward)
    event = {"session": "d", "e_value": 1, "n_obs": 2, "decision": "hold"}
    forward["empirical"][0]["strategies"] = {"a": event, "b": event}
    backward["empirical"][0]["strategies"] = {"b": event, "a": event}
    forward["empirical"][1]["scenarios"] = {"x": {"k": 1}, "y": {"k": 2}}
    backward["empirical"][1]["scenarios"] = {"y": {"k": 2}, "x": {"k": 1}}
    assert markdown(forward) == markdown(backward)


def test_blocked_and_empty_monitor_rows_keep_their_blockers():
    doc = document()
    event = {"session": "d", "e_value": 1, "n_obs": 2, "decision": "hold"}
    doc["empirical"] = [
        {"kind": "monitor", "stage": "m1", "strategies": {}, "blockers": ["no_sessions"]},
        {"kind": "monitor", "stage": "m2", "strategies": {"s": event}, "blockers": ["stale"]},
    ]
    text = markdown(doc)
    assert "| monitor | m1 | - | - | - | no_sessions |" in text
    assert "| monitor | m2 | s | d | e_value 1, n_obs 2, hold | stale |" in text


def test_a_row_with_no_scenarios_keeps_its_status_and_blockers():
    doc = document()
    doc["empirical"] = [
        {
            "kind": "candidates",
            "stage": "evaluate",
            "id": "s-1",
            "status": "rejected",
            "blockers": ["thin_sample"],
            "scenarios": {},
        }
    ]
    text = markdown(doc)
    assert '| candidates | evaluate | s-1 | - | {"status": "rejected"} | thin_sample |' in text


def test_scenario_rows_show_status_and_tolerate_missing_blockers():
    doc = document()
    doc["empirical"] = [
        {"kind": "study", "stage": "e", "id": "i", "status": "ok", "scenarios": {"b": {"k": 1}}}
    ]
    assert '| study | e | i | b | {"k": 1, "status": "ok"} | - |' in markdown(doc)


def test_none_fields_render_as_dashes():
    doc = document()
    doc["lineage"]["chains"][0]["program_id"] = None
    doc["conceptual"][0]["claims"][0]["claim"]["horizon"] = None
    text = markdown(doc)
    assert "None" not in text
