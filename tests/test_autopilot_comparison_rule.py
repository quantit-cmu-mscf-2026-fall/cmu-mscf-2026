"""The frozen comparison rule is a pure function of the section and the study's report."""

import pytest

from capstone.pipeline.autopilot import ap5_rules


def rows():
    def row(id_, profile, mode, kind="full_pool", status="evaluated"):
        return {"id": id_, "profile": profile, "mode": mode, "kind": kind, "status": status}

    return {
        "comparisons": [
            row("a1", "long_only", "equal_weight", status="blocked"),
            row("a2", "long_only", "shrunk_max_sharpe"),
            row("a3", "long_only", "equal_weight", kind="zero_cash"),
            row("b1", "long_short", "equal_weight"),
            row("b2", "long_short", "shrunk_max_sharpe"),
        ]
    }


def test_comparison_rule_is_a_pure_function_of_the_section_and_the_report(tmp_path):
    for id_ in ("a1", "a2", "b1", "b2"):
        (tmp_path / "comparisons" / id_).mkdir(parents=True)
        (tmp_path / "comparisons" / id_ / ap5_rules.SPEC).write_text("x", encoding="utf-8")
    report = rows()

    def chosen(rule, comparison):
        row, why = ap5_rules.choose(rule, comparison, report, tmp_path)
        return (row or {}).get("id"), why

    assert chosen("named", "b2") == ("b2", None)  # an id
    assert chosen("named", "long-short:Equal-Weight") == ("b1", None)  # profile:mode
    assert chosen("named", "shrunk_max_sharpe")[1].endswith("ambiguous: a2, b2")
    blocked = "comparison a1 is 'blocked', not evaluated"
    assert chosen("named", "long_only:equal_weight")[1] == blocked
    assert chosen("named", "zero_cash")[1].endswith("matches no registered comparison")
    assert chosen("first_full_pool", None) == ("a2", None)  # registered order, evaluated
    assert chosen("first_full_pool", "long_short") == ("b1", None)
    (tmp_path / "comparisons" / "b1" / ap5_rules.SPEC).unlink()
    unsaved = "comparison b1 has no saved portfolio-spec.toml"
    assert chosen("first_full_pool", "long_short")[1] == unsaved
    for rule, comparison in (("best_sharpe", None), ("named", None)):
        with pytest.raises(ap5_rules.RuleError):
            ap5_rules.choose(rule, comparison, report, tmp_path)


def test_the_rule_never_reads_results_so_it_cannot_pick_a_winner(tmp_path):
    for id_ in ("a2", "b1"):
        (tmp_path / "comparisons" / id_).mkdir(parents=True)
        (tmp_path / "comparisons" / id_ / ap5_rules.SPEC).write_text("x", encoding="utf-8")
    plain = rows()
    flashy = {
        "comparisons": [
            {**row, "sharpe": 9.0 if row["id"] == "b1" else -1.0} for row in plain["comparisons"]
        ]
    }
    chosen = [
        ap5_rules.choose("first_full_pool", None, r, tmp_path)[0]["id"] for r in (plain, flashy)
    ]
    assert chosen == ["a2", "a2"]  # registered order, not the better number
