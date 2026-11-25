"""The analysis output folder: every statement cites a number, or nothing is written."""

import json

import pandas as pd
import pytest

from capstone.analysis.outputs import Stat, Statement, output_dir, write_outputs

STATS = {"n_families": Stat(15, "hypothesis families in the matrix", ("matrix.parquet",))}


def test_round_trip(tmp_path):
    table = pd.DataFrame({"family": ["a", "b"], "k": [1, 2]}).set_index("family")
    out = write_outputs(
        tmp_path / "unit",
        title="Example",
        tables={"families": table},
        stats=STATS | {"missing": Stat(float("nan"), "a NaN is written as null")},
        statements=[Statement("15 families were searched.", "n_families", "one run")],
    )
    assert pd.read_parquet(out / "data" / "families.parquet").equals(table)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["n_families"] == {
        "value": 15,
        "definition": "hypothesis families in the matrix",
        "inputs": ["matrix.parquet"],
    }
    assert stats["missing"]["value"] is None
    text = (out / "STATEMENTS.md").read_text()
    assert "S1. 15 families were searched. (`stats.json: n_families`)" in text
    assert "Caveat: one run" in text


@pytest.mark.parametrize(
    "statement, match",
    [
        (Statement("No source.", ""), "no source"),
        (Statement("Cites a missing number.", "n_variants"), "not in stats"),
        (Statement("  ", "n_families"), "empty"),
    ],
)
def test_an_unsourced_statement_writes_nothing(tmp_path, statement, match):
    with pytest.raises(ValueError, match=match):
        write_outputs(tmp_path / "unit", title="x", stats=STATS, statements=[statement])
    assert not (tmp_path / "unit").exists()


def test_a_stat_needs_a_definition(tmp_path):
    with pytest.raises(ValueError, match="definition"):
        write_outputs(tmp_path / "u", title="x", stats={"a": Stat(1, " ")}, statements=[])


def test_output_dir_is_per_run_and_unit(tmp_path):
    assert output_dir("crsp-v1", "funnel", tmp_path) == tmp_path / "crsp-v1" / "funnel"
