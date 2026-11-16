"""research/memory_calibration.py: the veto simulation and the correlation-cap table.

Synthetic throughout: seeded draws, a made-up evaluation run in `tmp_path`, a temporary
ledger. No model is called and no data is read.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone.factors.llm import load_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "memory_calibration.py"
_spec = importlib.util.spec_from_file_location("memory_calibration", SCRIPT)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)

VETO = {
    "attempts": [5, 20],
    "null_failure": [0.86],
    "null_scored": [1.0],
    "bad_residual": -0.5,
    "bad_failure": 0.95,
    "bad_scored": 1.0,
    "kappa": [5.0],
    "confidence": [0.3, 0.6],
    "target_false_veto": 0.5,
}


def params(**more):
    base = {"seed": 0, "reps": 400, "veto": VETO, "correlation": {"run": ""}}
    return {**base, **more}


def test_the_shipped_parameters_parse_and_skip_the_correlation_table():
    shipped = tomllib.loads(cal.PARAMS.read_text(encoding="utf-8"))
    assert shipped["correlation"]["run"] == ""
    assert cal.correlation_table(shipped) is None


def test_veto_rate_is_seeded_and_a_bad_edit_is_vetoed_more_than_a_null_one():
    config = load_config()
    kwargs = {"attempts": 20, "p_fail": 0.95, "share_scored": 1.0, "reps": 300, "seed": 1}
    null = cal.veto_rate(config, residual=0.0, **kwargs)
    bad = cal.veto_rate(config, residual=-0.5, **kwargs)
    assert cal.veto_rate(config, residual=0.0, **kwargs) == null
    assert 0.0 <= null <= bad <= 1.0 and bad > null


def test_veto_table_shape_and_a_stricter_confidence_vetoes_less():
    table = cal.veto_table(params())
    assert len(table) == 2
    assert {"kappa", "tau_c", "worst_false_veto", "bad@5", "bad@20", "meets_target"} <= set(table)
    low, high = table.sort_values("tau_c")["worst_false_veto"]
    assert low >= high
    assert table["meets_target"].tolist() == (table["worst_false_veto"] <= 0.5).tolist()


def test_a_pair_that_is_never_scored_is_eventually_vetoed():
    table = cal.veto_table(params())
    assert table["never_scored_at"].notna().all()
    assert (table["never_scored_at"] >= 1).all() and (table["never_scored_at"] <= 40).all()


def test_correlation_table_separates_variants_from_different_factors(tmp_path):
    rng = np.random.default_rng(0)
    base = {name: rng.normal(size=250) for name in ("f1", "f2")}
    columns = {
        "a": base["f1"],
        "b": base["f1"] + rng.normal(scale=0.05, size=250),  # a variant of f1
        "c": base["f2"],
    }
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(columns).to_parquet(run / "matrix.parquet")
    pd.DataFrame(
        {"factor_id": ["f1", "f1", "f2"], "hold": [21] * 3, "neutral": ["none"] * 3},
        index=["a", "b", "c"],
    ).to_csv(run / "summary.csv")
    shared = params(
        correlation={
            "run": "run",
            "spec": {"hold": 21, "neutral": "none"},
            "caps": [0.7, 0.9],
        }
    )
    table = cal.correlation_table(shared, root=tmp_path)
    assert table["cap"].tolist() == [0.7, 0.9]
    assert table["blocks_same_factor"].tolist() == [1.0, 1.0]
    assert table["blocks_different_factors"].tolist() == [0.0, 0.0]
    assert table["blocks_different_at_spec"].tolist() == [0.0, 0.0]


def test_main_logs_one_synthetic_run_before_printing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(cal.tomllib, "loads", lambda text: params(reps=50))
    cal.main()
    printed = capsys.readouterr().out
    assert "worst_false_veto" in printed
    (ledger,) = tmp_path.glob("*.jsonl")
    (row,) = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert row["name"] == "memory-calibration" and row["seed"] == 0
    assert "synthetic" in row["tags"] and row["metrics"]["correlation"] is None
    assert len(row["metrics"]["veto"]) == 2


def test_the_script_calls_no_model():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "make_client" not in text and "call_tool" not in text
    with pytest.raises(AttributeError):
        cal.make_client  # noqa: B018
