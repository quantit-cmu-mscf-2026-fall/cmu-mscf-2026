"""Tests for capstone.alpha_gpt.run (scripted LLM; Claude is never called)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from capstone.alpha_gpt.run import build_panel, load_config, main

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = """
name = "tiny"
idea = "yesterday's losers bounce"
seed = 1

[data]
n_dates = 400
n_assets = 40
pattern = "reversal"
strength = 0.08

[loop]
n_rounds = 2
n_alphas = 2
"""


def _llm():
    replies = iter(
        [
            json.dumps(
                {
                    "type": "seed_proposal",
                    "alphas": [
                        {"expression": "ts_mean(volume, 10)", "rationale": "r"},
                        {"expression": "minus(high, low)", "rationale": "r"},
                    ],
                }
            ),
            json.dumps(
                {
                    "type": "analyst_review",
                    "summary": "no effect",
                    "diagnoses": [{"alpha_id": "r1a1", "note": "flat"}],
                    "revised_idea": "one-day reversal",
                }
            ),
            json.dumps(
                {
                    "type": "seed_proposal",
                    "alphas": [
                        {"expression": "neg(returns)", "rationale": "r"},
                        {"expression": "neg(ts_delta(close, 3))", "rationale": "r"},
                    ],
                }
            ),
        ]
    )
    return lambda prompt: next(replies)


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "tiny.toml"
    path.write_text(CONFIG)
    return path


def test_end_to_end_run_writes_run_dir_and_ledger(config_path, tmp_path, ledger, capsys):
    out = tmp_path / "out"
    code = main(["--config", str(config_path), "--out-dir", str(out), "--run-id", "t1"], llm=_llm())

    assert code == 0
    run_dir = out / "t1"
    for name in ["config.json", "llm_calls.jsonl", "trials.jsonl", "reviews.jsonl", "events.jsonl"]:
        assert (run_dir / name).exists(), name
    final = json.loads((run_dir / "final_test.json").read_text())
    assert final["selected"][0]["expression"] == "neg(returns)"

    entries = ledger()
    assert [e["tags"][-1] for e in entries] == ["search"] * 4 + ["test"]
    assert {e["params"]["config_name"] for e in entries} == {"tiny"}

    stdout = capsys.readouterr().out
    assert "selected r2a1: neg(returns)" in stdout
    assert "TEST (evaluated once)" in stdout


def test_skip_test_leaves_test_untouched(config_path, tmp_path, ledger):
    out = tmp_path / "out"
    main(
        ["--config", str(config_path), "--out-dir", str(out), "--run-id", "t2", "--skip-test"],
        llm=_llm(),
    )
    assert not (out / "t2" / "final_test.json").exists()
    assert all(e["tags"][-1] == "search" for e in ledger())


def test_existing_run_dir_is_not_overwritten(config_path, tmp_path, ledger):
    (tmp_path / "out" / "t3").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        main(
            ["--config", str(config_path), "--out-dir", str(tmp_path / "out"), "--run-id", "t3"],
            llm=_llm(),
        )


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(CONFIG + "\n[loop2]\nn_rounds = 1\n")
    with pytest.raises(ValidationError, match="loop2"):
        load_config(path)


@pytest.mark.parametrize("name", ["baseline_synth", "baseline_null"])
def test_shipped_configs_are_valid(name):
    cfg = load_config(REPO_ROOT / "experiments" / "alpha_gpt" / f"{name}.toml")
    assert cfg.name == name
    assert cfg.data.pattern == ("reversal" if name == "baseline_synth" else "none")
    assert build_panel(cfg).descriptor["n_dates"] == 1260
