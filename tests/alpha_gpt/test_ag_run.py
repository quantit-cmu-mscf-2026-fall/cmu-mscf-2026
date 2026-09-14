"""Tests for capstone.alpha_gpt.run (scripted LLM; Claude is never called)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from capstone.alpha_gpt.loop import TestAlreadyEvaluated
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

[cv]
n_splits = 2

[loop]
n_rounds = 2
n_alphas = 2
"""


def _proposal(*expressions):
    alphas = [{"expression": e, "rationale": "r"} for e in expressions]
    return json.dumps({"type": "seed_proposal", "alphas": alphas})


def _llm():
    """Round 1 proposes noise, the Analyst suggests reversal, round 2 proposes it."""
    review = json.dumps(
        {
            "type": "analyst_review",
            "summary": "no effect",
            "diagnoses": [{"alpha_id": "r1a1", "note": "flat"}],
            "revised_idea": "one-day reversal",
        }
    )

    def reply(prompt):
        if prompt.startswith("# Role\nYou are the Analyst"):
            return review
        if "after round 1." in prompt:
            return _proposal("neg(returns)", "neg(ts_delta(close, 3))")
        return _proposal("ts_mean(volume, 10)", "minus(high, low)")

    return reply


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "tiny.toml"
    path.write_text(CONFIG)
    return path


def _main(config_path, out, run_id, *flags):
    return main(
        ["--config", str(config_path), "--out-dir", str(out), "--run-id", run_id, *flags],
        llm=_llm(),
    )


def test_end_to_end_run_writes_run_dir_and_ledger(config_path, tmp_path, ledger, capsys):
    assert _main(config_path, tmp_path / "out", "t1") == 0

    run_dir = tmp_path / "out" / "t1"
    for name in [
        "config.json",
        "summary.json",
        "final_test.json",
        "cv_folds.jsonl",
        "fold1/llm_calls.jsonl",
        "fold2/trials.jsonl",
        "final/reviews.jsonl",
    ]:
        assert (run_dir / name).exists(), name
    final = json.loads((run_dir / "final_test.json").read_text())
    assert final["selected"][0]["expression"] == "neg(returns)"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["family_size"] == 12
    assert summary["cross_validation"]["n_folds"] == 2

    entries = ledger()
    tags = [e["tags"][-1] for e in entries]
    assert tags.count("search") == 12 and tags.count("cv_valid") == 2 and tags.count("test") == 1
    assert {e["params"]["config_name"] for e in entries} == {"tiny"}

    stdout = capsys.readouterr().out
    assert "cross-validation" in stdout
    assert "TEST (evaluated once)" in stdout
    assert "overfitting check" in stdout
    assert "of train IC retained" in stdout


def test_skip_test_leaves_test_untouched(config_path, tmp_path, ledger):
    _main(config_path, tmp_path / "out", "t2", "--skip-test")
    run_dir = tmp_path / "out" / "t2"
    assert not (run_dir / "final_test.json").exists()
    assert json.loads((run_dir / "summary.json").read_text())["test"] is None
    assert all(e["tags"][-1] != "test" for e in ledger())


def test_existing_run_dir_is_not_overwritten(config_path, tmp_path, ledger):
    (tmp_path / "out" / "t3").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        _main(config_path, tmp_path / "out", "t3")


def test_second_run_on_the_same_data_needs_reuse_test(config_path, tmp_path, ledger):
    _main(config_path, tmp_path / "out", "first")
    with pytest.raises(TestAlreadyEvaluated):
        _main(config_path, tmp_path / "out", "second")
    _main(config_path, tmp_path / "out", "third", "--reuse-test")

    test_tags = [e["tags"] for e in ledger() if e["tags"][-1] == "test"]
    assert len(test_tags) == 2
    assert "test_reuse" not in test_tags[0] and "test_reuse" in test_tags[1]


def test_label_horizon_beyond_the_embargo_is_refused(tmp_path, ledger):
    path = tmp_path / "long.toml"
    path.write_text(CONFIG + "\n[scoring]\nlabel_horizon = 10\n")
    with pytest.raises(ValueError, match="exceeds embargo_days"):
        _main(path, tmp_path / "out", "t4")


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
    assert cfg.cv.n_splits == 4 and cfg.scoring.label_horizon == 1
    assert build_panel(cfg).descriptor["n_dates"] == 1260
