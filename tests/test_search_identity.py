"""The search config's digest (`capstone.pipeline.search_identity`): what enters it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from capstone.funnel import Thresholds, thresholds_dict
from capstone.pipeline.config import load_config
from capstone.pipeline.rows import REPO
from capstone.pipeline.search_identity import config_digest

CONFIGS = ["e2e_synthetic.toml", "e2e_crsp.toml"]


def _digest_without_optional_fields(config):
    """The digest rule written out again: optional fields at their default are absent."""
    papers = hashlib.sha256(Path(config.papers).read_bytes()).hexdigest()
    content = asdict(config) | {
        "thresholds": thresholds_dict(config.thresholds),
        "papers": f"sha256:{papers}",
    }
    optional = (("deepen_on_budget", "raise"), ("paper_mode", "sequential"))
    for name, default in (*optional, ("regime_operators", False)):
        if content[name] == default:
            del content[name]
    del content["concurrency"]
    text = json.dumps(content, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name", CONFIGS)
def test_the_digest_is_stable_and_follows_the_written_out_rule(name):
    config = load_config(REPO / "config" / name)
    assert config_digest(config) == config_digest(load_config(REPO / "config" / name))
    assert config_digest(config) == _digest_without_optional_fields(config)
    assert len(config_digest(config)) == 64


@pytest.mark.parametrize("name", CONFIGS)
def test_concurrency_never_enters_the_digest_but_paper_mode_does_when_set(name):
    config = load_config(REPO / "config" / name)
    assert config.paper_mode == "sequential"
    digest = config_digest(config)
    assert config_digest(replace(config, concurrency=7)) == digest
    assert config_digest(replace(config, paper_mode="isolated")) != digest
    assert config_digest(replace(config, deepen_on_budget="stop")) != config_digest(
        replace(config, deepen_on_budget="raise")
    )


def test_what_can_change_a_result_changes_the_digest():
    config = load_config()
    digest = config_digest(config)
    assert config_digest(replace(config, seed=1)) != digest
    assert config_digest(replace(config, max_model_calls=config.max_model_calls + 1)) != digest
    assert config_digest(replace(config, thresholds=Thresholds(q1=0.05, stage1="rho"))) != digest
    assert config_digest(replace(config, thresholds=Thresholds(stage1="bootstrap"))) != digest


def test_the_paper_file_enters_by_content_not_by_path(tmp_path):
    config = load_config()
    original = Path(config.papers).read_bytes()
    moved = tmp_path / "elsewhere.toml"
    moved.write_bytes(original)
    assert config_digest(replace(config, papers=str(moved))) == config_digest(config)
    moved.write_bytes(original + b"\n# an edited paper list\n")
    assert config_digest(replace(config, papers=str(moved))) != config_digest(config)
