"""A placebo search's config switch: `[run] placebo`, its seed, its tag and its digest, and
the module's guard and panel reading a real `E2EConfig`. Fixture rows only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from capstone.funnel import thresholds_dict
from capstone.pipeline import placebo
from capstone.pipeline.config import load_config
from capstone.pipeline.rows import REPO
from capstone.pipeline.search_identity import config_digest

METHOD = "cross_section_permutation"
CONFIGS = ["e2e_crsp.toml", "e2e_synthetic.toml"]


def _write_config(tmp_path, extra, name="e2e_crsp.toml"):
    text = (REPO / "config" / name).read_text(encoding="utf-8")
    path = tmp_path / "placebo.toml"
    path.write_text(text.replace("[run]\n", "[run]\n" + extra, 1), encoding="utf-8")
    return path


def test_the_config_reads_placebo_and_tags_the_run(tmp_path):
    config = load_config(_write_config(tmp_path, f'placebo = "{METHOD}"\nplacebo_seed = 3\n'))
    real = load_config(REPO / "config" / "e2e_crsp.toml")
    assert config.placebo == METHOD and config.placebo_seed == 3
    assert config.run_tag == f"{real.run_tag}:placebo"
    assert config_digest(config) != config_digest(real)
    assert config_digest(replace(config, placebo_seed=4)) != config_digest(config)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (f'placebo = "{METHOD}"\n', "placebo_seed"),
        ("placebo_seed = 3\n", "placebo must be"),
        ('placebo = "x"\nplacebo_seed = 3\n', "placebo must be"),
        (f'placebo = "{METHOD}"\nplacebo_seed = -1\n', "placebo_seed"),
        (f'placebo = "{METHOD}"\nplacebo_seed = true\n', "placebo_seed"),
    ],
)
def test_a_placebo_needs_a_known_method_and_a_seed(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, extra))


def _digest_without_placebo_fields(config):
    """`config_digest` as it was before the placebo fields existed: neither is in the
    payload, so a config that is not a placebo keeps the digest it had."""
    papers = hashlib.sha256(Path(config.papers).read_bytes()).hexdigest()
    content = asdict(config) | {
        "thresholds": thresholds_dict(config.thresholds),
        "papers": f"sha256:{papers}",
    }
    for name in ("placebo", "placebo_seed"):
        assert content.pop(name) is None
    optional = (("deepen_on_budget", "raise"), ("paper_mode", "sequential"))
    for name, default in (*optional, ("regime_operators", False)):
        if content[name] == default:
            del content[name]
    del content["concurrency"], content["resources"], content["thresholds_source"]
    assert content.pop("deepen_max_children_per_family") is None  # unset: not in the digest
    text = json.dumps(content, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name", CONFIGS)
def test_configs_without_a_placebo_keep_their_tag_and_digest(name):
    config = load_config(REPO / "config" / name)
    assert config.placebo is None and config.placebo_seed is None
    assert config.run_tag == f"e2e-run:{config.run_id}"
    assert config_digest(config) == _digest_without_placebo_fields(config)


def _rows():
    dates = pd.bdate_range("2001-01-02", periods=5)
    return pd.DataFrame(
        {
            "date": dates.repeat(6),
            "permno": list(range(10000, 10006)) * 5,
            "dlyret": [0.001 * i for i in range(30)],
        }
    )


def test_guard_and_apply_follow_the_config(tmp_path, monkeypatch):
    real = load_config(REPO / "config" / "e2e_crsp.toml")
    config = replace(real, placebo=METHOD, placebo_seed=5)
    rows = _rows()
    # a real config is untouched: no ledger check, rows and version unchanged
    main = tmp_path / "experiments"
    monkeypatch.setattr(placebo, "main_ledger_dir", lambda: main)
    monkeypatch.delenv("CAPSTONE_LEDGER_DIR", raising=False)
    placebo.guard(real)
    same, version = placebo.apply(real, rows, "v1")
    assert same is rows and version == "v1"
    # a placebo needs its own ledger, and never has a holdout
    with pytest.raises(placebo.PlaceboRefused):
        placebo.guard(config)
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(placebo.main_ledger_dir()))
    with pytest.raises(placebo.PlaceboRefused):
        placebo.guard(config)
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(main / "placebo"))
    placebo.guard(config)
    with pytest.raises(ValueError, match="no holdout"):
        placebo.guard(config, holdout=True)
    permuted, version = placebo.apply(config, rows, "v1")
    assert version == f"placebo:{METHOD}:5:v1"
    assert permuted.equals(placebo.permute_cross_section(rows, 5))
    assert not permuted["dlyret"].equals(rows.sort_values(["date", "permno"])["dlyret"])
