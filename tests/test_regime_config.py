"""The regime-operator switch in the run config (`[factors] regime_operators`): off by
default, read from the factors section, and in the digest only when set. The grammar and
evaluation tests of the operators themselves are `tests/test_regime_operators.py`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from capstone.factors.tree import parse
from capstone.funnel import thresholds_dict
from capstone.pipeline.config import DEFAULT_CONFIG, load_config
from capstone.pipeline.search_identity import config_digest


def _digest_before_the_field(config) -> str:
    """`config_digest` as it was before `regime_operators` existed: not in the payload."""
    papers = hashlib.sha256(Path(config.papers).read_bytes()).hexdigest()
    content = asdict(config) | {
        "thresholds": thresholds_dict(config.thresholds),
        "papers": f"sha256:{papers}",
    }
    del content["regime_operators"]
    for name, default in (("deepen_on_budget", "raise"), ("paper_mode", "sequential")):
        if content[name] == default:
            del content[name]
    content.pop("concurrency")
    text = json.dumps(content, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_off_the_config_digest_is_unchanged():
    config = load_config()
    assert config.regime_operators is False
    assert config_digest(config) == _digest_before_the_field(config)
    assert config_digest(replace(config, regime_operators=True)) != _digest_before_the_field(config)


def test_the_flag_is_read_from_the_factors_section(tmp_path):
    text = Path(DEFAULT_CONFIG).read_text(encoding="utf-8")
    assert "[factors]\n" in text
    path = tmp_path / "on.toml"
    path.write_text(
        text.replace("[factors]\n", "[factors]\nregime_operators = true\n", 1), encoding="utf-8"
    )
    config = load_config(path)
    assert config.regime_operators is True
    assert config.factors == load_config().factors  # FactorConfig itself is unchanged
    with pytest.raises(ValueError, match="true or false"):
        replace(config, regime_operators="yes")


def test_the_flag_decides_whether_the_grammar_admits_the_operators():
    off, on = load_config(), replace(load_config(), regime_operators=True)
    with pytest.raises(ValueError):
        parse("volscale(ts_mean(returns, 5))", regime_operators=off.regime_operators)
    assert parse("volscale(ts_mean(returns, 5))", regime_operators=on.regime_operators)
