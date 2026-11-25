"""`[deepen] max_children_per_family` in the run config: parsed, validated, and in the
digest only when set (so every existing config keeps its digest). The driver that hands it
to `Deepener(max_children_per_family=...)` is a follow-up."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from capstone.pipeline.config import DEFAULT_CONFIG, load_config
from capstone.pipeline.search_identity import config_digest


def _with_cap(tmp_path, line):
    text = Path(DEFAULT_CONFIG).read_text(encoding="utf-8")
    path = tmp_path / "cap.toml"
    path.write_text(text.replace("[deepen]\n", f"[deepen]\n{line}\n", 1), encoding="utf-8")
    return path


def test_the_key_is_read_from_the_deepen_section(tmp_path):
    assert load_config().deepen_max_children_per_family is None
    config = load_config(_with_cap(tmp_path, "max_children_per_family = 4"))
    assert config.deepen_max_children_per_family == 4


@pytest.mark.parametrize("bad", [0, -2, True, 1.5, "3"])
def test_invalid_values_raise(bad):
    with pytest.raises(ValueError, match="max_children_per_family"):
        replace(load_config(), deepen_max_children_per_family=bad)


def test_a_zero_in_the_toml_raises(tmp_path):
    with pytest.raises(ValueError, match="max_children_per_family"):
        load_config(_with_cap(tmp_path, "max_children_per_family = 0"))


def test_unset_leaves_the_digest_and_a_set_value_changes_it():
    base = load_config()
    digest = config_digest(base)
    assert config_digest(replace(base, deepen_max_children_per_family=None)) == digest
    capped = {config_digest(replace(base, deepen_max_children_per_family=n)) for n in (1, 2)}
    assert digest not in capped and len(capped) == 2
