"""The strategy contract: a frozen spec and its TOML file.

Synthetic expressions only; no data is read.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from capstone.factors.evaluation import TradingSpec
from capstone.factors.store import variant_id
from capstone.factors.tree import factor_id, parse
from capstone.strategy.contract import StrategySpec, load_spec, save_spec

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

EXPRESSION = "rank(ts_mean(volume, 5)) - rank(delta(close, 3))"


def make_spec(**overrides) -> StrategySpec:
    values = {
        "expression": EXPRESSION,
        "trading": TradingSpec(hold=5, neutral="sector"),
        "data_version": "synthetic-v1",
        "anchor": "2015-01-02",
        "gross": 1.0,
        "hypothesis_id": "H-001",
        "gate_report": "reports/gate/H-001.md",
    }
    values.update(overrides)
    return StrategySpec(**values)


# ---------------------------------------------------------------------------
# The spec and its file


def test_the_spec_is_frozen():
    spec = make_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.gross = 2.0


def test_the_spec_derives_its_provenance_and_data_fields_from_its_content():
    spec = make_spec()
    tree = parse(EXPRESSION)
    assert spec.factor_id == factor_id(tree)
    assert spec.variant_id == variant_id(factor_id(tree), TradingSpec(5, "sector").as_dict())
    assert spec.fields == ("close", "volume")


def test_the_file_format_is_fixed_and_the_id_hashes_everything_else():
    spec = make_spec()
    fid = factor_id(parse(EXPRESSION))
    vid = variant_id(fid, TradingSpec(5, "sector").as_dict())
    body = (
        "# Strategy spec (capstone.strategy.contract). Written by code; "
        "the id is a hash of the rest.\n"
        'expression = "(rank(ts_mean(volume, 5)) - rank(delta(close, 3)))"\n'
        'anchor = "2015-01-02"\n'
        "gross = 1.0\n"
        "\n"
        "[trading]\n"
        "hold = 5\n"
        'neutral = "sector"\n'
        "\n"
        "[data]\n"
        'fields = ["close", "volume"]\n'
        'version = "synthetic-v1"\n'
        "\n"
        "[provenance]\n"
        f'factor_id = "{fid}"\n'
        f'variant_id = "{vid}"\n'
        'hypothesis_id = "H-001"\n'
        'gate_report = "reports/gate/H-001.md"\n'
    )
    expected_id = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    first, rest = body.split("\n", 1)
    assert spec.id == expected_id
    assert spec.to_toml() == f'{first}\nid = "{expected_id}"\n{rest}'


def test_the_spec_round_trips_through_its_file_byte_for_byte(tmp_path):
    spec = make_spec(gross=0.5, hypothesis_id='odd "quoted" \\ name\n')
    path = tmp_path / "spec.toml"
    save_spec(spec, path)
    loaded = load_spec(path)
    assert loaded == spec
    assert loaded.id == spec.id
    again = tmp_path / "again.toml"
    save_spec(loaded, again)
    assert again.read_bytes() == path.read_bytes()
    assert b"\r\n" not in path.read_bytes()


def test_equal_content_gives_an_equal_id_and_any_change_a_new_one():
    base = make_spec()
    assert make_spec().id == base.id
    # The same expression spelled differently is the same content.
    assert make_spec(expression="rank(ts_mean(volume,5))-rank(delta(close,3))").id == base.id
    changes = [
        {"expression": "rank(volume)"},
        {"trading": TradingSpec(hold=1, neutral="sector")},
        {"trading": TradingSpec(hold=5, neutral="none")},
        {"data_version": "synthetic-v2"},
        {"anchor": "2015-01-05"},
        {"gross": 2.0},
        {"hypothesis_id": "H-002"},
        {"gate_report": "reports/gate/other.md"},
    ]
    ids = {make_spec(**change).id for change in changes}
    assert base.id not in ids and len(ids) == len(changes)


def test_a_tampered_file_is_refused(tmp_path):
    path = tmp_path / "spec.toml"
    save_spec(make_spec(), path)
    path.write_bytes(path.read_bytes().replace(b"hold = 5", b"hold = 1"))
    with pytest.raises(ValueError, match="id"):
        load_spec(path)


@pytest.mark.parametrize(
    "bad",
    [
        {"expression": "close +"},
        {"gross": 0.0},
        {"gross": float("nan")},
        {"anchor": "not a date"},
        {"data_version": ""},
    ],
)
def test_an_invalid_spec_is_refused(bad):
    with pytest.raises(ValueError):
        make_spec(**bad)


def test_a_banded_spec_round_trips_its_file_and_band_zero_writes_no_band_line():
    plain = make_spec(trading=TradingSpec(5, "sector"))
    assert "band" not in plain.to_toml()
    banded = make_spec(trading=TradingSpec(5, "sector", 0.005))
    text = banded.to_toml()
    assert "\nband = 0.005\n" in text
    again = StrategySpec.from_toml(text)
    assert again == banded and again.id == banded.id != plain.id
    assert again.variant_id == variant_id(
        banded.factor_id, TradingSpec(5, "sector", 0.005).as_dict()
    )


def test_a_file_that_adds_a_band_without_changing_its_id_is_refused(tmp_path):
    path = tmp_path / "spec.toml"
    save_spec(make_spec(), path)
    text = path.read_bytes().decode("utf-8")
    text = text.replace('neutral = "sector"\n', 'neutral = "sector"\nband = 0.01\n')
    path.write_bytes(text.encode("utf-8"))
    with pytest.raises(ValueError, match="id"):
        load_spec(path)


def test_the_spec_id_is_the_same_in_a_fresh_process(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(TESTS)]),
        "PYTHONHASHSEED": "777",
        "CAPSTONE_LEDGER_DIR": str(tmp_path),
    }
    code = "import test_strategy_contract as t; print(t.make_spec().id)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == make_spec().id
