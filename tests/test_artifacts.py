"""Canonical identity and atomic JSON writes."""

import hashlib
import json

import pytest

from capstone.artifacts import atomic_json, canonical, digest


def test_canonical_ignores_mapping_order_and_whitespace():
    a = {"b": [1, 2], "a": {"y": 1, "x": None}}
    b = {"a": {"x": None, "y": 1}, "b": [1, 2]}
    assert canonical(a) == canonical(b) == '{"a":{"x":null,"y":1},"b":[1,2]}'


def test_canonical_keeps_sequence_order():
    assert canonical([1, 2]) != canonical([2, 1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_canonical_refuses_nonfinite_numbers(bad):
    with pytest.raises(ValueError):
        canonical({"x": bad})


def test_digest_is_the_sha256_of_the_canonical_text():
    value = {"b": 1, "a": "x"}
    assert digest(value) == hashlib.sha256(b'{"a":"x","b":1}').hexdigest()
    assert digest({"a": 1}) != digest({"a": 2})


def test_atomic_json_writes_canonical_text_and_creates_parents(tmp_path):
    path = tmp_path / "deep" / "dir" / "artifact.json"
    atomic_json(path, {"b": 1, "a": [1.5]})
    assert path.read_text(encoding="utf-8") == '{"a":[1.5],"b":1}\n'
    assert {p.name for p in path.parent.iterdir()} == {"artifact.json"}


def test_atomic_json_failure_leaves_the_old_file_and_no_partial_write(tmp_path):
    path = tmp_path / "artifact.json"
    atomic_json(path, {"v": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"v": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}
    assert {p.name for p in tmp_path.iterdir()} <= {"artifact.json", "artifact.json.tmp"}
    atomic_json(path, {"v": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}
    assert {p.name for p in tmp_path.iterdir()} == {"artifact.json"}
