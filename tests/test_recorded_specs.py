"""Old source-bound identities remain inspectable but cannot trade under new code.

The testbed's full recorded-spec suite also drives the strategy contract, replay and bundle
routes; those routes are ported with their modules. These tests cover the record itself.
"""

import json
import tomllib
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest

from capstone.strategy.recorded import (
    ImplementationMismatch,
    RecordedExecutionBlocked,
    RecordedStrategySpec,
    load_compatible_spec,
    require_current_implementation,
)
from capstone.strategy.research import canonical, digest

CURRENT_HASH = "a" * 64


@dataclass(frozen=True)
class FakeSpec:
    """A current implementation whose hash no longer matches what an old record names."""

    claim_id: str
    data_version: str
    anchor: str
    gross: float = 1.0
    tail: float = 0.1
    method_revision: int = 1
    method_hash: str = CURRENT_HASH

    def __post_init__(self):
        if self.claim_id == "":
            raise ValueError("claim required")

    @classmethod
    def from_toml(cls, text):
        document = tomllib.loads(text)
        payload = json.loads(document["payload"])
        if payload["method_hash"] != CURRENT_HASH:
            raise ImplementationMismatch("recorded implementation differs")
        return cls(**payload)


def record_text(**change):
    payload = asdict(FakeSpec("c1", "fixture", "2000-12-29")) | {"method_hash": "b" * 64} | change
    return (
        f'schema_version = 2\nid = "{digest(payload)[:16]}"\n'
        f"payload = {json.dumps(canonical(payload))}\n"
    )


def test_old_implementation_loads_as_an_inspection_only_record_with_original_bytes():
    for newline in ("\n", "\r\n"):
        text = record_text().replace("\n", newline)
        spec = load_compatible_spec(FakeSpec, text)
        assert isinstance(spec, RecordedStrategySpec)
        assert spec.id == digest(spec.payload)[:16]
        assert spec.implementation_status == "recorded_implementation_unavailable"
        assert spec.to_toml() == text
        spec.payload["method_hash"] = "0" * 64  # the payload property returns a copy
        assert spec.payload["method_hash"] == "b" * 64
        assert not any(hasattr(spec, name) for name in ("side", "data_version", "method_hash"))


def test_current_implementation_still_loads_strictly():
    text = record_text(method_hash=CURRENT_HASH)
    assert load_compatible_spec(FakeSpec, text) == FakeSpec("c1", "fixture", "2000-12-29")


@pytest.mark.parametrize(
    "change",
    [
        {"method_hash": "bad"},
        {"method_hash": "A" * 64},
        {"data_version": 123},
        {"gross": True},
        {"method_revision": 3.0},
        {"gross": float("inf")},
        {"extra": "not-a-field"},
    ],
)
def test_archive_fallback_cannot_hide_malformed_fields(change):
    payload = asdict(FakeSpec("c1", "fixture", "2000-12-29")) | {"method_hash": "b" * 64} | change
    # allow_nan=False would refuse an infinite value upstream; write it through json directly.
    text = (
        f'schema_version = 2\nid = "{"0" * 16}"\n'
        f"payload = {json.dumps(json.dumps(payload, sort_keys=True))}\n"
    )
    with pytest.raises((ValueError, TypeError)):
        load_compatible_spec(FakeSpec, text)


def test_stale_identity_and_noncanonical_payload_are_not_archived():
    text = record_text()
    identifier = tomllib.loads(text)["id"]
    with pytest.raises(ValueError, match="altered"):
        load_compatible_spec(FakeSpec, text.replace(identifier, "0" * 16))
    with pytest.raises(ValueError, match="altered"):
        load_compatible_spec(FakeSpec, text.replace("payload = ", "payload  = "))
    with pytest.raises(ValueError, match="schema"):
        load_compatible_spec(FakeSpec, text.replace("schema_version = 2", "schema_version = 9"))


def test_records_refuse_current_construction_even_when_nested():
    record = load_compatible_spec(FakeSpec, record_text())
    for spec in (record, SimpleNamespace(specs=(SimpleNamespace(specs=(record,)),))):
        with pytest.raises(RecordedExecutionBlocked, match="recorded implementation"):
            require_current_implementation(spec)
    require_current_implementation(FakeSpec("c1", "fixture", "2000-12-29"))
    require_current_implementation(SimpleNamespace(specs=None))
