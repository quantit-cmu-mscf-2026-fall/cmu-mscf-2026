"""Read historical identities without authorizing execution under changed code."""

from __future__ import annotations

import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass, fields

from capstone.strategy.research import canonical, digest


class ImplementationMismatch(ValueError):
    """A recorded specification requires a different implementation."""


class RecordedExecutionBlocked(ValueError):
    """An inspection-only historical record cannot authorize current execution."""


@dataclass(frozen=True)
class RecordedStrategySpec:
    """Immutable content-checked record; intentionally supplies no trading interface."""

    schema_version: int
    payload_json: str
    recorded_id: str
    original_text: str

    def __post_init__(self):
        payload = json.loads(self.payload_json)
        if (
            type(self.schema_version) is not int
            or self.schema_version not in {2, 3}
            or not isinstance(payload, dict)
            or self.payload_json != canonical(payload)
            or self.recorded_id != digest(payload)[:16]
            or not isinstance(payload.get("method_hash"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", payload["method_hash"])
            or self.original_text.replace("\r\n", "\n") != self._canonical_toml()
        ):
            raise ValueError("recorded specification content or identifier was altered")

    @property
    def id(self):
        return self.recorded_id

    @property
    def payload(self):
        return json.loads(self.payload_json)

    @property
    def implementation_status(self):
        return "recorded_implementation_unavailable"

    def to_toml(self):
        return self.original_text

    def _canonical_toml(self):
        return (
            f"schema_version = {self.schema_version}\nid = {json.dumps(self.id)}\n"
            f"payload = {json.dumps(self.payload_json)}\n"
        )


def load_compatible_spec(cls, text):
    """Keep strict current loading; only typed code mismatches permit archival loading."""
    try:
        return cls.from_toml(text)
    except ImplementationMismatch:
        document = tomllib.loads(text)
        schema = document.get("schema_version")
        if (
            type(schema) is not int
            or schema not in {2, 3}
            or set(document) != {"schema_version", "id", "payload"}
        ):
            raise ValueError("invalid recorded specification schema") from None
        if not isinstance(document["payload"], str):
            raise ValueError("recorded specification payload must be text") from None
        payload = json.loads(document["payload"])
        if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError("invalid recorded specification fields") from None
        numeric = {"gross", "tail", "method_revision"}
        for key, value in payload.items():
            if key in numeric:
                if type(value) not in {int, float} or not math.isfinite(value):
                    raise ValueError("invalid recorded numeric field") from None
                if key == "method_revision" and type(value) is not int:
                    raise ValueError("invalid recorded method revision") from None
            elif not isinstance(value, str):
                raise ValueError("invalid recorded text field") from None
        if not re.fullmatch(r"[0-9a-f]{64}", payload["method_hash"]):
            raise ValueError("invalid recorded implementation hash") from None
        # Validate all non-hash construction fields through the strict current constructor.
        current = cls(**(payload | {"method_hash": ""}))
        expected = asdict(current)
        expected["method_hash"] = payload["method_hash"]
        if canonical(expected) != canonical(payload):
            raise ValueError("noncanonical recorded construction fields") from None
        record = RecordedStrategySpec(schema, canonical(payload), digest(payload)[:16], text)
        if record.id != document["id"]:
            raise ValueError("recorded specification content or identifier was altered") from None
        return record


def require_current_implementation(spec):
    if isinstance(spec, RecordedStrategySpec):
        raise RecordedExecutionBlocked(
            "recorded implementation unavailable; inspect the historical identity or review "
            "a new version before current construction, planning or evidence reuse"
        )
    members = getattr(spec, "specs", ())
    if members is not None:
        for member in members:
            require_current_implementation(member)
