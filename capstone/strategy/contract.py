"""The strategy contract: what a deployed strategy is.

A `StrategySpec` is a validated variant (a factor tree traded by a
`TradingSpec`) frozen into a file: its expression, how it is traded, the data
it was validated on, the date its rebalance calendar counts from, its gross
exposure, and where it came from (factor, variant, hypothesis, gate report).
Its id is a hash of everything else in its file, so the same content always
has the same id and any change makes a new one.

The file is written by code, never by hand: UTF-8, LF line endings, the same
bytes every time. `load_spec` refuses a file that is not exactly what its
spec would write (an edited value, a stale id or derived field). The file
carries a `band` line only for a spec with a no-trade band, so specs written
without one keep their bytes and id.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capstone.factors.evaluation import TradingSpec
from capstone.factors.store import variant_id as _variant_id
from capstone.factors.tree import Node, factor_id, features_used, parse, unparse

HEADER = (
    "# Strategy spec (capstone.strategy.contract). Written by code; the id is a hash of the rest."
)


@dataclass(frozen=True)
class StrategySpec:
    """A validated variant, frozen for trading. See the module docstring.

    `expression` is stored in `unparse`'s canonical spelling and `anchor` as
    YYYY-MM-DD, so two spellings of the same content share one id. The data
    fields, factor id and variant id are derived from the content, never
    passed in, so they cannot disagree with it.
    """

    expression: str
    trading: TradingSpec
    data_version: str  # the shared data version the variant was validated on
    anchor: str  # first rebalance date; the calendar counts trading days from it
    gross: float = 1.0  # sum of |weight| of the book
    hypothesis_id: str = ""
    gate_report: str = ""  # path or link to the gate report; required to register

    def __post_init__(self) -> None:
        object.__setattr__(self, "expression", unparse(parse(self.expression)))
        if not isinstance(self.trading, TradingSpec):
            raise ValueError("trading must be a TradingSpec")
        if not self.data_version:
            raise ValueError("a spec must name the data version it was validated on")
        try:
            anchor = pd.Timestamp(self.anchor)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"anchor {self.anchor!r} is not a date") from exc
        if anchor != anchor.normalize():
            raise ValueError(f"anchor {self.anchor!r} must be a date, not a time")
        object.__setattr__(self, "anchor", anchor.strftime("%Y-%m-%d"))
        gross = float(self.gross)
        if not (math.isfinite(gross) and gross > 0):
            raise ValueError(f"gross must be a positive number, got {self.gross!r}")
        object.__setattr__(self, "gross", gross)
        for name in ("data_version", "hypothesis_id", "gate_report"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")

    @property
    def tree(self) -> Node:
        return parse(self.expression)

    @property
    def fields(self) -> tuple[str, ...]:
        """The data fields the expression reads, sorted."""
        return tuple(sorted(features_used(self.tree)))

    @property
    def factor_id(self) -> str:
        return factor_id(self.tree)

    @property
    def variant_id(self) -> str:
        return _variant_id(self.factor_id, self.trading.as_dict())

    @property
    def id(self) -> str:
        """16 hex digits of the SHA-256 of the spec's file without its id line."""
        return hashlib.sha256(self._body().encode("utf-8")).hexdigest()[:16]

    def _body(self) -> str:
        fields = ", ".join(basic_string(f) for f in self.fields)
        # No line without a band, so specs written before bands keep their bytes and id.
        band = f"band = {self.trading.band!r}\n" if self.trading.band > 0 else ""
        return (
            f"{HEADER}\n"
            f"expression = {basic_string(self.expression)}\n"
            f"anchor = {basic_string(self.anchor)}\n"
            f"gross = {self.gross!r}\n"
            "\n[trading]\n"
            f"hold = {self.trading.hold}\n"
            f"neutral = {basic_string(self.trading.neutral)}\n"
            f"{band}"
            "\n[data]\n"
            f"fields = [{fields}]\n"
            f"version = {basic_string(self.data_version)}\n"
            "\n[provenance]\n"
            f"factor_id = {basic_string(self.factor_id)}\n"
            f"variant_id = {basic_string(self.variant_id)}\n"
            f"hypothesis_id = {basic_string(self.hypothesis_id)}\n"
            f"gate_report = {basic_string(self.gate_report)}\n"
        )

    def to_toml(self) -> str:
        """The spec's file: its id on the line after the header, then the content."""
        header, rest = self._body().split("\n", 1)
        return f"{header}\nid = {basic_string(self.id)}\n{rest}"

    @classmethod
    def from_toml(cls, text: str) -> StrategySpec:
        """The spec a file holds. A file that is not exactly what its spec writes
        (an edited value, a stale id or derived field) is refused."""
        values = tomllib.loads(text)
        try:
            spec = cls(
                expression=values["expression"],
                trading=TradingSpec(
                    hold=values["trading"]["hold"],
                    neutral=values["trading"]["neutral"],
                    band=values["trading"].get("band", 0.0),
                ),
                data_version=values["data"]["version"],
                anchor=values["anchor"],
                gross=values["gross"],
                hypothesis_id=values["provenance"]["hypothesis_id"],
                gate_report=values["provenance"]["gate_report"],
            )
        except KeyError as exc:
            raise ValueError(f"spec file is missing {exc}") from exc
        if values.get("id") != spec.id:
            raise ValueError(f"spec file id {values.get('id')!r} does not match its content")
        if spec.to_toml() != text.replace("\r\n", "\n"):
            raise ValueError(f"spec file {spec.id} is not the file its content writes")
        return spec


def basic_string(text: str) -> str:
    """A TOML basic string (also a YAML double-quoted scalar)."""
    out = []
    for ch in text:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def save_spec(spec: StrategySpec, path: Path) -> None:
    """Write the spec's file: UTF-8, LF line endings, the same bytes every time."""
    Path(path).write_bytes(spec.to_toml().encode("utf-8"))


def load_spec(path: Path) -> StrategySpec:
    """The spec a file holds (refused if the file is not exactly what its spec writes)."""
    return StrategySpec.from_toml(Path(path).read_bytes().decode("utf-8"))
