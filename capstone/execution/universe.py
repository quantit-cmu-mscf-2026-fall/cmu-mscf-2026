"""The trading universe after 2025: an explicit, versioned permno <-> symbol map.

CRSP ends in 2025, so after it nothing in the shared data says which stocks
may be held or what a permno is called at the trading venue. A consumer
therefore requires `deploy/universe.csv`, reviewed by PR like the strategy
registry, and refuses to trade (or to hold) a name that is not in it.

Format: `#` comment lines, then `version = <label>`, then the header
`permno,symbol,venue_symbol`, then one row per stock. `symbol` is the usual
ticker ("BRK.B"); `venue_symbol` is how the venue spells it ("BRK B"), the
name used for every price request, position and order. Permnos, symbols and
venue symbols are each unique. Anything else is refused.

A version starting "TEMPLATE" marks the committed placeholder: it lists no
stocks, and consumers refuse it (`is_template`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

UNIVERSE_FILE = Path("deploy") / "universe.csv"
HEADER = "permno,symbol,venue_symbol"
TEMPLATE_PREFIX = "TEMPLATE"
SYMBOL = re.compile(r"[A-Z][A-Z0-9.]{0,11}")
VENUE_SYMBOL = re.compile(r"(?=.{1,12}$)[A-Z][A-Z0-9]*( [A-Z0-9]+)?")


class UniverseError(ValueError):
    """A malformed universe file, or a name outside the universe."""


@dataclass(frozen=True)
class Listing:
    permno: int
    symbol: str
    venue_symbol: str


@dataclass(frozen=True)
class Universe:
    version: str
    listings: tuple[Listing, ...]

    def __post_init__(self) -> None:
        if not self.version:
            raise UniverseError("the universe file has no version")
        for name in ("permno", "symbol", "venue_symbol"):
            values = [getattr(row, name) for row in self.listings]
            if len(set(values)) != len(values):
                raise UniverseError(f"the universe names a {name} more than once")

    @property
    def is_template(self) -> bool:
        return self.version.upper().startswith(TEMPLATE_PREFIX)

    @property
    def venue_symbols(self) -> tuple[str, ...]:
        return tuple(row.venue_symbol for row in self.listings)

    def _row(self, permno: int) -> Listing:
        for row in self.listings:
            if row.permno == permno:
                return row
        raise UniverseError(f"permno {permno} is not in universe {self.version}; it is not traded")

    def symbol(self, permno: int) -> str:
        return self._row(permno).symbol

    def venue_symbol(self, permno: int) -> str:
        return self._row(permno).venue_symbol

    def permno(self, venue_symbol: str) -> int:
        """The permno of a name as the venue spells it (positions come back that way)."""
        for row in self.listings:
            if row.venue_symbol == venue_symbol:
                return row.permno
        raise UniverseError(
            f"venue symbol {venue_symbol} is not in universe {self.version}; it is not traded"
        )


def parse_universe(text: str) -> Universe:
    lines = [
        line.strip()
        for line in text.replace("\r\n", "\n").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines or not lines[0].startswith("version"):
        raise UniverseError("the universe file must start with `version = <label>`")
    key, sep, version = lines[0].partition("=")
    if key.strip() != "version" or not sep:
        raise UniverseError(f"bad version line {lines[0]!r}")
    if len(lines) < 2 or lines[1].replace(" ", "") != HEADER:
        raise UniverseError(f"the universe file's header must be `{HEADER}`")
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.split(",")]
        if (
            len(cells) != 3
            or not cells[0].isdigit()
            or not SYMBOL.fullmatch(cells[1])
            or not VENUE_SYMBOL.fullmatch(cells[2])
        ):
            raise UniverseError(
                f"bad universe row {line!r}: expected <permno>,<SYMBOL>,<VENUE SYMBOL>"
            )
        rows.append(Listing(int(cells[0]), cells[1], cells[2]))
    return Universe(version.strip(), tuple(sorted(rows, key=lambda r: r.permno)))
