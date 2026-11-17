"""Market data for the paper runner: a `Panel` up to the run's date.

`MarketDataSource.panel(fields, as_of)` is the strategy contract's input
(`capstone.factors.evaluation.Panel`, dates x permno) holding the fields a
spec reads, every trading date it has through `as_of` and none after. The
history must reach back to the spec's anchor: `target_weights` counts its
rebalance calendar from there.

- `FrozenSnapshotSource` serves a panel built beforehand (tests: synthetic
  rows relabelled onto later dates).
- A bar source (daily open, high, low, close and volume, no shares, no
  fundamentals, no sectors) can serve only `BAR_FIELDS`; `check_servable`
  refuses, naming it, any field or neutrality it cannot, before any data is
  asked for. `BarClient` is the one call such a source needs from a vendor
  connection. No vendor client lives here: a source built on one is its own
  change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import pandas as pd

from capstone.factors.evaluation import Panel

OHLCV = ("open", "high", "low", "close", "volume")
BAR_FIELDS = (*OHLCV, "returns")
_NO_SHARES = "shares outstanding, which daily bars do not carry"
WHY_NOT = {
    "shares": _NO_SHARES,
    "cap": f"price x {_NO_SHARES}",
    "mkt_return": f"the universe's cap-weighted return, so {_NO_SHARES}",
}


class MarketDataSource(Protocol):
    def panel(self, fields: tuple[str, ...], as_of: pd.Timestamp) -> Panel: ...


class FrozenSnapshotSource:
    """A fixed panel, served up to `as_of`."""

    def __init__(self, panel: Panel) -> None:
        self._panel = panel

    def panel(self, fields: tuple[str, ...], as_of: pd.Timestamp) -> Panel:
        source = self._panel
        missing = sorted(set(fields) - set(source.fields))
        if missing:
            raise ValueError(f"the snapshot has no field(s) {missing}")
        as_of = pd.Timestamp(as_of)
        if as_of not in pd.DatetimeIndex(source.universe.index):
            raise ValueError(f"{as_of.date()} is not a trading date of the snapshot")

        def cut(frame):
            # Mask, not a label slice: `.loc[:as_of]` assumes date-sorted rows.
            return None if frame is None else frame.loc[frame.index <= as_of].sort_index()

        return Panel(
            {name: cut(value) for name, value in source.fields.items()},
            cut(source.universe),
            cut(source.half_spread),
            cut(source.sic),
            source.data_version,
        )


class UnsupportedFieldError(ValueError):
    """A strategy needs data a bar source does not have."""


def check_servable(fields: Sequence[str], neutral: str = "none") -> None:
    """Refuse, naming it, any field (or neutrality) a bar source cannot serve."""
    for field in sorted(set(fields) - set(BAR_FIELDS)):
        why = WHY_NOT.get(field, "a field the bar source does not have")
        raise UnsupportedFieldError(
            f"the bar source cannot serve field {field!r}: it needs {why}. It serves "
            f"{', '.join(BAR_FIELDS)} only, so a strategy reading {field!r} cannot run on it."
        )
    if neutral != "none":
        raise UnsupportedFieldError(
            f"a {neutral}-neutral strategy needs SIC codes, which daily bars do not carry"
        )


class BarClient(Protocol):
    def historical_bars(self, symbol: str, request: Mapping[str, object]) -> Sequence:
        """Daily bars (objects with date, open, high, low, close, volume), oldest first."""
