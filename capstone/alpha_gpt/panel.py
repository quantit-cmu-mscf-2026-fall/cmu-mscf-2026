"""The data an alpha expression is evaluated on: aligned date x asset fields.

Source-neutral on purpose. The synthetic generator and the CRSP adapter both
produce an `OHLCVPanel`, so nothing downstream (operators, metrics, the loop)
knows or cares where the numbers came from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: The field every panel must carry: metrics score an alpha against it.
RETURNS_FIELD = "returns"


@dataclass(frozen=True)
class OHLCVPanel:
    """Aligned date x asset frames plus static asset groupings.

    Attributes:
        fields: name -> DataFrame (DatetimeIndex x assets). Every frame shares
            one index and one column set. `returns[t]` is the simple return
            realised from close t-1 to close t.
        groups: name -> Series (asset -> label), e.g. {"sector": ...}, used by
            grouped_* operators. Every asset must have a label.
        descriptor: provenance (source, seed, pattern, ...) recorded in the
            ledger. Never shown to the LLM.
    """

    fields: Mapping[str, pd.DataFrame]
    groups: Mapping[str, pd.Series] = field(default_factory=dict)
    descriptor: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if RETURNS_FIELD not in self.fields:
            raise ValueError(f"panel must include a {RETURNS_FIELD!r} field")

        reference = self.fields[RETURNS_FIELD]
        if not isinstance(reference.index, pd.DatetimeIndex):
            raise ValueError("panel index must be a DatetimeIndex")
        if not (reference.index.is_monotonic_increasing and reference.index.is_unique):
            raise ValueError("panel dates must be unique and increasing")

        for name, frame in self.fields.items():
            if not frame.index.equals(reference.index) or not frame.columns.equals(
                reference.columns
            ):
                raise ValueError(f"field {name!r} is not aligned with {RETURNS_FIELD!r}")

        for name, labels in self.groups.items():
            aligned = labels.reindex(reference.columns)
            if aligned.isna().any():
                missing = aligned.index[aligned.isna()].tolist()
                raise ValueError(f"group {name!r} has no label for assets {missing[:5]}")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields[RETURNS_FIELD].index

    @property
    def assets(self) -> pd.Index:
        return self.fields[RETURNS_FIELD].columns

    def mask(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> OHLCVPanel:
        """The panel with every field NaN on dates in [start, end].

        `returns` is also NaN on the first date after `end`, because that return
        is measured from `end`'s close. Everything computed from the masked rows
        — a label that realises inside them, a rolling feature whose window
        reaches into them — comes out NaN too, which is how a cross-validation
        fold is kept out of the procedure trained around it: the data is absent,
        not merely skipped by an index.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start > end:
            raise ValueError("mask start is after its end")
        hidden = (self.dates >= start) & (self.dates <= end)
        after = np.flatnonzero(self.dates > end)

        fields = {}
        for name, frame in self.fields.items():
            masked = frame.copy()
            masked.loc[hidden] = np.nan
            if name == RETURNS_FIELD and after.size:
                masked.iloc[after[0]] = np.nan
            fields[name] = masked

        descriptor = dict(self.descriptor)
        descriptor["masked"] = [
            *descriptor.get("masked", []),
            [start.date().isoformat(), end.date().isoformat()],
        ]
        return OHLCVPanel(fields=fields, groups=dict(self.groups), descriptor=descriptor)

    def until(self, date: str | pd.Timestamp) -> OHLCVPanel:
        """The panel restricted to dates strictly before `date`.

        This is how the search phase is kept away from TEST: it is handed a
        panel in which the test period does not exist, rather than a panel it
        is trusted not to look at.
        """
        keep = self.dates < pd.Timestamp(date)
        return OHLCVPanel(
            fields={name: frame.loc[keep] for name, frame in self.fields.items()},
            groups=dict(self.groups),
            descriptor=dict(self.descriptor),
        )
