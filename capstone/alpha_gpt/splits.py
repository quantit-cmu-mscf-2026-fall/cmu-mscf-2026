"""TRAIN / VALIDATION / TEST date ranges, separated by an embargo.

TRAIN is for search fitness, VALIDATION for selection and the Analyst's
feedback, TEST for one final evaluation (docs/experiments/alpha_gpt.md).

An alpha observed at the close of day t is scored against the return realised
on day t+1, so the last signal of a split reads one day past the split's end.
The embargo (at least one day) keeps that day out of the next split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from capstone.alpha_gpt.panel import OHLCVPanel

Split = Literal["train", "valid", "test"]
SPLITS: tuple[Split, ...] = ("train", "valid", "test")

#: Fewest dates a split may have; `backtest.summarize` refuses shorter series.
MIN_SPLIT_DAYS = 60


@dataclass(frozen=True)
class SplitSpec:
    """Inclusive (start, end) date ranges for each split."""

    train: tuple[pd.Timestamp, pd.Timestamp]
    valid: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]
    embargo_days: int

    def __post_init__(self) -> None:
        if self.embargo_days < 1:
            raise ValueError("embargo_days must be >= 1: a signal at t is scored on t+1")
        for name in SPLITS:
            start, end = getattr(self, name)
            if start > end:
                raise ValueError(f"{name} starts after it ends")
        if not self.train[1] < self.valid[0] or not self.valid[1] < self.test[0]:
            raise ValueError("splits must be ordered train < valid < test without overlap")

    @classmethod
    def from_fractions(
        cls,
        dates: pd.DatetimeIndex,
        *,
        train_frac: float = 0.6,
        valid_frac: float = 0.2,
        embargo_days: int = 5,
    ) -> SplitSpec:
        """Chronological splits of `dates`, with `embargo_days` dates dropped between each.

        The fractions apply to the dates left after removing the two embargo gaps;
        TEST takes the remainder.
        """
        if embargo_days < 1:
            raise ValueError("embargo_days must be >= 1: a signal at t is scored on t+1")
        if not (0 < train_frac < 1 and 0 < valid_frac < 1 and train_frac + valid_frac < 1):
            raise ValueError("need 0 < train_frac, valid_frac and train_frac + valid_frac < 1")

        usable = len(dates) - 2 * embargo_days
        n_train = int(usable * train_frac)
        n_valid = int(usable * valid_frac)
        n_test = usable - n_train - n_valid
        if min(n_train, n_valid, n_test) < MIN_SPLIT_DAYS:
            raise ValueError(
                f"each split needs >= {MIN_SPLIT_DAYS} dates; got train={n_train}, "
                f"valid={n_valid}, test={n_test} from {len(dates)} dates"
            )

        valid_start = n_train + embargo_days
        test_start = valid_start + n_valid + embargo_days
        return cls(
            train=(dates[0], dates[n_train - 1]),
            valid=(dates[valid_start], dates[valid_start + n_valid - 1]),
            test=(dates[test_start], dates[-1]),
            embargo_days=embargo_days,
        )

    def dates(self, dates: pd.DatetimeIndex, split: Split) -> pd.DatetimeIndex:
        start, end = getattr(self, split)
        return dates[(dates >= start) & (dates <= end)]

    def search_panel(self, panel: OHLCVPanel) -> OHLCVPanel:
        """The panel with the TEST period removed — all the search phase ever gets."""
        return panel.until(self.test[0])

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            name: [
                getattr(self, name)[0].date().isoformat(),
                getattr(self, name)[1].date().isoformat(),
            ]
            for name in SPLITS
        }
        out["embargo_days"] = self.embargo_days
        return out
