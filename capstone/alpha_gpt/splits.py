"""DEVELOPMENT / TEST date ranges, separated by an embargo.

TEST is evaluated once, at the very end (docs/experiments/alpha_gpt.md).
Everything before it is DEVELOPMENT, which `capstone.cv.PurgedKFold` divides
into TRAIN (where a discovery procedure searches, takes feedback and selects)
and VALIDATION (held-out folds that measure how that procedure does out of
sample). See `capstone/alpha_gpt/loop.py`.

A label at date t is realised over the next `horizon` returns, so the embargo
between DEVELOPMENT and TEST must be at least the horizon; otherwise the last
development labels would read TEST returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from capstone.alpha_gpt.panel import OHLCVPanel

Split = Literal["development", "test"]
SPLITS: tuple[Split, ...] = ("development", "test")

#: Fewest dates a split may have; `backtest.summarize` refuses shorter series.
MIN_SPLIT_DAYS = 60


@dataclass(frozen=True)
class SplitSpec:
    """Inclusive (start, end) date ranges for DEVELOPMENT and TEST."""

    development: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]
    embargo_days: int

    def __post_init__(self) -> None:
        if self.embargo_days < 1:
            raise ValueError("embargo_days must be >= 1: a signal at t is scored on t+1")
        for name in SPLITS:
            start, end = getattr(self, name)
            if start > end:
                raise ValueError(f"{name} starts after it ends")
        if not self.development[1] < self.test[0]:
            raise ValueError("splits must be ordered development < test without overlap")

    @classmethod
    def from_fractions(
        cls,
        dates: pd.DatetimeIndex,
        *,
        test_frac: float = 0.2,
        embargo_days: int = 5,
    ) -> SplitSpec:
        """DEVELOPMENT then TEST, with `embargo_days` dates dropped between them."""
        if embargo_days < 1:
            raise ValueError("embargo_days must be >= 1: a signal at t is scored on t+1")
        if not 0 < test_frac < 1:
            raise ValueError("test_frac must be in (0, 1)")

        usable = len(dates) - embargo_days
        n_test = int(usable * test_frac)
        n_development = usable - n_test
        if min(n_development, n_test) < MIN_SPLIT_DAYS:
            raise ValueError(
                f"each split needs >= {MIN_SPLIT_DAYS} dates; got development={n_development}, "
                f"test={n_test} from {len(dates)} dates"
            )
        return cls(
            development=(dates[0], dates[n_development - 1]),
            test=(dates[n_development + embargo_days], dates[-1]),
            embargo_days=embargo_days,
        )

    def check_horizon(self, horizon: int) -> None:
        """Refuse a label horizon whose last DEVELOPMENT labels would reach into TEST."""
        if horizon > self.embargo_days:
            raise ValueError(
                f"label horizon {horizon} exceeds embargo_days {self.embargo_days}: "
                "the last development labels would read TEST returns"
            )

    def dates(self, dates: pd.DatetimeIndex, split: Split) -> pd.DatetimeIndex:
        start, end = getattr(self, split)
        return dates[(dates >= start) & (dates <= end)]

    def search_panel(self, panel: OHLCVPanel) -> OHLCVPanel:
        """The panel with the TEST period removed — all a discovery procedure ever gets."""
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
