"""Purged, embargoed k-fold cross-validation for time-series labels.

Follows López de Prado, *Advances in Financial Machine Learning* (2018), ch. 7,
with sample weights from ch. 4. Plain k-fold assumes observations are IID. Two
things break that on return data, and each leaks test information into training:

1. **Labels span time.** A label observed at date t is realised over the next
   `horizon` returns, so a training label that ends inside the test fold shares
   returns with test labels. *Purging* drops every training date whose label
   interval overlaps any test label interval.
2. **Features are serially correlated.** Observations right after the test fold
   are built from data that overlaps it. The *embargo* also drops a band of
   training dates immediately after each test fold.

Overlapping labels are also not independent observations. `average_uniqueness`
weights each label by how much of its interval it shares with no other label;
use the same weights when selecting on training folds and when scoring test
folds, or the two are measuring different things.

Convention: the label at date t is realised over the returns on dates
t+1 .. t1 (t1 = t + horizon observations), i.e. the half-open interval (t, t1].
With horizon 1, no two labels share a return, so every weight is 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd


def label_end_times(dates: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """Map each labellable date t to t1, the date its `horizon`-step label is realised.

    The last `horizon` dates have no complete label and are omitted.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be a positive int, got {horizon!r}")
    if len(dates) <= horizon:
        raise ValueError(f"need more than {horizon} dates for horizon {horizon}")
    return pd.Series(dates[horizon:], index=dates[:-horizon])


class PurgedKFold:
    """k contiguous test folds; training dates purged of label overlap and embargoed.

    Args:
        n_splits: number of folds (>= 2).
        horizon: label horizon in observations (see module docstring).
        embargo_pct: share of labelled dates embargoed after each test fold. The
            embargo is never shorter than `horizon`.
    """

    def __init__(self, n_splits: int = 4, *, horizon: int = 1, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError(f"horizon must be a positive int, got {horizon!r}")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_pct = embargo_pct

    def embargo_size(self, n_labelled: int) -> int:
        return max(self.horizon, math.ceil(self.embargo_pct * n_labelled))

    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield `(train_dates, test_dates)` for each fold, in chronological fold order."""
        dates = pd.DatetimeIndex(dates)
        if not (dates.is_monotonic_increasing and dates.is_unique):
            raise ValueError("dates must be unique and increasing")
        t1 = label_end_times(dates, self.horizon)
        labelled = t1.index
        if len(labelled) < 2 * self.n_splits:
            raise ValueError(f"too few labelled dates ({len(labelled)}) for {self.n_splits} folds")

        label_end = t1.to_numpy()
        embargo = self.embargo_size(len(labelled))
        positions = pd.Series(np.arange(len(dates)), index=dates)

        for fold in np.array_split(np.arange(len(labelled)), self.n_splits):
            test_dates = labelled[fold]
            test_start = test_dates[0]
            test_label_end = label_end[fold].max()

            # (t, t1] overlaps (test_start, test_label_end]
            #   <=>  t < test_label_end  and  t1 > test_start
            overlaps = (labelled < test_label_end) & (label_end > test_start)

            # Embargo `embargo` dates starting at the test labels' last return date:
            # a feature observed there already contains that return (AFML's
            # train indices resume at searchsorted(max t1) + embargo).
            end_pos = positions[test_label_end]
            embargo_end = dates[min(end_pos + embargo - 1, len(dates) - 1)]
            embargoed = (labelled >= test_label_end) & (labelled <= embargo_end)

            in_test = np.zeros(len(labelled), dtype=bool)
            in_test[fold] = True
            train = labelled[~(overlaps | embargoed | in_test)]
            yield train, test_dates


def average_uniqueness(t1: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Per-label sample weight: mean over the label's returns of 1 / concurrency.

    Concurrency of a return date is the number of labels whose interval (t, t1]
    contains it. A label that shares none of its returns has weight 1; a label on
    a stretch where `h` labels overlap everywhere has weight about 1/h.

    Args:
        t1: label end times, e.g. from `label_end_times`, indexed by label date.
        dates: the full date index both `t1.index` and `t1` values belong to.
    """
    dates = pd.DatetimeIndex(dates)
    position = pd.Series(np.arange(len(dates)), index=dates)
    start = position.reindex(t1.index).to_numpy()
    end = position.reindex(pd.DatetimeIndex(t1.to_numpy())).to_numpy()
    if np.isnan(start).any() or np.isnan(end).any() or (end <= start).any():
        raise ValueError("every label must start and end on `dates`, with t1 after t")
    start, end = start.astype(int), end.astype(int)

    # Label i covers return positions start+1 .. end (inclusive).
    delta = np.zeros(len(dates) + 1)
    np.add.at(delta, start + 1, 1.0)
    np.add.at(delta, end + 1, -1.0)
    concurrency = np.cumsum(delta)[: len(dates)]

    inverse = np.where(concurrency > 0, 1.0 / np.where(concurrency > 0, concurrency, 1.0), 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])
    weights = (cumulative[end + 1] - cumulative[start + 1]) / (end - start)
    return pd.Series(weights, index=t1.index)
