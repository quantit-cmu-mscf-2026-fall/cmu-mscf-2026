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

Which splitter:

- `PurgedKFold` when data is short and the model has no memory of time: every
  date is tested once, and training may use dates on both sides of the test
  fold, made safe by purging and the embargo.
- `WalkForward` when the strategy must only ever learn from the past, or to
  simulate how it would be refitted live: each test window follows its
  training window, which either expands (all the past) or rolls (a fixed
  look-back). It uses fewer test dates, and the early windows train on less.

Either way the out-of-fold returns are what go into the performance matrix
that validation judges (`docs/validation.md`); validation never calls a
splitter. A strategy with nothing fitted needs neither. PBO's combinatorial
splits (`pbo.py`) answer a different question and are kept separate.
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


class WalkForward:
    """Train only on the past, test on the next window, then step forward.

    The last `n_splits * test_size` labelled dates form `n_splits` consecutive
    test windows. Each window's training dates end where their labels stop
    touching the window: a label at t covers (t, t1], so a training label must
    be realised by the window's first date. That drops the `horizon` dates
    just before the window, plus `gap` more if asked. Nothing after the test
    window trains, so no embargo is needed.

    Args:
        n_splits: number of test windows (>= 1).
        test_size: labelled dates per test window. Default:
            n_labelled // (n_splits + 1), so the first window has at least as
            much history before it as one test window.
        train_size: None for an expanding window (all eligible past dates), or
            a rolling window of at most this many labelled dates. Early
            windows with less history use what exists; set `min_train_size`
            to require a full window.
        horizon: label horizon in observations (see module docstring).
        gap: extra labelled dates dropped between training and test, beyond
            the purge, for features that look back further than the label.
        min_train_size: raise if any training window has fewer dates.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: int | None = None,
        train_size: int | None = None,
        horizon: int = 1,
        gap: int = 0,
        min_train_size: int = 1,
    ) -> None:
        for name, value, low in (
            ("n_splits", n_splits, 1),
            ("horizon", horizon, 1),
            ("gap", gap, 0),
            ("min_train_size", min_train_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{name} must be an int >= {low}, got {value!r}")
        for name, value in (("test_size", test_size), ("train_size", train_size)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be None or an int >= 1, got {value!r}")
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.horizon = horizon
        self.gap = gap
        self.min_train_size = min_train_size

    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield `(train_dates, test_dates)` for each window, in chronological order."""
        dates = pd.DatetimeIndex(dates)
        if not (dates.is_monotonic_increasing and dates.is_unique):
            raise ValueError("dates must be unique and increasing")
        labelled = label_end_times(dates, self.horizon).index
        n = len(labelled)
        test_size = self.test_size if self.test_size is not None else n // (self.n_splits + 1)
        first_test = n - self.n_splits * test_size
        if test_size < 1 or first_test < 1:
            raise ValueError(
                f"too few labelled dates ({n}) for {self.n_splits} windows of {test_size}"
            )
        for k in range(self.n_splits):
            start = first_test + k * test_size
            # Label i (a position in `labelled`, equal to its position in
            # `dates`) is realised at position i + horizon; it may train only
            # if i + horizon <= start, and `gap` drops more before the window.
            train_end = start - self.horizon - self.gap  # inclusive
            train_start = 0 if self.train_size is None else max(0, train_end - self.train_size + 1)
            n_train = train_end - train_start + 1
            if n_train < self.min_train_size:
                raise ValueError(
                    f"window {k} has {max(n_train, 0)} training dates; "
                    f"min_train_size is {self.min_train_size}"
                )
            yield labelled[train_start : train_end + 1], labelled[start : start + test_size]


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
