"""Tests for capstone.cv.WalkForward (QUANTIT-45).

Convention, as in `cv.py`: the label at position i covers the returns at
positions i+1 .. i+h, the half-open interval (t, t1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.cv import WalkForward, label_end_times
from capstone.synth import make_return_matrix

DATES = make_return_matrix(n_obs=1000, n_candidates=2, seed=0).returns.index
_rng = np.random.default_rng(1)
IRREGULAR = pd.bdate_range("2010-01-04", periods=1250)[_rng.random(1250) > 0.2]

CASES = [
    {"horizon": h, "gap": g, "train_size": ts}
    for h in (1, 5, 21)
    for g in (0, 3)
    for ts in (None, 250)
]


def _positions(dates, splits):
    return [(dates.get_indexer(tr), dates.get_indexer(te)) for tr, te in splits]


@pytest.mark.parametrize("dates", [DATES, IRREGULAR], ids=["matrix", "irregular"])
@pytest.mark.parametrize(
    "kwargs", CASES, ids=lambda k: "h{horizon}-gap{gap}-train{train_size}".format(**k)
)
class TestWalkForward:
    def test_training_labels_are_realised_before_the_test_window(self, dates, kwargs):
        # Train label (i, i+h] must end by the window's first date s: i+h <= s.
        h = kwargs["horizon"]
        for tr, te in _positions(dates, WalkForward(5, **kwargs).split(dates)):
            assert tr.max() < te.min()
            assert tr.max() + h <= te.min()

    def test_the_gap_is_exact(self, dates, kwargs):
        # The last training date sits exactly horizon + gap before the window.
        h, g = kwargs["horizon"], kwargs["gap"]
        for tr, te in _positions(dates, WalkForward(5, **kwargs).split(dates)):
            assert te.min() - tr.max() == h + g

    def test_windows_move_forward_and_tile_the_end_of_the_sample(self, dates, kwargs):
        splits = _positions(dates, WalkForward(5, **kwargs).split(dates))
        tests = np.concatenate([te for _, te in splits])
        n_labelled = len(dates) - kwargs["horizon"]
        assert (np.diff(tests) == 1).all()  # consecutive, no overlap, no holes
        assert tests[-1] == n_labelled - 1  # the last window ends at the last label
        ends = [tr.max() for tr, _ in splits]
        assert all(b > a for a, b in zip(ends, ends[1:], strict=False))

    def test_expanding_or_rolling(self, dates, kwargs):
        splits = _positions(dates, WalkForward(5, **kwargs).split(dates))
        for tr, _ in splits:
            assert (np.diff(tr) == 1).all()  # one contiguous block of the past
        if kwargs["train_size"] is None:
            assert all(tr[0] == 0 for tr, _ in splits)
            sizes = [len(tr) for tr, _ in splits]
            assert all(b > a for a, b in zip(sizes, sizes[1:], strict=False))
        else:
            # At most train_size; a full window once enough history exists.
            for tr, _ in splits:
                assert len(tr) == min(kwargs["train_size"], tr.max() + 1)
            starts = [tr[0] for tr, _ in splits]
            assert all(b >= a for a, b in zip(starts, starts[1:], strict=False))
            assert starts[-1] > starts[0]

    def test_every_date_has_a_complete_label(self, dates, kwargs):
        t1 = label_end_times(dates, kwargs["horizon"])
        for tr, te in WalkForward(5, **kwargs).split(dates):
            assert tr.isin(t1.index).all() and te.isin(t1.index).all()


class TestSizes:
    def test_default_test_size(self):
        n_labelled = len(DATES) - 1
        splits = list(WalkForward(4).split(DATES))
        assert all(len(te) == n_labelled // 5 for _, te in splits)

    def test_min_train_size_can_require_a_full_rolling_window(self):
        # 999 labels, 5 windows of 166: the first has 169 dates before it.
        with pytest.raises(ValueError, match="training dates"):
            list(WalkForward(5, train_size=250, min_train_size=250).split(DATES))

    def test_explicit_test_size(self):
        splits = list(WalkForward(3, test_size=40).split(DATES))
        assert [len(te) for _, te in splits] == [40, 40, 40]

    def test_too_few_dates(self):
        with pytest.raises(ValueError, match="too few"):
            list(WalkForward(10, test_size=200).split(DATES))

    def test_min_train_size_is_enforced(self):
        # With 999 labels, 5 windows of 190 leave 49 dates before the first;
        # minus the 1-date purge and a 10-date gap, 39 can train.
        with pytest.raises(ValueError, match="training dates"):
            list(WalkForward(5, test_size=190, gap=10, min_train_size=100).split(DATES))
        assert len(list(WalkForward(5, test_size=190, gap=10, min_train_size=39).split(DATES))) == 5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_splits": 0},
            {"horizon": 0},
            {"gap": -1},
            {"test_size": 0},
            {"train_size": 0},
            {"min_train_size": 0},
            {"horizon": True},
            {"train_size": 1.5},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            WalkForward(**kwargs)

    def test_dates_must_be_increasing(self):
        with pytest.raises(ValueError, match="increasing"):
            list(WalkForward(3).split(DATES[::-1]))


def _walk_forward_selection(matrix, **kwargs):
    """Pick the best training Sharpe before each window; score it on the window."""
    returns = matrix.returns
    in_sample, out_of_sample = [], []
    for train, test in WalkForward(5, **kwargs).split(returns.index):
        fit = returns.loc[train]
        sharpe = fit.mean() / fit.std() * np.sqrt(252)
        best = sharpe.idxmax()
        held = returns.loc[test, best]
        in_sample.append(sharpe[best])
        out_of_sample.append(held.mean() / held.std() * np.sqrt(252))
    return np.mean(in_sample), np.mean(out_of_sample)


def test_the_best_of_many_nulls_scores_near_zero_going_forward():
    # 100 null candidates, 5 years, AR(1) 0.3, expanding window. The winner
    # looks like a Sharpe of about 2.5 on its training window; over 30 seeds
    # at three ranges its forward mean was -0.10 to 0.15 (se 0.11-0.14).
    runs = np.array(
        [
            _walk_forward_selection(
                make_return_matrix(n_obs=1260, n_candidates=100, ar1=0.3, seed=s)
            )
            for s in range(30)
        ]
    )
    assert runs[:, 0].mean() > 1.2
    assert abs(runs[:, 1].mean()) < 0.45
