"""PurgedKFold on the performance matrix (QUANTIT-44).

`tests/test_cv.py` checks the splitter on a plain business-day calendar. These
check it the way strategies use it: splitting the index of a performance
matrix from `synth.make_return_matrix`, over several horizons and fold counts,
on an irregular calendar too, and end to end, by choosing the best of many
candidates on training folds and scoring it out of fold.

Convention, as in `cv.py`: the label at position i covers the returns at
positions i+1 .. i+h, the half-open interval (t, t1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.cv import PurgedKFold, label_end_times
from capstone.synth import make_return_matrix

MATRIX_DATES = make_return_matrix(n_obs=1000, n_candidates=2, seed=0).returns.index
# An irregular calendar: business days with a random fifth of them missing.
_rng = np.random.default_rng(1)
IRREGULAR = pd.bdate_range("2010-01-04", periods=1250)[_rng.random(1250) > 0.2]


def _audit(dates, horizon, n_splits, embargo_pct=0.01):
    """For each fold: train and test positions, the fold's purge zone and embargo zone."""
    cv = PurgedKFold(n_splits, horizon=horizon, embargo_pct=embargo_pct)
    n_labelled = len(dates) - horizon
    embargo = cv.embargo_size(n_labelled)
    folds = []
    for train, test in cv.split(dates):
        tr, te = dates.get_indexer(train), dates.get_indexer(test)
        assert (tr >= 0).all() and (te >= 0).all()
        s, e = te.min(), te.max()
        # Train label (i, i+h] overlaps the test labels' returns (s, e+h].
        overlap = (np.arange(n_labelled) < e + horizon) & (np.arange(n_labelled) + horizon > s)
        embargoed = (np.arange(n_labelled) >= e + horizon) & (
            np.arange(n_labelled) <= e + horizon + embargo - 1
        )
        folds.append({"train": tr, "test": te, "overlap": overlap, "embargo": embargoed})
    return folds, n_labelled


CASES = [(h, k) for h in (1, 5, 21) for k in (3, 5, 10)]


@pytest.mark.parametrize("dates", [MATRIX_DATES, IRREGULAR], ids=["matrix", "irregular"])
@pytest.mark.parametrize("horizon, n_splits", CASES)
class TestOnTheMatrixIndex:
    def test_test_folds_partition_the_labelled_dates_in_order(self, dates, horizon, n_splits):
        folds, n_labelled = _audit(dates, horizon, n_splits)
        tests = np.concatenate([f["test"] for f in folds])
        assert (tests == np.arange(n_labelled)).all()
        for f in folds:
            assert (np.diff(f["test"]) == 1).all()

    def test_no_training_label_shares_a_return_with_a_test_label(self, dates, horizon, n_splits):
        folds, _ = _audit(dates, horizon, n_splits)
        for f in folds:
            assert not f["overlap"][f["train"]].any()

    def test_the_embargo_band_after_each_fold_is_out_of_training(self, dates, horizon, n_splits):
        folds, _ = _audit(dates, horizon, n_splits)
        for f in folds:
            assert not f["embargo"][f["train"]].any()

    def test_nothing_else_is_dropped(self, dates, horizon, n_splits):
        # Purging and the embargo are the only exclusions: every other
        # labelled date trains, so the splitter wastes no data.
        folds, n_labelled = _audit(dates, horizon, n_splits)
        for f in folds:
            in_test = np.zeros(n_labelled, dtype=bool)
            in_test[f["test"]] = True
            expected = np.flatnonzero(~(in_test | f["overlap"] | f["embargo"]))
            assert (np.sort(f["train"]) == expected).all()

    def test_labels_the_splitter_trains_on_exist(self, dates, horizon, n_splits):
        # Every training date has a complete label inside the index.
        t1 = label_end_times(dates, horizon)
        for train, _ in PurgedKFold(n_splits, horizon=horizon).split(dates):
            assert train.isin(t1.index).all()


def _select_and_score(matrix, n_splits=5):
    """Pick the best training-fold Sharpe in each fold; score it on the test fold."""
    returns = matrix.returns
    in_sample, out_of_sample, picked_real = [], [], []
    for train, test in PurgedKFold(n_splits, horizon=1).split(returns.index):
        fit = returns.loc[train]
        sharpe = fit.mean() / fit.std() * np.sqrt(252)
        best = sharpe.idxmax()
        held = returns.loc[test, best]
        in_sample.append(sharpe[best])
        out_of_sample.append(held.mean() / held.std() * np.sqrt(252))
        picked_real.append(bool(matrix.truth[best]))
    return np.mean(in_sample), np.mean(out_of_sample), np.mean(picked_real)


class TestSelectionOutOfFold:
    def test_the_best_of_many_nulls_scores_near_zero_out_of_fold(self):
        # 100 null candidates, 5 years, AR(1) 0.3. The in-sample winner looks
        # like a Sharpe of about 1.7; out of fold it is noise. Over 30 seeds at
        # three ranges the out-of-fold mean was -0.26 to 0.07 (se about 0.15).
        runs = np.array(
            [
                _select_and_score(make_return_matrix(n_obs=1260, n_candidates=100, ar1=0.3, seed=s))
                for s in range(30)
            ]
        )
        assert runs[:, 0].mean() > 1.2
        assert abs(runs[:, 1].mean()) < 0.45

    def test_a_planted_signal_survives_out_of_fold(self):
        # 5 real candidates at Sharpe 2.0 among 100: chosen in about 97% of
        # folds, out-of-fold Sharpe 1.7 to 2.3 over 30 seeds.
        runs = np.array(
            [
                _select_and_score(
                    make_return_matrix(
                        n_obs=1260, n_candidates=100, n_real=5, sharpe_real=2.0, ar1=0.3, seed=s
                    )
                )
                for s in range(30)
            ]
        )
        assert runs[:, 1].mean() > 1.0
        assert runs[:, 2].mean() > 0.9
