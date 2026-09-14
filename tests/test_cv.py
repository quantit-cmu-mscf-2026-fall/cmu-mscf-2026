"""Tests for capstone.cv: purging, embargo, uniqueness weights, and the leak they prevent."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from capstone.cv import PurgedKFold, average_uniqueness, label_end_times

DATES = pd.bdate_range("2015-01-02", periods=500)


class TestLabelEndTimes:
    def test_horizon_one_maps_to_next_date(self):
        t1 = label_end_times(DATES, 1)
        assert len(t1) == len(DATES) - 1
        assert t1.iloc[0] == DATES[1]
        assert t1.index[-1] == DATES[-2]

    def test_longer_horizon(self):
        t1 = label_end_times(DATES, 5)
        assert t1.iloc[10] == DATES[15]
        assert len(t1) == len(DATES) - 5

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5])
    def test_invalid_horizon(self, bad):
        with pytest.raises(ValueError, match="horizon"):
            label_end_times(DATES, bad)


def _splits(horizon, n_splits=5, embargo_pct=0.01):
    cv = PurgedKFold(n_splits, horizon=horizon, embargo_pct=embargo_pct)
    return cv, list(cv.split(DATES))


class TestPurgedKFold:
    @pytest.mark.parametrize("horizon", [1, 5, 20])
    def test_test_folds_are_contiguous_and_cover_every_labelled_date_once(self, horizon):
        _, splits = _splits(horizon)
        tests = [test for _, test in splits]
        labelled = label_end_times(DATES, horizon).index
        assert pd.DatetimeIndex(np.concatenate(tests)).equals(labelled)
        for test in tests:
            positions = labelled.get_indexer(test)
            assert (np.diff(positions) == 1).all()

    @pytest.mark.parametrize("horizon", [1, 5, 20])
    def test_no_training_label_shares_a_return_with_a_test_label(self, horizon):
        t1 = label_end_times(DATES, horizon)
        position = pd.Series(np.arange(len(DATES)), index=DATES)
        _, splits = _splits(horizon)
        for train, test in splits:
            assert train.intersection(test).empty
            test_returns = set()
            for t in test:
                test_returns.update(range(position[t] + 1, position[t1[t]] + 1))
            for t in train:
                train_returns = range(position[t] + 1, position[t1[t]] + 1)
                assert test_returns.isdisjoint(train_returns), t

    def test_horizon_one_purges_nothing_before_the_fold(self):
        cv, splits = _splits(1)
        train, test = splits[2]
        before = DATES[DATES.get_loc(test[0]) - 1]
        assert before in train

    def test_longer_horizon_purges_h_minus_one_dates_before_the_fold(self):
        _, splits = _splits(5)
        train, test = splits[2]
        start = DATES.get_loc(test[0])
        assert DATES[start - 5] in train
        for k in range(1, 5):
            assert DATES[start - k] not in train

    def test_embargo_follows_the_fold_only(self):
        cv, splits = _splits(1, embargo_pct=0.05)
        train, test = splits[2]
        embargo = cv.embargo_size(len(DATES) - 1)
        assert embargo == math.ceil(0.05 * 499)
        label_end = DATES.get_loc(test[-1]) + 1
        dropped_after = DATES[label_end : label_end + embargo]
        assert train.intersection(dropped_after).empty
        assert DATES[label_end + embargo] in train

    def test_embargo_is_at_least_the_horizon(self):
        assert PurgedKFold(4, horizon=10, embargo_pct=0.0).embargo_size(1000) == 10

    @pytest.mark.parametrize(
        "kwargs",
        [{"n_splits": 1}, {"horizon": 0}, {"embargo_pct": 1.0}, {"embargo_pct": -0.1}],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            PurgedKFold(**kwargs)

    def test_too_few_dates(self):
        with pytest.raises(ValueError, match="too few"):
            list(PurgedKFold(5).split(DATES[:8]))


class TestAverageUniqueness:
    def test_horizon_one_gives_unit_weights(self):
        t1 = label_end_times(DATES, 1)
        assert (average_uniqueness(t1, DATES) == 1.0).all()

    def test_overlapping_labels_weigh_about_one_over_h_inside_the_sample(self):
        t1 = label_end_times(DATES, 5)
        weights = average_uniqueness(t1, DATES)
        assert weights.iloc[100:-100].to_numpy() == pytest.approx(0.2)
        assert weights.iloc[0] > 0.2  # near the edge fewer labels overlap
        assert ((weights > 0) & (weights <= 1)).all()

    def test_hand_computed_example(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        # label A covers returns on d1,d2; label B covers d2 only.
        t1 = pd.Series([dates[2], dates[2]], index=[dates[0], dates[1]])
        weights = average_uniqueness(t1, dates)
        assert weights.iloc[0] == pytest.approx((1.0 + 0.5) / 2)
        assert weights.iloc[1] == pytest.approx(0.5)

    def test_labels_must_lie_on_dates(self):
        t1 = pd.Series([pd.Timestamp("2030-01-01")], index=[DATES[0]])
        with pytest.raises(ValueError):
            average_uniqueness(t1, DATES)


def _slow_features_and_labels(seed, n_dates=600, n_features=5, horizon=20, phi=0.995):
    """Slow-moving features and h-day forward returns of pure noise: nothing is predictable."""
    rng = np.random.default_rng(seed)
    features = np.empty((n_dates, n_features))
    features[0] = rng.standard_normal(n_features)
    for t in range(1, n_dates):
        shock = rng.standard_normal(n_features)
        features[t] = phi * features[t - 1] + math.sqrt(1 - phi**2) * shock
    cumulative = np.concatenate([[0.0], np.cumsum(rng.standard_normal(n_dates))])
    labels = cumulative[1 + horizon :] - cumulative[1 : n_dates + 1 - horizon]
    dates = pd.bdate_range("2015-01-02", periods=n_dates)
    return dates, features[: n_dates - horizon], labels


def _naive_kfold(dates, n_splits, horizon):
    labelled = label_end_times(dates, horizon).index
    for fold in np.array_split(np.arange(len(labelled)), n_splits):
        test = labelled[fold]
        yield labelled.difference(test), test


def _weighted_corr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    var_x = np.average((x - mx) ** 2, weights=w)
    var_y = np.average((y - my) ** 2, weights=w)
    return cov / math.sqrt(var_x * var_y)


def _knn_out_of_fold_scores(splits, dates, features, labels, weights, k=5):
    """Fit k-nearest-neighbours on each training set; score its predictions on the fold.

    Sample weights are used on both sides: to average neighbour labels when fitting
    and to weight the correlation when scoring.
    """
    labelled = dates[: len(labels)]
    scores = []
    for train, test in splits:
        tr, te = labelled.get_indexer(train), labelled.get_indexer(test)
        distance = ((features[te, None, :] - features[None, tr, :]) ** 2).sum(axis=2)
        nearest = np.argsort(distance, axis=1)[:, :k]
        neighbour_weight = weights[tr][nearest]
        prediction = (labels[tr][nearest] * neighbour_weight).sum(axis=1) / neighbour_weight.sum(
            axis=1
        )
        scores.append(_weighted_corr(prediction, labels[te], weights[te]))
    return np.array(scores)


def test_naive_kfold_leaks_and_purged_kfold_does_not():
    """AFML ch. 7 in one experiment, on a target nothing can predict.

    A nearest-neighbours model on slow-moving features finds, for each test date,
    training dates next to it in time. Under naive k-fold those neighbours' 20-day
    labels share most of their returns with the test label, so the model looks
    predictive out of fold. Purging and embargoing remove exactly those neighbours.
    """
    horizon, n_splits = 20, 20
    naive, purged = [], []
    for seed in range(5):
        dates, features, labels = _slow_features_and_labels(seed, horizon=horizon)
        weights = average_uniqueness(label_end_times(dates, horizon), dates).to_numpy()
        naive.append(
            _knn_out_of_fold_scores(
                _naive_kfold(dates, n_splits, horizon), dates, features, labels, weights
            )
        )
        cv = PurgedKFold(n_splits, horizon=horizon, embargo_pct=0.01)
        purged.append(_knn_out_of_fold_scores(cv.split(dates), dates, features, labels, weights))

    def tstat(scores):
        scores = np.concatenate(scores)
        return scores.mean() / scores.std(ddof=1) * math.sqrt(len(scores))

    assert tstat(naive) > 3
    assert abs(tstat(purged)) < 3
    assert np.concatenate(naive).mean() > np.concatenate(purged).mean()
