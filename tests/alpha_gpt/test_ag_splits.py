"""Tests for capstone.alpha_gpt.splits."""

from __future__ import annotations

import pandas as pd
import pytest

from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel

DATES = pd.bdate_range("2015-01-02", periods=1260)


class TestFromFractions:
    def test_ordered_disjoint_with_exact_embargo_gaps(self):
        spec = SplitSpec.from_fractions(DATES, train_frac=0.6, valid_frac=0.2, embargo_days=5)
        train = spec.dates(DATES, "train")
        valid = spec.dates(DATES, "valid")
        test = spec.dates(DATES, "test")

        assert train[0] == DATES[0] and test[-1] == DATES[-1]
        assert DATES.get_loc(valid[0]) - DATES.get_loc(train[-1]) == 6
        assert DATES.get_loc(test[0]) - DATES.get_loc(valid[-1]) == 6
        assert len(train) + len(valid) + len(test) + 10 == len(DATES)
        assert len(train) == int(1250 * 0.6)

    def test_as_dict_is_iso_dates(self):
        spec = SplitSpec.from_fractions(DATES)
        out = spec.as_dict()
        assert out["train"][0] == "2015-01-02"
        assert out["embargo_days"] == 5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"embargo_days": 0},
            {"train_frac": 0.9, "valid_frac": 0.1},
            {"train_frac": 0.0},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            SplitSpec.from_fractions(DATES, **kwargs)

    def test_too_few_dates_for_a_split(self):
        with pytest.raises(ValueError, match=">= 60 dates"):
            SplitSpec.from_fractions(DATES[:200])


class TestDirectConstruction:
    def test_overlapping_splits_raise(self):
        with pytest.raises(ValueError, match="ordered"):
            SplitSpec(
                train=(DATES[0], DATES[100]),
                valid=(DATES[100], DATES[200]),
                test=(DATES[210], DATES[300]),
                embargo_days=1,
            )

    def test_zero_embargo_raises(self):
        with pytest.raises(ValueError, match="embargo"):
            SplitSpec(
                train=(DATES[0], DATES[99]),
                valid=(DATES[100], DATES[199]),
                test=(DATES[200], DATES[300]),
                embargo_days=0,
            )


def test_search_panel_has_no_test_dates_but_keeps_the_embargo():
    panel = make_ohlcv_panel(n_dates=400, n_assets=10, seed=0)
    spec = SplitSpec.from_fractions(panel.dates, embargo_days=3)
    search = spec.search_panel(panel)

    assert search.dates.max() < spec.test[0]
    # The day after VALIDATION's last signal is present: its return is needed to score it.
    assert search.dates.max() > spec.valid[1]
