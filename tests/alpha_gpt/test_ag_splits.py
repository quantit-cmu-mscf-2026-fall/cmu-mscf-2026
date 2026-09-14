"""Tests for capstone.alpha_gpt.splits."""

from __future__ import annotations

import pandas as pd
import pytest

from capstone.alpha_gpt.splits import SplitSpec
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel

DATES = pd.bdate_range("2015-01-02", periods=1260)


class TestFromFractions:
    def test_development_then_test_with_an_exact_embargo_gap(self):
        spec = SplitSpec.from_fractions(DATES, test_frac=0.2, embargo_days=5)
        development = spec.dates(DATES, "development")
        test = spec.dates(DATES, "test")

        assert development[0] == DATES[0] and test[-1] == DATES[-1]
        assert DATES.get_loc(test[0]) - DATES.get_loc(development[-1]) == 6
        assert len(development) + len(test) + 5 == len(DATES)
        assert len(test) == int(1255 * 0.2)

    def test_as_dict_is_iso_dates(self):
        out = SplitSpec.from_fractions(DATES).as_dict()
        assert set(out) == {"development", "test", "embargo_days"}
        assert out["development"][0] == "2015-01-02"
        assert out["embargo_days"] == 5

    @pytest.mark.parametrize(
        "kwargs", [{"embargo_days": 0}, {"test_frac": 0.0}, {"test_frac": 1.0}]
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            SplitSpec.from_fractions(DATES, **kwargs)

    def test_too_few_dates_for_a_split(self):
        with pytest.raises(ValueError, match=">= 60 dates"):
            SplitSpec.from_fractions(DATES[:100])


class TestDirectConstruction:
    def test_overlapping_splits_raise(self):
        with pytest.raises(ValueError, match="ordered"):
            SplitSpec(
                development=(DATES[0], DATES[200]),
                test=(DATES[200], DATES[300]),
                embargo_days=1,
            )

    def test_zero_embargo_raises(self):
        with pytest.raises(ValueError, match="embargo"):
            SplitSpec(
                development=(DATES[0], DATES[199]),
                test=(DATES[200], DATES[300]),
                embargo_days=0,
            )


def test_label_horizon_longer_than_the_embargo_is_refused():
    spec = SplitSpec.from_fractions(DATES, embargo_days=5)
    spec.check_horizon(5)
    with pytest.raises(ValueError, match="exceeds embargo_days"):
        spec.check_horizon(6)


def test_search_panel_has_no_test_dates_but_keeps_the_embargo():
    panel = make_ohlcv_panel(n_dates=400, n_assets=10, seed=0)
    spec = SplitSpec.from_fractions(panel.dates, embargo_days=3)
    search = spec.search_panel(panel)

    assert search.dates.max() < spec.test[0]
    # The days after DEVELOPMENT's last signal are present: their returns score it.
    assert search.dates.max() > spec.development[1]
