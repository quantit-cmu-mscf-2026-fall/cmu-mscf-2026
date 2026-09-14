"""Tests for capstone.alpha_gpt.panel and capstone.alpha_gpt.synth_ohlcv."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.alpha_gpt.panel import OHLCVPanel
from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel

FIELDS = {"open", "high", "low", "close", "volume", "vwap", "returns"}


def _next_day_rank_ic_tstat(signal: pd.DataFrame, returns: pd.DataFrame) -> float:
    """t-stat of the daily Spearman IC between signal[t] and returns[t+1]."""
    ic = signal.rank(axis=1).corrwith(returns.shift(-1).rank(axis=1), axis=1).dropna()
    return float(ic.mean() / ic.std() * np.sqrt(len(ic)))


@pytest.fixture(scope="module")
def null_panel():
    return make_ohlcv_panel(n_dates=500, n_assets=60, pattern="none", seed=0)


class TestSyntheticShape:
    def test_fields_groups_and_alignment(self, null_panel):
        assert set(null_panel.fields) == FIELDS
        assert null_panel.dates.shape == (500,)
        assert list(null_panel.assets[:2]) == ["S000", "S001"]
        for frame in null_panel.fields.values():
            assert frame.shape == (500, 60)
        assert null_panel.groups["sector"].index.equals(null_panel.assets)

    def test_ohlc_ordering(self, null_panel):
        f = null_panel.fields
        body_low = np.minimum(f["open"], f["close"])
        body_high = np.maximum(f["open"], f["close"])
        assert (f["low"] <= body_low).all().all()
        assert (body_high <= f["high"]).all().all()

    def test_volume_positive_and_returns_from_close(self, null_panel):
        f = null_panel.fields
        assert (f["volume"] > 0).all().all()
        pd.testing.assert_frame_equal(f["returns"], f["close"].pct_change())
        assert f["returns"].iloc[0].isna().all()

    def test_same_seed_same_panel_different_seed_different(self):
        a = make_ohlcv_panel(n_dates=80, n_assets=10, seed=3)
        b = make_ohlcv_panel(n_dates=80, n_assets=10, seed=3)
        c = make_ohlcv_panel(n_dates=80, n_assets=10, seed=4)
        for name in FIELDS:
            pd.testing.assert_frame_equal(a.fields[name], b.fields[name])
        assert not a.fields["close"].equals(c.fields["close"])

    def test_descriptor_records_provenance(self):
        panel = make_ohlcv_panel(n_dates=50, n_assets=5, pattern="reversal", strength=0.1, seed=7)
        assert panel.descriptor["pattern"] == "reversal"
        assert panel.descriptor["strength"] == 0.1
        assert panel.descriptor["seed"] == 7

    @pytest.mark.parametrize(
        "kwargs",
        [{"pattern": "momentum"}, {"strength": 1.0}, {"strength": -0.1}, {"n_assets": 1}],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            make_ohlcv_panel(n_dates=50, **kwargs)


class TestPlantedPattern:
    def test_reversal_is_detectable(self):
        panel = make_ohlcv_panel(
            n_dates=500, n_assets=100, pattern="reversal", strength=0.05, seed=1
        )
        returns = panel.fields["returns"]
        assert _next_day_rank_ic_tstat(-returns, returns) > 5

    def test_null_panel_has_no_reversal(self, null_panel):
        returns = null_panel.fields["returns"]
        assert abs(_next_day_rank_ic_tstat(-returns, returns)) < 3


class TestPanelContainer:
    def _frames(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        return dates, returns

    def test_requires_returns_field(self):
        _, returns = self._frames()
        with pytest.raises(ValueError, match="'returns'"):
            OHLCVPanel(fields={"close": returns})

    def test_rejects_misaligned_fields(self):
        _, returns = self._frames()
        with pytest.raises(ValueError, match="not aligned"):
            OHLCVPanel(fields={"returns": returns, "close": returns.iloc[:-1]})

    def test_rejects_non_datetime_or_unsorted_index(self):
        _, returns = self._frames()
        with pytest.raises(ValueError, match="DatetimeIndex"):
            OHLCVPanel(fields={"returns": returns.reset_index(drop=True)})
        with pytest.raises(ValueError, match="increasing"):
            OHLCVPanel(fields={"returns": returns.iloc[::-1]})

    def test_rejects_group_missing_an_asset(self):
        _, returns = self._frames()
        with pytest.raises(ValueError, match="no label"):
            OHLCVPanel(fields={"returns": returns}, groups={"sector": pd.Series({"A": "x"})})

    def test_until_drops_the_cutoff_and_everything_after(self, null_panel):
        cutoff = null_panel.dates[300]
        head = null_panel.until(cutoff)
        assert head.dates.max() < cutoff
        assert len(head.dates) == 300
        for name, frame in head.fields.items():
            pd.testing.assert_frame_equal(frame, null_panel.fields[name].iloc[:300])
