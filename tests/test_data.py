"""Tests for capstone.data. Network-dependent — skip cleanly offline."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from capstone.data import load_french, load_industry_returns

pytestmark = pytest.mark.network


def _load_or_skip(loader, *args, **kwargs):
    try:
        return loader(*args, **kwargs)
    except requests.RequestException as exc:
        pytest.skip(f"network unavailable: {exc}")


def test_load_french_factors_daily_shape_and_columns():
    frame = _load_or_skip(load_french, "factors_daily")

    assert not frame.empty
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert "Mkt-RF" in frame.columns
    assert "RF" in frame.columns
    # Catches a missing percent-to-decimal conversion.
    assert frame["Mkt-RF"].abs().max() < 1.0


def test_load_french_missing_value_sentinel_is_gone():
    frame = _load_or_skip(load_french, "factors_daily")

    assert not (frame == -0.9999).to_numpy().any()
    assert frame["Mkt-RF"].min() > -0.99


def test_load_french_index_strictly_increasing_no_duplicates():
    frame = _load_or_skip(load_french, "factors_daily")

    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates


def test_load_industry_returns_12_columns():
    frame = _load_or_skip(load_industry_returns, 12)
    assert frame.shape[1] == 12
