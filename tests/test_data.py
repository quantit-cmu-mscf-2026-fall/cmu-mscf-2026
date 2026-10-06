"""Tests for capstone.data. Network-marked tests skip cleanly offline."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from capstone import data
from capstone.data import load_ff5_momentum, load_french, load_industry_returns


def _load_or_skip(loader, *args, **kwargs):
    try:
        return loader(*args, **kwargs)
    except requests.RequestException as exc:
        pytest.skip(f"network unavailable: {exc}")


@pytest.mark.network
def test_load_french_factors_daily_shape_and_columns():
    frame = _load_or_skip(load_french, "factors_daily")

    assert not frame.empty
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert "Mkt-RF" in frame.columns
    assert "RF" in frame.columns
    # Catches a missing percent-to-decimal conversion.
    assert frame["Mkt-RF"].abs().max() < 1.0


@pytest.mark.network
def test_load_french_missing_value_sentinel_is_gone():
    frame = _load_or_skip(load_french, "factors_daily")

    assert not (frame == -0.9999).to_numpy().any()
    assert frame["Mkt-RF"].min() > -0.99


@pytest.mark.network
def test_load_french_index_strictly_increasing_no_duplicates():
    frame = _load_or_skip(load_french, "factors_daily")

    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates


@pytest.mark.network
def test_load_industry_returns_12_columns():
    frame = _load_or_skip(load_industry_returns, 12)
    assert frame.shape[1] == 12


FF5_MOMENTUM_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]


@pytest.mark.network
def test_load_ff5_momentum_columns_and_decimal_units():
    frame = _load_or_skip(load_ff5_momentum)

    assert list(frame.columns) == FF5_MOMENTUM_COLUMNS
    assert frame["Mom"].notna().any()
    # Percent-scaled daily factors would routinely exceed 1.0 in absolute value.
    assert (frame.abs().max() < 1.0).all()


@pytest.mark.network
def test_load_ff5_momentum_missing_value_sentinels_are_gone():
    frame = _load_or_skip(load_ff5_momentum)

    values = frame.to_numpy()
    assert not ((values == -0.9999) | (values == -9.99)).any()
    assert (frame.min() > -0.99).all()


@pytest.mark.network
def test_load_ff5_momentum_index_monotonic_and_unique():
    frame = _load_or_skip(load_ff5_momentum)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates


@pytest.mark.network
def test_load_ff5_momentum_spans_only_the_overlap():
    frame = _load_or_skip(load_ff5_momentum)
    ff5 = load_french("factors5_daily")
    mom = load_french("momentum_daily")

    assert frame.index[0] == max(ff5.index[0], mom.index[0])
    assert frame.index[-1] == min(ff5.index[-1], mom.index[-1])
    assert frame.index.isin(ff5.index).all()
    assert frame.index.isin(mom.index).all()


_FAKE_FRENCH = {
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip": (
        "notes\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "19630701,-0.67,0.00,-0.33,-0.01,0.16,0.01\n"
        "19630702,0.79,-0.26,0.26,-0.07,-0.20,0.01\n"
        "19630703,0.63,-0.17,-0.10,0.18,-0.34,0.01\n"
    ),
    "F-F_Momentum_Factor_daily_CSV.zip": (
        "notes\n\n,Mom   \n19630628,0.10\n19630701,0.55\n19630702,-99.99\n"
    ),
}


def test_load_ff5_momentum_inner_joins_on_date_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_download_french_csv", _FAKE_FRENCH.__getitem__)

    frame = load_ff5_momentum(use_cache=False)

    assert list(frame.columns) == FF5_MOMENTUM_COLUMNS
    # 1963-06-28 is Mom-only and 1963-07-03 is FF5-only; both are dropped.
    assert list(frame.index) == list(pd.to_datetime(["1963-07-01", "1963-07-02"]))
    assert frame.loc["1963-07-01", "Mom"] == pytest.approx(0.0055)
    assert frame.loc["1963-07-01", "Mkt-RF"] == pytest.approx(-0.0067)
    # The -99.99 sentinel becomes NaN, not a -99.99% return.
    assert pd.isna(frame.loc["1963-07-02", "Mom"])
