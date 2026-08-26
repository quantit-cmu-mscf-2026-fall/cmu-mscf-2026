from __future__ import annotations

import pandas as pd

from capstone.forecaster_momentum import momentum_signal


def test_momentum_signal_shape_and_columns():
    prices = pd.DataFrame(
        [[100.0, 200.0], [101.0, 205.0], [100.5, 210.0], [102.0, 212.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        columns=["A", "B"],
    )

    signal = momentum_signal(prices, lookback=2, skip=1)

    assert isinstance(signal, pd.DataFrame)
    assert signal.shape == prices.shape
    assert list(signal.columns) == list(prices.columns)
    assert list(signal.index) == list(prices.index)


def test_momentum_signal_is_point_in_time():
    prices = pd.DataFrame(
        [[100.0, 200.0], [101.0, 205.0], [100.5, 210.0], [102.0, 212.0], [103.0, 214.0]],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        ),
        columns=["A", "B"],
    )

    signal = momentum_signal(prices, lookback=2, skip=1, t="2024-01-03")
    future_signal = momentum_signal(prices, lookback=2, skip=1, t="2024-01-05")

    assert signal.index.max() <= pd.Timestamp("2024-01-03")
    assert future_signal.index.max() <= pd.Timestamp("2024-01-05")
    assert len(signal) <= len(prices)
    assert pd.isna(signal.loc["2024-01-02", "A"])
    assert pd.isna(signal.loc["2024-01-02", "B"])
    assert signal.loc["2024-01-03"].notna().any()
    assert "2024-01-05" not in signal.index
