from __future__ import annotations

import pandas as pd

from capstone.pit import available_at


def test_available_at_excludes_rows_after_t():
    t = pd.Timestamp("2024-01-20")
    prices = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        columns=["A", "B"],
    )
    factors = pd.DataFrame(
        [[10.0], [20.0], [30.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        columns=["mkt"],
    )
    fundamentals = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "period_end": pd.to_datetime(["2023-12-31", "2024-01-15", "2024-01-31"]),
            "filed": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-15"]),
            "tag": ["Revenues", "NetIncomeLoss", "Revenues"],
            "value": [10.0, 20.0, 30.0],
        }
    )

    prices_at, fundamentals_at, factors_at = available_at(t, prices, fundamentals, factors)

    assert prices_at.index.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert factors_at.index.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert fundamentals_at["filed"].max() <= t
    assert set(fundamentals_at["ticker"]) == {"A"}


def test_available_at_excludes_fundamental_filed_after_t():
    t = pd.Timestamp("2024-01-15")
    prices = pd.DataFrame(
        [[1.0]],
        index=pd.to_datetime(["2024-01-01"]),
        columns=["A"],
    )
    factors = pd.DataFrame(
        [[10.0]],
        index=pd.to_datetime(["2024-01-01"]),
        columns=["mkt"],
    )
    fundamentals = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "period_end": pd.to_datetime(["2023-12-31", "2024-01-10"]),
            "filed": pd.to_datetime(["2024-01-05", "2024-01-20"]),
            "tag": ["Revenues", "NetIncomeLoss"],
            "value": [10.0, 20.0],
        }
    )

    _, fundamentals_at, _ = available_at(t, prices, fundamentals, factors)

    assert len(fundamentals_at) == 1
    assert fundamentals_at.iloc[0]["filed"] == pd.Timestamp("2024-01-05")
    assert fundamentals_at.iloc[0]["tag"] == "Revenues"
