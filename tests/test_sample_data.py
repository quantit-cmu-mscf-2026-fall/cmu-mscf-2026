"""The sample panel must be deterministic, well-formed, and actually structured.

The last one matters most: a panel of independent random walks would pass shape
checks and teach nothing, because no cross-sectional signal can work on it. The
correlation tests below are what make this data a usable stand-in for CRSP
rather than noise with tickers attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capstone.sample_data import (
    FUNDAMENTAL_TAGS,
    SAMPLE_SECTORS,
    SAMPLE_TICKERS,
    generate_sample_fundamentals,
    generate_sample_prices,
)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return generate_sample_prices()


def test_shape_and_wellformedness(prices):
    assert len(prices) == 126
    assert list(prices.columns) == SAMPLE_TICKERS
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert not prices.isna().any().any()
    assert (prices > 0).all().all()


def test_deterministic_given_seed(prices):
    pd.testing.assert_frame_equal(prices, generate_sample_prices())
    other = generate_sample_prices(seed=7)
    assert not np.allclose(prices.to_numpy(), other.to_numpy())


def test_market_factor_is_present(prices):
    """A shared market factor should leave every pair positively correlated."""
    returns = prices.pct_change().dropna()
    corr = returns.corr().to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    assert off_diagonal.mean() > 0.15


def test_sector_structure_exceeds_market_structure(prices):
    """Same-sector names must co-move more than cross-sector ones.

    If this fails the sector factor is not wired in, and any sector-neutral
    exercise built on this data would be measuring nothing.
    """
    returns = prices.pct_change().dropna()
    corr = returns.corr()

    within, across = [], []
    for i, a in enumerate(SAMPLE_TICKERS):
        for b in SAMPLE_TICKERS[i + 1 :]:
            value = corr.loc[a, b]
            if SAMPLE_SECTORS[a] == SAMPLE_SECTORS[b]:
                within.append(value)
            else:
                across.append(value)

    assert within and across
    assert np.mean(within) > np.mean(across)


def test_every_ticker_has_a_sector():
    assert set(SAMPLE_SECTORS) == set(SAMPLE_TICKERS)
    assert len(set(SAMPLE_SECTORS.values())) >= 5


def test_filing_lag_prevents_look_ahead(prices):
    """`filed` must always postdate `period_end`, by a realistic reporting lag."""
    facts = generate_sample_fundamentals(prices)
    gap = (facts["filed"] - facts["period_end"]).dt.days
    assert (gap > 0).all()
    assert gap.between(30, 75).all()


def test_all_tags_present_for_every_ticker(prices):
    facts = generate_sample_fundamentals(prices)
    for ticker in SAMPLE_TICKERS:
        tags = set(facts.loc[facts["ticker"] == ticker, "tag"])
        assert set(FUNDAMENTAL_TAGS) <= tags
