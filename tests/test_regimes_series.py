"""The point-in-time regime series (`capstone.regimes`).

Look-ahead safety (one-day shift, future perturbation, truncation), warm-up, known
answers on synthetic market paths, and the holdout refusal. Synthetic series only:
no data, no model, no network.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from capstone import regimes
from capstone.factors.evaluation import DISCOVERY_END, HoldoutError, Panel

N = 1400


def _dates(n: int, end: str = DISCOVERY_END) -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=n)


def _market(n: int = N, seed: int = 3) -> pd.Series:
    """A market path with calm and stormy spells and a long down leg, so every column
    (bear and panic included) has defined, varying values."""
    rng = np.random.default_rng(seed)
    sd = np.where((np.arange(n) // 200) % 2 == 0, 0.008, 0.02)
    drift = np.where((np.arange(n) > 500) & (np.arange(n) < 1000), -0.0015, 0.0006)
    return pd.Series(drift + rng.normal(0.0, 1.0, n) * sd, index=_dates(n), name="mkt_return")


@pytest.fixture(scope="module")
def market() -> pd.Series:
    return _market()


@pytest.fixture(scope="module")
def full(market) -> pd.DataFrame:
    out = regimes.regime_series(market)
    # The fixture exercises every state, so the tests below test something.
    assert out["bear"].eq(1).any() and out["bear"].eq(0).any()
    assert out["high_vol"].eq(1).any() and out["high_vol"].eq(0).any()
    assert out["panic"].gt(0).any() and out["panic"].eq(0).any()
    return out


CHECK_DATES = [126, 127, 300, 504, 505, 506, 640, 777, 1000, 1200, N - 1]


def test_columns_and_index(market, full):
    assert list(full.columns) == list(regimes.COLUMNS)
    assert full.index.equals(pd.DatetimeIndex(market.index))


@pytest.mark.parametrize("t", CHECK_DATES)
@pytest.mark.parametrize("today", [1e6, -0.99, np.nan])
def test_value_at_t_uses_data_through_t_minus_1_only(market, full, t, today):
    """One-day shift: the row at t is the same when day t's own return is anything at
    all and nothing after t exists."""
    cut = market.iloc[: t + 1].copy()
    cut.iloc[-1] = today
    pd.testing.assert_series_equal(
        regimes.regime_series(cut).iloc[-1], full.iloc[t], check_exact=True
    )


@pytest.mark.parametrize("t", CHECK_DATES)
def test_future_perturbation_leaves_the_past_unchanged(market, full, t):
    rng = np.random.default_rng(t)
    moved = market.copy()
    after = len(moved) - t - 1
    moved.iloc[t + 1 :] = rng.normal(0.0, 0.05, after) - 0.01
    if after:
        moved.iloc[-1] = np.nan
    out = regimes.regime_series(moved)
    pd.testing.assert_frame_equal(out.iloc[: t + 1], full.iloc[: t + 1], check_exact=True)


@pytest.mark.parametrize("k", [1, 50, 126, 127, 504, 505, 900, N])
def test_truncation_invariance(market, full, k):
    pd.testing.assert_frame_equal(
        regimes.regime_series(market.iloc[:k]), full.iloc[:k], check_exact=True
    )


def test_warm_up_is_nan_never_zero(full):
    assert full["rv126"].iloc[:126].isna().all()
    assert full["rv126"].iloc[126:].notna().all()
    assert full["high_vol"].iloc[:127].isna().all()  # needs one earlier rv126
    assert full["high_vol"].iloc[127:].notna().all()
    assert full["bear"].iloc[:504].isna().all()
    assert full["bear"].iloc[504:].notna().all()
    assert full["panic"].iloc[:505].isna().all()  # needs one earlier panic_raw


def test_panic_is_nan_until_a_bear_day_has_been_seen():
    """Before the first bear day the expanding mean of bear * rv^2 is 0: panic is NaN
    (undefined), never 0."""
    s = pd.Series(2.0**-8, index=_dates(700))  # a market that only goes up
    out = regimes.regime_series(s)
    assert out["bear"].iloc[504:].eq(0).all()
    assert out["panic"].isna().all()


def test_nan_return_blanks_its_windows_only():
    s = _market(800)
    s.iloc[300] = np.nan
    out = regimes.regime_series(s)
    rv = out["rv126"]
    assert rv.iloc[301 : 301 + 126].isna().all()  # windows holding day 300
    assert rv.iloc[300] == rv.iloc[300] and rv.iloc[427] == rv.iloc[427]  # not NaN
    assert out["bear"].iloc[504:805].isna().all()  # every 504-day window holds day 300


def test_known_answer_alternating_path():
    """Returns +a, -a, +a, ...: every 126-day window has mean 0 and sd a*sqrt(126/125);
    every 504-day window compounds to (1 - a^2)^252 < 1, so it is always a bear market;
    rv is constant, so it is never above its own median, and panic is 1."""
    a = 2.0**-6
    n = 800
    s = pd.Series(np.where(np.arange(n) % 2 == 0, a, -a), index=_dates(n))
    out = regimes.regime_series(s)
    expected_rv = a * math.sqrt(126 / 125) * math.sqrt(252)
    assert np.allclose(out["rv126"].iloc[126:], expected_rv, rtol=1e-14, atol=0)
    assert out["rv126"].iloc[126:].nunique() == 1
    assert out["high_vol"].iloc[127:].eq(0).all()
    assert out["bear"].iloc[504:].eq(1).all()
    assert np.allclose(out["panic"].iloc[505:], 1.0, rtol=1e-15, atol=0)  # mean of equals


def test_known_answer_volatility_step():
    """Calm (+-a) then stormy (+-4a) from day 400: rv126 is 4x higher once the window
    is all stormy, which is above every earlier value, so high_vol is 1."""
    a = 2.0**-7
    n = 700
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    size = np.where(np.arange(n) < 400, a, 4 * a)
    out = regimes.regime_series(pd.Series(sign * size, index=_dates(n)))
    calm = a * math.sqrt(126 / 125) * math.sqrt(252)
    assert math.isclose(out["rv126"].iloc[399], calm, rel_tol=1e-14)
    assert math.isclose(out["rv126"].iloc[526], 4 * calm, rel_tol=1e-14)
    assert out["high_vol"].iloc[127:401].eq(0).all()  # constant calm rv, never above
    assert out["high_vol"].iloc[401:].eq(1).all()  # rising, then above the calm median


def test_known_answer_bear_flip_and_panic_normalisation():
    """Up 2^-8 a day for 600 days, then down 2^-6 a day. bear at t is 1 exactly when the
    compounded return of days t-504..t-1 is below 0 (checked by brute force); panic is
    bear * rv^2 over the mean of earlier bear * rv^2 values."""
    n = 900
    r = np.where(np.arange(n) < 600, 2.0**-8, -(2.0**-6))
    out = regimes.regime_series(pd.Series(r, index=_dates(n)))
    for t in range(504, n):
        growth = 1.0
        for x in r[t - 504 : t]:
            growth *= 1.0 + x
        assert out["bear"].iloc[t] == (1.0 if growth < 1.0 else 0.0)
    first = int(np.flatnonzero(out["bear"].to_numpy() == 1)[0])
    assert 600 < first < 900
    raw = (out["bear"] * out["rv126"] ** 2).to_numpy()
    for t in range(first + 1, n):
        prior = raw[504:t]
        expected = raw[t] / (math.fsum(prior) / len(prior))
        assert out["panic"].iloc[t] == expected


def test_input_checks():
    s = _market(300)
    with pytest.raises(ValueError, match="ascending"):
        regimes.regime_series(s.iloc[::-1])
    with pytest.raises(ValueError, match="ascending"):
        regimes.regime_series(pd.concat([s, s.iloc[-1:]]))


def test_nullable_input_matches_float():
    s = _market(700)
    nullable = s.astype("Float64")
    nullable.iloc[10] = pd.NA
    plain = s.copy()
    plain.iloc[10] = np.nan
    pd.testing.assert_frame_equal(
        regimes.regime_series(nullable), regimes.regime_series(plain), check_exact=True
    )


def test_holdout_refused_before_anything_is_computed(monkeypatch):
    s = pd.Series(0.001, index=pd.bdate_range("2020-06-01", "2021-01-04"))

    def boom(*_args, **_kwargs):
        raise AssertionError("computed on holdout data")

    monkeypatch.setattr(regimes, "_rv", boom)
    with pytest.raises(HoldoutError):
        regimes.regime_series(s)


def test_series_ending_on_discovery_end_is_allowed():
    s = pd.Series(0.001, index=_dates(200))
    assert s.index.max() == pd.Timestamp(DISCOVERY_END)
    assert len(regimes.regime_series(s)) == 200


def _panel(index: pd.DatetimeIndex) -> Panel:
    mkt = pd.Series(0.001, index=index)
    universe = pd.DataFrame(True, index=index, columns=[1, 2])
    return Panel({"mkt_return": mkt}, universe, universe.astype(float) * 0.0)


def test_panel_regimes_uses_the_panel_market_return():
    panel = _panel(_dates(300))
    pd.testing.assert_frame_equal(
        regimes.panel_regimes(panel),
        regimes.regime_series(panel.fields["mkt_return"]),
        check_exact=True,
    )


def test_panel_regimes_refuses_a_holdout_panel():
    panel = _panel(pd.bdate_range("2020-11-01", "2021-02-01"))
    with pytest.raises(HoldoutError):
        regimes.panel_regimes(panel)
    # A clean market series does not get through on a panel whose universe is past the cut.
    panel.fields["mkt_return"] = panel.fields["mkt_return"].loc[:DISCOVERY_END]
    with pytest.raises(HoldoutError):
        regimes.panel_regimes(panel)


def test_require_discovery_refuses_only_a_holdout_universe():
    from capstone.factors.evaluation import require_discovery

    require_discovery(_panel(_dates(50)))  # ends on DISCOVERY_END: allowed
    with pytest.raises(HoldoutError):
        require_discovery(_panel(pd.bdate_range("2020-12-20", "2021-01-15")))
