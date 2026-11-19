import numpy as np
import pandas as pd
import pytest

from capstone.factors.evaluation import Panel
from capstone.factors.reference import (
    ff3_residual_volatility,
    formation_return,
    high_nearness,
    january_diagnostic,
    mean_turnover,
    month_ends,
)


def make_panel(fields):
    frame = next(iter(fields.values()))
    return Panel(
        fields,
        frame.notna(),
        frame * 0 + 0.001,
        data_version="synthetic-reference",
        calendar=frame.index,
        availability={n: pd.Series(f.index, index=f.index) for n, f in fields.items()},
    )


def test_month_ends_use_exchange_calendar_not_last_loaded_row():
    calendar = pd.DatetimeIndex(["2000-12-28", "2000-12-29", "2001-01-02", "2001-01-03"])
    assert list(month_ends(calendar, "2001-01-02")) == [pd.Timestamp("2000-12-29")]
    with pytest.raises(ValueError, match="next session"):
        month_ends(calendar, "2001-01-03")


def test_calendar_formation_return_ignores_future_and_rejects_gaps():
    dates = pd.bdate_range("1998-01-01", "2001-01-02")
    frame = pd.DataFrame(0.001, index=dates, columns=[1, 2])
    panel = make_panel({"returns": frame})
    before = formation_return(panel, "2000-12-29", 6)
    frame.loc["2001-01-02"] = 1000
    pd.testing.assert_series_equal(before, formation_return(panel, "2000-12-29", 6))
    frame.loc["2000-11-01", 1] = np.nan
    assert np.isnan(formation_return(panel, "2000-12-29", 6)[1])


def test_ff3_uses_excess_returns_and_residuals_not_market_adjustment():
    dates = pd.bdate_range("2000-01-03", "2000-01-31")
    rng = np.random.default_rng(42)
    f = rng.normal(0, 0.01, (len(dates), 3))
    fields = {
        name: pd.DataFrame(np.repeat(f[:, i, None], 2, axis=1), index=dates, columns=[1, 2])
        for i, name in enumerate(("mkt_excess", "smb", "hml"))
    }
    fields["rf"] = pd.DataFrame(0.0001, index=dates, columns=[1, 2])
    x = np.column_stack([np.ones(len(dates)), f])
    noise = rng.normal(0, 0.003, len(dates))
    noise -= x @ np.linalg.lstsq(x, noise, rcond=None)[0]
    fields["returns"] = pd.DataFrame(
        {1: 0.0001 + f @ [1.5, 2, -1] + noise, 2: 0.0001 + f @ [-1, 1, 2]}, index=dates
    )
    result = ff3_residual_volatility(make_panel(fields), dates[-1])
    assert result[1] == pytest.approx(noise.std(ddof=1))
    assert result[2] < 1e-15
    fields["hml"].iloc[:8] = np.nan
    assert ff3_residual_volatility(make_panel(fields), dates[-1]).isna().all()


def test_high_uses_closing_prices_and_requires_full_history():
    close = pd.DataFrame(
        np.arange(1.0, 254.0), index=pd.bdate_range("2000-01-03", periods=253), columns=[1]
    )
    assert high_nearness(make_panel({"close": close}), close.index[-1])[1] == 1
    with pytest.raises(ValueError, match="252"):
        high_nearness(make_panel({"close": close}), close.index[100])


def test_january_finding_preserves_cohort_age():
    rows = pd.DataFrame(
        {
            "date": ["2001-01-02", "2001-02-01", "2002-01-02"],
            "formation": ["2000-12-29"] * 3,
            "long_return": [0.1, 0.02, 0.03],
            "short_return": [0, 0, 0],
        }
    )
    report = january_diagnostic(rows)
    assert report.age_months.tolist() == [1, 13]


def test_complete_window_starts_on_first_declared_session_and_rejects_dropped_dates():
    dates = pd.bdate_range("2000-01-03", "2000-07-03").difference(
        pd.to_datetime(["2000-01-17", "2000-05-29"])
    )
    frame = pd.DataFrame(0.001, index=dates, columns=[1])
    panel = make_panel({"returns": frame})
    actual = formation_return(panel, "2000-06-30", 6)
    assert actual[1] == pytest.approx(1.001 ** len(frame.loc[:"2000-06-30"]) - 1)
    panel.fields["returns"] = frame.drop(pd.Timestamp("2000-02-01"))
    with pytest.raises(ValueError, match="missing declared calendar"):
        formation_return(panel, "2000-06-30", 6)


def test_negative_volume_is_rejected_instead_of_becoming_a_low_turnover_signal():
    dates = pd.bdate_range("2000-01-03", "2000-02-01")
    volume = pd.DataFrame(10.0, index=dates, columns=[1])
    volume.loc["2000-01-05", 1] = -10
    panel = make_panel({"volume": volume, "shares": volume * 0 + 100})
    with pytest.raises(ValueError, match="negative trading volume"):
        mean_turnover(panel, "2000-01-31", 1)
