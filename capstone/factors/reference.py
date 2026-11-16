"""Point-in-time reference features. Dates are exchange sessions, never row counts for months."""

from __future__ import annotations

import numpy as np
import pandas as pd

from capstone.factors.evaluation import Panel


def sessions(calendar, as_of=None) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(calendar)
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("calendar must contain unique increasing sessions")
    if dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("sessions must be timezone-free dates")
    if as_of is not None:
        at = pd.Timestamp(as_of)
        if at not in dates or not (dates > at).any():
            raise ValueError("calendar must contain as_of and at least the next session")
    return dates


def month_ends(calendar, as_of) -> pd.DatetimeIndex:
    dates = sessions(calendar, as_of)
    # The full exchange calendar, including the next session, establishes completeness.
    ends = pd.Series(dates, index=dates.to_period("M")).groupby(level=0).max()
    return pd.DatetimeIndex(ends[ends <= pd.Timestamp(as_of)].values)


def field(panel: Panel, name: str, as_of) -> pd.DataFrame:
    if name not in panel.fields:
        raise ValueError(f"required feature input {name!r} is unavailable")
    frame = panel.fields[name]
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"duplicate dates or instruments in {name}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"dates must increase in {name}")
    out = frame.loc[: pd.Timestamp(as_of)].astype(float)
    if panel.calendar is not None and not out.empty:
        calendar = sessions(panel.calendar)
        expected = calendar[(calendar >= out.index[0]) & (calendar <= pd.Timestamp(as_of))]
        if not expected.isin(out.index).all():
            raise ValueError(f"{name} is missing declared calendar sessions")
    if panel.availability is None or name not in panel.availability:
        raise ValueError(f"point-in-time availability is required for {name}")
    available = pd.to_datetime(panel.availability[name].reindex(out.index))
    if available.isna().any() or (available.dt.normalize() > pd.Timestamp(as_of)).any():
        raise ValueError(f"{name} includes observations unavailable at the decision date")
    if np.isinf(out.to_numpy()).any():
        raise ValueError(f"infinite input in {name}")
    return out


def _month_window(panel, frame, end, months):
    if isinstance(months, bool) or not isinstance(months, int) or months <= 0:
        raise ValueError("formation months must be positive integers")
    start = (end.to_period("M") - months + 1).start_time
    calendar = sessions(panel.calendar)
    expected = calendar[(calendar >= start) & (calendar <= end)]
    if (
        expected.empty
        or expected[0].to_period("M") != start.to_period("M")
        or not expected.isin(frame.index).all()
    ):
        raise ValueError(f"insufficient history for {months} calendar months")
    return frame.loc[expected]


def formation_return(panel: Panel, formation, months: int, *, market_adjusted=False):
    end = pd.Timestamp(formation)
    window = _month_window(panel, field(panel, "returns", end), end, months)
    if (window < -1).any().any():
        raise ValueError("total returns cannot be below -1")
    if market_adjusted:
        market = field(panel, "mkt_return", end).reindex_like(window)
        # De Bondt-Thaler ranks cumulative monthly market-adjusted residuals.
        monthly = (1 + window).groupby(window.index.to_period("M")).prod(min_count=1) - 1
        market_monthly = (1 + market).groupby(market.index.to_period("M")).prod(min_count=1) - 1
        result = (monthly - market_monthly).sum(skipna=False)
    else:
        result = (1 + window).prod(skipna=False) - 1
    return result.where(window.notna().all())


def mean_turnover(panel: Panel, formation, months=6):
    end = pd.Timestamp(formation)
    volume, shares = field(panel, "volume", end), field(panel, "shares", end)
    turnover = volume / shares.where(shares > 0)
    window = _month_window(panel, turnover, end, months)
    if (window < 0).any().any():
        raise ValueError("turnover inputs cannot contain negative trading volume")
    return window.mean().where(window.notna().all())


def high_nearness(panel: Panel, formation):
    close = field(panel, "close", formation).tail(252)
    if len(close) < 252:
        raise ValueError("52-week reference needs 252 adjusted closing observations")
    close = close.where(close > 0)
    return (close.iloc[-1] / close.max()).where(close.notna().all())


def ff3_residual_volatility(panel: Panel, formation):
    end = pd.Timestamp(formation)
    returns = field(panel, "returns", end)
    dates = returns.index[returns.index.to_period("M") == end.to_period("M")]
    factors = []
    for name in ("rf", "mkt_excess", "smb", "hml"):
        values = field(panel, name, end).reindex(index=dates, columns=returns.columns)
        if (values.nunique(axis=1, dropna=False) != 1).any():
            raise ValueError(f"{name} must be the same market-wide series for every instrument")
        factors.append(values.iloc[:, 0])
    regressors = pd.concat(factors, axis=1)
    out = pd.Series(np.nan, index=returns.columns, dtype=float)
    for name in out.index:
        data = pd.concat([returns.loc[dates, name], regressors], axis=1).dropna()
        if len(data) < 17:
            continue
        y = data.iloc[:, 0].to_numpy() - data.iloc[:, 1].to_numpy()
        x = np.column_stack([np.ones(len(data)), data.iloc[:, 2:].to_numpy()])
        if np.linalg.matrix_rank(x) != 4:
            continue
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        out.loc[name] = residual.std(ddof=1)
    return out


def january_diagnostic(cohort_returns: pd.DataFrame) -> pd.DataFrame:
    """Describe returns; this never creates an executable January-only strategy."""
    required = {"date", "formation", "long_return", "short_return"}
    if not required <= set(cohort_returns):
        raise ValueError(f"cohort diagnostic requires {sorted(required)}")
    rows = cohort_returns.copy()
    rows["date"], rows["formation"] = pd.to_datetime(rows.date), pd.to_datetime(rows.formation)
    if (rows.date <= rows.formation).any():
        raise ValueError("cohort returns must occur after formation")
    rows["age_months"] = (
        (rows.date.dt.year - rows.formation.dt.year) * 12
        + rows.date.dt.month
        - rows.formation.dt.month
    )
    return rows.loc[rows.date.dt.month == 1].sort_values(["formation", "date"])
