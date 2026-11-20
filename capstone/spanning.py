"""Stage 3 spanning test: does a candidate earn alpha beyond daily FF5 + Momentum?

Each candidate's daily returns are regressed on the Fama-French five factors
plus Momentum, and the intercept is tested one-sided with Newey-West standard
errors. Design and the choices behind it: `docs/validation/incremental-value.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from capstone.data import FF5_MOMENTUM_REGRESSORS
from capstone.evaluate import newey_west_lags

MAX_MEDIAN_GAP_DAYS = 4

BETA_COLUMNS = [f"beta_{name}" for name in FF5_MOMENTUM_REGRESSORS]
OUTPUT_COLUMNS = [
    "alpha",
    "se",
    "tstat",
    "pvalue_greater",
    "n_obs",
    "n_dropped",
    "lags",
    *BETA_COLUMNS,
]

_N_PARAMS = len(FF5_MOMENTUM_REGRESSORS) + 1
_DEGENERATE_RESIDUAL_SCALE = float(np.sqrt(np.finfo(float).eps))


def factor_alpha(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    returns_are_excess: bool,
    lags: int | None = None,
) -> pd.DataFrame:
    """Alpha of each candidate against daily FF5 + Momentum, tested for alpha > 0.

    For each candidate, r_t = alpha + sum_k beta_k F_k,t + e_t is fitted by OLS
    on the regressors `FF5_MOMENTUM_REGRESSORS`; `RF` is never a regressor. The
    intercept's standard error is Newey-West (Bartlett kernel, no small-sample
    correction) with `lags`, or `evaluate.newey_west_lags(n_obs)` per candidate
    when `lags` is None. `pvalue_greater` is the one-sided normal p-value of
    H0: alpha = 0 against H1: alpha > 0.

    Args:
        returns: daily decimal returns, dates x candidates, every date before
            `gate.HOLDOUT_OPENS` (the search period only).
        factors: daily decimal factor returns with at least the columns
            `FF5_MOMENTUM_REGRESSORS`, plus `RF` when `returns_are_excess` is
            False; the frame from `data.load_ff5_momentum()`. Never downloaded
            here.
        returns_are_excess: True when `returns` are already excess returns
            (dollar-neutral long-short, the backtest default) and are used as
            given; False to subtract each date's `RF` once before fitting.
        lags: Newey-West truncation lag, a non-negative int, for every
            candidate; None for the per-candidate default.

    Returns:
        One row per candidate (same order as `returns.columns`) with columns
        `OUTPUT_COLUMNS`: `alpha` (per period, decimal), `se`, `tstat`,
        `pvalue_greater`, `n_obs` (dates in the regression), `n_dropped` (the
        candidate's non-missing dates with no complete factor row, so
        `n_obs + n_dropped` is its non-missing count), `lags` (nullable int),
        and one `beta_<factor>` per regressor. A candidate that cannot be
        estimated (`n_obs` at most the 7 parameters, `lags` not below `n_obs`,
        a rank-deficient design, or residuals with no variance) gets NaN in
        every statistic and keeps its counts, so the batch still runs.

    Raises:
        TypeError: `returns_are_excess` is not a bool.
        ValueError: on structurally invalid input: either frame without a
            unique, increasing, timezone-naive `DatetimeIndex`, or not daily
            (gaps under one day, or a median gap above `MAX_MEDIAN_GAP_DAYS`
            calendar days, which allows holidays and missing sessions);
            `returns` and `factors` sharing no dates; non-numeric or infinite
            values in `returns` or in the factor columns used (`RF` only when
            it is subtracted); duplicate candidate names; missing or
            duplicated factor columns; a candidate date on or after
            `gate.HOLDOUT_OPENS`; or a negative or non-integer `lags`.
    """
    if not isinstance(returns_are_excess, (bool, np.bool_)):
        raise TypeError("returns_are_excess must be a bool: say whether RF is already out")
    if lags is not None and (
        isinstance(lags, (bool, np.bool_)) or not isinstance(lags, (int, np.integer)) or lags < 0
    ):
        raise ValueError(f"lags must be a non-negative int or None, got {lags!r}")

    needed = [*FF5_MOMENTUM_REGRESSORS, *([] if returns_are_excess else ["RF"])]
    _check_daily_frame(returns, "returns")
    _check_daily_frame(factors, "factors")
    if returns.columns.duplicated().any():
        duplicated = returns.columns[returns.columns.duplicated()].unique().tolist()
        raise ValueError(f"returns has duplicate candidate names: {duplicated[:5]}")
    if factors.columns.duplicated().any():
        raise ValueError("factors has duplicate column names")
    missing = [name for name in needed if name not in factors.columns]
    if missing:
        raise ValueError(f"factors is missing columns {missing}; expected load_ff5_momentum()")
    _check_numeric(returns, "returns")
    _check_numeric(factors[needed], "factors")
    _check_search_period(returns)
    if not returns.index.isin(factors.index).any():
        raise ValueError("returns and factors share no dates: check their date stamps")

    complete_factors = factors[needed].dropna()
    rows = {
        name: _fit_candidate(returns[name], complete_factors, returns_are_excess, lags)
        for name in returns.columns
    }
    result = pd.DataFrame.from_dict(rows, orient="index", columns=OUTPUT_COLUMNS)
    result = result.reindex(returns.columns)
    result.index.name = returns.columns.name
    result[["n_obs", "n_dropped"]] = result[["n_obs", "n_dropped"]].astype(int)
    result["lags"] = result["lags"].astype("Int64")
    return result


def _check_daily_frame(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} must be a DataFrame, got {type(frame).__name__}")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{label} must have a DatetimeIndex")
    if index.tz is not None:
        raise ValueError(f"{label} has a timezone-aware index; pass timezone-naive dates")
    if not index.is_unique:
        raise ValueError(f"{label} has duplicate dates")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label} dates must be increasing")
    if len(index) < 2:
        raise ValueError(f"{label} needs at least two dates to check it is daily")
    gaps_days = np.diff(index.to_numpy()) / np.timedelta64(1, "D")
    if gaps_days.min() < 1:
        raise ValueError(f"{label} has dates less than one day apart: not daily")
    median_gap = float(np.median(gaps_days))
    if median_gap > MAX_MEDIAN_GAP_DAYS:
        raise ValueError(
            f"{label} has a median gap of {median_gap:g} days: not daily "
            f"(at most {MAX_MEDIAN_GAP_DAYS} allowed)"
        )


def _check_numeric(frame: pd.DataFrame, label: str) -> None:
    non_numeric = [
        name
        for name, dtype in frame.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
    ]
    if non_numeric:
        raise ValueError(f"{label} has non-numeric columns: {non_numeric[:5]}")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} has infinite values")


def _check_search_period(returns: pd.DataFrame) -> None:
    from capstone.gate import HOLDOUT_OPENS

    holdout_opens = pd.Timestamp(HOLDOUT_OPENS)
    if returns.index.max() >= holdout_opens:
        raise ValueError(
            f"returns run to {returns.index.max().date()}, into the holdout that opens "
            f"{holdout_opens.date()}: stage 3 sees the search period only"
        )


def _fit_candidate(
    candidate: pd.Series,
    complete_factors: pd.DataFrame,
    returns_are_excess: bool,
    lags: int | None,
) -> dict:
    available = candidate.dropna()
    dates = available.index[available.index.isin(complete_factors.index)]
    n_obs = len(dates)
    row = dict.fromkeys(OUTPUT_COLUMNS, np.nan)
    row.update(n_obs=n_obs, n_dropped=len(available) - n_obs, lags=pd.NA)

    if lags is not None:
        row["lags"] = int(lags)
    elif n_obs >= 2:
        row["lags"] = newey_west_lags(n_obs)
    if n_obs <= _N_PARAMS or row["lags"] is pd.NA or row["lags"] >= n_obs:
        return row

    aligned = complete_factors.loc[dates]
    y = available.loc[dates].to_numpy(dtype=float)
    if not returns_are_excess:
        y = y - aligned["RF"].to_numpy(dtype=float)
    design = sm.add_constant(
        aligned[FF5_MOMENTUM_REGRESSORS].to_numpy(dtype=float), has_constant="add"
    )
    if np.linalg.matrix_rank(design) < _N_PARAMS:
        return row

    fit = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": row["lags"]}, use_t=False)
    residual_rms = float(np.sqrt(fit.ssr / n_obs))
    alpha, se = float(fit.params[0]), float(fit.bse[0])
    if residual_rms <= _DEGENERATE_RESIDUAL_SCALE * float(np.abs(y).max()) or not (
        np.isfinite(se) and se > 0
    ):
        return row

    tstat = alpha / se
    row.update(
        alpha=alpha,
        se=se,
        tstat=tstat,
        pvalue_greater=float(stats.norm.sf(tstat)),
        **dict(zip(BETA_COLUMNS, (float(b) for b in fit.params[1:]), strict=True)),
    )
    return row
