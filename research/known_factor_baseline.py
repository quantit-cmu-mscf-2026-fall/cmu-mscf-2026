"""A known factor for checking the funnel: 12-1 momentum and its dated periods.

A sanity check, not a target: momentum, written in the grammar
(`shift(ts_sum(returns, 231), 21)`), is a factor the funnel should be able to
say something about. This module builds its inputs and nothing more; it never
imports the validation gate, and the registered holdout dates are never read.

- `momentum_market`: a CRSP-shaped synthetic market whose momentum premium is
  earned by its top-minus-bottom tercile book (daily rows as the shared file).
- `momentum_factors`: FF5 + Mom on a panel's dates, built from the panel
  itself; Mom is that tercile book, using only returns through each day.
- `synthetic_periods`, `crsp_periods`: the search period and the registered
  holdout dates. The search data only: nothing here reaches past `search_end`.

Parameters are in research/known_factor_baseline.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.factors.evaluation import DISCOVERY_END, Panel
from capstone.factors.interpret import evaluate
from capstone.factors.tree import parse

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "research" / "known_factor_baseline.toml"
FAMILY = "known-momentum-12-1"
FACTOR_COLUMNS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
LOOKBACK, SKIP = 252, 21  # momentum on day t: returns t-251 .. t-21


def load_params(path: Path = PARAMS) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The synthetic market and its factors


def terciles(values: np.ndarray) -> np.ndarray:
    """+1 for the top third of each row, -1 for the bottom third, 0 otherwise
    (and 0 on a row with any missing value). Ties go by position."""
    out = np.zeros(values.shape)
    complete = np.isfinite(values).all(axis=1)
    n = values.shape[1]
    third = n // 3
    ranks = values[complete].argsort(axis=1).argsort(axis=1)
    out[complete] = np.where(ranks >= n - third, 1.0, np.where(ranks < third, -1.0, 0.0))
    return out


def momentum_market(
    *, n_stocks: int, n_days: int, premium: float, seed: int, start: str = "2000-01-03"
) -> pd.DataFrame:
    """CRSP-shaped daily rows (as the shared file) with a momentum premium.

    Returns are a market factor times a beta, plus noise, plus `premium` for
    the stocks in the top tercile of 12-1 momentum the day before and minus it
    for the bottom tercile.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    beta = rng.uniform(0.8, 1.2, n_stocks)
    market = rng.normal(0.0003, 0.01, n_days)
    noise = rng.normal(0.0, 0.02, (n_days, n_stocks))
    volume = np.exp(rng.normal(13, 0.5, (n_days, n_stocks)))
    ret = np.empty((n_days, n_stocks))
    total = np.zeros((n_days + 1, n_stocks))  # total[t + 1] = sum of returns 0..t
    for t in range(n_days):
        ret[t] = beta * market[t] + noise[t]
        s = t - 1  # yesterday's momentum
        if s >= LOOKBACK - 1:
            mom = total[s - SKIP + 1] - total[s - LOOKBACK + 1]
            ret[t] += premium * terciles(mom[None, :])[0]
        total[t + 1] = total[t] + ret[t]
    price = 50 * np.cumprod(1 + ret, axis=0)
    spread = 0.002
    flat = price.T.ravel()  # permno-major, as fake_rows
    return pd.DataFrame(
        {
            "date": np.tile(dates, n_stocks),
            "permno": np.repeat(10000 + np.arange(n_stocks), n_days),
            "in_universe": True,
            "dlyret": ret.T.ravel(),
            "dlyretmissflg": None,
            "dlydelflg": "N",
            "delactiontype": None,
            "dlyopen": flat,
            "dlyhigh": flat * 1.01,
            "dlylow": flat * 0.99,
            "dlyprc": flat,
            "dlyvol": volume.T.ravel(),
            "shrout": 1000.0,
            "dlycap": flat * 1000.0,
            "dlycumfacpr": 1.0,
            "dlycumfacshr": 1.0,
            "dlybid": flat * (1 - spread / 2),
            "dlyask": flat * (1 + spread / 2),
            "siccd": np.repeat(np.array([2834, 3674, 6020])[np.arange(n_stocks) % 3], n_days),
        }
    )


def momentum_factors(panel: Panel, *, seed: int, expression: str | None = None) -> pd.DataFrame:
    """FF5 + Mom on the panel's dates, built from the panel itself.

    Mom is the equal-weighted top-minus-bottom tercile of 12-1 momentum,
    rebalanced daily (as Ken French's, a sort, not the signal's weights);
    Mkt-RF is the panel's `mkt_return`; SMB, HML, RMW and CMA are seeded
    N(0, 0.5%) noise. Dates where Mom is not yet defined are left out.
    """
    expression = expression or load_params()["expression"]
    mom = evaluate(parse(expression), panel.fields).where(panel.universe)
    returns = panel.fields["returns"].to_numpy(dtype="float64")
    side = terciles(mom.to_numpy(dtype="float64"))[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        top = (returns[1:] * (side > 0)).sum(axis=1) / (side > 0).sum(axis=1)
        bottom = (returns[1:] * (side < 0)).sum(axis=1) / (side < 0).sum(axis=1)
    dates = panel.universe.index
    rng = np.random.default_rng([seed, 3])
    frame = pd.DataFrame(rng.normal(0.0, 0.005, (len(dates), 4)), index=dates)
    frame.columns = list(FACTOR_COLUMNS[1:5])
    frame.insert(0, "Mkt-RF", panel.fields["mkt_return"].to_numpy(dtype="float64"))
    frame["Mom"] = np.concatenate(
        [[np.nan], np.where(np.isfinite(top - bottom), top - bottom, np.nan)]
    )
    return frame.dropna().rename_axis("date")


def synthetic_periods(panel: Panel) -> dict:
    """The synthetic search period is the whole panel; the registered holdout
    dates come after it and are never read."""
    last = panel.universe.index[-1]
    return {
        "search_end": str(last.date()),
        "holdout_start": str((last + pd.offsets.BDay(1)).date()),
        "holdout_end": str((last + pd.DateOffset(years=1)).date()),
    }


def crsp_periods(params: dict | None = None) -> dict:
    periods = dict((params or load_params())["crsp"])
    if pd.Timestamp(periods["search_end"]) > pd.Timestamp(DISCOVERY_END):
        raise ValueError("the baseline's search period must end by the discovery period's end")
    return periods
