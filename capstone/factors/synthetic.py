"""A synthetic panel with one planted factor, and a scorer for it.

Learned memory search is checked here before it sees real data: the planted
factor is known, so "did the deepen move find it?" is a measurement.

`make_panel` draws every field of the factor grammar for a set of stocks and
days (the annual accounting fields last, so the daily draws do not depend on
them). Returns are noise plus the planted part: the planted factor's
cross-sectional z-score on day t, times `beta`, is added to day t+1's
return. The planted factor is short-term reversal, -ts_sum(returns, 5),
computed day by day from the returns drawn so far. With beta = 0 the panel is
the null panel: the same draws, no signal.

`PanelScorer` is a `deepen.Scorer` for such a panel: Q is |ICIR| of the rank
IC between the factor and the next h days' returns, on every `ic_step`-th
day, and the values it returns are the factor on a fixed sample of cells,
for correlations. Real data gets its scorer from evaluation (QUANTIT-81).
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone.factors.deepen import Scored
from capstone.factors.interpret import evaluate
from capstone.factors.tree import FUNDAMENTAL_FIELDS, Node, identity_key

PLANTED = "-ts_sum(returns, 5)"


@dataclass(frozen=True)
class PanelSpec:
    n_stocks: int = 100
    n_days: int = 1260
    beta: float = 0.0  # planted strength; 0 is the null panel
    return_sd: float = 0.02
    seed: int = 0


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - x.mean(axis=1, keepdims=True)) / sd
    return np.nan_to_num(z)


def make_panel(spec: PanelSpec) -> dict[str, pd.DataFrame | pd.Series]:
    """Every field of the grammar, deterministically from `spec.seed`."""
    rng = np.random.default_rng(spec.seed)
    n, t_max = spec.n_stocks, spec.n_days
    level = rng.normal(np.log(1e6), 0.5, n)
    log_volume = np.empty((t_max, n))
    log_volume[0] = level
    shocks = rng.normal(0.0, 0.3, (t_max, n))
    for t in range(1, t_max):
        log_volume[t] = level + 0.9 * (log_volume[t - 1] - level) + shocks[t]
    shares = np.exp(rng.normal(np.log(5e7), 0.5, n))
    noise = rng.normal(0.0, spec.return_sd, (t_max, n))

    returns = np.zeros((t_max, n))
    week = np.zeros(n)  # running sum of the last 5 returns
    for t in range(t_max):
        drift = spec.beta * _zscore_rows(-week[None, :])[0] if t >= 5 else 0.0
        returns[t] = drift + noise[t]
        week += returns[t] - (returns[t - 5] if t >= 5 else 0.0)

    price0 = rng.uniform(10, 100, n)
    close = price0 * np.cumprod(1 + returns, axis=0)
    prev = np.vstack([price0, close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.005, (t_max, n)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, (t_max, n))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, (t_max, n))))
    shares_panel = np.broadcast_to(shares, (t_max, n)).copy()
    cap = close * shares_panel
    prev_cap = np.vstack([price0 * shares, cap[:-1]])
    mkt = (prev_cap * returns).sum(axis=1) / prev_cap.sum(axis=1)

    # Drawn last, so every daily field is what it was before fundamentals existed.
    fundamentals = _fundamentals(rng, t_max, n)

    dates = pd.bdate_range("2000-01-03", periods=t_max)
    stocks = [f"S{i:03d}" for i in range(n)]

    def frame(x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(x, index=dates, columns=stocks)

    return {
        "open": frame(open_),
        "high": frame(high),
        "low": frame(low),
        "close": frame(close),
        "volume": frame(np.exp(log_volume)),
        "returns": frame(returns),
        "shares": frame(shares_panel),
        "cap": frame(cap),
        "mkt_return": pd.Series(mkt, index=dates),
        **{name: frame(values) for name, values in fundamentals.items()},
    }


# A synthetic fiscal year is 252 rows; its values are visible 63 rows (about
# 90 calendar days) after it ends, as `sd.available_from`'s annual rule.
FISCAL_YEAR, REPORT_LAG = 252, 63


def _fundamentals(rng: np.random.Generator, t_max: int, n: int) -> dict[str, np.ndarray]:
    """Annual book, earnings, sales and assets: the year ending at row 252k - 1
    is visible from row 252k - 1 + 63 to the next year's, NaN before the first.
    No planted signal."""
    visible = np.arange(REPORT_LAG - 1, t_max, FISCAL_YEAR)
    assets = np.exp(rng.normal(8.0, 0.5, n))
    out = {name: np.full((t_max, n), np.nan) for name in FUNDAMENTAL_FIELDS}
    for start in visible:
        assets = assets * np.exp(rng.normal(0.05, 0.1, n))
        year = {
            "book": assets * rng.uniform(0.2, 0.6, n),
            "earnings": assets * rng.normal(0.05, 0.03, n),
            "sales": assets * rng.uniform(0.5, 1.5, n),
            "assets": assets,
        }
        for name in FUNDAMENTAL_FIELDS:
            out[name][start:] = year[name]
    return out


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Sum of returns from t+1 to t+h, on day t; NaN where that runs past the data."""
    return returns.rolling(horizon).sum().shift(-horizon)


def icir(values: pd.DataFrame, forward: pd.DataFrame, step: int = 1, min_stocks: int = 20):
    """mean / std of the cross-sectional rank IC on every `step`-th day; NaN below two days."""
    v = values.to_numpy(dtype=float)[::step]
    f = forward.to_numpy(dtype=float)[::step]
    both = np.isfinite(v) & np.isfinite(f)
    v = pd.DataFrame(np.where(both, v, np.nan)).rank(axis=1).to_numpy()
    f = pd.DataFrame(np.where(both, f, np.nan)).rank(axis=1).to_numpy()
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # days with no finite pair
        v = v - np.nanmean(v, axis=1, keepdims=True)
        f = f - np.nanmean(f, axis=1, keepdims=True)
        ic = np.nansum(v * f, axis=1) / np.sqrt(np.nansum(v * v, axis=1) * np.nansum(f * f, axis=1))
    ic = ic[(both.sum(axis=1) >= min_stocks) & np.isfinite(ic)]
    if len(ic) < 2 or ic.std(ddof=1) == 0:
        return float("nan")
    return float(ic.mean() / ic.std(ddof=1))


class PanelScorer:
    """Q = |ICIR| at horizon h, and the factor's values on a fixed sample of cells.

    Results are cached by identity key: each distinct factor is computed once.
    """

    def __init__(
        self,
        panel: dict,
        horizon: int = 20,
        *,
        ic_step: int = 5,
        sample_cells: int = 20_000,
        sample_seed: int = 0,
    ):
        self.panel = panel
        self.ic_step = ic_step
        self.forward = forward_returns(panel["returns"], horizon)
        rng = np.random.default_rng(sample_seed)
        size = self.forward.size
        self.sample = np.sort(rng.choice(size, size=min(sample_cells, size), replace=False))
        self.cache: OrderedDict[str, Scored | None] = OrderedDict()
        self.computed = 0

    def __call__(self, tree: Node) -> Scored | None:
        key = identity_key(tree)
        if key not in self.cache:
            self.computed += 1
            self.cache[key] = self._score(tree)
        return self.cache[key]

    def _score(self, tree: Node) -> Scored | None:
        try:
            frame = evaluate(tree, self.panel)
        except (ValueError, ArithmeticError):
            return None
        q = abs(icir(frame, self.forward, self.ic_step))
        if not np.isfinite(q):
            return None
        with np.errstate(over="ignore"):  # beyond float32 is inf, which counts as missing
            values = frame.to_numpy(dtype=np.float32).ravel()[self.sample]
        return Scored(q, values)
