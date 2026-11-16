"""Run an incubation over a returns series: the monitor's scale and a synthetic live series.

The gate gives each accepted family an expected mean and a scale (its
search-period sd). Serially correlated returns break the monitor's
assumption 1 (`capstone.incubation.monitor`): AR(1) 0.2 lifts its false-retire rate to
6.5%. `incubate` inflates the scale by the
long-run sd factor of the search-period returns. There is no live data
source yet, so `live_continuation` makes one from the search returns.

With `confirming` (incubation-only families) the confirming e-process
(`capstone.incubation.confirm`) runs beside the decay monitor, both to
`confirm_horizon`, with the same inflated scale: promote, retire or continue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone.evaluate import newey_west_lags
from capstone.incubation.confirm import ConfirmConfig, ConfirmState, confirm
from capstone.incubation.monitor import MonitorConfig, MonitorState, monitor


@dataclass(frozen=True)
class IncubationSettings:
    """The decay monitor's alpha and horizon, and the confirming process's.

    An incubation-only family also runs the confirming e-process at `confirm_alpha`
    and is promoted or retired within `confirm_horizon` observed periods (its decay
    monitor runs to that horizon too). The values are the private testbed's; adopting
    an incubation policy (horizon, alpha, who decides) is the team's call.
    """

    alpha: float = 0.05
    horizon: int = 252
    block: int = 20
    confirm_alpha: float = 0.05
    confirm_horizon: int = 1260

    def __post_init__(self) -> None:
        for name in ("alpha", "confirm_alpha"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        for name in ("horizon", "confirm_horizon", "block"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Incubation:
    family: str
    config: MonitorConfig
    lr_factor: float  # the long-run sd factor the gate's scale was inflated by
    state: MonitorState  # the decay monitor
    confirm: ConfirmState | None = None  # with `confirming`; its `decay` is `state`

    @property
    def decision(self) -> str:
        """continue / retire / keep (decay only) or continue / promote / retire."""
        return self.state.decision if self.confirm is None else self.confirm.decision


def long_run_sd_factor(returns: pd.Series, lags: int | None = None) -> float:
    """sqrt(long-run variance / variance) of a series, at least 1.

    Newey-West (Bartlett) with `evaluate.newey_west_lags` by default:
    sqrt((1 + phi) / (1 - phi)) for an AR(1) phi. Floored at 1 so a series
    with negative autocorrelation never shrinks the monitor's scale.
    """
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if x.size == 0:
        return 1.0
    x = x - x.mean()
    n = len(x)
    gamma0 = float(x @ x) / n if n else 0.0
    if not gamma0 > 0:
        return 1.0
    lags = newey_west_lags(n) if lags is None else lags
    lrv = gamma0 + 2 * sum(
        (1 - j / (lags + 1)) * float(x[j:] @ x[:-j]) / n for j in range(1, min(lags, n - 1) + 1)
    )
    return max(1.0, math.sqrt(lrv / gamma0)) if lrv > 0 else 1.0


def live_continuation(
    search_returns: pd.Series, *, mean: float, periods: int, seed: int, block: int
) -> np.ndarray:
    """A synthetic live series: the search returns' deviations from their mean,
    resampled in circular blocks (keeping their serial correlation), plus `mean`."""
    x = np.asarray(pd.Series(search_returns).dropna(), dtype=float)
    deviations = x - x.mean()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), size=-(-periods // block))
    index = (starts[:, None] + np.arange(block)[None, :]).ravel()[:periods] % len(x)
    return mean + deviations[index]


def incubate(
    settings: IncubationSettings,
    *,
    family: str,
    expected_mean: float,
    scale: float,
    search_returns: pd.Series,
    live,
    confirming: bool = False,
) -> Incubation:
    """Monitor `live` against the gate's expected mean; the scale is the gate's
    times `long_run_sd_factor(search_returns)`. With `confirming`, also the
    confirming e-process at that scale, both to `settings.confirm_horizon`.
    Pure: logs nothing."""
    factor = long_run_sd_factor(search_returns)
    config = MonitorConfig(
        expected_mean=float(expected_mean),
        scale=float(scale) * factor,
        alpha=settings.alpha,
        horizon=settings.confirm_horizon if confirming else settings.horizon,
    )
    if not confirming:
        return Incubation(family, config, factor, monitor(live, MonitorState.start(config)))
    confirm_config = ConfirmConfig(
        scale=config.scale, alpha=settings.confirm_alpha, horizon=settings.confirm_horizon
    )
    state = confirm(live, ConfirmState.start(confirm_config, config))
    return Incubation(family, config, factor, state.decay, state)
