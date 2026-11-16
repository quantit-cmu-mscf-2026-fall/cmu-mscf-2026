"""Anytime-valid incubation monitor: has a live strategy's edge decayed?

Each period's realised return ``r_t`` arrives one at a time. The monitor tests

    H0: the strategy still earns its expected mean,  E[r_t | past] >= mu
    H1: its edge has decayed,                         E[r_t | past] <  mu

with an e-process: a running statistic whose chance of ever reaching 1/alpha
under H0 is at most alpha, however often it is looked at and whenever the
looking stops (Ville's inequality). Checking it every period therefore costs
nothing in false retirements.

Construction: a mixture of Catoni-style betting supermartingales (Catoni 2012;
Wang & Ramdas 2023, "Catoni-style confidence sequences for heavy-tailed mean
estimation", Stochastic Processes and their Applications 163). With the
standardised shortfall ``x_t = (mu - r_t) / scale`` and Catoni's influence
function ``phi(u) = sign(u) * log(1 + |u| + u**2 / 2)``, each bet ``lam > 0``
has wealth

    W_t(lam) = prod_{i <= t} exp(phi(lam * x_i) - lam**2 / 2)

and the e-process is the equal-weight average of ``W_t`` over a fixed grid of
bets (an average of e-processes is one; Ramdas, Grunwald, Vovk & Shafer 2023,
"Game-theoretic statistics and safe anytime-valid inference", Statistical
Science). The grid spans the shortfalls worth detecting without tuning to one.

Why it is valid: ``phi(u) <= log(1 + u + u**2 / 2)`` for every u, so
``E[exp(phi(lam x)) | past] <= 1 + lam E[x | past] + lam**2 E[x**2 | past] / 2
<= exp(lam**2 / 2)`` whenever the assumptions below hold; each ``W_t`` is then a
non-negative supermartingale starting at 1.

Assumptions under H0, conditional on the past:

1. ``E[r_t] >= mu`` (``expected_mean``; the composite null includes better).
2. ``E[(r_t - mu)**2] <= scale**2``: ``scale`` bounds the second moment about
   ``mu``. No distributional shape is assumed: fat tails are fine as long as
   the variance is finite, and ``phi``'s logarithmic growth keeps one extreme
   period from deciding on its own.
3. The bets are fixed before monitoring starts.

Serial correlation breaks assumption 1 conditionally (positively
autocorrelated returns drift away from ``mu`` given the past). Measured on
AR(1) 0.2 nulls (2,000 paths of 1,000 periods, alpha 0.05): 0.065 unadjusted,
0.015 with ``scale`` inflated by ``sqrt((1 + phi) / (1 - phi))``
(tests/test_incubation.py).
A rise in volatility alone also violates assumption 2 and can retire a
strategy whose mean is intact.

Decisions: ``retire`` once the e-value reaches 1/alpha; ``keep`` once
``horizon`` observed periods pass without that (not a confirmation of the
edge, only the end of the monitoring budget without evidence of decay);
``continue`` otherwise. Both final decisions are sticky. Periods with no
position (NaN) are skipped: they neither count towards the horizon nor move
the e-value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

Decision = Literal["continue", "retire", "keep"]

DEFAULT_BETS: tuple[float, ...] = tuple(2.0**-k for k in range(8))  # 1, 1/2, ..., 1/128


@dataclass(frozen=True)
class MonitorConfig:
    """What the monitor tests and when it stops.

    Args:
        expected_mean: per-period mean the strategy is expected to earn (H0's
            boundary). Pass a haircut figure rather than the raw backtest mean
            when the backtest was selected from many trials.
        scale: bound on the per-period standard deviation about
            ``expected_mean`` (assumption 2 of the module docstring).
        alpha: false-retirement probability over the whole run.
        horizon: observed periods after which an unretired strategy is kept.
        bets: the grid of bet sizes, in units of ``scale``, averaged equally.
    """

    expected_mean: float
    scale: float
    alpha: float = 0.05
    horizon: int = 252
    bets: tuple[float, ...] = field(default=DEFAULT_BETS)

    def __post_init__(self) -> None:
        if not math.isfinite(self.expected_mean):
            raise ValueError("expected_mean must be finite")
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError("scale must be positive and finite")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if int(self.horizon) != self.horizon or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        bets = tuple(float(b) for b in self.bets)
        if not bets or not all(math.isfinite(b) and b > 0 for b in bets):
            raise ValueError("bets must be a non-empty set of positive, finite sizes")
        object.__setattr__(self, "bets", bets)
        object.__setattr__(self, "horizon", int(self.horizon))

    @classmethod
    def from_backtest(
        cls, backtest_returns: Iterable[float], *, alpha: float = 0.05, horizon: int = 252
    ) -> MonitorConfig:
        """Expected mean and scale from a backtest's per-period returns (NaNs dropped).

        The scale is the sample standard deviation (ddof 1). Both are estimates
        taken as known; the guarantee is exact only if they are right.
        """
        values = np.asarray(list(backtest_returns), dtype=float)
        values = values[~np.isnan(values)]
        if len(values) < 2:
            raise ValueError("need at least two backtest returns")
        return cls(
            expected_mean=float(values.mean()),
            scale=float(values.std(ddof=1)),
            alpha=alpha,
            horizon=horizon,
        )


def log_increments(x: np.ndarray, bets: tuple[float, ...]) -> list[np.ndarray]:
    """Per-bet log-wealth increments ``phi(lam * x) - lam**2 / 2`` of standardised
    observations ``x``; NaN periods give 0. Shared with `incubation.confirm`."""
    observed = ~np.isnan(x)
    out = []
    for lam in bets:
        u = lam * np.where(observed, x, 0.0)
        step = np.sign(u) * np.log1p(np.abs(u) + u * u / 2) - lam * lam / 2
        out.append(np.where(observed, step, 0.0))
    return out


def _increments(config: MonitorConfig, returns: np.ndarray) -> list[np.ndarray]:
    """The decay bets' increments: ``x`` is the standardised shortfall ``(mu - r) / scale``."""
    return log_increments((config.expected_mean - returns) / config.scale, config.bets)


def log_mixture(log_wealth: list[np.ndarray] | list[float], n_bets: int) -> np.ndarray:
    """Log of the equal-weight average of the bets' wealth."""
    total = log_wealth[0]
    for w in log_wealth[1:]:
        total = np.logaddexp(total, w)
    return np.asarray(total) - math.log(n_bets)


def e_process(config: MonitorConfig, returns: ArrayLike) -> np.ndarray:
    """E-values after each period, for a series (periods,) or many (periods, paths).

    The batch form of feeding ``MonitorState.update`` one period at a time,
    without stopping at a decision. A NaN period carries the e-value forward.
    """
    values = np.asarray(returns, dtype=float)
    if np.isinf(values).any():
        raise ValueError("returns must be finite or NaN")
    log_wealth = [np.cumsum(step, axis=0) for step in _increments(config, values)]
    return np.exp(log_mixture(log_wealth, len(config.bets)))


@dataclass(frozen=True)
class MonitorState:
    """The monitor after ``n_periods`` periods, ``n_obs`` of them with a return.

    Immutable: ``update`` returns the next state. ``to_dict`` / ``from_dict``
    round-trip exactly (through JSON too), so monitoring can resume anywhere.
    """

    config: MonitorConfig
    log_wealth: tuple[float, ...]
    n_obs: int = 0
    n_periods: int = 0
    decision: Decision = "continue"

    @classmethod
    def start(cls, config: MonitorConfig) -> MonitorState:
        return cls(config=config, log_wealth=(0.0,) * len(config.bets))

    @property
    def e_value(self) -> float:
        return float(np.exp(log_mixture(list(self.log_wealth), len(self.log_wealth))))

    def update(self, realised_return: float) -> MonitorState:
        """Take one period's realised return (NaN: no position that period)."""
        if self.decision != "continue":
            raise ValueError(f"monitoring has ended with '{self.decision}'")
        r = float(realised_return)
        if math.isinf(r):
            raise ValueError("returns must be finite or NaN")
        if math.isnan(r):
            return replace(self, n_periods=self.n_periods + 1)
        steps = _increments(self.config, np.array(r))
        log_wealth = tuple(float(w + s) for w, s in zip(self.log_wealth, steps, strict=True))
        state = replace(
            self, log_wealth=log_wealth, n_obs=self.n_obs + 1, n_periods=self.n_periods + 1
        )
        if state.e_value >= 1 / self.config.alpha:
            return replace(state, decision="retire")
        if state.n_obs >= self.config.horizon:
            return replace(state, decision="keep")
        return state

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["config"]["bets"] = list(self.config.bets)
        data["log_wealth"] = list(self.log_wealth)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonitorState:
        config = MonitorConfig(**{**data["config"], "bets": tuple(data["config"]["bets"])})
        log_wealth = tuple(float(w) for w in data["log_wealth"])
        if len(log_wealth) != len(config.bets):
            raise ValueError("log_wealth must have one entry per bet in bets")
        if data["decision"] not in ("continue", "retire", "keep"):
            raise ValueError(f"unknown decision {data['decision']!r}")
        return cls(
            config=config,
            log_wealth=log_wealth,
            n_obs=int(data["n_obs"]),
            n_periods=int(data["n_periods"]),
            decision=data["decision"],
        )


def monitor(returns: Iterable[float], state: MonitorState) -> MonitorState:
    """Feed a returns series (any iterable: list, array, Series) until a final decision.

    Periods after a final decision are not consumed; ``state.n_periods`` says
    how many were.
    """
    for r in returns:
        if state.decision != "continue":
            break
        state = state.update(r)
    return state
