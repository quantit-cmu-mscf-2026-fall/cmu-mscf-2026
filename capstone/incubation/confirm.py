"""Anytime-valid confirmation: has an incubating strategy shown an edge live?

The decay monitor (`incubation.monitor`) tests whether a strategy still earns
its expected mean. For an incubation-only strategy that is the wrong question:
the gate's expected mean for it is haircut to about 0, so a zero live edge is
the decay monitor's null and it is (almost) never retired. This module adds
the other one-sided test,

    H0: no edge,          E[r_t | past] <= 0
    H1: a positive edge,  E[r_t | past] >  0

with the same construction as the decay monitor: a mixture of Catoni-style
betting supermartingales (Catoni 2012; Wang & Ramdas 2023, "Catoni-style
confidence sequences for heavy-tailed mean estimation", Stochastic Processes
and their Applications 163), averaged equally over a fixed grid of bets
(Ramdas, Grunwald, Vovk & Shafer 2023, "Game-theoretic statistics and safe
anytime-valid inference", Statistical Science). With ``x_t = r_t / scale`` and
``phi(u) = sign(u) * log(1 + |u| + u**2 / 2)``, each bet ``lam > 0`` has wealth

    W_t(lam) = prod_{i <= t} exp(phi(lam * x_i) - lam**2 / 2)

Why it is valid: ``phi(u) <= log(1 + u + u**2 / 2)``, so
``E[exp(phi(lam x)) | past] <= 1 + lam E[x | past] + lam**2 E[x**2 | past] / 2
<= exp(lam**2 / 2)`` under the assumptions below; each ``W_t`` is a
non-negative supermartingale from 1 and by Ville's inequality the mixture
reaches 1/alpha with probability at most alpha, however often it is checked.

Assumptions under H0, conditional on the past:

1. ``E[r_t] <= 0`` (the composite null includes a losing strategy).
2. ``E[r_t**2] <= scale**2``: ``scale`` bounds the second moment about 0,
   which at the boundary (mean 0) is the variance. A variance bound is enough
   for any mean ``m`` in ``[-2 scale / max(bets), 0]`` (``lam m + lam**2
   (v + m**2) / 2 <= lam**2 v / 2`` iff ``m (1 + lam m / 2) <= 0``), i.e. any
   daily mean above -16 standard deviations for the default grid. Fat tails
   are fine while the variance is finite.
3. The bets are fixed before monitoring starts.

Serial correlation breaks assumption 1 conditionally, as for the decay
monitor; the caller inflates ``scale`` by the long-run sd factor of the
search-period returns. Measured on
AR(1) 0.2 nulls in tests/test_incubation_confirm.py.

The bets (`CONFIRM_BETS`, in units of ``scale``) are smaller than the decay
monitor's: the edges worth confirming are small per period (an annualised
Sharpe of 0.25-2 is a daily one of 0.016-0.126, and the best fixed bet is about
the per-period Sharpe), and each bet in the mixture costs log(len(bets)) of
evidence. The price of anytime validity is time: the power-per-horizon table in
tests/test_incubation_confirm.py shows how long a given Sharpe takes to confirm.

Incubation for an incubation-only strategy (`ConfirmState`) runs this and the
decay monitor side by side on the same returns. After each observed period:

- ``retire`` ("decayed") once the decay e-value reaches 1/alpha_decay;
- else ``promote`` ("confirmed") once the confirming e-value reaches
  1/alpha. This only reports: promotion still needs a human / registry change;
- else ``retire`` ("not confirmed") once ``horizon`` observed periods pass;
- else ``continue``.

So with no edge it is promoted with probability at most alpha (the decay
process only removes chances) and retired otherwise by the horizon. Both final
decisions are sticky; periods with no position (NaN) are skipped.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from capstone.incubation.monitor import (
    MonitorConfig,
    MonitorState,
    e_process,
    log_increments,
    log_mixture,
)

Decision = Literal["continue", "promote", "retire"]
Reason = Literal["", "confirmed", "decayed", "not confirmed"]

CONFIRM_BETS: tuple[float, ...] = tuple(2.0**-k for k in range(3, 7))  # 1/8, ..., 1/64
DEFAULT_HORIZON = 1260  # five years of daily periods


@dataclass(frozen=True)
class ConfirmConfig:
    """What the confirming process tests and when incubation stops.

    Args:
        scale: bound on the per-period standard deviation (assumption 2).
        alpha: false-promotion probability over the whole run.
        horizon: observed periods after which an unconfirmed strategy is retired.
        bets: the grid of bet sizes, in units of ``scale``, averaged equally.
    """

    scale: float
    alpha: float = 0.05
    horizon: int = DEFAULT_HORIZON
    bets: tuple[float, ...] = field(default=CONFIRM_BETS)

    def __post_init__(self) -> None:
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


def _finite_or_nan(returns: ArrayLike) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if np.isinf(values).any():
        raise ValueError("returns must be finite or NaN")
    return values


def confirm_e_process(config: ConfirmConfig, returns: ArrayLike) -> np.ndarray:
    """Confirming e-values after each period, for (periods,) or (periods, paths).

    The batch form of the confirming half of `ConfirmState.update`, without
    stopping. A NaN period carries the e-value forward.
    """
    values = _finite_or_nan(returns)
    steps = log_increments(values / config.scale, config.bets)
    return np.exp(log_mixture([np.cumsum(s, axis=0) for s in steps], len(config.bets)))


def decide_paths(
    config: ConfirmConfig, decay: MonitorConfig, returns: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """(decision, observed periods used) per path of a (periods, paths) matrix:
    what `confirm` from `ConfirmState.start(config, decay)` returns on each
    column, vectorised (Monte Carlo)."""
    values = _finite_or_nan(returns)
    values = values[:, None] if values.ndim == 1 else values
    _check_horizons(config, decay)
    observed = ~np.isnan(values)
    n_obs = np.cumsum(observed, axis=0)
    decayed = observed & (e_process(decay, values) >= 1 / decay.alpha)
    confirmed = observed & (confirm_e_process(config, values) >= 1 / config.alpha)
    ended = observed & (n_obs >= config.horizon)
    stop = decayed | confirmed | ended
    stopped = stop.any(axis=0)
    first = np.argmax(stop, axis=0)
    paths = np.arange(values.shape[1])
    decisions = np.where(
        ~stopped,
        "continue",
        np.where(
            decayed[first, paths] | ~confirmed[first, paths],
            "retire",
            "promote",
        ),
    )
    used = np.where(stopped, n_obs[first, paths], n_obs[-1] if len(values) else 0)
    return decisions, used


def _check_horizons(config: ConfirmConfig, decay: MonitorConfig) -> None:
    if decay.horizon != config.horizon:
        raise ValueError(
            f"the decay monitor's horizon ({decay.horizon}) must be the confirm "
            f"horizon ({config.horizon}): both run on the same periods"
        )


@dataclass(frozen=True)
class ConfirmState:
    """Incubation of an incubation-only strategy: the decay monitor and the
    confirming process after the same periods.

    Immutable: ``update`` returns the next state. ``to_dict`` / ``from_dict``
    round-trip exactly (through JSON too), so incubation can resume anywhere.
    """

    config: ConfirmConfig
    decay: MonitorState
    log_wealth: tuple[float, ...]
    decision: Decision = "continue"
    reason: Reason = ""

    @classmethod
    def start(cls, config: ConfirmConfig, decay: MonitorConfig) -> ConfirmState:
        _check_horizons(config, decay)
        return cls(
            config=config, decay=MonitorState.start(decay), log_wealth=(0.0,) * len(config.bets)
        )

    @property
    def e_value(self) -> float:
        """The confirming e-value (the decay one is ``decay.e_value``)."""
        return float(np.exp(log_mixture(list(self.log_wealth), len(self.log_wealth))))

    @property
    def n_obs(self) -> int:
        return self.decay.n_obs

    @property
    def n_periods(self) -> int:
        return self.decay.n_periods

    def update(self, realised_return: float) -> ConfirmState:
        """Take one period's realised return (NaN: no position that period)."""
        if self.decision != "continue":
            raise ValueError(f"incubation has ended with '{self.decision}' ({self.reason})")
        r = float(realised_return)
        decay = self.decay.update(r)  # raises on an infinite return
        if math.isnan(r):
            return replace(self, decay=decay)
        steps = log_increments(np.array(r / self.config.scale), self.config.bets)
        log_wealth = tuple(float(w + s) for w, s in zip(self.log_wealth, steps, strict=True))
        state = replace(self, decay=decay, log_wealth=log_wealth)
        if decay.decision == "retire":
            return replace(state, decision="retire", reason="decayed")
        if state.e_value >= 1 / self.config.alpha:
            return replace(state, decision="promote", reason="confirmed")
        if state.n_obs >= self.config.horizon:
            return replace(state, decision="retire", reason="not confirmed")
        return state

    def to_dict(self) -> dict[str, Any]:
        config = asdict(self.config)
        config["bets"] = list(self.config.bets)
        return {
            "config": config,
            "decay": self.decay.to_dict(),
            "log_wealth": list(self.log_wealth),
            "decision": self.decision,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfirmState:
        config = ConfirmConfig(**{**data["config"], "bets": tuple(data["config"]["bets"])})
        decay = MonitorState.from_dict(data["decay"])
        _check_horizons(config, decay.config)
        log_wealth = tuple(float(w) for w in data["log_wealth"])
        if len(log_wealth) != len(config.bets):
            raise ValueError("log_wealth must have one entry per bet in bets")
        if data["decision"] not in ("continue", "promote", "retire"):
            raise ValueError(f"unknown decision {data['decision']!r}")
        if data["reason"] not in ("", "confirmed", "decayed", "not confirmed"):
            raise ValueError(f"unknown reason {data['reason']!r}")
        return cls(config, decay, log_wealth, data["decision"], data["reason"])


def confirm(returns: Iterable[float], state: ConfirmState) -> ConfirmState:
    """Feed a returns series (any iterable) until a final decision; periods after
    it are not consumed (``state.n_periods`` says how many were)."""
    for r in returns:
        if state.decision != "continue":
            break
        state = state.update(r)
    return state
