"""Synthetic panels whose true signal set is known by construction.

Real data cannot tell you whether a discovery procedure works, because the true
number of real signals in the market is unknown. Data you generated yourself
can: you know exactly which signals were planted, so "does our gate control the
false-discovery rate?" becomes something you measure rather than assert.

This module is offered, not prescribed. If you would rather build your own
generator — or argue that a different control is better — do that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SyntheticPanel:
    """A generated panel together with the truth used to generate it.

    Attributes:
        returns: asset returns, dates x assets.
        signals: candidate signals, one column per candidate.
        truth: True for candidates that were planted with real predictive power.
        params: the settings used, so a result can be traced to its generator.
    """

    returns: pd.DataFrame
    signals: pd.DataFrame
    truth: pd.Series
    params: dict = field(default_factory=dict)

    @property
    def n_true(self) -> int:
        return int(self.truth.sum())

    @property
    def n_null(self) -> int:
        return int((~self.truth).sum())

    @property
    def n_candidates(self) -> int:
        # signals has MultiIndex columns (candidate, asset), so its width is
        # candidates x assets — count the candidate level, not the columns.
        return len(self.truth)

    def __repr__(self) -> str:
        return (
            f"SyntheticPanel({len(self.returns)} dates x {self.returns.shape[1]} assets, "
            f"{self.n_candidates} candidates, {self.n_true} real / {self.n_null} null)"
        )


def make_panel(
    n_dates: int = 2520,
    n_assets: int = 100,
    n_candidates: int = 200,
    n_real: int = 0,
    effect_size: float = 0.02,
    vol: float = 0.015,
    seed: int | None = 0,
) -> SyntheticPanel:
    """Generate a panel with a known number of genuinely predictive signals.

    Args:
        n_dates: number of periods (2520 ~ ten years of trading days).
        n_assets: width of the cross-section.
        n_candidates: how many candidate signals to generate.
        n_real: how many of those candidates actually predict returns.
            The default of 0 gives a pure null panel — the most useful
            single test case in this module.
        effect_size: predictive strength of the real signals, as the
            correlation between the signal and next-period returns.
        vol: per-period return standard deviation.
        seed: RNG seed. Pass a value to make results reproducible.

    Returns:
        SyntheticPanel carrying the data and the ground truth.

    Note that signals predict the NEXT period's return: `signals.loc[t]` is
    information available at `t`, targeting the return realised at `t+1`. This
    matches what `backtest.run_backtest` assumes.
    """
    if n_real > n_candidates:
        raise ValueError("n_real cannot exceed n_candidates")
    if not 0.0 <= effect_size < 1.0:
        raise ValueError("effect_size must be in [0, 1)")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_dates)
    assets = [f"A{i:03d}" for i in range(n_assets)]

    noise = rng.standard_normal((n_dates, n_assets))
    returns = pd.DataFrame(noise * vol, index=dates, columns=assets)

    truth = pd.Series(False, index=[f"cand_{i:04d}" for i in range(n_candidates)])
    if n_real:
        truth.iloc[rng.choice(n_candidates, size=n_real, replace=False)] = True

    signals = {}
    for name, is_real in truth.items():
        raw = rng.standard_normal((n_dates, n_assets))
        if is_real:
            # Blend in next period's return so the signal genuinely leads it.
            future = np.vstack([noise[1:], rng.standard_normal((1, n_assets))])
            blended = effect_size * future + np.sqrt(1 - effect_size**2) * raw
            signals[name] = pd.DataFrame(blended, index=dates, columns=assets)
        else:
            signals[name] = pd.DataFrame(raw, index=dates, columns=assets)

    panel = pd.concat(signals, axis=1)

    return SyntheticPanel(
        returns=returns,
        signals=panel,
        truth=truth,
        params={
            "n_dates": n_dates,
            "n_assets": n_assets,
            "n_candidates": n_candidates,
            "n_real": n_real,
            "effect_size": effect_size,
            "vol": vol,
            "seed": seed,
        },
    )


def candidate_frames(panel: SyntheticPanel):
    """Iterate over (name, signal_frame, is_real) for each candidate."""
    for name in panel.truth.index:
        yield name, panel.signals[name], bool(panel.truth[name])


def bootstrap_from_real(
    returns: pd.DataFrame,
    n_candidates: int = 200,
    seed: int | None = 0,
) -> SyntheticPanel:
    """Build a null panel that keeps a real panel's statistical character.

    Resamples observed return rows with replacement, which preserves the
    cross-sectional covariance and the fat tails of the source while destroying
    any time-series structure. Every candidate here is null by construction, so
    anything a procedure "finds" in the result is a false positive.

    Useful when a reviewer objects that Gaussian synthetic data is too easy.
    """
    rng = np.random.default_rng(seed)
    clean = returns.dropna(how="all")
    if clean.empty:
        raise ValueError("returns contain no usable rows")

    rows = rng.integers(0, len(clean), size=len(clean))
    resampled = pd.DataFrame(clean.to_numpy()[rows], index=clean.index, columns=clean.columns)

    truth = pd.Series(False, index=[f"cand_{i:04d}" for i in range(n_candidates)])
    signals = {
        name: pd.DataFrame(
            rng.standard_normal(resampled.shape),
            index=resampled.index,
            columns=resampled.columns,
        )
        for name in truth.index
    }

    return SyntheticPanel(
        returns=resampled,
        signals=pd.concat(signals, axis=1),
        truth=truth,
        params={"source": "bootstrap", "n_candidates": n_candidates, "seed": seed},
    )
