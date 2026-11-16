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
from scipy.signal import lfilter


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


@dataclass
class ReturnMatrix:
    """Candidate return series generated directly, with the truth used to make them.

    The input every validation method takes: one column of per-period strategy
    returns per candidate. `SyntheticPanel` plus `backtest.candidate_returns`
    produces the same shape the long way round; this skips the signals, which is
    what makes thousands of candidates, and the properties real strategy returns
    have and `make_panel` cannot give them, cheap to generate.

    Attributes:
        returns: per-period returns, dates x candidates.
        truth: True for candidates planted with a positive expected return.
        params: the settings used, so a result can be traced to its generator.
    """

    returns: pd.DataFrame
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
        return len(self.truth)

    def __repr__(self) -> str:
        return (
            f"ReturnMatrix({len(self.returns)} dates x {self.n_candidates} candidates, "
            f"{self.n_true} real / {self.n_null} null)"
        )


def make_return_matrix(
    n_obs: int = 2520,
    n_candidates: int = 200,
    n_real: int = 0,
    *,
    sharpe_real: float = 1.0,
    vol: float = 0.01,
    rho: float = 0.0,
    ar1: float = 0.0,
    t_df: float | None = None,
    garch: tuple[float, float] | None = None,
    periods_per_year: int = 252,
    seed: int | None = 0,
) -> ReturnMatrix:
    """Generate candidate return series with known truth and realistic defects.

    Each option adds one property that breaks an assumption some validation
    method makes, so a method can be tested against exactly that property:

    - `rho`: every pair of candidates has return correlation `rho`, through one
      common factor. Candidates from one search are correlated; BH's guarantee,
      Bonferroni's power and the "independent trials" in the deflated Sharpe
      ratio all assume otherwise.
    - `ar1`: each series is AR(1) with this coefficient. The IID Sharpe
      standard error is wrong under autocorrelation.
    - `t_df`: innovations are Student-t with this many degrees of freedom,
      rescaled to unit variance. Fat tails widen the Sharpe estimate's spread.
    - `garch`: `(alpha, beta)` of a GARCH(1,1) variance process with
      unconditional variance 1. Volatility clustering is what an IID bootstrap
      destroys.

    All options keep each column's unconditional mean and standard deviation
    where the arguments put them, so a planted candidate's population Sharpe is
    `sharpe_real` and a null candidate's is 0 whatever else is switched on.
    Under `garch`, pairwise correlation comes out somewhat below `rho`, because
    each column's variance moves on its own.

    Args:
        n_obs: number of periods (2520 ~ ten years of trading days).
        n_candidates: number of candidate series.
        n_real: how many carry a genuine positive expected return. The default
            of 0 is a pure null matrix.
        sharpe_real: annualised population Sharpe of the planted candidates.
        vol: per-period return standard deviation of every candidate.
        rho: pairwise return correlation, in [0, 1).
        ar1: AR(1) coefficient of each series, in (-1, 1).
        t_df: Student-t degrees of freedom (> 2), or None for normal innovations.
        garch: GARCH(1,1) `(alpha, beta)`, both >= 0 with alpha + beta < 1, or None.
        periods_per_year: annualisation factor behind `sharpe_real`.
        seed: RNG seed. Pass a value to make results reproducible.

    Returns:
        ReturnMatrix carrying the returns and the ground truth.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")
    if not 0 <= n_real <= n_candidates:
        raise ValueError("n_real must be in [0, n_candidates]")
    if vol <= 0:
        raise ValueError("vol must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if not -1.0 < ar1 < 1.0:
        raise ValueError("ar1 must be in (-1, 1)")
    if t_df is not None and t_df <= 2:
        raise ValueError("t_df must exceed 2, or the variance is infinite")
    if garch is not None:
        alpha, beta = garch
        if alpha < 0 or beta < 0 or alpha + beta >= 1:
            raise ValueError("garch needs alpha, beta >= 0 and alpha + beta < 1")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1")

    rng = np.random.default_rng(seed)

    def unit_innovations(shape: tuple[int, int]) -> np.ndarray:
        if t_df is None:
            return rng.standard_normal(shape)
        return rng.standard_t(t_df, shape) / np.sqrt(t_df / (t_df - 2))

    common = unit_innovations((n_obs, 1))
    own = unit_innovations((n_obs, n_candidates))
    shocks = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * own

    if garch is not None:
        alpha, beta = garch
        omega = 1.0 - alpha - beta  # unconditional variance of 1
        variance = np.ones(n_candidates)
        for t in range(n_obs):
            shocks[t] *= np.sqrt(variance)
            variance = omega + alpha * shocks[t] ** 2 + beta * variance

    if ar1 != 0.0:
        # u_t = ar1 * u_{t-1} + sqrt(1 - ar1^2) * e_t keeps unit variance; starting
        # from u_0 = e_0 starts the process in its stationary distribution.
        rest = lfilter(
            [np.sqrt(1.0 - ar1**2)], [1.0, -ar1], shocks[1:], axis=0, zi=ar1 * shocks[:1]
        )[0]
        shocks = np.vstack([shocks[:1], rest])

    names = [f"cand_{i:04d}" for i in range(n_candidates)]
    truth = pd.Series(False, index=names)
    if n_real:
        truth.iloc[rng.choice(n_candidates, size=n_real, replace=False)] = True

    mean = np.where(truth.to_numpy(), sharpe_real / np.sqrt(periods_per_year) * vol, 0.0)
    returns = pd.DataFrame(
        mean + vol * shocks,
        index=pd.bdate_range("2015-01-01", periods=n_obs),
        columns=pd.Index(names, name="candidate"),
    )

    return ReturnMatrix(
        returns=returns,
        truth=truth,
        params={
            "n_obs": n_obs,
            "n_candidates": n_candidates,
            "n_real": n_real,
            "sharpe_real": sharpe_real,
            "vol": vol,
            "rho": rho,
            "ar1": ar1,
            "t_df": t_df,
            "garch": list(garch) if garch is not None else None,
            "periods_per_year": periods_per_year,
            "seed": seed,
        },
    )
