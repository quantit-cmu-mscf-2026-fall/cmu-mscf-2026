"""Data-generating process for the hypothesis-level multiple-testing experiment.

Implements the model DESIGN.md §4 Phase B leaves open ("the mathematical
return-generating equation will be finalized in the next revision") under the
constraints §7 sets out: separate common-time, family, and idiosyncratic shocks,
with knobs for within- and between-family correlation, serial dependence,
sparsity, effect size, and family size.

The model
---------
For family `h`, specification `j`, time `t`::

    r[h,j,t] = theta[h,j] + sigma * z[h,j,t]

    z[h,j,t] = sqrt(rho_B)          * c[t]        # common across everything
             + sqrt(rho_W - rho_B)  * f[h,t]      # shared within family h
             + sqrt(1 - rho_W)      * e[h,j,t]    # idiosyncratic

with `c`, `f`, `e` mutually independent, each unit-variance AR(1) in time::

    x[t] = phi * x[t-1] + sqrt(1 - phi^2) * eta[t],   eta ~ N(0, 1)

Why this parameterisation
-------------------------
The three weights are chosen so the correlation targets come out exactly, with
no normalisation left to chance::

    Var(z)                   = rho_B + (rho_W - rho_B) + (1 - rho_W) = 1
    Corr(z[h,j], z[h,k])     = rho_B + (rho_W - rho_B)               = rho_W
    Corr(z[h,j], z[h',k])    = rho_B                                  (h != h')

Note the middle line requires `rho_W >= rho_B` for the weight to be real. That
is not an implementation detail — it is exactly validity rule 1 in DESIGN.md
§4 ("only use correlation pairs satisfying 0 <= rho_B <= rho_W <= 1"), which
falls out of the model rather than being imposed on top of it.

The AR(1) scaling `sqrt(1 - phi^2)` keeps each component at unit unconditional
variance, so `phi` changes the serial dependence without changing the
cross-sectional correlation structure or the marginal scale. Every knob is
therefore orthogonal to the others, which is what makes the simulation grid
interpretable.

Effect size
-----------
DESIGN.md prescribes the local-alternative parameterisation::

    theta[h,j] = delta * sigma / sqrt(T)   for a non-null candidate, else 0

so the standardised effect `delta = sqrt(T) * theta / sigma` is held fixed as
`T` varies and the statistical difficulty stays comparable. A family is
non-null **iff at least one** of its candidates has `theta > 0` (§4 Phase B.3);
`is_nonnull_family` is derived from the candidate effects, never set
independently.

Sparsity
--------
`n_signals_per_family` controls how many of a non-null family's `m` candidates
actually carry the effect — the "sparse versus dense signals within a non-null
family" axis §4 requires. Sparse is the hard and realistic case: one working
implementation among twenty searched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- DESIGN.md §4 Phase B parameter grid ------------------------------------
H_FAMILIES = 100
M_PER_FAMILY = 20
T_LENGTHS = (500, 1000)
PI1_LEVELS = (0.0, 0.1, 0.3)
RHO_W_LEVELS = (0.0, 0.3, 0.7, 0.9)
RHO_B_LEVELS = (0.0, 0.1)
PHI_LEVELS = (0.0, 0.2, 0.5)
DELTA_LEVELS = (0.0, 2.0, 3.0, 4.0)
Q_LEVELS = (0.05, 0.10)

SIGMA = 1.0  # marginal volatility; delta is scale-free so this is a unit choice
N_SIGNALS_PER_FAMILY = 2  # sparse default: 2 of 20 specifications actually work


@dataclass
class Panel:
    """One simulated dataset together with the truth used to generate it.

    Attributes:
        returns: (T, H*M) candidate return series, column-major by family so
            that columns `h*M : (h+1)*M` belong to family `h`.
        family_of: (H*M,) family index of each candidate column.
        theta: (H*M,) true population effect of each candidate.
        is_nonnull_candidate: (H*M,) True where `theta > 0`.
        is_nonnull_family: (H,) True where the family holds >= 1 real candidate.
            Derived from `theta`, never assigned independently, so the two
            levels of truth cannot disagree.
        params: the configuration used, so a result traces to its generator.
    """

    returns: np.ndarray
    family_of: np.ndarray
    theta: np.ndarray
    is_nonnull_candidate: np.ndarray
    is_nonnull_family: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return self.returns.shape[0]

    @property
    def n_candidates(self) -> int:
        return self.returns.shape[1]

    @property
    def n_families(self) -> int:
        return len(self.is_nonnull_family)

    def __repr__(self) -> str:
        return (
            f"Panel(T={self.n_obs}, H={self.n_families}, m={self.n_candidates // self.n_families}, "
            f"{int(self.is_nonnull_family.sum())} non-null families, "
            f"{int(self.is_nonnull_candidate.sum())} non-null candidates)"
        )


def _ar1(rng: np.random.Generator, shape: tuple[int, ...], phi: float) -> np.ndarray:
    """Unit-variance AR(1) noise along axis 0.

    `x[t] = phi*x[t-1] + sqrt(1-phi^2)*eta[t]`, started from its stationary
    distribution so there is no burn-in transient. Returns exactly `shape`.

    The `sqrt(1-phi^2)` innovation scaling is what keeps the unconditional
    variance at 1 for every `phi`, so serial dependence can be varied without
    also varying the marginal scale.
    """
    if phi == 0.0:
        return rng.standard_normal(shape, dtype=np.float32)

    n_obs = shape[0]
    out = np.empty(shape, dtype=np.float32)
    innovation_sd = np.sqrt(1.0 - phi * phi, dtype=np.float32)
    out[0] = rng.standard_normal(shape[1:], dtype=np.float32)  # stationary start
    noise = rng.standard_normal((n_obs - 1, *shape[1:]), dtype=np.float32)
    for t in range(1, n_obs):
        out[t] = phi * out[t - 1] + innovation_sd * noise[t - 1]
    return out


def make_panel(
    *,
    n_obs: int = 500,
    n_families: int = H_FAMILIES,
    m_per_family: int = M_PER_FAMILY,
    pi1: float = 0.1,
    delta: float = 3.0,
    rho_w: float = 0.7,
    rho_b: float = 0.0,
    phi: float = 0.2,
    sigma: float = SIGMA,
    n_signals_per_family: int = N_SIGNALS_PER_FAMILY,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> Panel:
    """Simulate one Monte Carlo dataset with known family- and candidate-level truth.

    Args:
        n_obs: time-series length `T`.
        n_families: number of hypothesis families `H`.
        m_per_family: specifications searched within each family `m_h`.
        pi1: proportion of families that are truly non-null.
        delta: standardised effect `sqrt(T)*theta/sigma` of a real candidate.
        rho_w: within-family correlation of candidate shocks.
        rho_b: between-family correlation. Must satisfy `rho_b <= rho_w`.
        phi: AR(1) serial-correlation coefficient.
        sigma: marginal volatility.
        n_signals_per_family: non-null candidates inside a non-null family.
        rng: supply a Generator to draw many panels from one stream.
        seed: convenience for a standalone panel.

    Returns:
        A `Panel` carrying the data and both levels of truth.

    Raises:
        ValueError: if the correlation pair or the truth configuration is
            invalid. `rho_b > rho_w` is rejected here rather than silently
            producing a complex weight.
    """
    if not 0.0 <= rho_b <= rho_w <= 1.0:
        raise ValueError(
            f"require 0 <= rho_b <= rho_w <= 1 (DESIGN.md §4 validity rule 1); "
            f"got rho_w={rho_w}, rho_b={rho_b}"
        )
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi must lie in [0, 1); got {phi}")
    if n_signals_per_family > m_per_family:
        raise ValueError("n_signals_per_family cannot exceed m_per_family")
    if delta > 0.0 and pi1 == 0.0:
        raise ValueError("delta > 0 with pi1 = 0 plants no signal; use delta = 0 for global null")

    generator = rng if rng is not None else np.random.default_rng(seed)
    n_candidates = n_families * m_per_family
    family_of = np.repeat(np.arange(n_families), m_per_family)

    # --- truth ---------------------------------------------------------------
    n_nonnull_families = int(round(pi1 * n_families))
    theta = np.zeros(n_candidates, dtype=np.float64)
    if n_nonnull_families > 0 and delta > 0.0:
        chosen = generator.choice(n_families, size=n_nonnull_families, replace=False)
        effect = delta * sigma / np.sqrt(n_obs)
        for h in chosen:
            offsets = generator.choice(m_per_family, size=n_signals_per_family, replace=False)
            theta[h * m_per_family + offsets] = effect

    is_nonnull_candidate = theta > 0.0
    # Derived, per DESIGN.md §4 Phase B.3 — never assigned separately.
    is_nonnull_family = np.zeros(n_families, dtype=bool)
    if is_nonnull_candidate.any():
        is_nonnull_family[np.unique(family_of[is_nonnull_candidate])] = True

    # --- three-component shocks ---------------------------------------------
    common = _ar1(generator, (n_obs, 1), phi)
    family = _ar1(generator, (n_obs, n_families), phi)
    idio = _ar1(generator, (n_obs, n_candidates), phi)

    w_common = np.sqrt(rho_b, dtype=np.float32)
    w_family = np.sqrt(max(rho_w - rho_b, 0.0), dtype=np.float32)
    w_idio = np.sqrt(max(1.0 - rho_w, 0.0), dtype=np.float32)

    z = w_idio * idio
    if w_family > 0:
        z += w_family * family[:, family_of]
    if w_common > 0:
        z += w_common * common

    returns = (theta.astype(np.float32) + np.float32(sigma) * z).astype(np.float32)

    return Panel(
        returns=returns,
        family_of=family_of,
        theta=theta,
        is_nonnull_candidate=is_nonnull_candidate,
        is_nonnull_family=is_nonnull_family,
        params={
            "n_obs": n_obs,
            "n_families": n_families,
            "m_per_family": m_per_family,
            "pi1": pi1,
            "delta": delta,
            "rho_w": rho_w,
            "rho_b": rho_b,
            "phi": phi,
            "sigma": sigma,
            "n_signals_per_family": n_signals_per_family,
        },
    )


def newey_west_se(returns: np.ndarray, lags: int | None = None) -> np.ndarray:
    """HAC standard error of the sample mean, per column.

    Under serial correlation the i.i.d. standard error `s/sqrt(T)` is wrong —
    too small when `phi > 0` — and every marginal p-value built on it would be
    anti-conservative for a reason that has nothing to do with multiplicity.
    That would confound the comparison DESIGN.md actually wants to make, so the
    marginal p-values feeding Bonferroni / flat BH / naive-min use this instead.

    Bartlett kernel with the standard automatic bandwidth
    `floor(4 * (T/100)^(2/9))`::

        Var(mean) = (1/T) * [ gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k ]

    Args:
        returns: (T, N) array of series, one candidate per column.
        lags: truncation lag `L`; the automatic rule is used when omitted.

    Returns:
        (N,) HAC standard errors of the column means.
    """
    n_obs = returns.shape[0]
    if lags is None:
        lags = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    centred = (returns - returns.mean(axis=0, keepdims=True)).astype(np.float64)

    variance = (centred * centred).sum(axis=0) / n_obs  # gamma_0
    for k in range(1, min(lags, n_obs - 1) + 1):
        weight = 1.0 - k / (lags + 1.0)
        gamma_k = (centred[k:] * centred[:-k]).sum(axis=0) / n_obs
        variance += 2.0 * weight * gamma_k

    variance = np.maximum(variance, 1e-300)  # Bartlett guarantees >= 0, but be safe
    return np.sqrt(variance / n_obs)


def candidate_statistics(panel: Panel, hac_lags: int | None = None):
    """Per-candidate estimates, studentised statistics and marginal p-values.

    DESIGN.md Phase C: the same estimated statistics feed every method.

    Returns a dict with:
        theta_hat: (N,) sample means.
        sd: (N,) sample standard deviations (ddof=1).
        t_plain: (N,) `sqrt(T)*theta_hat/sd` — the statistic the block bootstrap
            calibrates. Serial dependence is handled by the resampling scheme,
            not by this statistic, so a plain `sd` is the internally consistent
            choice here (see `bootstrap.py`).
        t_hac: (N,) `theta_hat / se_hac` — the statistic whose null distribution
            is approximately standard normal even under serial correlation.
        p_marginal: (N,) one-sided right-tail p-values from `t_hac`.
    """
    from scipy import stats

    returns = panel.returns
    n_obs = returns.shape[0]

    theta_hat = returns.mean(axis=0, dtype=np.float64)
    sd = returns.std(axis=0, ddof=1, dtype=np.float64)
    sd = np.maximum(sd, 1e-300)

    t_plain = np.sqrt(n_obs) * theta_hat / sd
    se_hac = newey_west_se(returns, lags=hac_lags)
    t_hac = theta_hat / se_hac
    p_marginal = stats.norm.sf(t_hac)

    return {
        "theta_hat": theta_hat,
        "sd": sd,
        "t_plain": t_plain,
        "t_hac": t_hac,
        "p_marginal": p_marginal,
    }
