"""Circular block bootstrap, family max-omnibus and per-candidate p-values.

Implements DESIGN.md §3.1. Three structural rules make it correct, and each is
easy to violate without noticing:

1. **One common resampling scheme, applied jointly** (§3.1 step 1, §5 rule 5).
   A single time-index draw per bootstrap replication is applied to every
   candidate in every family at once. Resampling each candidate independently
   would destroy the dependence the method exists to exploit.

2. **The null is imposed by centring** (§3.1 step 3):
   `T* = sqrt(T)(theta_hat* - theta_hat)/sd*`. Without the centring the
   bootstrap reproduces the observed effects, not the null.

3. **The internal search is repeated inside each bootstrap sample** (§3.1
   step 4). The family maximum is retaken every replication, so the reference
   distribution is that of *the winner of a search of this size and this
   correlation* — the search cannot manufacture significance.

Why this also produces the MARGINAL p-values
--------------------------------------------
DESIGN.md §3.2 requires the baselines' candidate p-values to be "marginally
valid ... using the same studentization and time-series inference assumptions as
the main method". A Newey-West normal approximation does not meet that bar here:
measured over 12 panels at T=500, `P(p <= 0.01)` came out at 0.012 / 0.015 /
0.023 for phi = 0 / 0.2 / 0.5 — the familiar finite-sample downward bias of HAC
variance estimates, which taking a minimum over 20 candidates then compounds.
Bonferroni built on those marginals over-rejected at the global null for a
reason that has nothing to do with multiplicity, confounding the one comparison
the experiment exists to make.

So the same bootstrap that calibrates the family maximum also calibrates each
candidate, at essentially no extra cost — the per-candidate exceedance counter
rides along inside the loop that is already computing `T*`. Every method then
shares identical time-series inference and differs *only* in how it aggregates
across the internal search, which is exactly the isolation §3's preamble asks
for.

Resolution
----------
`p = (1 + #exceedances) / (B + 1)` can never be smaller than `1/(B+1)`. Outer BH
across `H` families tests its most extreme rank at `q/H`, so the bootstrap must
resolve below that: **`B + 1 > H / q`**. At H = 100 and q = 0.05 that means
B >= 1999, and DESIGN.md's final B = 4999 leaves headroom. `check_resolution`
makes the requirement explicit rather than leaving it to be discovered as
mysteriously absent power.

Performance
-----------
The naive implementation gathers `X[idx]`, moving `B * T * N` floats. Since a
circular block sum is a difference of prefix sums, only `B * n_blocks * N` need
move — an `l`-fold reduction for block length `l`. That is what makes B = 4999
reachable rather than theoretical.
"""

from __future__ import annotations

import numpy as np


def block_length_rule(n_obs: int) -> int:
    """Default block length `ceil(2 * T^(1/3))` — 16 at T=500, 20 at T=1000.

    DESIGN.md §4 Phase A.3 asks for a pre-specified block-length rule and Phase
    F.6 for evidence the conclusions do not hinge on one arbitrary choice. This
    default was chosen by running that check rather than by folklore. Marginal
    p-value calibration at the global null, T=500, measured as `P(p <= 0.01)`
    against a target of 0.010::

        block:        8        16        25        50
        phi=0.0    0.0121    0.0132    0.0159    0.0227
        phi=0.2    0.0141    0.0144    0.0169    0.0234
        phi=0.5    0.0198    0.0172    0.0189    0.0245

    The trade-off is visible in both directions: short blocks sever serial
    dependence at block boundaries and under-state the long-run variance (worst
    at phi=0.5), while long blocks leave too few distinct blocks and the
    resampled variance itself becomes noisy (worst everywhere by block 50). The
    minimum sits near 16 at T=500, which `ceil(2*T^(1/3))` reproduces.

    Residual anti-conservatism of roughly 1.3-1.7x at the 1% level remains at
    every block length. That is a finite-sample property of the studentised
    block bootstrap, not a defect of this implementation, and it applies
    identically to every method here — so absolute levels are mildly optimistic
    while the comparison between methods stays fair. Figure 1 measures the
    family-level consequence directly.
    """
    return max(1, int(np.ceil(2.0 * n_obs ** (1.0 / 3.0))))


def check_resolution(n_boot: int, n_families: int, q: float, m_per_family: int = 1) -> bool:
    """Can `B` bootstrap draws resolve the most extreme outer-BH threshold?

    Outer BH tests rank 1 of `H` at `q/H`. A method whose family p-value is a
    bootstrap quantity multiplied by `m` — Bonferroni is exactly this,
    `min(1, m * min_j p_hj)` — has a floor of `m/(B+1)`, not `1/(B+1)`. So::

        m / (B + 1)  <=  q / H        i.e.   B >= m*H/q - 1

    **This is not a fussy detail; it silently inverted a headline result.** With
    `m=20, H=100, q=0.10` the bootstrap max needs only `B >= 999` but Bonferroni
    needs `B >= 19999`. At the pilot's `B=1999` Bonferroni could not be rejected
    below rank 10 — it needed ten families at the floor simultaneously, and only
    3-6% got there — so its measured power collapsed to ~0 and the bootstrap
    appeared to win enormously. Raising `B` to 19999 on the same design::

        rho_W=0.0:  bonf 0.024 -> 0.268   (boot-minus-bonf flips +0.204 -> -0.108)
        rho_W=0.9:  bonf 0.000 -> 0.196   (boot-minus-bonf halves +0.272 -> +0.120)

    Coarse resolution also inflates the bootstrap's own power by creating ties at
    the floor that BH's step-up then sweeps up wholesale, which is why
    `boot_max_bh` itself moved (0.228 -> 0.160 at rho_W=0). Both arms are
    affected, in opposite directions.

    Pass `m_per_family` to check the Bonferroni-style requirement; leave it at 1
    for a bare bootstrap p-value.
    """
    return (m_per_family / (n_boot + 1.0)) <= (q / n_families)


def _circular_prefix(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prefix sums of the doubled series, for O(1) circular block sums.

    Returns `(P1, P2)` with shape `(2T+1, N)`, where `P1[k] = sum(Xd[:k])` and
    `P2[k] = sum(Xd[:k]**2)` over the circularly doubled array `Xd`. A block of
    length `l` starting at `s < T` then sums to `P1[s+l] - P1[s]`.
    """
    doubled = np.concatenate([data, data], axis=0)
    p1 = np.zeros((doubled.shape[0] + 1, doubled.shape[1]), dtype=np.float64)
    p2 = np.zeros_like(p1)
    np.cumsum(doubled, axis=0, out=p1[1:], dtype=np.float64)
    np.cumsum(doubled.astype(np.float64) ** 2, axis=0, out=p2[1:])
    return p1, p2


def bootstrap_pvalues(
    returns: np.ndarray,
    family_of: np.ndarray,
    n_families: int,
    *,
    n_boot: int = 1999,
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
    batch_size: int = 64,
) -> dict:
    """Bootstrap family max-omnibus and per-candidate p-values in one pass.

    Args:
        returns: (T, N) candidate return series.
        family_of: (N,) family index per candidate; candidates must be laid out
            contiguously by family.
        n_families: number of families `H`.
        n_boot: bootstrap replications `B`.
        block_length: circular block length; `block_length_rule(T)` if omitted.
        rng: random generator.
        batch_size: replications per vectorised pass — affects memory and speed
            only, never the result.

    Returns:
        dict with `p_family` (H,), `p_candidate` (N,), `t_obs` (N,),
        `m_obs` (H,), and `block_length`.
    """
    generator = rng if rng is not None else np.random.default_rng()
    n_obs, n_candidates = returns.shape
    block = block_length if block_length is not None else block_length_rule(n_obs)

    m_per_family = n_candidates // n_families
    if not np.array_equal(family_of, np.repeat(np.arange(n_families), m_per_family)):
        raise ValueError("bootstrap requires candidates laid out contiguously by family")

    data = np.asarray(returns, dtype=np.float32)

    # --- observed statistics --------------------------------------------------
    theta_hat = data.mean(axis=0, dtype=np.float64)
    sd = np.maximum(data.std(axis=0, ddof=1, dtype=np.float64), 1e-300)
    t_obs = np.sqrt(n_obs) * theta_hat / sd
    m_obs = t_obs.reshape(n_families, m_per_family).max(axis=1)

    # --- prefix sums so a block sum is a subtraction ---------------------------
    p1, p2 = _circular_prefix(data)
    n_full = n_obs // block
    remainder = n_obs - n_full * block

    exceed_family = np.zeros(n_families, dtype=np.int64)
    exceed_candidate = np.zeros(n_candidates, dtype=np.int64)

    done = 0
    while done < n_boot:
        size = min(batch_size, n_boot - done)
        # One common set of block starts per replication, shared by every
        # candidate — rule 1 above.
        starts = generator.integers(0, n_obs, size=(size, n_full))
        s1 = (p1[starts + block] - p1[starts]).sum(axis=1)
        s2 = (p2[starts + block] - p2[starts]).sum(axis=1)
        if remainder:
            tail = generator.integers(0, n_obs, size=(size, 1))
            s1 += (p1[tail + remainder] - p1[tail]).sum(axis=1)
            s2 += (p2[tail + remainder] - p2[tail]).sum(axis=1)

        mean_star = s1 / n_obs
        var_star = np.maximum((s2 - n_obs * mean_star**2) / (n_obs - 1), 1e-300)
        # Centre on the observed estimate: impose the null (rule 2).
        t_star = np.sqrt(n_obs) * (mean_star - theta_hat) / np.sqrt(var_star)

        # Per-candidate exceedances: the marginal p-values, free of charge.
        exceed_candidate += (t_star >= t_obs[None, :]).sum(axis=0)
        # Retake the internal search inside the bootstrap world (rule 3).
        m_star = t_star.reshape(size, n_families, m_per_family).max(axis=2)
        exceed_family += (m_star >= m_obs[None, :]).sum(axis=0)

        done += size

    return {
        "p_family": (1.0 + exceed_family) / (n_boot + 1.0),
        "p_candidate": (1.0 + exceed_candidate) / (n_boot + 1.0),
        "t_obs": t_obs,
        "m_obs": m_obs,
        "block_length": block,
    }
