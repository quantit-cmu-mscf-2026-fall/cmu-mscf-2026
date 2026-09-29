"""The six testing procedures compared in DESIGN.md §3, plus BH/BY primitives.

Every method maps candidate-level inputs to decisions at BOTH levels, because
DESIGN.md §5 rule 2 insists the two are different estimands and must be reported
separately. A method that only targets one level still gets scored on both — flat
BH's family-level column is its *induced* behaviour, not a claim about what it
controls.

Method order is fixed by DESIGN.md §6 and used everywhere downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone.evaluate import benjamini_hochberg

# DESIGN.md §6: fixed order, fixed labels, used in every table and figure.
METHOD_ORDER = (
    "boot_max_bh",
    "boot_max_by",
    "bonf_bh",
    "bonf_by",
    "flat_bh",
    "naive_min_bh",
)
METHOD_LABEL = {
    "boot_max_bh": "Bootstrap max omnibus + BH",
    "boot_max_by": "Bootstrap max omnibus + BY",
    "bonf_bh": "Bonferroni omnibus + BH",
    "bonf_by": "Bonferroni omnibus + BY",
    "flat_bh": "Flat BH over all specifications",
    "naive_min_bh": "Naive minimum-p + BH",
}
# §6: "clearly mark flat BH as a different-target method and naive minimum-p as
# a negative control". Carried as data so every figure marks them consistently.
METHOD_ROLE = {
    "boot_max_bh": "primary",
    "boot_max_by": "primary",
    "bonf_bh": "same-target baseline",
    "bonf_by": "same-target baseline",
    "flat_bh": "different-target",
    "naive_min_bh": "negative control",
}


@dataclass
class Decision:
    """Rejections from one method, at both levels.

    Attributes:
        families: (H,) True where the hypothesis family was discovered.
        candidates: (N,) True where an individual specification was rejected.
            Only flat BH rejects candidates directly; the family-level methods
            leave this all-False, because DESIGN.md §5 rule 3 is explicit that a
            family rejection does NOT validate any particular specification
            inside it. Recording zeros here rather than "the winner" is the
            code-level expression of that rule.
        method: method key from `METHOD_ORDER`.
    """

    families: np.ndarray
    candidates: np.ndarray
    method: str


def _bh(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up at level `q`, via the kit's implementation."""
    if pvalues.size == 0:
        return np.zeros(0, dtype=bool)
    series = pd.Series(pvalues, index=np.arange(pvalues.size))
    return benjamini_hochberg(series, alpha=q).to_numpy()


def by_constant(n_tests: int) -> float:
    """Benjamini-Yekutieli harmonic constant `c(n) = sum_{i=1..n} 1/i`."""
    return float(np.sum(1.0 / np.arange(1, n_tests + 1))) if n_tests > 0 else 1.0


def _by(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Yekutieli: BH run at the deflated level `q / c(n)`.

    Controls FDR under ARBITRARY dependence, where BH needs positive regression
    dependence. The price is the `c(n)` factor — at H = 100 that is ~5.19, so BY
    tests at roughly one fifth of BH's level and is correspondingly
    conservative. DESIGN.md §3.1 wants both reported: BH as the higher-power
    default when dependence assumptions are plausible, BY as the robustness
    check.
    """
    return _bh(pvalues, q / by_constant(pvalues.size))


def _families_from_candidates(
    candidate_rejected: np.ndarray, family_of: np.ndarray, n_families: int
) -> np.ndarray:
    """Induced family discovery: a family counts as found if any of its specs was."""
    families = np.zeros(n_families, dtype=bool)
    if candidate_rejected.any():
        families[np.unique(family_of[candidate_rejected])] = True
    return families


def family_min_pvalues(p_marginal: np.ndarray, n_families: int, m_per_family: int):
    """Per-family minimum marginal p-value (candidates contiguous by family)."""
    return p_marginal.reshape(n_families, m_per_family).min(axis=1)


def bonferroni_family_pvalues(p_marginal: np.ndarray, n_families: int, m_per_family: int):
    """DESIGN.md §3.2: `p_h = min(1, m_h * min_j p_hj)`.

    A valid family-level p-value under the global null — the Bonferroni
    correction pays in full for the internal search of `m_h` specifications. It
    does so without using the dependence among those specifications, which is
    why it is expected to be conservative when they are highly correlated. That
    conservatism is precisely the quantity the bootstrap method is trying to
    recover, which is why this is the fair same-target baseline rather than a
    straw man.
    """
    return np.minimum(1.0, m_per_family * family_min_pvalues(p_marginal, n_families, m_per_family))


def apply_all_methods(
    *,
    p_boot: np.ndarray,
    p_marginal: np.ndarray,
    family_of: np.ndarray,
    n_families: int,
    m_per_family: int,
    q: float,
) -> dict[str, Decision]:
    """Run every method in `METHOD_ORDER` on one replication's shared statistics.

    DESIGN.md Phase C.2 and §3 preamble: identical inputs for every method, so a
    difference in outcome is attributable to the multiple-testing procedure and
    nothing else.

    Args:
        p_boot: (H,) bootstrap max omnibus p-values.
        p_marginal: (N,) HAC-studentised one-sided candidate p-values.
        family_of: (N,) family index per candidate.
        n_families: `H`.
        m_per_family: `m_h`, assumed equal across families.
        q: nominal FDR level.

    Returns:
        `{method_key: Decision}` for every key in `METHOD_ORDER`.
    """
    no_candidates = np.zeros(p_marginal.size, dtype=bool)
    p_bonf = bonferroni_family_pvalues(p_marginal, n_families, m_per_family)
    p_naive = family_min_pvalues(p_marginal, n_families, m_per_family)

    out: dict[str, Decision] = {}

    # --- primary: bootstrap max omnibus, outer BH and BY ---------------------
    out["boot_max_bh"] = Decision(_bh(p_boot, q), no_candidates.copy(), "boot_max_bh")
    out["boot_max_by"] = Decision(_by(p_boot, q), no_candidates.copy(), "boot_max_by")

    # --- same-target baseline: Bonferroni omnibus ----------------------------
    out["bonf_bh"] = Decision(_bh(p_bonf, q), no_candidates.copy(), "bonf_bh")
    out["bonf_by"] = Decision(_by(p_bonf, q), no_candidates.copy(), "bonf_by")

    # --- different-target: flat BH over ALL specifications -------------------
    # This one genuinely rejects candidates; its family column is induced, and
    # DESIGN.md §5 rule 2 forbids reading it as hypothesis-level control.
    flat_candidates = _bh(p_marginal, q)
    out["flat_bh"] = Decision(
        _families_from_candidates(flat_candidates, family_of, n_families),
        flat_candidates,
        "flat_bh",
    )

    # --- negative control: naive minimum-p, uncorrected for the search -------
    out["naive_min_bh"] = Decision(_bh(p_naive, q), no_candidates.copy(), "naive_min_bh")

    return out
