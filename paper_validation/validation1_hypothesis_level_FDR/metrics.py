"""Performance measures, exactly as defined in DESIGN.md §6.

Formulas, for replication `r`, with `D_H` the rejected families, `H_0` the true
nulls and `H_1` the true non-nulls::

    V_H   = |D_H  intersect H_0|
    S_H   = |D_H  intersect H_1|
    FDP_H = V_H / max(|D_H|, 1)
    Pow_H = S_H / |H_1|

Monte Carlo estimates average those across `R` replications, and every reported
average carries

    MCSE(m_hat) = sd(m_1, ..., m_R) / sqrt(R)

with 95% bars at `m_hat +- 1.96 * MCSE`. DESIGN.md is careful about what these
bars mean and so is this module: **they quantify simulation uncertainty, not
sampling uncertainty within one generated dataset.** A tight bar says the
simulation has converged, not that the finding would replicate on new data.

The `max(|D_H|, 1)` denominator is the standard FDR convention: the false
discovery proportion is defined as zero when nothing was rejected. That matters
for the negative control — a procedure that rejects nothing scores a perfect
0.0 FDP and 0.0 power, so the two must always be read together, which is why
`no_discovery_rate` is carried alongside.

Under the global null (`pi1 = 0`) there are no true non-nulls, so:
  * power is undefined and reported as NaN, never as 0.0;
  * `FDP_H` collapses to `1{|D_H| > 0}` and hypothesis-level FDR coincides with
    FWER, which is the identity DESIGN.md §6 Figure 1 relies on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

Z_95 = 1.959963985


@dataclass
class ReplicationScore:
    """All measures from one replication, at both levels."""

    n_family_discoveries: int
    family_fdp: float
    family_power: float
    family_any_false: float
    n_candidate_discoveries: int
    candidate_fdp: float
    candidate_power: float
    no_discovery: float

    def as_dict(self) -> dict:
        return asdict(self)


def _fdp(selected: np.ndarray, truth: np.ndarray) -> float:
    """`V / max(R, 1)` — zero when nothing was selected."""
    n_selected = int(selected.sum())
    if n_selected == 0:
        return 0.0
    return float((selected & ~truth).sum()) / n_selected


def _power(selected: np.ndarray, truth: np.ndarray) -> float:
    """`S / |H_1|` — NaN when there is nothing to find, never 0.0."""
    n_true = int(truth.sum())
    if n_true == 0:
        return float("nan")
    return float((selected & truth).sum()) / n_true


def score_decision(decision, panel) -> ReplicationScore:
    """Score one method's `Decision` against one `Panel`'s two levels of truth."""
    fam_sel, fam_true = decision.families, panel.is_nonnull_family
    cand_sel, cand_true = decision.candidates, panel.is_nonnull_candidate

    return ReplicationScore(
        n_family_discoveries=int(fam_sel.sum()),
        family_fdp=_fdp(fam_sel, fam_true),
        family_power=_power(fam_sel, fam_true),
        # Under the global null this equals FDP_H exactly; away from it the two
        # differ, so both are kept rather than one being inferred from the other.
        family_any_false=float(bool((fam_sel & ~fam_true).any())),
        n_candidate_discoveries=int(cand_sel.sum()),
        candidate_fdp=_fdp(cand_sel, cand_true),
        candidate_power=_power(cand_sel, cand_true),
        no_discovery=float(fam_sel.sum() == 0),
    )


def aggregate(scores: list[ReplicationScore]) -> dict:
    """Monte Carlo means with standard errors and 95% bars.

    Returns `{metric: value, metric_mcse: ..., metric_lo: ..., metric_hi: ...}`.
    `nanmean`/`nanstd` skip replications where a measure is undefined (power
    under the global null), so those come back as NaN rather than poisoning the
    average with an invented zero.
    """
    if not scores:
        raise ValueError("no replications to aggregate")

    out: dict[str, float] = {}
    n_reps = len(scores)
    for key in ReplicationScore.__dataclass_fields__:
        values = np.array([getattr(s, key) for s in scores], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[key] = float("nan")
            out[f"{key}_mcse"] = float("nan")
            out[f"{key}_lo"] = float("nan")
            out[f"{key}_hi"] = float("nan")
            continue
        mean = float(finite.mean())
        mcse = float(finite.std(ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0
        out[key] = mean
        out[f"{key}_mcse"] = mcse
        out[f"{key}_lo"] = mean - Z_95 * mcse
        out[f"{key}_hi"] = mean + Z_95 * mcse
    out["n_reps"] = float(n_reps)
    return out


def paired_difference(treatment: np.ndarray, baseline: np.ndarray) -> dict:
    """Paired Monte Carlo difference, for DESIGN.md §6 Figure 4.

    Every method sees the same generated dataset within a replication, so the
    difference is paired and

        Var(D) = Var(X) + Var(Y) - 2 Cov(X, Y)

    is much smaller than the unpaired sum when the arms are positively
    correlated — which they are, since they read the same statistics. Treating
    them as independent would inflate the MCSE several-fold and hide real
    differences.

    Returns mean, MCSE, 95% bounds, and whether the interval excludes zero.
    """
    diff = np.asarray(treatment, dtype=float) - np.asarray(baseline, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {
            "mean": float("nan"),
            "mcse": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "significant": False,
        }
    mean = float(diff.mean())
    mcse = float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0
    lo, hi = mean - Z_95 * mcse, mean + Z_95 * mcse
    return {"mean": mean, "mcse": mcse, "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)}
