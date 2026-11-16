"""The two-tier evaluation's rules, shared by a real evaluation and by calibration.

A procedure opts in with a version-2 `ProcedureSpec` whose `evaluation` section registers
the rule (`procedure_contracts.two_tier_section`). Tier 1 is a vectorized returns-matrix
screen over every variant; tier 2 is the exact replay, run for the tier-1 survivors and
for a sample drawn without looking at any result. The Spearman rank agreement of tier 1
against tier 2 on that sample switches the screen off when it is below the registered
minimum: every variant then gets the exact replay. Every tier-1 variant is a ledger
trial (`log_screen`).

A frozen evaluation and the whole-procedure calibration (`calibration_columns`) both use
these rules, which are added in later changes. A version-1 procedure never reaches this
module, so its outputs are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from capstone import runlog
from capstone.factors.procedure_contracts import (
    TWO_TIER_RULE,
    TWO_TIER_STATISTIC,
    two_tier_section,
)
from capstone.factors.research_contracts import digest

SCREEN_STAGE = "two-tier-screen-v1"  # ledger name: one row per tier-1 variant
PERIODS = 252  # sessions per year, for the annualized tier-1 statistic


def section(
    *,
    tier1_id,
    survivors_per_family,
    agreement_sample,
    min_rank_agreement,
    calibration_tier1_error,
    agreement_seed=0,
):
    """A validated `evaluation` section for `ProcedureSpec.freeze(..., evaluation=...)`."""
    row = {
        "rule": TWO_TIER_RULE,
        "tier1_id": tier1_id,
        "statistic": TWO_TIER_STATISTIC,
        "survivors_per_family": survivors_per_family,
        "agreement_sample": agreement_sample,
        "agreement_seed": agreement_seed,
        "min_rank_agreement": min_rank_agreement,
        "calibration_tier1_error": calibration_tier1_error,
    }
    two_tier_section(row)
    return row


def statistic(matrix):
    """The registered tier statistic of every column at once: annualized Sharpe.

    A column with fewer than two finite values or no variation has no statistic (NaN):
    the screen cannot rank it, so it always survives."""
    values = np.asarray(matrix, dtype=float)
    finite = np.isfinite(values)
    n = finite.sum(axis=0)
    filled = np.where(finite, values, 0.0)
    mean = filled.sum(axis=0) / np.maximum(n, 1)
    centred = np.where(finite, values - mean, 0.0)
    var = (centred**2).sum(axis=0) / np.maximum(n - 1, 1)
    sd = np.sqrt(var)
    ok = (n >= 2) & (sd > 1e-15 * np.maximum(np.abs(mean), 1.0))
    out = np.full(values.shape[1], np.nan)
    out[ok] = mean[ok] / sd[ok] * np.sqrt(PERIODS)
    columns = matrix.columns if isinstance(matrix, pd.DataFrame) else range(values.shape[1])
    return pd.Series(out, index=columns, dtype=float)


def survivors(scores, family, keep):
    """The best `keep` variants of each family by tier-1 statistic (ties: first listed).

    A variant without a statistic always survives (it cannot be screened out)."""
    family = pd.Series(family).reindex(scores.index)
    if family.isna().any():
        raise ValueError("every screened variant needs a family")
    out = set()
    for label in pd.unique(family):
        ids = [v for v in scores.index if family[v] == label]
        out |= {v for v in ids if not np.isfinite(scores[v])}
        scored = sorted((v for v in ids if np.isfinite(scores[v])), key=lambda v: -scores[v])
        out |= set(scored[:keep])  # sorted is stable: ties keep the listed order
    return [v for v in scores.index if v in out]


def sample(variants, size, seed):
    """The agreement sample: `size` variants ordered by a seeded digest of their ids.

    It reads ids only, never a result, so it is fixed before anything is scored."""
    chosen = sorted(variants, key=lambda v: digest({"seed": seed, "variant": str(v)}))[:size]
    return [v for v in variants if v in set(chosen)]


def spearman(a, b):
    """Spearman rank correlation of two aligned series over the pairs both define."""
    pairs = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pairs) < 3:
        return None
    ranks = pairs.rank(method="average")
    if ranks["a"].std() == 0 or ranks["b"].std() == 0:
        return None
    return float(np.corrcoef(ranks["a"], ranks["b"])[0, 1])


def agreement(tier1, tier2, rule):
    """The registered rank-agreement check on the sample: the screen stays on only when
    at least three sampled variants have both statistics and their Spearman correlation
    is at least `min_rank_agreement` (compared as recorded, to 12 decimals)."""
    rho = spearman(tier1, tier2)
    rho = None if rho is None else round(rho, 12)
    pairs = int(pd.DataFrame({"a": tier1, "b": tier2}).dropna().shape[0])
    passed = rho is not None and rho >= rule["min_rank_agreement"]
    return {
        "pairs": pairs,
        "spearman": rho,
        "min_rank_agreement": rule["min_rank_agreement"],
        "screen_on": bool(passed),
    }


def decide(scores, family, rule):
    """Which variants get the exact replay while the screen is on: survivors + sample."""
    keep = survivors(scores, family, rule["survivors_per_family"])
    drawn = sample(list(scores.index), rule["agreement_sample"], rule["agreement_seed"])
    replay = [v for v in scores.index if v in set(keep) | set(drawn)]
    return {"survivors": keep, "sample": drawn, "replay": replay}


def screen_id(procedure_id, scope_id):
    return digest({"stage": SCREEN_STAGE, "procedure_id": procedure_id, "scope_id": scope_id})


def screen_trials(key):
    """The trial count of one screen: its distinct variants on the ledger."""
    return len(
        {
            (e.get("params") or {}).get("variant_id")
            for e in runlog.read_entries()
            if e.get("name") == SCREEN_STAGE and (e.get("params") or {}).get("screen_id") == key
        }
    )


def log_screen(procedure_id, scope_id, variants, *, rule, data_version, seed):
    """One ledger row per tier-1 variant, logged once per (screen, variant).

    Called after the tier-1 matrix is built and before any statistic of it is read. A
    resumed screen finds its rows and does not log them again. Returns the rows logged."""
    key = screen_id(procedure_id, scope_id)
    done = {
        (e.get("params") or {}).get("variant_id")
        for e in runlog.read_entries()
        if e.get("name") == SCREEN_STAGE and (e.get("params") or {}).get("screen_id") == key
    }
    logged = 0
    for variant in variants:
        if variant in done:
            continue
        runlog.log_run(
            SCREEN_STAGE,
            params={
                "screen_id": key,
                "procedure_id": procedure_id,
                "scope_id": scope_id,
                "variant_id": variant,
                "tier1_id": rule["tier1_id"],
                "statistic": rule["statistic"],
                "data_version": data_version,
            },
            seed=seed,
            tags=[SCREEN_STAGE, "tier1", "registered-before-results"],
            notes="Tier-1 screen: every screened variant is a trial.",
        )
        logged += 1
    return logged


# --- calibration (synthetic worlds, zero model calls) ---------------------------------------


def calibration_columns(returns, columns, family, n_search, seed, stream, rule):
    """The columns one arm's batch keeps under the screen, and the screen's record.

    Tier 2 is the world's own (exact) returns. Tier 1 is tier 2 plus independent noise
    with sd `calibration_tier1_error` x each column's search-period sd: the error the
    registered screen is calibrated at. Survivors and the sample are chosen exactly as
    the real path chooses them (search-period statistic, same rule); when the rank
    agreement on the sample fails, every column is kept."""
    if not columns:
        return [], {"screened": 0, "replayed": 0, "survivors": 0, "sample": 0, "agreement": None}
    exact = returns[columns]
    search = exact.iloc[:n_search]
    rng = np.random.default_rng([seed, 1515, stream])  # 1515: tier-1 noise stream; one per arm
    sd = search.std(ddof=1).to_numpy()
    noise = rng.standard_normal(search.shape) * rule["calibration_tier1_error"] * sd
    tier1 = statistic(pd.DataFrame(search.to_numpy() + noise, columns=columns))
    tier2 = statistic(search)
    plan = decide(tier1, family[columns], rule)
    check = agreement(tier1[plan["sample"]], tier2[plan["sample"]], rule)
    kept = plan["replay"] if check["screen_on"] else list(columns)
    return kept, {
        "screened": len(columns),
        "replayed": len(kept),
        "survivors": len(plan["survivors"]),
        "sample": len(plan["sample"]),
        "agreement": check,
    }


def calibration_events(spec, returns, family, events, n_search, seed):
    """`(events, screens)` for one calibration world.

    A version-1 procedure: `(events, None)`, untouched. A two-tier procedure: each arm's
    evaluated slots the screen withholds from the exact replay are dropped from what the
    funnel and the portfolio see (they stay trials in the accounting, which ran before),
    and `screens` holds each arm's screen record."""
    rule = spec.two_tier
    if rule is None:
        return events, None
    kept, screens = set(), {}
    for stream, arm in enumerate(spec.to_record()["arms"]):
        mine = [e for e in events if e["arm"] == arm["name"]]
        live = [e["column"] for e in mine if e["status"] == "evaluated"]
        columns, screens[arm["name"]] = calibration_columns(
            returns, live, family, n_search, seed, stream, rule
        )
        kept |= set(columns)
    return [e for e in events if e["status"] != "evaluated" or e["column"] in kept], screens
