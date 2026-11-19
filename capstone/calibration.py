"""Calibrating the validation pipeline on signals with known truth.

`docs/validation/framework.md` gives each method one role and each stage one
gate. This module measures whether that works: it feeds the pipeline simulated
hypothesis families where we know which are real, and reports, for every gate
and for the whole pipeline,

- the **false-pass rate** on signal-free candidates (is the gate calibrated?),
- the **pass rate on real signals** (is it too strict?),
- the **false-discovery rate** of what comes out (the target is 10%, #50 Q5),
- the **true discoveries**, and the share of accepted candidates that are real.

It compares three ways of deciding (#50, Q5):

- **naive stack:** every screening method as a gate at once, on the whole
  search period, then the final holdout test;
- **baseline:** BH at q on each family's search-adjusted p-value over the
  search period, the rule that decides until the funnel beats it;
- **funnel:** stage 1 screens families, stage 2 checks survivors (by default
  on the same search period, with no gate: the design chosen in
  `docs/validation/framework.md`; "split" confirms on years stage 1 didn't
  use instead), stage 3 (spanning) is a placeholder that passes everything
  and is reported as not measured, stage 4 tests the survivors on the
  holdout.

Each simulated data set is summarised once per hypothesis (`family_statistics`,
which runs `capstone.funnel`'s dated statistics, the gate's code path);
every pipeline and threshold setting is then a cheap filter over that summary,
so threshold sweeps cost almost nothing. `scripts/calibrate_validation.py`
runs the grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone import funnel
from capstone.evaluate import benjamini_hochberg
from capstone.funnel import STAGE1_METHODS, Thresholds, _final_gate, run_funnel, stage1_pvalues
from capstone.synth import make_return_matrix

__all__ = [
    "STAGE1_METHODS",
    "FamilySet",
    "Thresholds",
    "family_statistics",
    "gate_report",
    "gates_alone",
    "idle_days",
    "make_family_set",
    "run_baseline",
    "run_funnel",
    "run_naive_stack",
    "score",
]

# ---------------------------------------------------------------------------
# Test signals with known truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilySet:
    """Simulated hypothesis families, split into periods by position.

    returns: dates x variants. family: hypothesis label per variant column.
    truth: per hypothesis, True when its variants carry a real signal.
    n_search / n_holdout: the first n_search rows are the search period
    (agents and stages 1-3), the last n_holdout the holdout (stage 4 only).
    """

    returns: pd.DataFrame
    family: pd.Series
    truth: pd.Series
    n_search: int
    n_holdout: int

    @property
    def search(self) -> pd.DataFrame:
        return self.returns.iloc[: self.n_search]

    @property
    def holdout(self) -> pd.DataFrame:
        return self.returns.iloc[self.n_search : self.n_search + self.n_holdout]


def make_family_set(
    n_families: int,
    k: int | Sequence[int],
    share_real: float,
    sharpe_real: float,
    *,
    within_rho: float = 0.9,
    n_search: int = 31 * 252,
    n_holdout: int = 5 * 252,
    seed: int = 0,
    idle_share: float = 0.0,
    idle_run: float = 21.0,
    **defects,
) -> FamilySet:
    """Families of `k` variants of one idea, with realistic defects.

    `k` is every family's variant count, or a sequence of `n_families` counts
    (families in order); a scalar `k` and `[k] * n_families` give the same
    data. Each variant is sqrt(within_rho) x its family's series + sqrt(1 - within_rho)
    x its own, both from `synth.make_return_matrix` with the same `defects`
    (`rho`, `ar1`, `t_df`, `garch`, `vol`), so variants of one idea are
    near-copies and every defect carries through. A real family's series is
    scaled so that each of its *variants* has annualised Sharpe `sharpe_real`.
    The default periods follow #43: 31 years of search, 5 of holdout.

    `idle_share` is the long-run share of days a family holds no position, its
    variants' returns NaN then (`backtest` leaves a flat date NaN, not 0.0).
    A family's variants are idle on the same days, in runs of `idle_run` days
    on average (a two-state Markov chain started from its stationary state).
    Active days keep their returns, so a real variant's Sharpe per active day
    is still `sharpe_real`; it just has fewer days to show it.
    """
    if not 0.0 < within_rho <= 1.0:
        raise ValueError("within_rho must be in (0, 1]")
    if not 0.0 <= idle_share < 1.0:
        raise ValueError(f"idle_share must be in [0, 1), got {idle_share}")
    if not idle_run >= 1.0:
        raise ValueError(f"idle_run must be at least 1 day, got {idle_run}")
    counts = np.full(n_families, k) if np.ndim(k) == 0 else np.asarray(k)
    if len(counts) != n_families or (counts < 1).any():
        raise ValueError(f"k must be >= 1 for each of the {n_families} families, got {k!r}")
    n_obs = n_search + n_holdout
    n_real = int(round(share_real * n_families))
    base = make_return_matrix(
        n_obs=n_obs,
        n_candidates=n_families,
        n_real=n_real,
        sharpe_real=sharpe_real / np.sqrt(within_rho),
        seed=seed,
        **defects,
    )
    own = make_return_matrix(
        n_obs=n_obs, n_candidates=int(counts.sum()), n_real=0, seed=seed + 1_000_003, **defects
    )
    # Variant columns family by family: column j of family f is own column
    # sum(counts[:f]) + j, as the scalar case's reshape laid them out.
    which = np.repeat(np.arange(n_families), counts)
    b = base.returns.to_numpy()[:, which]
    x = np.sqrt(within_rho) * b + np.sqrt(1.0 - within_rho) * own.returns.to_numpy()
    if idle_share > 0.0:
        idle = idle_days(n_obs, n_families, idle_share, idle_run, seed=seed + 2_000_003)
        x[idle[:, which]] = np.nan
    hypotheses = list(base.returns.columns)
    columns = [f"{h}_v{j}" for h, n in zip(hypotheses, counts, strict=True) for j in range(n)]
    returns = pd.DataFrame(x, index=base.returns.index, columns=columns)
    family = pd.Series(np.repeat(hypotheses, counts), index=columns, name="hypothesis")
    return FamilySet(returns, family, base.truth.rename("truth"), n_search, n_holdout)


def idle_days(n_obs: int, n_series: int, share: float, run: float, *, seed: int) -> np.ndarray:
    """n_obs x n_series booleans, True on idle days: each column a two-state
    Markov chain idle `share` of the time in runs of mean length `run`."""
    leave_idle = 1.0 / run
    enter_idle = share * leave_idle / (1.0 - share)
    if enter_idle > 1.0:
        raise ValueError(f"idle_share {share} needs idle runs longer than {run} days")
    rng = np.random.default_rng(seed)
    u = rng.random((n_obs, n_series))
    out = np.empty((n_obs, n_series), dtype=bool)
    state = u[0] < share
    out[0] = state
    for t in range(1, n_obs):
        state = np.where(state, u[t] >= leave_idle, u[t] < enter_idle)
        out[t] = state
    return out


# ---------------------------------------------------------------------------
# One summary per hypothesis, computed once per data set
# ---------------------------------------------------------------------------


def family_statistics(
    fs: FamilySet,
    *,
    stage2_years: float = 7.0,
    n_boot: int = 1999,
    seed: int = 0,
    q: float = 0.05,
    designs: tuple[str, ...] = ("split", "reuse"),
) -> dict[str, pd.DataFrame]:
    """Everything the pipelines need, per hypothesis, for both stage-2 designs.

    Returns {"reuse": ..., "split": ...}. In "reuse", the chosen design
    (#50, "Stage 2"), both stages see the whole search period and stage 2 is
    not a gate. In "split", stage 1 sees the search period minus its last
    `stage2_years`, and stage 2 confirms on those years; calibration found it
    costs power. The baseline and the naive stack always use the whole search
    period, so they read "reuse".

    The work is `capstone.funnel.family_statistics`, the code the gate runs
    on real candidates: the set's positional periods become dates (the
    search period ends on its last search row, the holdout starts on its
    first holdout row, "split" confirms on the last `stage2_years` x 252
    search rows). Columns are that function's, plus `truth`.
    """
    n_b = int(round(stage2_years * 252))
    if not 0 < n_b < fs.n_search:
        raise ValueError("stage2_years must leave data for both stages")
    search = fs.search
    first, last = search.iloc[: fs.n_search - n_b], search.iloc[fs.n_search - n_b :]
    # Every variant is tested in each period the pipelines use; name the first
    # one that can't be, rather than failing deep inside a Sharpe test.
    for name, window in (("search", first), ("stage 2", last), ("holdout", fs.holdout)):
        thin = window.columns[window.notna().sum() < 3]
        if len(thin):
            raise ValueError(
                f"hypothesis {fs.family[thin[0]]!r} has no variant with data in the "
                f"{name} period ({thin[0]!r} has fewer than 3 values)"
            )
    # `q` is the smallest stage-1 BH level the caller will use (it sets how
    # many bootstrap draws are enough). Computing only the designs asked for
    # halves the bootstrap's cost when one design is being calibrated.
    if "reuse" not in designs or not set(designs) <= {"split", "reuse"}:
        raise ValueError("designs must include 'reuse' and contain only 'split' or 'reuse'")
    out = funnel.family_statistics(
        fs.returns.iloc[: fs.n_search + fs.n_holdout],
        fs.family,
        search_end=search.index[-1],
        holdout_start=fs.holdout.index[0],
        stage2_start=last.index[0] if "split" in designs else None,
        n_boot=n_boot,
        seed=seed,
        q=q,
    )
    for s in out.values():
        s["truth"] = fs.truth.reindex(s.index)
    return {d: out[d] for d in ("split", "reuse") if d in out}


# ---------------------------------------------------------------------------
# The three pipelines, as filters over the summary (the funnel's Thresholds,
# stages and final gate are capstone.funnel's)
# ---------------------------------------------------------------------------


def run_baseline(stats: pd.DataFrame, *, q: float = 0.10, stage1: str = "rho") -> pd.Series:
    """BH at `q` on each family's search-adjusted p-value (#50, Q5).

    Pass "reuse" statistics, so the p-values cover the whole search period.
    Reads the p-values through `funnel.stage1_pvalues`, so a missing one
    raises here as it does in the funnel, and the two never run over
    different family sets.
    """
    return benjamini_hochberg(stage1_pvalues(stats, stage1), q).rename("accepted")


def run_naive_stack(stats: pd.DataFrame, th: Thresholds) -> pd.Series:
    """Every screening gate at once, then the same final holdout test."""
    alive = stats["naive_pass"].astype(bool)
    final = _final_gate(stats, alive, th, prefix="nv_")
    return (alive & final).rename("accepted")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(accepted: pd.Series, truth: pd.Series) -> dict:
    """Outcome of one pipeline on one data set.

    `fdr` is false discoveries / max(discoveries, 1), so a run that accepts
    nothing scores 0, and its average over runs is the FDR proper.
    `power` is NaN when nothing is real.
    """
    accepted = accepted.reindex(truth.index).fillna(False).astype(bool)
    truth = truth.astype(bool)
    n = int(accepted.sum())
    true = int((accepted & truth).sum())
    false = n - true
    return {
        "discoveries": n,
        "true": true,
        "false": false,
        "any_false": false > 0,
        "fdr": false / max(n, 1),
        "power": true / truth.sum() if truth.any() else np.nan,
        "precision": true / n if n else np.nan,
    }


def gate_report(stages: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    """Where real signals die and where noise leaks, stage by stage.

    For each stage: how many real and null hypotheses enter and pass it, and
    the pass rates among those that entered (in-pipeline power and false-pass
    rate). Stage 3 is flagged as not measured, and its pass rates are NaN.
    """
    truth = truth.reindex(stages.index).astype(bool)
    entering = pd.Series(True, index=stages.index)
    rows = []
    for stage in stages.columns:
        passed = stages[stage].astype(bool)
        # Stage 3 passes everything until the spanning test exists; a 1.0
        # pass rate there would read as a measurement.
        measured = stage != "stage3"
        real_in, null_in = int((entering & truth).sum()), int((entering & ~truth).sum())
        real_out, null_out = int((passed & truth).sum()), int((passed & ~truth).sum())
        rows.append(
            {
                "stage": stage,
                "real_in": real_in,
                "real_out": real_out,
                "null_in": null_in,
                "null_out": null_out,
                "pass_rate_real": real_out / real_in if real_in and measured else np.nan,
                "pass_rate_null": null_out / null_in if null_in and measured else np.nan,
                "measured": measured,
            }
        )
        entering = passed
    return pd.DataFrame(rows).set_index("stage")


def gates_alone(stats: pd.DataFrame, th: Thresholds, *, stage1: str = "rho") -> pd.DataFrame:
    """Each gate applied by itself to every hypothesis, for comparison with
    its pass rates inside the pipeline: a gate that looks fine alone can still
    repeat a penalty another gate already charged."""
    everyone = pd.Series(True, index=stats.index)
    alone = pd.DataFrame(
        {
            "stage1": benjamini_hochberg(stage1_pvalues(stats, stage1), th.q1),
            "stage2": stats["p2"] <= th.alpha2,
            "stage4": _final_gate(stats, everyone, th),
        }
    )
    truth = stats["truth"].astype(bool)
    return pd.DataFrame(
        {
            "pass_rate_real": alone[truth].mean() if truth.any() else np.nan,
            "pass_rate_null": alone[~truth].mean(),
        }
    )
