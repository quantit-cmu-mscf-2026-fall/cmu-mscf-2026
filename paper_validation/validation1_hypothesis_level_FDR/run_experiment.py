"""Monte Carlo driver for the hypothesis-level multiple-testing experiment.

Run:  python run_experiment.py                      # pilot, Phase F path
      python run_experiment.py --scope full         # adds T = 1000
      python run_experiment.py --boot 4999 --reps 1000 --scope full   # DESIGN.md final

Scope, and why it is not the full Cartesian grid
------------------------------------------------
DESIGN.md §4 validity rule 5 is explicit: *"Do not run the full Cartesian grid
until the global-null and a small alternative scenario pass the sanity checks in
Phase F."* The default scope is therefore exactly that path —

    global_null    pi1 = 0, delta = 0, over rho_W x phi        (Figure 1)
    calibration    the Figure 2 primary scenario               (Figure 2)
    power_grid     pi1 = 0.1, over rho_W x delta               (Figures 3, 4)

— not an abbreviation of the intended experiment but the stage the design says
to run first. `--scope full` adds T = 1000; `--boot` and `--reps` take the
run to the final B = 4999, R >= 1000 settings when there is time to spend.

Efficiency note
---------------
The nominal level `q` costs nothing to vary: it changes only the threshold
applied to p-values that are already computed. So each replication computes its
bootstrap and marginal p-values ONCE and every `q` in `Q_LEVELS` is evaluated
against them. This is not a shortcut — using the same p-values across `q` is
what makes the Figure 2 calibration curve a curve rather than a scatter of
independent runs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from bootstrap import block_length_rule, bootstrap_pvalues, check_resolution
from dgp import DELTA_LEVELS, Q_LEVELS, RHO_W_LEVELS, make_panel
from methods import METHOD_ORDER, apply_all_methods
from metrics import aggregate, paired_difference, score_decision

from capstone.runlog import log_run

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "results.json"

EXPERIMENT_NAME = "validation1_hypothesis_level_fdr"
SEED = 20260928
PILOT_BOOT = 1999
PILOT_REPS = 120
PHI_GLOBAL_NULL = (0.0, 0.2, 0.5)


@dataclass(frozen=True)
class Scenario:
    """One DGP configuration. `q` is not part of it — see the module docstring."""

    group: str
    n_obs: int
    pi1: float
    delta: float
    rho_w: float
    rho_b: float
    phi: float

    @property
    def key(self) -> str:
        return (
            f"{self.group}|T{self.n_obs}|pi{self.pi1}|d{self.delta}"
            f"|rw{self.rho_w}|rb{self.rho_b}|phi{self.phi}"
        )

    def as_dict(self) -> dict:
        return {
            "group": self.group,
            "T": self.n_obs,
            "pi1": self.pi1,
            "delta": self.delta,
            "rho_w": self.rho_w,
            "rho_b": self.rho_b,
            "phi": self.phi,
        }


def build_scenarios(scope: str) -> list[Scenario]:
    """The Phase F path, per DESIGN.md §4 rule 5 and the Figure specifications."""
    lengths = (500, 1000) if scope == "full" else (500,)
    scenarios: list[Scenario] = []

    # Figure 1 — global null. pi1 = 0 and delta = 0, T = 500, rho_B = 0.
    for rho_w in RHO_W_LEVELS:
        for phi in PHI_GLOBAL_NULL:
            scenarios.append(Scenario("global_null", 500, 0.0, 0.0, rho_w, 0.0, phi))

    # Figure 2 — calibration away from the global null. The one primary scenario.
    scenarios.append(Scenario("calibration", 500, 0.1, 3.0, 0.7, 0.1, 0.2))

    # Figures 3 & 4 — power vs within-family correlation. rho_B = 0 so the full
    # rho_W grid stays legal under rho_B <= rho_W.
    for n_obs in lengths:
        for rho_w in RHO_W_LEVELS:
            for delta in (d for d in DELTA_LEVELS if d > 0):
                scenarios.append(Scenario("power_grid", n_obs, 0.1, delta, rho_w, 0.0, 0.2))

    return scenarios


@dataclass
class CellAccumulator:
    """Per-replication scores for one (scenario, q, method) cell."""

    scores: list = field(default_factory=list)


def run_scenario(
    scenario: Scenario,
    *,
    n_reps: int,
    n_boot: int,
    q_levels: tuple[float, ...],
    block_length: int | None,
    rng: np.random.Generator,
) -> dict:
    """Run `n_reps` replications of one scenario, evaluating every q and method.

    All methods see the same generated dataset and the same estimated statistics
    within a replication (DESIGN.md §3 preamble, Phase C.2), so any difference
    is attributable to the testing procedure alone.
    """
    cells: dict[tuple[float, str], CellAccumulator] = {
        (q, m): CellAccumulator() for q in q_levels for m in METHOD_ORDER
    }

    for _ in range(n_reps):
        panel = make_panel(
            n_obs=scenario.n_obs,
            pi1=scenario.pi1,
            delta=scenario.delta,
            rho_w=scenario.rho_w,
            rho_b=scenario.rho_b,
            phi=scenario.phi,
            rng=rng,
        )
        boot = bootstrap_pvalues(
            panel.returns,
            panel.family_of,
            panel.n_families,
            n_boot=n_boot,
            block_length=block_length,
            rng=rng,
        )
        m_per_family = panel.n_candidates // panel.n_families

        for q in q_levels:
            decisions = apply_all_methods(
                p_boot=boot["p_family"],
                p_marginal=boot["p_candidate"],
                family_of=panel.family_of,
                n_families=panel.n_families,
                m_per_family=m_per_family,
                q=q,
            )
            for name, decision in decisions.items():
                cells[(q, name)].scores.append(score_decision(decision, panel))

    # Aggregate, and keep the paired per-replication family power for Figure 4.
    out: dict = {"scenario": scenario.as_dict(), "by_q": {}}
    for q in q_levels:
        per_method = {}
        power_by_method = {}
        for name in METHOD_ORDER:
            scores = cells[(q, name)].scores
            per_method[name] = aggregate(scores)
            power_by_method[name] = np.array([s.family_power for s in scores], dtype=float)
        # DESIGN.md §6 Figure 4: the primary paired comparison.
        delta_power = paired_difference(power_by_method["boot_max_bh"], power_by_method["bonf_bh"])
        out["by_q"][str(q)] = {"methods": per_method, "boot_minus_bonf_power": delta_power}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="run_experiment.py",
        description="Hypothesis-level multiple-testing Monte Carlo (DESIGN.md).",
    )
    parser.add_argument("--reps", type=int, default=PILOT_REPS, help="Monte Carlo replications R")
    parser.add_argument("--boot", type=int, default=PILOT_BOOT, help="bootstrap replications B")
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--block-length", type=int, default=None, help="override the block rule")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    for q in Q_LEVELS:
        if not check_resolution(args.boot, 100, q):
            parser.error(
                f"B={args.boot} cannot resolve outer BH's rank-1 threshold q/H = {q / 100:.5f} "
                f"(resolution 1/(B+1) = {1 / (args.boot + 1):.5f}). "
                f"Raise --boot above {int(100 / q) - 1}."
            )
        # Bonferroni multiplies by m, so its floor is m/(B+1). Failing this does
        # not invalidate the bootstrap arm, but it cripples the BASELINE and so
        # makes the headline power comparison meaningless. Warn, loudly.
        if not check_resolution(args.boot, 100, q, m_per_family=20):
            needed = int(20 * 100 / q)
            print(
                f"WARNING: B={args.boot} is too coarse for the Bonferroni baseline at q={q}. "
                f"Its p-value floor m/(B+1)={20 / (args.boot + 1):.4f} exceeds the rank-1 "
                f"threshold q/H={q / 100:.4f}, so it cannot be rejected below rank "
                f"{int(np.ceil((20 / (args.boot + 1)) / (q / 100)))}. Power comparisons against "
                f"it will overstate the bootstrap. Use --boot {needed - 1} or more.",
                flush=True,
            )

    scenarios = build_scenarios(args.scope)
    rng = np.random.default_rng(args.seed)

    started = time.time()
    results: dict[str, dict] = {}
    for i, scenario in enumerate(scenarios, 1):
        t0 = time.time()
        results[scenario.key] = run_scenario(
            scenario,
            n_reps=args.reps,
            n_boot=args.boot,
            q_levels=Q_LEVELS,
            block_length=args.block_length,
            rng=rng,
        )
        print(
            f"[{i:3d}/{len(scenarios)}] {scenario.key}  ({time.time() - t0:.1f}s)",
            flush=True,
        )
    elapsed = time.time() - started

    payload = {
        "meta": {
            "experiment": EXPERIMENT_NAME,
            "scope": args.scope,
            "n_reps": args.reps,
            "n_boot": args.boot,
            "seed": args.seed,
            "q_levels": list(Q_LEVELS),
            "block_length": args.block_length or block_length_rule(500),
            "n_scenarios": len(scenarios),
            "elapsed_seconds": round(elapsed, 1),
        },
        "results": results,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # Log BEFORE reporting anything, per CLAUDE.md: the ledger's length is the
    # trial count every correction depends on, so the entry must not be
    # contingent on the result being interesting.
    headline = {}
    for key, res in results.items():
        if not key.startswith("global_null"):
            continue
        for method in METHOD_ORDER:
            headline[f"{key}|q0.1|{method}|fwer"] = res["by_q"]["0.1"]["methods"][method][
                "family_any_false"
            ]
    entry = log_run(
        EXPERIMENT_NAME,
        params={
            "design": "DESIGN.md v0.1 — hypothesis-level FDR",
            "scope": args.scope,
            "n_reps": args.reps,
            "n_boot": args.boot,
            "n_scenarios": len(scenarios),
            "H": 100,
            "m": 20,
            "q_levels": list(Q_LEVELS),
        },
        metrics=headline,
        seed=args.seed,
        tags=["paper-validation", "validation1", "hypothesis-level-fdr", "bootstrap"],
        notes=(
            "Phase F path per DESIGN.md §4 rule 5: global null + calibration + power grid. "
            "Circular block bootstrap, HAC marginal p-values. Pilot scale unless --boot/--reps "
            "raised to B=4999, R>=1000."
        ),
    )

    print(f"\nscenarios: {len(scenarios)}   elapsed: {elapsed / 60:.1f} min")
    print(f"results written to {RESULTS_JSON.name}")
    print(f"ledger entry: {entry['ts_utc']}  (git {entry['git_sha']})")


if __name__ == "__main__":
    main()
