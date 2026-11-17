"""Incubation as the second filter: what does promote="incubate" let through?

Open decision (D-4): the team has not adopted an "incubate" outcome or an
incubation policy, so this script is **off by default**: with `q4_incubate`
null in `promotion_rates.json` the gate runs with the registered default
(`promote="accept"`), no family is labelled incubate, and the incubate and
monitor sections report empty (NaN) rates. Set `q4_incubate` in the parameter
file, or pass `--q4-incubate`, to measure the labelled outcome
(`capstone.promotion`).

Runs the registered gate (`gate.run_gate`, default thresholds, plus
promote="incubate" at `q4_incubate` when that is set) on synthetic worlds from
`calibration.make_family_set` (31 years of search, a 5-year holdout,
`n_families` families of `k` variants) and reports, with the parameters in
`promotion_rates.json`:

- **planted** worlds (`share_real` of the families earn annualised Sharpe
  `sharpe` per variant): real families accepted (what the 5-year holdout
  alone passes) and real families reaching incubation (accepted or
  incubate); null families accepted and incubate (false promotion);
- **null** worlds (no real family): null families accepted and incubate;
- **the monitor on null incubation-only families**: each is monitored as
  `capstone.incubate.incubate` does (`MonitorConfig` at the gate's expected
  mean, scale times the long-run sd factor of its search returns) on
  `live_continuation(mean=0)`: its own search returns' deviations,
  block-bootstrapped and re-centred on 0, because its true edge is 0. Retire
  rate per horizon. As a contrast the same families are also monitored at
  their raw search-period mean.
- **the confirming e-process on the same null families**: decay monitor and
  confirming process side by side (`incubation.confirm.decide_paths`), on
  `live_draws` continuations per family: retire / promote / undecided rate
  per confirm horizon (retire about 1 - alpha);
- **its power**: on the planted worlds' real incubate families (live
  re-centred on their true mean, Sharpe `sharpe` times their search sd over
  sqrt(252)), and on iid normal paths of known annualised Sharpe (`power`):
  promote rate within each horizon and the years to 50% / 80% power.

Logged to the ledger as "promotion-rates" (tags "synthetic", "promotion")
before anything is printed. Point CAPSTONE_LEDGER_DIR at a scratch directory.

    python scripts/promotion_rates.py            # the full run (about 10 minutes)
    python scripts/promotion_rates.py --quick    # 5 + 5 worlds, a smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capstone import gate, grading, promotion
from capstone.calibration import make_family_set
from capstone.funnel import Thresholds
from capstone.incubate import live_continuation, long_run_sd_factor
from capstone.incubation.confirm import ConfirmConfig, confirm_e_process, decide_paths
from capstone.incubation.monitor import MonitorConfig, MonitorState, monitor
from capstone.runlog import log_run
from capstone.scorecard import TrialScope

PARAMS = Path(__file__).with_suffix(".json")
TRIAL_NAME = "promotion-sim-variant"
# Synthetic worlds: the "look" the gate records is at a simulated holdout, not a real one.
AUTHORIZER = "promotion_rates.py (synthetic worlds)"


def world(p: dict, seed: int, share_real: float, thresholds: Thresholds) -> pd.DataFrame:
    """One world through the registered gate: per family its truth and outcome."""
    fs = make_family_set(p["n_families"], p["k"], share_real, p["sharpe"], seed=seed)
    dates = pd.bdate_range("1990-01-01", periods=len(fs.returns))
    returns = fs.returns.set_axis(dates)
    search_end, holdout_start = dates[fs.n_search - 1], dates[fs.n_search]
    config = gate.GateConfig(
        search_end=search_end,
        holdout_start=holdout_start,
        holdout_end=dates[-1],
        data_version=f"synthetic-family-set:{seed}:{share_real}:{p['sharpe']}",
        generator="promotion-rates",
        thresholds=thresholds,
        n_boot=p["n_boot"],
        seed=0,
        trial_scope=TrialScope(name=TRIAL_NAME),
    )
    gate.register(config)
    search = returns.loc[:search_end]

    def holdout(ids, start, end):
        return returns.loc[start:end, list(ids)]

    result = gate.run_gate(config, search, fs.family, holdout=holdout, authorizer=AUTHORIZER)
    # The gate itself labels no incubate family: `promotion` does, from its stage columns
    # (all False unless the thresholds were registered with promote="incubate").
    card = promotion.decide(result.scorecard.copy(), thresholds, grading.expected_mean)
    out = card[["best", "accepted", "incubate", "expected_mean", "is_mean", "scale"]].copy()
    out["real"] = fs.truth.reindex(card.index).astype(bool)
    out["seed"] = seed
    out["search"] = [search[b] for b in card["best"]]
    return out


def monitored(rows: pd.DataFrame, horizons, alpha: float, block: int, mean_column: str) -> dict:
    """Retire rate per horizon of `rows` on a zero-edge live continuation."""
    rates = {}
    for h in horizons:
        retired = 0
        for _, r in rows.iterrows():
            config = MonitorConfig(
                expected_mean=float(r[mean_column]),
                scale=float(r["scale"]) * long_run_sd_factor(r["search"]),
                alpha=alpha,
                horizon=h,
            )
            live = live_continuation(r["search"], mean=0.0, periods=h, seed=r["seed"], block=block)
            retired += monitor(live, MonitorState.start(config)).decision == "retire"
        rates[str(h)] = retired / len(rows) if len(rows) else float("nan")
    return rates


def confirming(rows: pd.DataFrame, p: dict, live_mean) -> dict:
    """Per confirm horizon: share of (family, live draw) runs retired, promoted and
    undecided, with the decay monitor at the gate's expected mean, both scales the
    gate's times the long-run sd factor, `live_mean(row)` the live edge."""
    rates = {}
    for h in p["confirm_horizons"]:
        if not len(rows):
            rates[str(h)] = dict.fromkeys(("retire", "promote", "continue"), float("nan"))
            continue
        decided = []
        for _, r in rows.iterrows():
            scale = float(r["scale"]) * long_run_sd_factor(r["search"])
            decay = MonitorConfig(
                expected_mean=float(r["expected_mean"]), scale=scale, alpha=p["alpha"], horizon=h
            )
            live = np.column_stack(
                [
                    live_continuation(
                        r["search"],
                        mean=live_mean(r),
                        periods=h,
                        seed=int(r["seed"]) * 1000 + j,
                        block=p["block"],
                    )
                    for j in range(p["live_draws"])
                ]
            )
            config = ConfirmConfig(scale=scale, alpha=p["alpha"], horizon=h)
            decided.extend(decide_paths(config, decay, live)[0])
        decided = np.asarray(decided)
        rates[str(h)] = {d: float((decided == d).mean()) for d in ("retire", "promote", "continue")}
    return rates


def confirm_power(q: dict, alpha: float) -> dict:
    """Promote rate within each horizon (years) and the years to 50% / 80% power,
    iid normal daily returns of annualised Sharpe s, scale = their sd (1)."""
    periods = max(q["years"]) * 252
    config = ConfirmConfig(scale=1.0, alpha=alpha, horizon=periods)
    rng = np.random.default_rng(q["seed"])
    shocks = [rng.standard_normal((periods, q["chunk"])) for _ in range(q["paths"] // q["chunk"])]
    out = {}
    for s in q["sharpes"]:
        times = []
        for z in shocks:
            hit = confirm_e_process(config, s / np.sqrt(252) + z) >= 1 / alpha
            times.append(np.where(hit.any(axis=0), np.argmax(hit, axis=0) + 1, np.inf))
        t = np.concatenate(times) / 252
        quantiles = {  # inverted_cdf: a first-crossing time, never inf - inf interpolated
            f"{int(100 * x)}%": float(np.quantile(t, x, method="inverted_cdf")) for x in (0.5, 0.8)
        }
        out[str(s)] = {
            "promote_within_years": {str(y): float((t <= y).mean()) for y in q["years"]},
            "years_to_power": {k: (v if np.isfinite(v) else None) for k, v in quantiles.items()},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5 + 5 worlds, a smoke test")
    parser.add_argument("--params", type=Path, default=PARAMS, help="parameter file")
    parser.add_argument(
        "--q4-incubate",
        type=float,
        default=None,
        help='opt in to the incubate label (D-4): promote="incubate" at this stage-4 level',
    )
    args = parser.parse_args()
    p = json.loads(args.params.read_text())
    if args.q4_incubate is not None:
        p["q4_incubate"] = args.q4_incubate
    n_planted, n_null = (5, 5) if args.quick else (p["planted_worlds"], p["null_worlds"])
    q4 = p.get("q4_incubate")
    thresholds = Thresholds() if q4 is None else Thresholds(promote="incubate", q4_incubate=q4)
    for i in range(p["n_families"] * p["k"]):  # every world's variants are trials
        log_run(TRIAL_NAME, params={"variant": i}, seed=i, tags=["synthetic", "promotion"])

    seeds = range(p["first_seed"], p["first_seed"] + max(n_planted, n_null))
    planted = pd.concat([world(p, s, p["share_real"], thresholds) for s in seeds[:n_planted]])
    null = pd.concat([world(p, s, 0.0, thresholds) for s in seeds[:n_null]])
    real, planted_null = planted[planted["real"]], planted[~planted["real"]]
    incubating = pd.concat([null[null["incubate"]], planted_null[planted_null["incubate"]]])

    def share(frame, column):
        return float(frame[column].mean()) if len(frame) else float("nan")

    metrics = {
        "planted_real_families": len(real),
        "planted_real_accepted": int(real["accepted"].sum()),
        "planted_real_incubate": int(real["incubate"].sum()),
        "planted_real_reaching_incubation": int((real["accepted"] | real["incubate"]).sum()),
        "planted_null_families": len(planted_null),
        "planted_null_accepted_rate": share(planted_null, "accepted"),
        "planted_null_incubate_rate": share(planted_null, "incubate"),
        "null_families": len(null),
        "null_accepted_rate": share(null, "accepted"),
        "null_incubate_rate": share(null, "incubate"),
        "null_worlds_with_an_incubate": int(null.groupby("seed")["incubate"].any().sum()),
        "null_incubation_only_monitored": len(incubating),
        "null_incubation_only_retire_rate": monitored(
            incubating, p["horizons"], p["alpha"], p["block"], "expected_mean"
        ),
        "null_incubation_only_retire_rate_at_raw_search_mean": monitored(
            incubating, p["horizons"], p["alpha"], p["block"], "is_mean"
        ),
        "null_incubation_only_max_expected_annual_sharpe": float(
            (incubating["expected_mean"] / incubating["scale"]).max() * 252**0.5
        )
        if len(incubating)
        else float("nan"),
        "null_incubation_only_confirming": confirming(incubating, p, lambda r: 0.0),
        "planted_real_incubate_families": int(real["incubate"].sum()),
        "planted_real_incubate_confirming": confirming(
            real[real["incubate"]],
            p,
            lambda r: p["sharpe"] / np.sqrt(252) * float(r["search"].std()),
        ),
        "confirm_power_iid_normal": confirm_power(p["power"], p["alpha"]),
    }
    log_run(
        "promotion-rates",
        params={**p, "planted_worlds": n_planted, "null_worlds": n_null, "data_version": "synth"},
        metrics=metrics,
        tags=["synthetic", "promotion"],
        notes="incubate outcome rates; decay and confirming incubation on them",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
