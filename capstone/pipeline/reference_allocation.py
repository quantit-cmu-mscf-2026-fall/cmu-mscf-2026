"""Frozen, offline walk-forward allocation of published factor-return sleeves."""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from capstone import runlog
from capstone.factors.research_artifacts import atomic_json
from capstone.pipeline.reference_inputs import POOL, fetch, validate_reference
from capstone.strategy.portfolio import allocation
from capstone.strategy.robust_allocation import AllocationPolicy


@dataclass(frozen=True)
class Reference:
    id: str
    side: str = "long_short"


def replay(frame, mode, policy, *, decision_provider=None):
    """Monthly synthetic funded sleeves; quarterly targets and monthly weight drift."""
    frame = validate_reference(frame)
    names = sorted(POOL)
    specs = [Reference(n) for n in names]
    current, wealth = np.zeros(len(names)), 100000.0
    rows, decisions = [], []
    for at, row in frame.loc["1996-01-01":].iterrows():
        start_wealth = wealth
        target, switching, turnover = current.copy(), 0.0, 0.0
        if at.month in {1, 4, 7, 10}:
            if decision_provider is None:
                decision = allocation(
                    specs,
                    frame[names],
                    at,
                    mode=mode,
                    return_basis="self_financing_excess",
                    policy=policy,
                    previous=dict(zip(names, current, strict=True)),
                )
            else:
                decision = decision_provider(at, current.copy())
                if set(decision["weights"]) != set(names) or pd.Timestamp(decision["as_of"]) != at:
                    raise ValueError("frozen decision must match the execution month and pool")
            decisions.append(decision)
            target = np.array([decision["weights"][n] for n in names])
            if not np.isfinite(target).all() or target.min() < 0 or target.sum() > 1 + 1e-8:
                raise ValueError("invalid frozen portfolio coefficients")
            # Fee and target dollars must reconcile on NAV after paying the fee.
            rate = policy.switch_bps / 10000
            switching = (
                brentq(
                    lambda c, rate=rate, target=target, current=current: (
                        c - rate * abs((1 - c) * target - current).sum()
                    ),
                    0,
                    0.1,
                    xtol=1e-15,
                )
                if rate
                else 0.0
            )
            turnover = float(abs((1 - switching) * target - current).sum())
        invested = target.sum()
        ongoing = policy.ongoing_bps / 10000 * invested
        factor_pnl = float(target @ row[names].to_numpy())
        inside_return = float(row.RF + factor_pnl - ongoing)
        net = (1 - switching) * (1 + inside_return) - 1
        if not np.isfinite(net) or net <= -1:
            raise ValueError("synthetic account insolvency or undefined return")
        wealth *= 1 + net
        current = (
            target
            * (1 + row.RF + row[names].to_numpy() - policy.ongoing_bps / 10000)
            / (1 + inside_return)
        )
        if current.min() < -1e-10 or current.sum() > 1 + 1e-8:
            raise ValueError("synthetic sleeve accounting failed")
        components = {
            "cash_interest_pnl": start_wealth * (1 - switching) * float(row.RF),
            "factor_pnl": start_wealth * (1 - switching) * factor_pnl,
            "ongoing_cost": start_wealth * (1 - switching) * ongoing,
            "switch_cost": start_wealth * switching,
        }
        residual = (
            components["cash_interest_pnl"]
            + components["factor_pnl"]
            - components["ongoing_cost"]
            - components["switch_cost"]
            - (wealth - start_wealth)
        )
        if abs(residual) > 1e-7:
            raise ValueError("P&L reconciliation failed")
        rows.append(
            {
                "month": at,
                "return": net,
                "excess": net - row.RF,
                "equity": wealth,
                "RF": row.RF,
                "invested": invested,
                "turnover": turnover,
                "active": int((target > 1e-6).sum()),
                **components,
                **{f"weight_{n}": float(w) for n, w in zip(names, target, strict=True)},
                **{
                    f"factor_pnl_{n}": start_wealth * (1 - switching) * float(w * row[n])
                    for n, w in zip(names, target, strict=True)
                },
            }
        )
    return pd.DataFrame(rows).set_index("month"), decisions


def statistics(history):
    returns, excess = history["return"], history.excess
    wealth = np.r_[1.0, np.cumprod(1 + returns)]
    vol = float(excess.std(ddof=1) * np.sqrt(12))
    return {
        "months": len(history),
        "cagr": float(wealth[-1] ** (12 / len(history)) - 1),
        "annual_excess_mean": float(excess.mean() * 12),
        "annual_volatility": vol,
        "sharpe": float(excess.mean() * 12 / vol) if vol > 1e-12 else None,
        "maximum_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "ending_value_from_100k": float(100000 * wealth[-1]),
        "mean_invested": float(history.invested.mean()),
        "mean_active": float(history.active.mean()),
        "annual_coefficient_turnover": float(history.turnover.mean() * 12)
        if "turnover" in history
        else None,
        "active_range": [int(history.active.min()), int(history.active.max())],
        "assumed_cost_dollars": float(history.ongoing_cost.sum() + history.switch_cost.sum()),
    }


def uncertainty(candidate, benchmark):
    """Paired 12-month moving blocks; exploratory intervals, not selection-adjusted tests."""
    rng = np.random.default_rng(404)
    x = np.column_stack([candidate.excess, benchmark.excess])
    n = len(x)
    indices = rng.integers(0, n - 12 + 1, size=(1999, int(np.ceil(n / 12))))
    indices = (indices[:, :, None] + np.arange(12)).reshape(1999, -1)[:, :n]
    sample = x[indices]
    vol = sample.std(axis=1, ddof=1)
    ratios = np.divide(
        sample.mean(axis=1) * np.sqrt(12), vol, out=np.full_like(vol, np.nan), where=vol > 1e-12
    )
    delta = ratios[:, 0] - ratios[:, 1]
    finite = delta[np.isfinite(delta)]
    return {
        "sharpe_difference_95_interval": np.quantile(finite, [0.025, 0.975]).tolist()
        if len(finite)
        else None,
        "draws": 1999,
        "block_months": 12,
        "seed": 404,
        "selection_adjusted": False,
    }


def run(inputs, registration, output):
    inputs, output = Path(inputs), Path(output)
    repo = Path(__file__).resolve().parents[2]
    config = json.loads(Path(registration).read_text())
    if config != json.loads((repo / "config/reference_allocation.json").read_text()):
        raise ValueError("incompatible frozen study registration")
    raw = inputs / "monthly.csv"
    sources = json.loads((inputs / "sources.json").read_text())
    if hashlib.sha256(raw.read_bytes()).hexdigest() != sources["monthly_sha256"]:
        raise ValueError("reference input hash mismatch")
    frame = validate_reference(
        pd.read_csv(raw, index_col="month", parse_dates=True, float_precision="round_trip")
    )
    output.mkdir(parents=True, exist_ok=False)
    code_paths = [
        Path(__file__),
        repo / "capstone/pipeline/reference_inputs.py",
        repo / "capstone/strategy/robust_allocation.py",
        repo / "capstone/strategy/portfolio.py",
        repo / "capstone/pipeline/reference_report.py",
        repo / "capstone/runlog.py",
        repo / "config/reference_allocation.json",
    ]
    manifest = {
        "registration": config,
        "sources": sources,
        "code": {
            p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code_paths
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
        ).stdout.strip(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    atomic_json(output / "manifest.json", manifest)  # Frozen before return evaluation.
    atomic_json(
        output / "trial-accounting.json",
        {
            "registration_sha256": hashlib.sha256(Path(registration).read_bytes()).hexdigest(),
            "planned_policy_cost_comparisons": len(config["policies"])
            * len(config["cost_scenarios"]),
            "standalone_descriptive_references": len(POOL),
            "quarterly_fits_per_comparison": 100,
            "optimizer_starts": "equal plus every single-factor vertex; no tuned restarts",
            "historical_reference_search_count": "unknown; no independent-trial claim",
            "selection_adjusted_admission": False,
            "relationship_to_prior_studies": "new registered comparison; originals preserved",
        },
    )
    results, histories = {}, {}
    for scenario, costs in config["cost_scenarios"].items():
        for mode in config["policies"]:
            policy = AllocationPolicy(
                periods=12,
                lookback=config["maximum_training_months"],
                minimum=config["minimum_training_months"],
                window=36,
                step=12,
                mean_shrink=config["expected_return_shrinkage"],
                holding_periods=3,
                switch_bps=costs["switch_bps"],
                ongoing_bps=costs["ongoing_monthly_bps"],
            )
            history, decisions = replay(frame, mode, policy)
            key = f"{scenario}/{mode}"
            histories[key] = history
            path = output / scenario / mode
            path.mkdir(parents=True)
            history.to_csv(path / "monthly.csv", float_format="%.17g", lineterminator="\n")
            atomic_json(path / "decisions.json", decisions)
            results[key] = {
                "full_1996_2020": statistics(history),
                "later_2016_2020": statistics(history.loc["2016-01-01":]),
                "five_year_periods": {
                    str(y): statistics(history.loc[f"{y}-01-01" : f"{y + 4}-12-31"])
                    for y in range(1996, 2021, 5)
                },
                "allocation_status_counts": pd.Series([d["status"] for d in decisions])
                .value_counts()
                .to_dict(),
                "contribution": {
                    n: {
                        "mean_coefficient": float(history[f"weight_{n}"].mean()),
                        "factor_pnl_dollars_before_proxy_costs": float(
                            history[f"factor_pnl_{n}"].sum()
                        ),
                    }
                    for n in POOL
                },
            }
            runlog.log_run(
                "reference-allocation",
                params={
                    "comparison": key,
                    "policy": asdict(policy),
                    "input_sha256": sources["monthly_sha256"],
                },
                metrics=results[key]["full_1996_2020"],
                seed=404,
                tags=["discovery", "reference-allocation", "not-admission"],
                ledger_dir=output / "ledger",
            )
            print(key, "completed", flush=True)
    for scenario in config["cost_scenarios"]:
        for mode in config["policies"]:
            key = f"{scenario}/{mode}"
            results[key]["paired_uncertainty_vs_equal"] = uncertainty(
                histories[key], histories[f"{scenario}/equal_weight"]
            )
    standalone = {}
    period = frame.loc["1996-01-01":]
    for name in POOL:
        h = pd.DataFrame(
            {
                "return": period[name] + period.RF,
                "excess": period[name],
                "invested": 1,
                "active": 1,
                "ongoing_cost": 0,
                "switch_cost": 0,
            }
        )
        standalone[name] = statistics(h)
    frame.loc["1996-01-01":, list(POOL)].corr().to_csv(output / "correlations.csv")
    benchmarks = {}
    for name in ("market", "cash"):
        excess = period["Mkt-RF"] if name == "market" else period.RF * 0
        h = pd.DataFrame(
            {
                "return": excess + period.RF,
                "excess": excess,
                "invested": int(name == "market"),
                "active": int(name == "market"),
                "ongoing_cost": 0,
                "switch_cost": 0,
            }
        )
        benchmarks[name] = statistics(h)
    atomic_json(
        output / "results.json",
        {"allocations": results, "standalone_gross": standalone, "benchmarks_gross": benchmarks},
    )
    from capstone.pipeline.reference_report import report

    report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    downloader = commands.add_parser("fetch")
    downloader.add_argument("destination", type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("--inputs", type=Path, required=True)
    runner.add_argument("--registration", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.destination)
    else:
        run(args.inputs, args.registration, args.output)


if __name__ == "__main__":
    main()
