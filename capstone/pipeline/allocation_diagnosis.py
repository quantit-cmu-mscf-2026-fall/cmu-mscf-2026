"""Frozen retrospective diagnosis; no new discovery, source downloads or broker access."""

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd

from capstone import runlog
from capstone.factors.research_artifacts import atomic_json
from capstone.pipeline.reference_allocation import replay, statistics, uncertainty
from capstone.pipeline.reference_inputs import POOL, validate_reference
from capstone.strategy.research import canonical, digest
from capstone.strategy.robust_allocation import (
    AllocationPolicy,
    AllocationPolicyV2,
    optimize_allocation,
    shrunk_covariance,
)

MODES = ("equal_weight", "shrunk_max_sharpe", "median_window_sharpe")
PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "config/allocation_diagnosis.json"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_policy(costs, *, corrected=False, lookback=120):
    cls = AllocationPolicyV2 if corrected else AllocationPolicy
    return cls(
        periods=12,
        lookback=lookback,
        minimum=60,
        window=36,
        step=12,
        holding_periods=3,
        ongoing_bps=costs["ongoing_monthly_bps"],
        switch_bps=costs["switch_bps"],
    )


def frozen_provider(decisions, frame, *, risk_limit=None, label="frozen_targets"):
    by_date = {pd.Timestamp(d["as_of"]): d for d in decisions}
    if len(by_date) != len(decisions):
        raise ValueError("duplicate frozen decision dates")

    def decide(at, _current):
        original = by_date[at]
        names = sorted(POOL)
        weights = np.array([original["weights"][n] for n in names], dtype=float)
        scale, observations = 1.0, None
        if risk_limit is not None:
            past = frame.loc[frame.index < at, names].sort_index().tail(120)
            if len(past) < 60 or not np.isfinite(past.to_numpy()).all():
                raise ValueError("prior history insufficient for risk overlay")
            covariance, _ = shrunk_covariance(past.to_numpy())
            vol = np.sqrt(max(float(weights @ covariance @ weights), 0) * 12)
            scale = min(1.0, risk_limit / max(float(vol), 1e-12))
            observations = len(past)
        weights *= scale
        result = {
            "as_of": str(at.date()),
            "weights": dict(zip(names, weights, strict=True)),
            "status": label,
            "source_decision_id": original.get("id"),
            "source_mode": original.get("mode"),
            "risk_scale": scale,
            "prior_observations": observations,
            "reoptimization": False,
            "diagnostic_only": True,
        }
        result["id"] = digest(result)
        return result

    return decide


def write_replay(root, key, history, decisions):
    target = root / key
    target.mkdir(parents=True)
    history.to_csv(target / "monthly.csv", float_format="%.17g", lineterminator="\n")
    atomic_json(target / "decisions.json", decisions)
    result = {
        "full_1996_2020": statistics(history),
        "later_2016_2020": statistics(history.loc["2016-01-01":]),
        "five_year_periods": {
            str(y): statistics(history.loc[f"{y}-01-01" : f"{y + 4}-12-31"])
            for y in range(1996, 2021, 5)
        },
        "status_counts": pd.Series([d["status"] for d in decisions]).value_counts().to_dict(),
    }
    runlog.log_run(
        "allocation-diagnosis",
        params={
            "comparison": key,
            "classification": "legacy_reproduction"
            if key.startswith("legacy-")
            else "registered_retrospective_diagnostic",
        },
        metrics=result["full_1996_2020"],
        seed=404,
        tags=["discovery", "diagnostic", "not-admission"],
        ledger_dir=root / "ledger",
    )
    return result


def run(baseline, registration, output):
    from capstone.pipeline.allocation_diagnostic_statistics import (
        concentration_diagnostics,
        forecast_audit,
        pool_evidence,
        risk_sharpe_decomposition,
    )

    baseline, output = Path(baseline), Path(output)
    repo = Path(__file__).resolve().parents[2]
    config = json.loads(Path(registration).read_text())
    if config != json.loads(PROTOCOL_PATH.read_text()):
        raise ValueError("diagnostic registration differs from committed protocol")
    old_manifest = json.loads((baseline / "study-1/manifest.json").read_text())
    if old_manifest["git_commit"] != config["baseline_commit"]:
        raise ValueError("wrong baseline study revision")
    source = baseline / "inputs/monthly.csv"
    if file_hash(source) != config["baseline_input_sha256"]:
        raise ValueError("wrong reference input bundle")
    inventory = json.loads((baseline / "evidence-sha256.json").read_text())
    if not all(file_hash(baseline / p) == h for p, h in inventory.items()):
        raise ValueError("original evidence changed")
    frame = validate_reference(
        pd.read_csv(source, index_col="month", parse_dates=True, float_precision="round_trip")
    )
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        repo / "capstone/pipeline/allocation_diagnostic_statistics.py",
        repo / "capstone/pipeline/allocation_diagnosis_report.py",
        repo / "capstone/pipeline/allocation_diagnosis_views.py",
        repo / "capstone/pipeline/reference_allocation.py",
        repo / "capstone/strategy/robust_allocation.py",
        repo / "capstone/strategy/portfolio.py",
        repo / "config/allocation_diagnosis.json",
        repo / "capstone/pipeline/reference_inputs.py",
        repo / "capstone/runlog.py",
        repo / "capstone/strategy/research.py",
        repo / "capstone/factors/research_artifacts.py",
        repo / "pyproject.toml",
    ]
    atomic_json(
        output / "manifest.json",
        {
            "registration": config,
            "baseline_inventory_sha256": file_hash(baseline / "evidence-sha256.json"),
            "baseline_files_verified": len(inventory),
            "input_sha256": file_hash(source),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
            ).stdout.strip(),
            "code": {p.relative_to(repo).as_posix(): file_hash(p) for p in paths},
            "runtime": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scipy": package_version("scipy"),
                "matplotlib": package_version("matplotlib"),
                "ruff": package_version("ruff"),
            },
        },
    )
    atomic_json(output / "trial-accounting.json", config["trial_accounting"])
    old_config = old_manifest["registration"]
    results = {
        "legacy_equivalence": {},
        "corrected_v2": {},
        "frozen_target_costs": {},
        "prior_risk_overlay": {},
        "hindsight_fixed_mix": {},
        "diagnostics": {},
    }
    histories, saved_decisions = {}, {}
    for scenario, costs in old_config["cost_scenarios"].items():
        for mode in MODES:
            key = f"{scenario}/{mode}"
            old_path = baseline / "study-1" / key
            legacy, decisions = replay(frame, mode, make_policy(costs))
            new_path = output / "legacy-reproduction" / key
            write_replay(output, "legacy-reproduction/" + key, legacy, decisions)
            same = file_hash(new_path / "monthly.csv") == file_hash(
                old_path / "monthly.csv"
            ) and file_hash(new_path / "decisions.json") == file_hash(old_path / "decisions.json")
            results["legacy_equivalence"][key] = same
            if not same:
                raise ValueError(f"legacy behavior changed: {key}")
            histories[key], saved_decisions[key] = legacy, decisions
            corrected, cd = replay(frame, mode, make_policy(costs, corrected=True))
            results["corrected_v2"][key] = write_replay(
                output, "corrected-v2/" + key, corrected, cd
            )
            histories["v2/" + key], saved_decisions["v2/" + key] = corrected, cd
            print("corrected", key, "complete", flush=True)
    for mode in MODES:
        key = "assumed_base/" + mode
        results["corrected_v2"][key]["paired_vs_old_equal"] = uncertainty(
            histories["v2/" + key], histories["assumed_base/equal_weight"]
        )
    for mode in MODES:
        key = "assumed_base/" + mode
        original = saved_decisions[key]
        results["frozen_target_costs"][mode] = {}
        for scenario, costs in old_config["cost_scenarios"].items():
            h, d = replay(
                frame, mode, make_policy(costs), decision_provider=frozen_provider(original, frame)
            )
            results["frozen_target_costs"][mode][scenario] = write_replay(
                output, f"frozen-targets/{mode}/{scenario}", h, d
            )
            if scenario == "assumed_base":
                if not np.array_equal(h.to_numpy(), histories[key].to_numpy()):
                    raise ValueError("frozen base replay differs from original")
        h, d = replay(
            frame,
            mode,
            make_policy(old_config["cost_scenarios"]["assumed_base"]),
            decision_provider=frozen_provider(
                original, frame, risk_limit=0.05, label="prior_5pct_risk_overlay"
            ),
        )
        results["prior_risk_overlay"][mode] = write_replay(output, "risk-overlay/" + mode, h, d)
        print("frozen cost/risk controls", mode, "complete", flush=True)
    names = sorted(POOL)
    for mode in MODES[1:]:
        decision = optimize_allocation(
            frame.loc["1996-01-01":, names],
            "2021-01-01",
            mode=mode,
            policy=make_policy(
                old_config["cost_scenarios"]["assumed_base"], corrected=True, lookback=300
            ),
        )
        schedule = [
            {**decision, "as_of": str(at.date())}
            for at in pd.date_range("1996-01-01", "2020-10-01", freq="QS")
        ]
        h, d = replay(
            frame,
            mode,
            make_policy(old_config["cost_scenarios"]["assumed_base"]),
            decision_provider=frozen_provider(
                schedule, frame, label="NONDEPLOYABLE_full_sample_mix"
            ),
        )
        results["hindsight_fixed_mix"][mode] = {
            **write_replay(output, "hindsight/" + mode, h, d),
            "uses_future_information": True,
            "upper_bound_guaranteed": False,
            "construction": "full-sample optimized fixed mix; no rolling forecast",
            "fit": decision,
        }
    for version in ("legacy", "v2"):
        for mode in MODES:
            key = "assumed_base/" + mode
            actual_key = key if version == "legacy" else "v2/" + key
            history = histories[actual_key]
            results["diagnostics"][version + "/" + mode] = {
                "risk_sharpe_identity": risk_sharpe_decomposition(
                    history, histories["assumed_base/equal_weight"]
                ),
                "concentration": concentration_diagnostics(history),
                "forecast": forecast_audit(frame[names], saved_decisions[actual_key], history),
                "solver": [
                    dict(as_of=d["as_of"], **d["solver"])
                    for d in saved_decisions[actual_key]
                    if "solver" in d
                ],
            }
    results["pool_evidence"] = pool_evidence(frame.loc["1996-01-01":, names])
    results["qualification"] = {
        "formal_admission": False,
        "historical_vintages_verified": False,
        "execution_costs_observed": False,
        "stock_gross_exposure_verified": False,
        "optimizer_winner_certified": False,
        "original_artifacts_preserved": True,
    }
    atomic_json(output / "results.json", results)
    ledger = output / "comparisons.jsonl"
    entries = []
    for section in ("corrected_v2", "prior_risk_overlay", "hindsight_fixed_mix"):
        entries.extend({"section": section, "comparison": key} for key in results[section])
    entries.extend(
        {"section": "frozen_target_costs", "comparison": mode + "/" + scenario}
        for mode in MODES
        for scenario in old_config["cost_scenarios"]
    )
    ledger.write_text("".join(canonical(e) + "\n" for e in entries), encoding="utf-8")
    from capstone.pipeline.allocation_diagnosis_report import report

    report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline, args.registration, args.output)


if __name__ == "__main__":
    main()
