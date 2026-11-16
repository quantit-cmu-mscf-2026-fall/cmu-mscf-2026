"""Offline proof and frozen order plans. No command connects to a broker.

python -m capstone.pipeline.offline prove --out <new-directory>
python -m capstone.pipeline.offline plan <bundle-directory> --out <plan.json>
python -m capstone.pipeline.offline simulate <bundle-directory> <plan.json> --out <new-directory>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from capstone import runlog
from capstone.execution.bundle import freeze_bundle, load_bundle
from capstone.execution.frozen import build_plan, read_plan, write_plan
from capstone.execution.journal import execute_plan
from capstone.execution.simulator import SimulatedBroker
from capstone.pipeline.diagnostics import cohort_diagnostics
from capstone.pipeline.evidence import replay_evidence, write_json
from capstone.pipeline.execution_proof import fault_proof
from capstone.pipeline.reference_fixture import reference_panel
from capstone.pipeline.reference_validation import discovery_evidence
from capstone.pipeline.replay import ReplayCosts, replay
from capstone.release import runtime_versions
from capstone.strategy.portfolio import PortfolioSpec
from capstone.strategy.reference import BenchmarkSpec, ReferenceStrategySpec
from capstone.strategy.research import canonical, contracts


def _synthetic(bundle, plan, market=None):
    """The plan's bundle, if the simulator may run it: explicitly synthetic only.

    A bundle made from current data is refused. A verifier for such bundles
    could be attached later as `capstone.pipeline.autopilot.live_receipt`; it is
    looked up lazily, and without it the path stays closed.
    """
    data, _, _, _ = load_bundle(bundle)
    if data["input_hash"] != plan["input_hash"]:
        raise ValueError("offline simulator requires this plan's explicitly synthetic input bundle")
    if not data["data_version"].startswith("synthetic-"):
        try:
            from capstone.pipeline.autopilot import live_receipt
        except ImportError as error:  # `from package import missing` is not ModuleNotFoundError
            if not (error.name or "").startswith("capstone.pipeline.autopilot"):
                raise
            raise ValueError(
                "offline simulator requires this plan's explicitly synthetic input bundle"
            ) from error
        live_receipt.verify(data, market)
    return data


def _fresh_broker(data, plan, market):
    return SimulatedBroker(
        data["account"],
        profile=plan["profile"],
        session=plan["execution_session"],
        execution_prices=(market or {}).get("prices"),
        commission_bps=(market or {}).get("commission_bps", 0),
        execution_adjustment_bps=(market or {}).get("execution_adjustment_bps", 0),
    )


def simulate(bundle, plan_path, out, *, market=None):
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    plan = read_plan(plan_path)
    data = _synthetic(bundle, plan, market)
    state = root / "simulated-broker.json"
    if not state.exists() and market is None:
        raise ValueError("a fresh simulation requires an explicit synthetic next-session market")
    if market is not None and market.get("session") != plan["execution_session"]:
        raise ValueError("synthetic execution market belongs to a different session")
    broker = SimulatedBroker.load(state) if state.exists() else _fresh_broker(data, plan, market)
    result = execute_plan(plan, broker, root / "execution.sqlite", simulate=True)
    expected = {int(n): q for n, q in plan["projected_shares"].items() if q}
    if result["complete"] and result["positions"] != expected:
        raise AssertionError("simulated fills differ from the frozen fixed-share targets")
    broker.save(state)
    result["synthetic_only"] = True
    write_json(root / "reconciliation.json", result)
    return result


def prove(out):
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("proof output must be a new directory")
    root.mkdir(parents=True, exist_ok=True)
    previous_ledger = os.environ.get("CAPSTONE_LEDGER_DIR")
    os.environ["CAPSTONE_LEDGER_DIR"] = str(root / "ledger")
    try:
        panel = reference_panel()
        catalog = contracts()
        write_json(root / "claim-audit.json", catalog)
        parameters = {
            "seed": 404,
            "start": "2005-01-03",
            "end": "2005-12-30",
            "capital": 100000.0,
            "synthetic_only": True,
            "costs": asdict(ReplayCosts(1, 5, 5, 200, 500)),
            "stress_multiplier": 2,
        }
        write_json(root / "parameters.json", parameters)
        write_json(
            root / "cohort-diagnostics.json",
            cohort_diagnostics(
                panel, anchor="2000-12-29", start=parameters["start"], end=parameters["end"]
            ),
        )
        specs = [
            ReferenceStrategySpec(c["id"], panel.data_version, "2000-12-29", side)
            for side in ("long_only", "long_short")
            for c in catalog["claims"]
            if c["status"] == "executable"
        ]
        base = ReplayCosts(**parameters["costs"])
        stress = ReplayCosts(**{k: 2 * v for k, v in parameters["costs"].items()})
        returns, reports = {}, []
        opens, closes = panel.fields["open"], panel.fields["close"]
        for spec in specs:
            for label, costs in (("base", base), ("stress", stress)):
                result = replay(
                    spec,
                    panel,
                    opens,
                    closes,
                    start=parameters["start"],
                    end=parameters["end"],
                    capital=parameters["capital"],
                    costs=costs,
                )
                curve = result["history"]["net"]
                reports.append(
                    {
                        "strategy_id": spec.id,
                        "claim_id": spec.claim_id,
                        "side": spec.side,
                        "cost_scenario": label,
                        **replay_evidence(root, result, label),
                    }
                )
                if label == "base":
                    returns[spec.id] = curve
        matrix = pd.DataFrame(returns)
        write_json(root / "discovery-statistics.json", discovery_evidence(matrix, specs))
        benchmarks = [
            BenchmarkSpec(kind, panel.data_version, "2000-12-29")
            for kind in (
                "cash",
                "cap_weight_universe_proxy",
                "equal_weight_universe_proxy",
                "known_momentum_12_1",
            )
        ]
        benchmarks.append(
            BenchmarkSpec("known_momentum_12_1", panel.data_version, "2000-12-29", "long_short")
        )
        for spec in benchmarks:
            result = replay(
                spec,
                panel,
                opens,
                closes,
                start=parameters["start"],
                end=parameters["end"],
                capital=parameters["capital"],
                costs=base,
            )
            reports.append(
                {
                    "benchmark": spec.kind,
                    "side": spec.side,
                    "strategy_id": spec.id,
                    **replay_evidence(root, result, "base"),
                }
            )
        portfolio_reports, profiles = [], []
        for side in ("long_only", "long_short"):
            selected = tuple(s for s in specs if s.side == side)
            handoff = None
            for mode in ("equal_family", "correlation_groups"):
                basket = PortfolioSpec(selected, matrix, mode)
                result = replay(
                    basket,
                    panel,
                    opens,
                    closes,
                    start=parameters["start"],
                    end=parameters["end"],
                    capital=parameters["capital"],
                    costs=base,
                )
                portfolio_reports.append(
                    {
                        "id": basket.id,
                        "profile": side,
                        "allocation": mode,
                        **replay_evidence(root, result, "base"),
                    }
                )
                if mode == "equal_family":
                    handoff = result
            destination = root / side
            prices = closes.loc[parameters["end"]].to_dict()
            instruments = {
                n: {
                    "symbol": f"S{n}",
                    "currency": "USD",
                    "sec_type": "STK",
                    "exchange": "SMART",
                    "qualified": False,
                }
                for n in prices
            }
            account = {
                "positions": handoff["positions"],
                "open_orders": [],
                "cash": handoff["cash"],
                "capital": float(handoff["history"].iloc[-1].equity),
            }
            freeze_bundle(
                destination / "inputs",
                panel,
                selected,
                parameters["end"],
                prices=prices,
                instruments=instruments,
                returns=matrix,
                execution_cost_bps=11.0,
                account=account,
            )
            plan = build_plan(destination / "inputs")
            if handoff["pending"] is None or canonical(plan["projected_shares"]) != canonical(
                handoff["pending"]["projected"]
            ):
                raise AssertionError(
                    "historical replay and offline planner produced different shares"
                )
            write_plan(plan, destination / "order-plan.json")
            again = build_plan(destination / "inputs")
            if plan["document_hash"] != again["document_hash"]:
                raise AssertionError("frozen plan is not repeatable")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "capstone.pipeline.offline",
                    "plan",
                    str(destination / "inputs"),
                    "--out",
                    str(destination / "fresh-process-plan.json"),
                ],
                check=True,
            )
            if canonical(read_plan(destination / "fresh-process-plan.json")) != canonical(plan):
                raise AssertionError("fresh-process order instructions changed")
            market = {
                "session": plan["execution_session"],
                "prices": opens.loc[plan["execution_session"]].to_dict(),
                "commission_bps": 1,
                "execution_adjustment_bps": 10,
            }
            faults = fault_proof(plan, account, destination / "faults", market=market)
            write_json(destination / "fault-results.json", faults)
            reconciled = simulate(
                destination / "inputs",
                destination / "order-plan.json",
                destination / "simulation",
                market=market,
            )
            profiles.append(
                {
                    "profile": side,
                    "faults_passed": all(f["passed"] for f in faults.values()),
                    "fresh_process_identical": True,
                    "replay_planner_quantities_identical": True,
                    "instructions": len(plan["orders"]),
                    "document_hash": plan["document_hash"],
                    "reconciliation": reconciled,
                    "planning_blockers": plan["planning_blockers"],
                    "validation_blockers": plan["validation_blockers"],
                    "broker_blockers": plan["broker_blockers"],
                }
            )
        matrix.to_parquet(root / "discovery-return-matrix.parquet")
        write_json(
            root / "strategy-evidence.json",
            {
                "synthetic_only": True,
                "formally_admitted": False,
                "trial_accounting_reconciled": False,
                "strategies_and_benchmarks": reports,
                "portfolios": portfolio_reports,
            },
        )
        readiness = {
            "software_proof_passed": all(
                p["reconciliation"]["complete"]
                and p["faults_passed"]
                and p["fresh_process_identical"]
                and not p["planning_blockers"]
                for p in profiles
            ),
            "synthetic_only": True,
            "real_strategy_admission": False,
            "real_broker_verified": False,
            "holdout_opened": False,
            "runtime": runtime_versions(),
            "profiles": profiles,
            "claims": {
                status: sum(c["status"] == status for c in catalog["claims"])
                for status in ("executable", "diagnostic", "blocked")
            },
            "registered_replays": sum(
                e["name"] == "reference-replay" for e in runlog.read_entries()
            ),
            "registered_ledger_entries": len(runlog.read_entries()),
        }
        write_json(root / "readiness.json", readiness)
        text = [
            "# Offline release evidence",
            "",
            "Synthetic implementation proof; no financial edge is certified.",
            f"Software proof passed: {readiness['software_proof_passed']}",
            "Real strategy admission: false. Real broker verification: false.",
            "Holdout opened: false.",
            "",
            "| Profile | Order instructions | Reconciled |",
            "|---|---:|---|",
        ]
        text += [
            f"| {p['profile']} | {p['instructions']} | {p['reconciliation']['complete']} |"
            for p in profiles
        ]
        text += [
            "",
            "See the claim audit, strategy evidence, parameters and profile order plans.",
            "",
            "Before any broker connection: reviewed release, gate reports and trial accounting;",
            "current universe and feature feeds; qualified contracts; paper login and market data;",
            "USD funds, margin and shorting checks; calendar and monitored read-only runs.",
        ]
        (root / "READINESS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
        return readiness
    finally:
        if previous_ledger is None:
            os.environ.pop("CAPSTONE_LEDGER_DIR", None)
        else:
            os.environ["CAPSTONE_LEDGER_DIR"] = previous_ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    proof = commands.add_parser("prove")
    proof.add_argument("--out", required=True, type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("--out", type=Path)
    plan = commands.add_parser("plan")
    plan.add_argument("bundle", type=Path)
    plan.add_argument("--out", required=True, type=Path)
    sim = commands.add_parser("simulate")
    sim.add_argument("bundle", type=Path)
    sim.add_argument("plan", type=Path)
    sim.add_argument("--out", required=True, type=Path)
    sim.add_argument("--market", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prove":
        result = prove(args.out)
        print(
            canonical(
                {"software_proof_passed": result["software_proof_passed"], "out": str(args.out)}
            )
        )
    elif args.command == "audit":
        write_json(args.out, contracts()) if args.out else print(canonical(contracts()))
    elif args.command == "plan":
        write_plan(build_plan(args.bundle), args.out)
    else:
        market = (
            None if args.market is None else json.loads(args.market.read_text(encoding="utf-8"))
        )
        print(canonical(simulate(args.bundle, args.plan, args.out, market=market)))


if __name__ == "__main__":
    main()
