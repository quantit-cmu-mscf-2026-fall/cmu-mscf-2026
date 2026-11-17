"""Archived-session planning, simulated fills and restart evidence. No broker imports.

`rehearse_bundle` plans a frozen bundle, proves the plan in a fresh process, then runs it
against the simulated broker for an archived market. The archived rehearsal driver that
builds these bundles from a completed historical basket belongs with its caller.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from capstone.execution.bundle import load_bundle
from capstone.execution.frozen import build_plan, read_plan, write_plan
from capstone.execution.journal import execute_plan
from capstone.execution.simulator import SimulatedBroker
from capstone.pipeline.evidence import write_json
from capstone.strategy.research import digest


def rehearse_bundle(bundle, out, market):
    allowed = {
        "session",
        "prices",
        "commission_bps",
        "execution_adjustment_bps",
        "fault",
        "fill_fraction",
    }
    if set(market) - allowed:
        raise ValueError("archived market contains unsupported metadata")
    bundle, root = Path(bundle).resolve(), Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data, _, _, _ = load_bundle(bundle)
    plan = build_plan(bundle)
    if pd.Timestamp(plan["execution_session"]) > pd.Timestamp("2020-12-31"):
        raise ValueError("archived rehearsal cannot execute into the sealed holdout")
    if market["session"] != plan["execution_session"]:
        raise ValueError("archived market session differs from the fixed plan")
    market_path = root / "market.json"
    if market_path.exists() and digest(json.loads(market_path.read_text())) != digest(market):
        raise ValueError("archived market changed; create a new rehearsal")
    if not market_path.exists():
        write_json(market_path, market)
    plan_path = root / "order-plan.json"
    if plan_path.exists():
        if digest(read_plan(plan_path)) != digest(plan):
            raise ValueError("rehearsal inputs changed")
    else:
        write_plan(plan, plan_path)
    fresh = root / "fresh-process-plan.json"
    if not fresh.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "capstone.pipeline.offline",
                "plan",
                str(bundle),
                "--out",
                str(fresh),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    if digest(read_plan(fresh)) != digest(plan):
        raise AssertionError("fresh-process canonical plan differs")
    state = root / "simulated-broker.json"
    broker = (
        SimulatedBroker.load(state)
        if state.exists()
        else SimulatedBroker(
            data["account"],
            profile=plan["profile"],
            session=market["session"],
            execution_prices=market["prices"],
            commission_bps=market["commission_bps"],
            execution_adjustment_bps=market["execution_adjustment_bps"],
            fault=market.get("fault"),
            fill_fraction=market.get("fill_fraction", 1.0),
        )
    )
    try:
        result = execute_plan(plan, broker, root / "execution.sqlite", simulate=True)
    finally:
        broker.save(state)
    expected = {int(n): q for n, q in plan["projected_shares"].items() if q}
    if result["complete"] and result["positions"] != expected:
        raise AssertionError("archived fills differ from fixed plan quantities")
    result.update(
        submissions=broker.submissions,
        fresh_process_identical=True,
        archived_only=True,
        real_broker_verified=False,
    )
    write_json(root / "reconciliation.json", result)
    return result
