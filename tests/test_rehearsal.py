"""Archived rehearsal: a fresh `offline plan` process must reproduce the plan before fills."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from reference_fakes import reference_panel

from capstone.execution.bundle import freeze_bundle
from capstone.execution.frozen import build_plan, identify_plan, read_plan
from capstone.execution.simulator import SimulatedBroker
from capstone.pipeline import rehearsal
from capstone.pipeline.rehearsal import rehearse_bundle
from capstone.strategy.reference import ReferenceStrategySpec


@pytest.fixture(scope="module")
def archived(tmp_path_factory):
    root = tmp_path_factory.mktemp("archived")
    panel = reference_panel()
    spec = ReferenceStrategySpec("716bdaf7915c65be", panel.data_version, "2000-12-29")
    at = pd.Timestamp("2005-12-30")
    prices = panel.fields["close"].loc[at].to_dict()
    bundle = freeze_bundle(
        root / "bundle",
        panel,
        [spec],
        at,
        prices=prices,
        instruments={
            n: {"symbol": f"S{n}", "currency": "USD", "sec_type": "STK", "exchange": "XSYN"}
            for n in prices
        },
        account={"positions": {}, "cash": 100000.0, "capital": 100000.0, "open_orders": []},
        execution_cost_bps=11,
    )
    market = {
        "session": "2006-01-02",
        "prices": prices,
        "commission_bps": 1,
        "execution_adjustment_bps": 10,
    }
    return bundle, market


def test_fresh_process_plan_matches_and_a_repeat_run_adds_no_submissions(
    archived, tmp_path, monkeypatch
):
    bundle, market = archived
    monkeypatch.chdir(tmp_path)  # The fresh process must use this code from any folder.
    first = rehearse_bundle(bundle, tmp_path, market)
    second = rehearse_bundle(bundle, tmp_path, market)
    plan = build_plan(bundle)
    assert first["complete"] and first["fresh_process_identical"] and first["archived_only"]
    assert first["real_broker_verified"] is False
    assert first == second and first["submissions"] == len(plan["orders"])
    assert read_plan(tmp_path / "fresh-process-plan.json")["document_hash"] == plan["document_hash"]
    assert json.loads((tmp_path / "reconciliation.json").read_text())["complete"]


def test_a_changed_market_or_session_is_refused(archived, tmp_path):
    bundle, market = archived
    rehearse_bundle(bundle, tmp_path, market)
    with pytest.raises(ValueError, match="changed"):
        rehearse_bundle(bundle, tmp_path, {**market, "commission_bps": 2})
    with pytest.raises(ValueError, match="session"):
        rehearse_bundle(bundle, tmp_path / "wrong", {**market, "session": "2006-01-03"})
    with pytest.raises(ValueError, match="unsupported metadata"):
        rehearse_bundle(bundle, tmp_path / "meta", {**market, "account": "x"})


def test_the_fresh_process_command_is_the_offline_plan_command(archived, tmp_path, monkeypatch):
    bundle, market = archived
    seen = []
    actual = subprocess.run

    def observe(command, **kwargs):
        seen.append(command)
        return actual(command, **kwargs)

    monkeypatch.setattr(rehearsal.subprocess, "run", observe)
    rehearse_bundle(bundle, tmp_path, market)
    assert len(seen) == 1
    assert seen[0][1:4] == ["-m", "capstone.pipeline.offline", "plan"]
    assert seen[0][5] == "--out" and seen[0][4] == str(bundle.resolve())


def test_a_fresh_process_that_plans_differently_stops_the_rehearsal_before_any_fill(
    archived, tmp_path, monkeypatch
):
    """Mutation: the offline command writes a plan whose canonical form differs."""
    bundle, market = archived
    actual = subprocess.run

    def tampered(command, **kwargs):
        done = actual(command, **kwargs)
        out = command[command.index("--out") + 1]
        plan = json.loads(Path(out).read_text(encoding="utf-8"))
        plan["execution_cost_bps"] += 1
        Path(out).write_text(json.dumps(identify_plan(plan)), encoding="utf-8")
        return done

    monkeypatch.setattr(rehearsal.subprocess, "run", tampered)
    with pytest.raises(AssertionError, match="fresh-process canonical plan differs"):
        rehearse_bundle(bundle, tmp_path, market)
    assert not (tmp_path / "execution.sqlite").exists()
    assert not (tmp_path / "simulated-broker.json").exists()


def test_a_corrupted_cached_proof_cannot_authorize_a_resumed_rehearsal(archived, tmp_path):
    bundle, market = archived
    rehearse_bundle(bundle, tmp_path, market)
    state = (tmp_path / "simulated-broker.json").read_bytes()
    fresh = tmp_path / "fresh-process-plan.json"
    data = json.loads(fresh.read_text())
    data["orders"] = []
    fresh.write_text(json.dumps(data))
    with pytest.raises((ValueError, AssertionError)):
        rehearse_bundle(bundle, tmp_path, market)
    assert (tmp_path / "simulated-broker.json").read_bytes() == state


def test_an_interrupted_plan_proof_is_required_again_before_simulated_submission(
    archived, tmp_path, monkeypatch
):
    bundle, market = archived
    actual = subprocess.run

    def interrupt(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="fixture interruption")

    monkeypatch.setattr(rehearsal.subprocess, "run", interrupt)
    with pytest.raises(subprocess.CalledProcessError):
        rehearse_bundle(bundle, tmp_path, market)
    assert (tmp_path / "order-plan.json").exists()
    assert not (tmp_path / "fresh-process-plan.json").exists()
    assert not (tmp_path / "execution.sqlite").exists()
    monkeypatch.setattr(rehearsal.subprocess, "run", actual)
    result = rehearse_bundle(bundle, tmp_path, market)
    assert (tmp_path / "fresh-process-plan.json").exists()
    assert result["complete"] and result["fresh_process_identical"]


def test_a_partial_fill_restart_does_not_duplicate_submissions(archived, tmp_path):
    bundle, market = archived
    first = rehearse_bundle(bundle, tmp_path, {**market, "fill_fraction": 0.5})
    assert not first["complete"]
    state = tmp_path / "simulated-broker.json"
    broker = SimulatedBroker.load(state)
    broker.fill(build_plan(bundle)["orders"][0]["instruction_id"])
    broker.fill_fraction = 1
    broker.save(state)
    second = rehearse_bundle(bundle, tmp_path, {**market, "fill_fraction": 0.5})
    assert second["complete"] and second["reconciled"]
    assert second["submissions"] == len(build_plan(bundle)["orders"])


def test_a_session_inside_the_sealed_holdout_is_refused_before_anything_is_written(
    archived, tmp_path, monkeypatch
):
    bundle, market = archived
    real = rehearsal.build_plan
    monkeypatch.setattr(
        rehearsal, "build_plan", lambda path: real(path) | {"execution_session": "2021-01-04"}
    )
    with pytest.raises(ValueError, match="sealed holdout"):
        rehearse_bundle(bundle, tmp_path, {**market, "session": "2021-01-04"})
    assert not (tmp_path / "order-plan.json").exists()
