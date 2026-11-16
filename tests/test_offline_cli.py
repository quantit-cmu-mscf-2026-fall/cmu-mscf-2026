"""The offline command: plan, simulate and prove on a synthetic bundle, with no broker."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from reference_fakes import reference_panel

from capstone.execution.bundle import freeze_bundle
from capstone.execution.frozen import build_plan, read_plan, write_plan
from capstone.pipeline import offline
from capstone.strategy.reference import ReferenceStrategySpec
from capstone.strategy.research import canonical, contracts

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2005-12-30"


def freeze(path, panel):
    spec = ReferenceStrategySpec("716bdaf7915c65be", panel.data_version, "2000-12-29")
    prices = panel.fields["close"].loc[AS_OF].to_dict()
    instruments = {
        n: {"symbol": f"S{n}", "currency": "USD", "sec_type": "STK", "exchange": "XSYN"}
        for n in prices
    }
    return freeze_bundle(
        path,
        panel,
        [spec],
        AS_OF,
        prices=prices,
        instruments=instruments,
        execution_cost_bps=11.0,
        account={"positions": {}, "open_orders": [], "cash": 100000.0, "capital": 100000.0},
    )


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return freeze(tmp_path_factory.mktemp("offline") / "bundle", reference_panel())


def market_for(plan):
    panel = reference_panel()
    return {
        "session": plan["execution_session"],
        "prices": panel.fields["open"].loc[plan["execution_session"]].to_dict(),
        "commission_bps": 1,
        "execution_adjustment_bps": 10,
    }


def test_plan_command_writes_the_plan_the_library_builds(bundle, tmp_path):
    offline.main(["plan", str(bundle), "--out", str(tmp_path / "plan.json")])
    plan = build_plan(bundle)
    assert canonical(read_plan(tmp_path / "plan.json")) == canonical(plan)
    assert plan["orders"] and not plan["transmittable"]


def test_plan_in_a_fresh_process_has_the_same_document_hash(bundle, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = tmp_path / "fresh.json"
    subprocess.run(
        [sys.executable, "-m", "capstone.pipeline.offline", "plan", str(bundle), "--out", str(out)],
        check=True,
        capture_output=True,
        env=env,
        cwd=ROOT,
    )
    assert read_plan(out)["document_hash"] == build_plan(bundle)["document_hash"]


def test_simulation_fills_the_fixed_share_targets_and_is_synthetic_only(bundle, tmp_path):
    plan = build_plan(bundle)
    write_plan(plan, tmp_path / "plan.json")
    result = offline.simulate(
        bundle, tmp_path / "plan.json", tmp_path / "sim", market=market_for(plan)
    )
    assert result["complete"] and result["synthetic_only"]
    expected = {int(n): q for n, q in plan["projected_shares"].items() if q}
    assert result["positions"] == expected
    assert json.loads((tmp_path / "sim" / "reconciliation.json").read_text())["complete"]
    assert (tmp_path / "sim" / "simulated-broker.json").exists()


def test_a_fresh_simulation_needs_an_explicit_market_for_the_planned_session(bundle, tmp_path):
    plan = build_plan(bundle)
    write_plan(plan, tmp_path / "plan.json")
    with pytest.raises(ValueError, match="explicit synthetic next-session market"):
        offline.simulate(bundle, tmp_path / "plan.json", tmp_path / "a")
    wrong = market_for(plan) | {"session": "2006-01-03"}
    with pytest.raises(ValueError, match="different session"):
        offline.simulate(bundle, tmp_path / "plan.json", tmp_path / "b", market=wrong)


def test_a_bundle_that_is_not_synthetic_is_refused_and_nothing_is_attached(tmp_path):
    panel = reference_panel()
    current = freeze(tmp_path / "bundle", replace(panel, data_version="vendor-current-fixture"))
    plan = build_plan(current)
    write_plan(plan, tmp_path / "plan.json")
    with pytest.raises(ValueError, match="explicitly synthetic input bundle"):
        offline.simulate(current, tmp_path / "plan.json", tmp_path / "sim", market=market_for(plan))
    assert not (tmp_path / "sim" / "execution.sqlite").exists()


def test_a_plan_from_another_bundle_is_refused(bundle, tmp_path):
    other = freeze(tmp_path / "other", replace(reference_panel(), data_version="synthetic-other"))
    write_plan(build_plan(other), tmp_path / "plan.json")
    with pytest.raises(ValueError, match="this plan's explicitly synthetic input bundle"):
        offline.simulate(bundle, tmp_path / "plan.json", tmp_path / "sim", market={})


def test_audit_prints_the_claim_catalog(capsys, tmp_path):
    offline.main(["audit"])
    assert capsys.readouterr().out.strip() == canonical(contracts())
    offline.main(["audit", "--out", str(tmp_path / "audit.json")])
    assert json.loads((tmp_path / "audit.json").read_text())["claims"]


def test_prove_refuses_a_directory_that_is_not_new(tmp_path):
    (tmp_path / "leftover").write_text("x")
    with pytest.raises(FileExistsError):
        offline.prove(tmp_path)
