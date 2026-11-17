import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone.pipeline import allocation_diagnosis as diagnosis
from capstone.pipeline.allocation_diagnosis import MODES, make_policy, run, write_replay
from capstone.pipeline.reference_allocation import replay
from capstone.pipeline.reference_inputs import POOL
from capstone.strategy.robust_allocation import AllocationPolicy, AllocationPolicyV2

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "config/allocation_diagnosis.json").read_text())
COSTS = json.loads((ROOT / "config/reference_allocation.json").read_text())["cost_scenarios"]
# One scenario keeps the end-to-end run fast; the report is covered by its own tests.
BASE = {"assumed_base": COSTS["assumed_base"]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel():
    index = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    x = np.random.default_rng(404).normal(0.004, 0.025, (len(index), len(POOL)))
    return pd.DataFrame(x, index=index, columns=POOL).assign(RF=0.001, **{"Mkt-RF": 0.004})


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """A synthetic stand-in for the preserved original study: inputs, replays, inventory."""
    root = tmp_path_factory.mktemp("baseline")
    (root / "inputs").mkdir()
    source = root / "inputs/monthly.csv"
    frame = panel()
    frame.to_csv(source, index_label="month", float_format="%.17g", lineterminator="\n")
    for scenario, costs in BASE.items():
        for mode in MODES:
            history, decisions = replay(frame, mode, make_policy(costs))
            write_replay(root, f"study-1/{scenario}/{mode}", history, decisions)
    (root / "study-1/manifest.json").write_text(
        json.dumps({"git_commit": "a" * 40, "registration": {"cost_scenarios": BASE}})
    )
    files = [p for p in (root / "study-1").rglob("*") if p.is_file()] + [source]
    inventory = {p.relative_to(root).as_posix(): sha(p) for p in files}
    (root / "evidence-sha256.json").write_text(json.dumps(inventory))
    config = {**PROTOCOL, "baseline_commit": "a" * 40, "baseline_input_sha256": sha(source)}
    registration = root / "registration.json"
    registration.write_text(json.dumps(config))
    return root, registration


@pytest.fixture
def protocol(baseline, monkeypatch):
    _, registration = baseline
    monkeypatch.setattr(diagnosis, "PROTOCOL_PATH", registration)


def test_policy_versions_are_explicit_and_opt_in():
    costs = COSTS["assumed_base"]
    legacy, corrected = make_policy(costs), make_policy(costs, corrected=True)
    assert type(legacy) is AllocationPolicy and isinstance(corrected, AllocationPolicyV2)
    assert (legacy.switch_bps, legacy.ongoing_bps) == (6, 10) and legacy.lookback == 120
    assert make_policy(costs, lookback=300).lookback == 300


def test_run_rejects_an_unregistered_baseline_or_changed_evidence(baseline, protocol, tmp_path):
    root, registration = baseline
    altered = json.loads(registration.read_text())
    altered["baseline_commit"] = "b" * 40
    wrong = tmp_path / "registration.json"
    wrong.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="differs from committed protocol"):
        run(root, wrong, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_rejects_a_tampered_original_study(baseline, protocol, tmp_path):
    root, registration = baseline
    copy = tmp_path / "baseline"
    copy.mkdir()
    for path in root.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    victim = copy / "study-1/assumed_base/equal_weight/decisions.json"
    victim.write_text(victim.read_text() + " ")
    with pytest.raises(ValueError, match="original evidence changed"):
        run(copy, registration, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_reproduces_the_original_and_writes_every_diagnostic(
    baseline, protocol, tmp_path, monkeypatch
):
    root, registration = baseline
    rendered = []
    monkeypatch.setattr("capstone.pipeline.allocation_diagnosis_report.report", rendered.append)
    out = tmp_path / "out"
    run(root, registration, out)
    results = json.loads((out / "results.json").read_text())
    assert len(results["legacy_equivalence"]) == 3 and all(results["legacy_equivalence"].values())
    assert len(results["corrected_v2"]) == 3
    assert set(results["frozen_target_costs"]) == set(MODES)
    assert set(results["prior_risk_overlay"]) == set(MODES)
    assert set(results["hindsight_fixed_mix"]) == set(MODES[1:])
    assert all(v["uses_future_information"] for v in results["hindsight_fixed_mix"].values())
    assert len(results["diagnostics"]) == 6
    assert results["qualification"]["formal_admission"] is False
    assert results["qualification"]["optimizer_winner_certified"] is False
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["baseline_files_verified"] > 0
    assert "config/allocation_diagnosis.json" in manifest["code"]
    assert rendered == [out] and (out / "comparisons.jsonl").is_file()
    # 3 legacy + 3 V2 + 3 frozen-target + 3 overlay + 2 hindsight replays, in the study ledger.
    ledger = (out / "ledger/runs.jsonl").read_text().splitlines()
    assert len(ledger) == 14
    assert all("not-admission" in json.loads(line)["tags"] for line in ledger)
    # The preserved original study is read, never rewritten.
    assert sha(root / "evidence-sha256.json") == manifest["baseline_inventory_sha256"]
