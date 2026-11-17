from types import SimpleNamespace

import pandas as pd
import pytest

from capstone.execution.contracts import ExecutionContractBlocked
from capstone.pipeline import benchmark_controls as bc
from capstone.pipeline.benchmark_controls import (
    CONTROLS,
    LATEST,
    POLICIES,
    benchmark_preflight,
    evaluate_controls,
    freeze_controls,
    verify_controls,
)


def metadata():
    days = pd.bdate_range("2019-01-01", "2019-12-31")
    return {"calendar": [str(day.date()) for day in days], "data_version": "synthetic-v1"}


def test_freeze_registers_every_control_once_without_selection():
    frozen = freeze_controls(metadata(), "2019-03-01", "2019-06-28")
    assert frozen["policy"] == LATEST
    assert [(row["kind"], row["profile"]) for row in frozen["controls"]] == list(CONTROLS)
    assert len({row["id"] for row in frozen["controls"]}) == len(CONTROLS)
    assert frozen["comparison_selection"] == "all_prespecified; no winner selection"
    for row in frozen["controls"]:
        assert row["registration_blockers"] == []
        assert row["formal_admission"] is False and row["source_model_slot"] is False
    assert freeze_controls(metadata(), "2019-03-01", "2019-06-28") == frozen


def test_freeze_without_prior_session_blocks_every_control():
    frozen = freeze_controls(metadata(), "2019-01-01", "2019-02-01")
    for row in frozen["controls"]:
        assert row["specification"] is None and row["execution_contract"] is None
        assert row["registration_blockers"]


@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_every_registered_policy_still_verifies(policy):
    meta = metadata()
    try:
        frozen = freeze_controls(meta, "2019-03-01", "2019-06-28", policy=policy)
    except ExecutionContractBlocked as blocked:  # prefix route not in this build
        pytest.skip(str(blocked))
    manifest = {"start": "2019-03-01", "end": "2019-06-28", "benchmark_controls": frozen}
    verify_controls(manifest, meta)


def test_verify_rejects_a_changed_registration():
    meta = metadata()
    controls = freeze_controls(meta, "2019-03-01", "2019-06-28")
    manifest = {"start": "2019-03-01", "end": "2019-06-28", "benchmark_controls": controls}
    controls["controls"][0]["formal_admission"] = True
    with pytest.raises(ValueError, match="changed or is incompatible"):
        verify_controls(manifest, meta)
    manifest["benchmark_controls"] = freeze_controls(meta, "2019-03-01", "2019-06-28")
    manifest["start"] = "2019-04-01"
    with pytest.raises(ValueError):
        verify_controls(manifest, meta)


def inputs(*, qualified=True):
    qualification = {
        "global_blockers": [],
        "source_hashes": {"daily": "abc"},
        "sources": {role: {"qualified": qualified} for role in ("calendar", "events")},
        "event_return_identities": {"verified": qualified},
    }
    return SimpleNamespace(
        qualification=qualification, panel=SimpleNamespace(fields={}, availability={})
    )


def test_preflight_names_each_missing_input():
    cap = SimpleNamespace(kind="cap_weight_universe_proxy")
    result = benchmark_preflight(cap, inputs(), "p")
    assert not result["source_preflight_passed"]
    assert result["blockers"] == ["availability_missing:cap", "feature_missing:cap"]
    cash = SimpleNamespace(kind="cash")
    assert benchmark_preflight(cash, inputs(), "p")["source_preflight_passed"]
    bad = benchmark_preflight(cash, inputs(qualified=False), "p")
    assert "calendar_source_unqualified" in bad["blockers"]
    assert "corporate_action_return_identity_unverified" in bad["blockers"]


def manifest_for(kinds):
    frozen = freeze_controls(metadata(), "2019-03-01", "2019-06-28")
    frozen["controls"] = [row for row in frozen["controls"] if row["kind"] in kinds]
    return {"scenarios": {"base": 1, "doubled_cost": 2}, "benchmark_controls": frozen}


def test_blocked_controls_never_run_a_scenario(monkeypatch):
    monkeypatch.setattr(bc, "benchmark_due", lambda *args: True)

    def runner(*args, **kwargs):
        raise AssertionError("scenario must not run for a blocked control")

    rows = evaluate_controls(None, inputs(qualified=False), manifest_for({"cash"}), runner)
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "data_or_initialization_blocked"
        assert set(row["scenarios"]) == {"base", "doubled_cost"}
        assert not row["scenarios"]["base"]["completed"]


def test_ready_controls_run_every_scenario_and_report_status(monkeypatch):
    monkeypatch.setattr(bc, "benchmark_due", lambda *args: True)
    manifest = manifest_for({"cash"})
    seen = []

    def runner(root, spec, data, manifest_, name, multiplier, *, execution_version):
        seen.append((name, multiplier, execution_version))
        return {"completed": True, "status": "ok"}, None

    rows = evaluate_controls(None, inputs(), manifest, runner)
    assert [row["status"] for row in rows] == ["evaluated", "evaluated"]
    version = manifest["benchmark_controls"]["execution_version"]
    assert seen.count(("doubled_cost", 2, version)) == 2


def test_unscheduled_formation_blocks_the_control(monkeypatch):
    monkeypatch.setattr(bc, "benchmark_due", lambda *args: False)
    rows = evaluate_controls(None, inputs(), manifest_for({"cash"}), lambda *a, **k: None)
    assert all(
        "benchmark_prior_session_is_not_scheduled_formation" in row["blockers"] for row in rows
    )
