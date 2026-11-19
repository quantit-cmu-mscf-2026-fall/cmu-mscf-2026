"""Temporary synthetic datasets only; the real authority and market data stay sealed.

Nothing here authorizes or opens the real dataset: every protocol names a `synthetic:`
dataset and a temporary store, and the run ledger is a temporary directory.
"""

import inspect
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from holdout_fakes import FakeConfig, approve, looks, make_protocol, open_look

from capstone import holdout_authority as authority
from capstone.artifacts import canonical, digest


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


def test_freeze_is_reviewable_readonly_and_does_not_authorize(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    assert authority.status(protocol, synthetic_root=root)["state"] == "SEALED"
    assert not root.exists()
    for output in (tmp_path / "run-A/final.json", tmp_path / "run-B/final.json"):
        authority.save_protocol(protocol, output)
        before = output.read_bytes()
        authority.save_protocol(protocol, output)
        assert output.read_bytes() == before
        assert json.loads(before)["authorized"] is False
    assert not root.exists()
    with pytest.raises(authority.HoldoutSealed):
        with open_look(protocol, root):
            pytest.fail("a prepared protocol grants no access")


def test_one_terminal_look_across_outputs_ledgers_and_restarts(tmp_path, monkeypatch):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    assert approve(protocol, root) == protocol.id
    for name in ("run-A", "run-B"):
        authority.save_protocol(protocol, tmp_path / name / "protocol.json")
    with open_look(protocol, root):
        pass
    assert authority.status(protocol, synthetic_root=root)["state"] == "SPENT"
    for ledger in ("new-ledger", "another-output"):
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / ledger))
        with pytest.raises(authority.HoldoutSpent):
            with open_look(protocol, root):
                pytest.fail("a new ledger/output cannot renew access")
    resumed = authority.FinalHoldoutProtocol(protocol.payload_json)
    with pytest.raises(authority.HoldoutSpent):
        approve(resumed, root)
    changed = make_protocol(FakeConfig(config_hash=digest("other config")))
    with pytest.raises(authority.HoldoutSealed, match="different frozen"):
        approve(changed, root)


def test_concurrent_processes_in_different_directories_have_one_look(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    saved = tmp_path / "protocol.json"
    authority.save_protocol(protocol, saved)
    # Each child prints its outcome then leaves with os._exit: a native extension's
    # teardown at interpreter exit has aborted a child that already printed OPENED in CI.
    script = """import json,os,sys
from capstone.artifacts import canonical
from capstone import holdout_authority as a
p=a.FinalHoldoutProtocol(canonical(json.load(open(sys.argv[1]))['protocol']))
try:
    with a.open_once(p,candidates=['a'*16],start='2021-01-04',
                     end='2025-12-31',synthetic_root=sys.argv[2]):
        pass
    outcome='OPENED'
except RuntimeError:
    outcome='REFUSED'
print(outcome,flush=True)
sys.stderr.flush()
os._exit(0)
"""
    env = dict(
        os.environ,
        CAPSTONE_LEDGER_DIR=str(tmp_path / "ledger"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(authority.__file__).resolve().parents[1]),
    )
    processes = []
    for i in range(4):
        folder = tmp_path / f"process-{i}"
        folder.mkdir()
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", script, str(saved), str(root)],
                cwd=folder,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    results = [(process.communicate(timeout=60), process.returncode) for process in processes]
    assert all(code == 0 for _, code in results), results
    assert sorted(output.strip() for (output, _), _ in results) == [
        "OPENED",
        "REFUSED",
        "REFUSED",
        "REFUSED",
    ]
    assert authority.status(protocol, synthetic_root=root)["state"] == "SPENT"
    assert len(looks()) == 1


def test_failed_or_uncertain_materialization_is_terminal(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    with pytest.raises(OSError, match="synthetic crash"):
        with open_look(protocol, root):
            raise OSError("synthetic crash after durable reservation")
    with pytest.raises(authority.HoldoutSpent):
        with open_look(protocol, root):
            pytest.fail("an uncertain look cannot be retried")


@pytest.mark.parametrize(
    "change",
    [
        "config_hash",
        "generator_id",
        "procedure_id",
        "source_version",
        "period",
        "code_id",
        "runtime_id",
    ],
)
def test_identity_changes_cannot_rebind_authorized_dataset(tmp_path, change):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    row = protocol.payload
    row[change] = (
        ["2022-01-01", "2025-12-31"]
        if change == "period"
        else (
            "other-version"
            if change == "source_version"
            else "synthetic:other"
            if change == "generator_id"
            else "0" * 64
        )
    )
    row["calibration"] = None
    changed = authority.FinalHoldoutProtocol(canonical(row))
    with pytest.raises(authority.HoldoutSealed):
        authority.status(changed, synthetic_root=root)


@pytest.mark.parametrize("changed", ["code_identity", "runtime_identity"])
def test_stale_implementation_or_runtime_cannot_open(tmp_path, monkeypatch, changed):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    monkeypatch.setattr(authority, changed, lambda: "0" * 64)
    with pytest.raises(authority.HoldoutSealed, match="stale"):
        with open_look(protocol, root):
            pytest.fail("stale authorization cannot open")


def test_code_identity_hashes_the_capstone_package_not_the_repository(tmp_path, monkeypatch):
    before = authority.code_identity()
    assert len(before) == 64
    # A file outside the package (tests, scripts, a scratch file) does not change it.
    outside = Path(authority.__file__).resolve().parents[1] / "scratch_identity_probe.py"
    outside.write_text("x = 1\n")
    try:
        assert authority.code_identity() == before
    finally:
        outside.unlink()


def test_authorization_and_terminal_receipt_tampering_is_refused(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    stored = next((root / "synthetic").glob("*.json"))
    row = json.loads(stored.read_text())
    row["state"] = "SPENT"
    stored.write_text(canonical(row) + "\n")
    with pytest.raises(authority.HoldoutSealed, match="changed"):
        authority.status(protocol, synthetic_root=root)


def test_frozen_protocol_cannot_inherit_old_generator_or_allocation_calibration():
    protocol = make_protocol()
    for name in ("generator_id", "procedure_id", "config_hash"):
        row = protocol.payload
        row["calibration"][name] = "old-generator" if name == "generator_id" else "0" * 64
        with pytest.raises(ValueError, match="calibration belongs"):
            authority.FinalHoldoutProtocol(canonical(row))
    pending = make_protocol(calibrated=False)
    with pytest.raises(authority.HoldoutSealed, match="calibration is pending"):
        approve(pending, Path("unused-synthetic-root"))
    row = protocol.payload
    row["generator_id"] = "synthetic:new-generator"
    row["calibration"]["generator_id"] = row["generator_id"]
    with pytest.raises(ValueError, match="evidence identity"):
        authority.FinalHoldoutProtocol(canonical(row))
    row["calibration"]["evidence"]["generator_id"] = row["generator_id"]
    with pytest.raises(ValueError, match="evidence identity"):
        authority.FinalHoldoutProtocol(canonical(row))


def test_real_protocol_period_cannot_be_shifted_or_subdivided():
    row = make_protocol(dataset=authority.REAL_DATASET).payload
    assert row["period"] == ["2021-01-01", "2025-12-31"]
    row["period"] = ["2022-01-01", "2025-12-31"]
    row["calibration"] = None
    with pytest.raises(ValueError, match="cannot be shifted"):
        authority.FinalHoldoutProtocol(canonical(row))


def test_real_dataset_refuses_a_caller_selected_store(tmp_path):
    protocol = make_protocol(dataset=authority.REAL_DATASET)
    with pytest.raises(authority.HoldoutSealed, match="caller-selected"):
        authority.status(protocol, synthetic_root=tmp_path)
    with pytest.raises(authority.HoldoutSealed, match="caller-selected"):
        approve(protocol, tmp_path)
    assert not list(tmp_path.iterdir())


def test_second_configuration_cannot_rebind_an_unspent_authorization(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    with pytest.raises(authority.HoldoutSealed, match="different frozen"):
        approve(make_protocol(FakeConfig(config_hash=digest("config 20"))), root)
    assert authority.status(protocol, synthetic_root=root)["state"] == "AUTHORIZED"


def test_reader_scope_is_checked_before_the_look_is_spent(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    for changes in ({"candidates": ["b" * 16]}, {"start": "2020-12-31"}, {"end": "2026-01-01"}):
        with pytest.raises(authority.HoldoutSealed):
            with open_look(protocol, root, **changes):
                pytest.fail("out-of-scope reader")
    assert authority.status(protocol, synthetic_root=root)["state"] == "AUTHORIZED"
    assert looks() == []


def test_returned_capability_cannot_replace_scope_or_clear_consumption(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    with open_look(protocol, root, end="2021-01-04") as access:
        with pytest.raises(FrozenInstanceError):
            access.receipt_id = "x"
        with pytest.raises(authority.HoldoutSealed, match="frozen dates"):
            authority.validate_access(access, end="2025-12-31")
        authority.validate_access(access, end="2021-01-04")
        assert authority.read_source(access, "SYNTHETIC_DAILY", "synthetic:v1") == "1" * 64
        with pytest.raises(authority.HoldoutSealed, match="already materialized"):
            authority.read_source(access, "synthetic_daily", "synthetic:v1")
        with pytest.raises(authority.HoldoutSealed, match="version changed"):
            authority.validate_access(access, source_version="other")
        with pytest.raises(authority.HoldoutSealed, match="foreign authority"):
            authority.validate_access(access, dataset=authority.REAL_DATASET)
        with pytest.raises(authority.HoldoutSealed, match="active durable"):
            authority.validate_access(replace(access))
    with pytest.raises(authority.HoldoutSealed, match="active durable"):
        authority.validate_access(access)


# The authorizer and the ledger record. The default (team decision pending): a named
# team member, the gate owner, authorizes; the look is recorded in the run ledger.


def test_authorizer_is_named_by_the_caller_and_recorded(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root, approved_by="A. Gate Owner", cosigned_by="B. Quantit Mentor")
    stored = json.loads(next((root / "synthetic").glob("*.json")).read_text())
    assert stored["authorization"]["approved_by"] == "A. Gate Owner"
    assert stored["authorization"]["cosigned_by"] == "B. Quantit Mentor"
    # The same people approving again is idempotent; a different approver is not.
    approve(protocol, root, approved_by="A. Gate Owner", cosigned_by="B. Quantit Mentor")
    with pytest.raises(authority.HoldoutSealed, match="immutable authorization"):
        approve(protocol, root, approved_by="C. Someone Else")


def test_cosign_is_optional(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    stored = json.loads(next((root / "synthetic").glob("*.json")).read_text())
    assert stored["authorization"]["cosigned_by"] is None


@pytest.mark.parametrize("name", ["", "  ", "unknown", "TBD", "OPEN MANAGER DECISION", None, 7])
def test_a_placeholder_or_missing_authorizer_is_refused(tmp_path, name):
    protocol, root = make_protocol(), tmp_path / "authority"
    with pytest.raises(authority.HoldoutSealed, match="named authorizer"):
        approve(protocol, root, approved_by=name)
    with pytest.raises(authority.HoldoutSealed, match="named authorizer"):
        approve(protocol, root, cosigned_by="TBD")
    assert not root.exists()


def test_authorize_has_no_default_authorizer():
    parameter = inspect.signature(authority.authorize).parameters["approved_by"]
    assert parameter.default is inspect.Parameter.empty
    assert "OPEN MANAGER DECISION".casefold() in authority._PLACEHOLDERS


def test_a_look_is_recorded_in_the_ledger_before_the_reader_runs(tmp_path):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root, approved_by="A. Gate Owner")
    with open_look(protocol, root) as access:
        # Already in the ledger while the reader is still running.
        (entry,) = looks()
        assert entry["params"]["receipt_id"] == access.receipt_id
    assert entry["params"]["approved_by"] == "A. Gate Owner"
    assert entry["params"]["dates"] == ["2021-01-04", "2025-12-31"]
    assert entry["params"]["data_version"] == "synthetic:v1"
    assert entry["metrics"] == {}
    assert "synthetic" in entry["tags"] and authority.LOOK_TAG in entry["tags"]
    with pytest.raises(authority.HoldoutSpent):
        with open_look(protocol, root):
            pytest.fail("spent")
    assert len(looks()) == 1


def test_a_spent_look_with_a_lost_ledger_entry_is_recovered_once(tmp_path, monkeypatch):
    protocol, root = make_protocol(), tmp_path / "authority"
    approve(protocol, root)
    assert authority.ensure_logged(protocol, synthetic_root=root) is False  # not spent
    real = authority.runlog.log_run

    def broken(*args, **kwargs):
        raise OSError("ledger disk full")

    monkeypatch.setattr(authority.runlog, "log_run", broken)
    with pytest.raises(OSError, match="ledger disk full"):
        with open_look(protocol, root):
            pytest.fail("no access is yielded when the record cannot be written")
    monkeypatch.setattr(authority.runlog, "log_run", real)
    assert authority.status(protocol, synthetic_root=root)["state"] == "SPENT"
    assert looks() == []
    assert authority.ensure_logged(protocol, synthetic_root=root) is True
    assert authority.ensure_logged(protocol, synthetic_root=root) is False
    assert len(looks()) == 1
    with pytest.raises(authority.HoldoutSpent):
        with open_look(protocol, root):
            pytest.fail("recovery does not renew the look")


def _home(monkeypatch, path):
    path.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("USERPROFILE", str(path))


def test_a_second_home_cannot_take_a_second_real_look_on_the_same_ledger(tmp_path, monkeypatch):
    protocol = make_protocol(dataset=authority.REAL_DATASET)
    _home(monkeypatch, tmp_path / "h1")
    approve(protocol, None)
    with open_look(protocol, None):
        pass
    assert authority.status(protocol)["state"] == "SPENT"
    _home(monkeypatch, tmp_path / "h2")
    assert authority.status(protocol)["state"] == "SEALED"  # the new HOME knows nothing
    with pytest.raises(authority.HoldoutSpent):
        approve(protocol, None)
    assert authority.status(protocol)["state"] == "SEALED"  # authorize left no state behind
    assert len(looks()) == 1


def test_open_once_also_refuses_when_the_ledger_holds_a_real_look(tmp_path, monkeypatch):
    protocol = make_protocol(dataset=authority.REAL_DATASET)
    _home(monkeypatch, tmp_path / "h2")
    approve(protocol, None)  # authorized on this HOME while the ledger is still empty
    authority._log_look(
        protocol,
        {"approved_by": "another machine", "cosigned_by": None},
        {"candidates": ["a" * 16], "dates": ["2021-01-04", "2025-12-31"], "id": "r" * 64},
    )
    with pytest.raises(authority.HoldoutSpent):
        with open_look(protocol, None):
            pytest.fail("a second real look was opened")
    assert authority.status(protocol)["state"] == "AUTHORIZED"
    assert len(looks()) == 1


def test_a_synthetic_look_in_the_ledger_does_not_spend_the_real_dataset(tmp_path, monkeypatch):
    synthetic = make_protocol()
    approve(synthetic, tmp_path / "authority")
    with open_look(synthetic, tmp_path / "authority"):
        pass
    _home(monkeypatch, tmp_path / "h1")
    real = make_protocol(dataset=authority.REAL_DATASET)
    approve(real, None)
    with open_look(real, None):
        pass
    assert len(looks()) == 2
