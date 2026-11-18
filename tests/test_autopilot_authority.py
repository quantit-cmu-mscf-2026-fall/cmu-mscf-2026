"""Authorization policy for current data: refusals, reuse and write-ahead creation.

Synthetic vendor, temporary store, no data and no feed. The research holdout is never
opened: windows touching 2021-2025 and an unconfigured store are refused.
"""

import json
from pathlib import Path

import pytest

from capstone import current_authority
from capstone.pipeline.autopilot import authority

SIGNER = "autopilot:test"
POLICY = {"vendor": "synthetic-vendor", "fields": ["close"], "start": "2026-01-02"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(current_authority.ROOT_ENV, str(tmp_path / "authority-store"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")  # no real holdout record
    return tmp_path / "run" / "current-authorization.json"


def test_nothing_is_authorized_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(current_authority.ROOT_ENV, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    target = tmp_path / "run" / "current-authorization.json"
    with pytest.raises(authority.PolicyRefused, match="no current feed is configured"):
        authority.ensure(target, POLICY, "2026-01-05", SIGNER)
    assert not target.exists() and not (tmp_path / "home").exists()
    assert (
        authority.main(
            ["--vendor", "v", "--field", "close", "--start", "2026-03-02"]
            + ["--authorized-by", "op"]
        )
        == 2
    )


def test_policy_refusals(store):
    with pytest.raises(
        authority.PolicyRefused, match="is empty; session 2026-01-05 is not authorized"
    ):
        authority.ensure(store, {}, "2026-01-05", SIGNER)
    with pytest.raises(authority.PolicyRefused, match="sealed research holdout"):
        authority.ensure(store, POLICY | {"start": "2025-06-02"}, "2026-01-05", SIGNER)
    with pytest.raises(authority.PolicyRefused, match="unknown keys"):
        authority.ensure(store, POLICY | {"extra": 1}, "2026-01-05", SIGNER)
    with pytest.raises(authority.PolicyRefused, match="fields"):
        authority.ensure(store, POLICY | {"fields": []}, "2026-01-05", SIGNER)
    with pytest.raises(authority.PolicyRefused, match="vendor is required"):
        authority.ensure(store, {"fields": ["close"]}, "2026-01-05", SIGNER)
    with pytest.raises(authority.PolicyRefused, match="outside the policy window"):
        authority.ensure(store, POLICY | {"end": "2026-01-31"}, "2026-02-02", SIGNER)
    assert not store.exists()


def test_a_record_is_created_once_reused_and_never_replaced(store):
    record, created = authority.ensure(store, POLICY, "2026-01-05", SIGNER)
    assert created and authority.ensure(store, POLICY, "2026-02-02", SIGNER) == (record, False)
    with pytest.raises(authority.PolicyRefused, match="changed"):
        authority.ensure(store, POLICY | {"fields": ["open"]}, "2026-02-02", SIGNER)
    current_authority.revoke_current(
        record, revoked_by="synthetic operator", reason="test", effective="2026-01-20"
    )
    with pytest.raises(authority.PolicyRefused, match="revoked"):
        authority.ensure(store, POLICY, "2026-02-02", SIGNER)
    assert json.loads(store.read_text(encoding="utf-8"))["record_id"] == record


def test_a_killed_attempt_is_adopted_not_duplicated(store, monkeypatch):
    def killed(point):
        raise RuntimeError(point)

    real = authority.fault
    monkeypatch.setattr(authority, "fault", killed)
    with pytest.raises(RuntimeError, match="autopilot:daily:authorized"):
        authority.ensure(store, POLICY, "2026-01-05", SIGNER)
    assert not store.exists()  # the intent is recorded, the run's store file is not
    folder = current_authority.store_root() / "authorizations"
    (written,) = list(folder.glob("*.json"))
    monkeypatch.setattr(authority, "fault", real)
    record, created = authority.ensure(store, POLICY, "2026-01-05", SIGNER)
    assert record == written.stem and created
    assert len(list(folder.glob("*.json"))) == 1


def test_a_revoked_record_from_a_killed_attempt_is_not_adopted(store, monkeypatch):
    def killed(point):
        raise RuntimeError(point)

    real = authority.fault
    monkeypatch.setattr(authority, "fault", killed)
    with pytest.raises(RuntimeError, match="autopilot:daily:authorized"):
        authority.ensure(store, POLICY, "2026-01-05", SIGNER)  # interrupt
    folder = current_authority.store_root() / "authorizations"
    (written,) = list(folder.glob("*.json"))
    current_authority.revoke_current(
        written.stem, revoked_by="synthetic operator", reason="test", effective="2026-01-05"
    )
    monkeypatch.setattr(authority, "fault", real)
    record, created = authority.ensure(store, POLICY, "2026-01-05", SIGNER)  # resume
    assert created and record != written.stem
    assert not current_authority.revocations(record)
    assert json.loads(store.read_text(encoding="utf-8"))["record_id"] == record


def test_a_policy_changed_after_a_killed_attempt_is_refused(store, monkeypatch):
    def killed(point):
        raise RuntimeError(point)

    real = authority.fault
    monkeypatch.setattr(authority, "fault", killed)
    with pytest.raises(RuntimeError):
        authority.ensure(store, POLICY, "2026-01-05", SIGNER)
    monkeypatch.setattr(authority, "fault", real)
    with pytest.raises(authority.PolicyRefused, match="a person reviews the store"):
        authority.ensure(store, POLICY | {"fields": ["open"]}, "2026-01-05", SIGNER)


def test_cli_creates_a_record_and_refuses_the_holdout(store, capsys):
    args = ["--vendor", "synthetic-vendor", "--field", "close", "--authorized-by", "op"]
    assert authority.main([*args, "--start", "2026-03-02"]) == 0
    record = json.loads(capsys.readouterr().out)["record_id"]
    assert current_authority.authorization(record)["authorized_by"] == "op"
    assert authority.main([*args, "--start", "2024-01-02", "--end", "2024-12-31"]) == 2
    error = capsys.readouterr().err.strip()
    assert error.startswith("authority:") and "holdout" in error and "\n" not in error
