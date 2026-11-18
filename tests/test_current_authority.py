"""Revocable authorizations for a current feed: synthetic vendor, temporary store, no data."""

import json
from pathlib import Path

import pytest

from capstone import current_authority
from capstone.artifacts import digest
from capstone.holdout_authority import HoldoutSealed


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(current_authority.ROOT_ENV, str(tmp_path / "current-authority"))
    # The real holdout record is looked up under the home directory: keep it synthetic.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return tmp_path / "current-authority"


def authorize(**changes):
    arguments = dict(
        start="2026-01-05", end=None, fields=["close", "volume"], authorized_by="A. Signer"
    )
    return current_authority.authorize_current("synthetic-vendor", **arguments | changes)


def test_nothing_is_configured_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(current_authority.ROOT_ENV, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    with pytest.raises(HoldoutSealed, match="no current feed is configured"):
        authorize()
    with pytest.raises(HoldoutSealed, match="no current feed is configured"):
        current_authority.authorization("0" * 64)
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("name", ["", "  ", "TBD", "unknown", None])
def test_a_named_signer_is_required(store, name):
    with pytest.raises(HoldoutSealed, match="names who authorized"):
        authorize(authorized_by=name)
    assert not store.exists()


@pytest.mark.parametrize("vendor", ["", "Vendor", "a b", "../x", "x" * 41, None])
def test_vendors_are_plain_lowercase_names(store, vendor):
    with pytest.raises(HoldoutSealed, match="lowercase"):
        current_authority.current_dataset(vendor)
    assert current_authority.current_dataset("synthetic-vendor") == "current:synthetic-vendor"


def test_the_authorization_is_immutable_and_declares_window_and_fields(store):
    record = authorize()
    row = current_authority.authorization(record)
    assert row["fields"] == ["close", "volume"]
    assert (row["start"], row["end"], row["authorized_by"]) == ("2026-01-05", None, "A. Signer")
    path = store / "authorizations" / f"{record}.json"
    path.write_text(json.dumps(row | {"authorized_by": "someone else"}), "utf-8")
    with pytest.raises(HoldoutSealed, match="record changed"):
        current_authority.authorization(record)
    with pytest.raises(HoldoutSealed, match="record changed"):
        authorize()  # an existing record is verified before it is reused


def test_reversed_windows_and_empty_fields_are_refused(store):
    with pytest.raises(HoldoutSealed, match="reversed"):
        authorize(start="2026-03-01", end="2026-02-01")
    for fields in ([], [""], [3]):
        with pytest.raises(HoldoutSealed, match="declares its fields"):
            authorize(fields=fields)


def test_the_id_leaves_the_recorded_time_out(store, monkeypatch):
    times = iter(["2026-01-05T21:00:00+00:00", "2027-02-03T04:05:06+00:00"])
    monkeypatch.setattr(current_authority, "_now", lambda: next(times))
    first = authorize()
    row = current_authority.authorization(first)
    assert (row["schema_version"], row["issue"], row["at"]) == (2, 1, "2026-01-05T21:00:00+00:00")
    assert first == digest({k: v for k, v in row.items() if k not in ("record_id", "at")})
    assert authorize() == first  # the same unrevoked record, its `at` unchanged
    assert current_authority.authorization(first)["at"] == "2026-01-05T21:00:00+00:00"
    assert authorize(fields=["close"]) != first
    assert len(list((store / "authorizations").glob("*.json"))) == 2


def test_a_revoked_authorization_is_issued_again_under_a_new_id(store):
    first = authorize()
    current_authority.revoke_current(
        first, revoked_by="A. Signer", reason="test", effective="2026-02-02"
    )
    second = authorize()
    assert second != first
    assert current_authority.authorization(second)["issue"] == 2
    assert not current_authority.revocations(second)
    current_authority.require_active(
        second, vendor="synthetic-vendor", start="2026-01-05", end="2026-03-02", fields=["close"]
    )
    assert authorize() == second  # the re-issue is reused while it is active


def test_revocation_is_appended_and_effective_from_its_date(store):
    record = authorize()
    check = dict(vendor="synthetic-vendor", start="2026-01-05", fields=["close"])
    current_authority.require_active(record, end="2026-01-30", **check)
    current_authority.revoke_current(
        record, revoked_by="A. Signer", reason="vendor terms changed", effective="2026-02-02"
    )
    # A frozen input before the effective date still verifies; new data does not.
    current_authority.require_valid_at(record, end="2026-02-01", **check)
    with pytest.raises(HoldoutSealed, match="revoked on or before"):
        current_authority.require_valid_at(record, end="2026-02-02", **check)
    with pytest.raises(HoldoutSealed, match="is revoked; new current data"):
        current_authority.require_active(record, end="2026-01-30", **check)
    assert len(current_authority.revocations(record)) == 1
    with pytest.raises(HoldoutSealed, match="who revoked it and why"):
        current_authority.revoke_current(record, revoked_by="", reason="x")
    log = store / "revocations.jsonl"
    log.write_text(log.read_text().replace("vendor terms", "other terms"), "utf-8")
    with pytest.raises(HoldoutSealed, match="revocation log changed"):
        current_authority.revocations(record)


def test_validation_is_bound_to_vendor_window_and_fields(store):
    record = authorize(start="2026-01-05", end="2026-06-30")
    ok = dict(vendor="synthetic-vendor", start="2026-02-01", end="2026-03-01", fields=["close"])
    current_authority.require_valid_at(record, **ok)
    for changes, message in (
        ({"vendor": "other-vendor"}, "another vendor"),
        ({"start": "2026-01-01"}, "outside its authorized window"),
        ({"end": "2026-07-01"}, "outside its authorized window"),
        ({"fields": ["close", "open"]}, "outside the authorized coverage"),
    ):
        with pytest.raises(HoldoutSealed, match=message):
            current_authority.require_valid_at(record, **ok | changes)


def test_a_schema_one_authorization_still_verifies(store):
    row = {
        "schema_version": 1,
        "vendor": "synthetic-vendor",
        "dataset_id": current_authority.current_dataset("synthetic-vendor"),
        "start": "2026-01-05",
        "end": None,
        "fields": ["close", "volume"],
        "authorized_by": "A. Signer",
        "at": "2026-01-05T21:00:00+00:00",
    }
    record = digest(row)
    (store / "authorizations").mkdir(parents=True)
    path = store / "authorizations" / f"{record}.json"
    path.write_text(json.dumps(row | {"record_id": record}), "utf-8")
    current_authority.require_valid_at(
        record, vendor="synthetic-vendor", start="2026-01-05", end="2026-01-30", fields=["close"]
    )
    assert authorize() != record  # a new schema-2 record of the same content is separate
    path.write_text(json.dumps(row | {"record_id": record, "at": "2026-01-06"}), "utf-8")
    with pytest.raises(HoldoutSealed, match="record changed"):
        current_authority.authorization(record)  # schema 1: `at` is an identity input


def test_the_sealed_holdout_is_never_opened_by_a_current_authorization(store):
    for start, end in (("2021-06-01", None), ("2020-01-01", "2022-01-01"), ("2025-12-31", None)):
        with pytest.raises(HoldoutSealed, match="sealed research holdout"):
            authorize(start=start, end=end)
    authorize(start="2026-01-01")  # after the sealed period: fine
    authorize(start="2019-01-01", end="2020-12-31")  # before it: not the holdout
    assert current_authority.real_holdout_spent() is False


def test_a_plain_string_of_fields_is_refused_not_split_into_letters(store):
    with pytest.raises(HoldoutSealed, match="not one string"):
        authorize(fields="close")
    assert not store.exists()
    record = authorize(fields=["close"])
    with pytest.raises(HoldoutSealed, match="not one string"):
        current_authority.require_valid_at(
            record, vendor="synthetic-vendor", start="2026-02-01", end="2026-03-01", fields="close"
        )


@pytest.mark.parametrize("value", ["   ", "relative/store", "."])
def test_the_store_root_must_be_a_nonblank_absolute_path(tmp_path, monkeypatch, value):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(current_authority.ROOT_ENV, value)
    with pytest.raises(HoldoutSealed, match="no current feed is configured|absolute"):
        authorize()
    assert not (tmp_path / "relative").exists()
    assert not (tmp_path / "authorizations").exists()


def test_only_the_hashed_fields_are_tamper_evident(store):
    record = authorize()
    path = store / "authorizations" / f"{record}.json"
    row = json.loads(path.read_text("utf-8"))
    path.write_text(json.dumps(row | {"at": "1999-01-01T00:00:00+00:00"}), "utf-8")
    assert current_authority.authorization(record)["at"] == "1999-01-01T00:00:00+00:00"
    path.write_text(json.dumps(row | {"authorized_by": "Someone Else"}), "utf-8")
    with pytest.raises(HoldoutSealed, match="record changed"):
        current_authority.authorization(record)


def test_revoking_an_unknown_record_is_refused_and_writes_nothing(store):
    authorize()
    with pytest.raises(HoldoutSealed, match="no current authorization"):
        current_authority.revoke_current("f" * 64, revoked_by="A. Signer", reason="mistake")
    assert not (store / "revocations.jsonl").exists()


def _write_record(store, row):
    record = current_authority.record_identity(row)
    (store / "authorizations").mkdir(parents=True, exist_ok=True)
    path = store / "authorizations" / f"{record}.json"
    path.write_text(json.dumps(row | {"record_id": record}), "utf-8")
    return record


def test_a_record_must_carry_exactly_its_schemas_keys(store):
    base = {
        "vendor": "synthetic-vendor",
        "dataset_id": current_authority.current_dataset("synthetic-vendor"),
        "start": "2026-01-05",
        "end": None,
        "fields": ["close"],
        "authorized_by": "A. Signer",
        "at": "2026-01-05T21:00:00+00:00",
    }
    for row in (
        base | {"schema_version": 2},  # schema 2 needs `issue`
        base | {"schema_version": 1, "issue": 1},  # schema 1 has no `issue`
        base | {"schema_version": 2, "issue": 1, "extra": "x"},
        base | {"schema_version": 3, "issue": 1},  # unknown schema
    ):
        record = _write_record(store, row)
        with pytest.raises(HoldoutSealed, match="record changed"):
            current_authority.authorization(record)


def test_a_revocation_line_must_carry_exactly_its_keys(store):
    record = authorize()
    current_authority.revoke_current(record, revoked_by="A. Signer", reason="rotated")
    log = store / "revocations.jsonl"
    row = json.loads(log.read_text("utf-8").splitlines()[0])
    for changed in (row | {"extra": 1}, {k: v for k, v in row.items() if k != "reason"}):
        log.write_text(json.dumps(changed), "utf-8")
        with pytest.raises(HoldoutSealed, match="revocation log changed"):
            current_authority.revocations(record)
