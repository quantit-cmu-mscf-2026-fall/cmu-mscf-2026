"""Freezing and loading an immutable input bundle: synthetic panel, no account, no network."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from reference_fakes import reference_panel

from capstone import current_authority, release
from capstone.execution import bundle
from capstone.execution.bundle import freeze_bundle, load_bundle
from capstone.execution.current_features import current_panel
from capstone.execution.data import FrozenSnapshotSource
from capstone.execution.deploy import Refused
from capstone.factors.evaluation import HoldoutError, Panel
from capstone.holdout_authority import HoldoutSealed
from capstone.strategy.reference import ReferenceStrategySpec

ACCOUNT = {"positions": {}, "open_orders": [], "cash": 100000.0, "capital": 100000.0}


def freeze(path, panel, spec, as_of, **changes):
    prices = panel.fields["close"].loc[as_of].to_dict()
    instruments = {
        n: {"symbol": f"S{n}", "currency": "USD", "sec_type": "STK", "exchange": "XSYN"}
        for n in prices
    }
    arguments = dict(
        prices=prices, instruments=instruments, execution_cost_bps=11.0, account=ACCOUNT
    )
    return freeze_bundle(path, panel, [spec], as_of, **arguments | changes)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    panel = reference_panel()
    spec = ReferenceStrategySpec("716bdaf7915c65be", panel.data_version, "2000-12-29")
    return freeze(tmp_path_factory.mktemp("frozen") / "bundle", panel, spec, "2005-12-30")


def test_a_frozen_bundle_loads_with_its_frames_specs_and_identity(frozen):
    data, panel, specs, returns = load_bundle(frozen)
    assert data["as_of"] == "2005-12-30" and data["schema_version"] == 2
    assert data["runtime"] == release.runtime_versions()
    assert data["release"]["commit"] and "current_provenance" not in data
    assert panel.provenance is None and panel.universe.index.max() == pd.Timestamp("2005-12-30")
    assert panel.data_version == "synthetic-reference-v2" and set(panel.fields) >= {"close"}
    assert [s.claim_id for s in specs] == ["716bdaf7915c65be"] and returns.empty
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    assert {"inputs.json", "strategy-returns.parquet"} <= set(manifest["files"])


def test_release_and_runtime_identity_are_the_shared_ones():
    assert bundle.runtime_versions is release.runtime_versions
    assert bundle.stamped_identity is release.stamped_identity
    assert bundle.release_identity()["deployment_tag"] is None
    with pytest.raises(Refused, match="there is no tag"):
        bundle.release_identity("deploy-2000.01.01")


def test_a_bundle_cannot_be_overwritten(frozen):
    with pytest.raises(FileExistsError):
        freeze_bundle(frozen, None, None, None, prices=None, instruments=None, account=None)


@pytest.mark.parametrize("name", ["inputs.json", "strategy-returns.parquet"])
def test_a_modified_file_is_refused(frozen, tmp_path, name):
    copy = tmp_path / "copy"
    shutil.copytree(frozen, copy)
    with (copy / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(copy)


def test_metadata_rewritten_with_a_matching_manifest_still_fails_its_identity(frozen, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(frozen, copy)
    data = json.loads((copy / "inputs.json").read_text(encoding="utf-8"))
    data["execution_cost_bps"] = 0.0
    text = bundle.canonical(data) + "\n"
    (copy / "inputs.json").write_text(text, encoding="utf-8", newline="\n")
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["inputs.json"] = bundle.hashlib.sha256(text.encode()).hexdigest()
    (copy / "manifest.json").write_text(bundle.canonical(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="input metadata changed"):
        load_bundle(copy)


def test_a_different_runtime_is_refused(frozen, monkeypatch):
    monkeypatch.setattr(bundle, "runtime_versions", lambda: {"python": "0.0"})
    with pytest.raises(ValueError, match="runtime differs"):
        load_bundle(frozen)


@pytest.mark.parametrize(
    "extra",
    [
        {"accountId": "SYNTHETIC_PRIVATE_VALUE"},
        {"metadata": {"credentials": "SYNTHETIC_PRIVATE_VALUE"}},
        {"positions": {"SYNTHETIC_PRIVATE_VALUE": 1}},
        {"cash": "SYNTHETIC_PRIVATE_VALUE"},
    ],
)
def test_private_account_metadata_is_rejected_before_writing(tmp_path, extra):
    root = tmp_path / "private"
    with pytest.raises(ValueError) as error:
        freeze_bundle(
            root, None, None, None, prices=None, instruments=None, account=ACCOUNT | extra
        )
    assert "SYNTHETIC_PRIVATE_VALUE" not in str(error.value)
    assert not root.exists()


def test_a_zero_position_needs_no_instrument_mapping_to_freeze(frozen, tmp_path):
    data, panel, specs, returns = load_bundle(frozen)
    path = freeze_bundle(
        tmp_path / "zero",
        panel,
        specs,
        data["as_of"],
        prices={int(n): p for n, p in data["prices"].items()},
        instruments={int(n): p for n, p in data["instruments"].items()},
        account=dict(data["account"], positions={999999: 0}),
        execution_cost_bps=data["execution_cost_bps"],
        returns=returns,
    )
    assert load_bundle(path)[0]["account"]["positions"] == {"999999": 0}


def test_a_snapshot_must_contain_the_decision_session(frozen):
    data, panel, specs, _ = load_bundle(frozen)
    with pytest.raises(ValueError, match="snapshot must contain"):
        freeze_bundle(
            frozen.parent / "late",
            panel,
            specs,
            "2006-02-03",  # a calendar session after the last panel row
            prices={},
            instruments={},
            account=ACCOUNT,
        )
    assert not (frozen.parent / "late").exists()


def test_threshold_records_are_not_needed_without_a_program_gate_report():
    assert bundle._record_thresholds(Path("unused"), {"a": {"report_path": None}}) is None
    with pytest.raises(ImportError):  # a verified report needs the registered-threshold module
        bundle._record_thresholds(
            Path("unused"),
            {"a": {"report_path": "report.json", "validation_status": bundle.SEARCH_STAGE}},
        )


# ---------------------------------------------------------------------------
# Past the discovery period: only with an active current-feature authorization.


def current_case(start="2026-01-01"):
    """The reference panel relabelled onto business days from `start`, cut at its as-of."""
    base = reference_panel()
    calendar = pd.bdate_range(start, periods=len(base.calendar))
    index = calendar[: len(base.universe)]
    old = pd.DatetimeIndex(base.universe.index)

    def move(frame):
        return None if frame is None else frame.set_axis(index, axis=0)

    moved = Panel(
        {name: move(value) for name, value in base.fields.items()},
        move(base.universe),
        move(base.half_spread),
        None,
        base.data_version,
        calendar,
        {name: pd.Series(index, index=index) for name in base.availability},
    )
    dates = dict(zip(old, index, strict=True))
    as_of = dates[pd.Timestamp("2005-12-30")]
    panel = FrozenSnapshotSource(moved).panel(tuple(moved.fields), as_of)
    spec = ReferenceStrategySpec(
        "716bdaf7915c65be", panel.data_version, str(dates[pd.Timestamp("2000-12-29")].date())
    )
    return panel, spec, as_of, [str(panel.universe.index[0].date()), str(as_of.date())]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(current_authority.ROOT_ENV, str(tmp_path / "current-authority"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return tmp_path / "current-authority"


def test_a_current_panel_without_authorization_is_not_frozen(tmp_path):
    panel, spec, as_of, _ = current_case()
    root = tmp_path / "bundle"
    with pytest.raises(HoldoutError, match="authorized current-feature route"):
        freeze(root, panel, spec, as_of)
    assert not root.exists()


def test_an_authorized_current_panel_freezes_and_keeps_its_provenance(tmp_path, store):
    panel, spec, as_of, period = current_case()
    record = current_authority.authorize_current(
        "synthetic-vendor",
        start=period[0],
        end=period[1],
        fields=list(panel.fields),
        authorized_by="A. Signer",
    )
    current = current_panel(panel, record)
    path = freeze(tmp_path / "bundle", current, spec, as_of)
    data, loaded, _, _ = load_bundle(path)
    assert data["current_provenance"] == current.provenance.to_dict()
    assert loaded.provenance == current.provenance
    # Tampering with the frozen provenance breaks the bundle's identity.
    inputs = json.loads((path / "inputs.json").read_text(encoding="utf-8"))
    inputs["current_provenance"]["source_version"] = "other"
    (path / "inputs.json").write_text(bundle.canonical(inputs) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(path)
    # Changed content, or a revoked authorization, is refused at freeze.
    edited = replace(current, fields={**current.fields, "close": current.fields["close"] * 2})
    with pytest.raises(HoldoutSealed, match="content changed"):
        freeze(tmp_path / "edited", edited, spec, as_of)
    current_authority.revoke_current(
        record, revoked_by="A. Signer", reason="test", effective=str(as_of.date())
    )
    with pytest.raises(HoldoutSealed, match="revoked"):
        freeze(tmp_path / "revoked", current, spec, as_of)
    assert not (tmp_path / "revoked").exists()
