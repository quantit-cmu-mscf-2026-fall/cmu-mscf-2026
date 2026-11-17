"""The current-feature gate is closed by default: synthetic panel, vendor and store, no data."""

import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone import current_authority
from capstone.execution import current_features as cf
from capstone.factors.evaluation import HoldoutError, Panel
from capstone.holdout_authority import HoldoutSealed

DATES = pd.bdate_range("2026-01-05", periods=5)
PERMNOS = [10001, 10002, 10003]


class Tripwire(dict):
    """A field mapping whose values must not be read."""

    def _refuse(self, *args, **kwargs):
        raise AssertionError("a feature was read before the authorization was checked")

    items = values = get = __getitem__ = _refuse


def frame(seed, dates=DATES):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(dates), 3)), index=dates, columns=PERMNOS)


def make_panel(dates=DATES, fields=None):
    return Panel(
        fields if fields is not None else {"close": frame(1, dates), "volume": frame(2, dates)},
        pd.DataFrame(True, index=dates, columns=PERMNOS),
        frame(3, dates).abs() * 0.001,
        None,
        "synthetic-v1",
        pd.DatetimeIndex(dates),
        {"close": pd.Series(dates, index=dates)},
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(current_authority.ROOT_ENV, str(tmp_path / "current-authority"))
    monkeypatch.setattr(
        Path, "home", lambda: tmp_path / "home"
    )  # keep the holdout record synthetic
    return tmp_path / "current-authority"


def authorize(**changes):
    arguments = dict(
        start="2026-01-05", end=None, fields=["close", "volume"], authorized_by="A. Signer"
    )
    return current_authority.authorize_current("synthetic-vendor", **arguments | changes)


def test_default_refuses_and_reads_no_feature(tmp_path, monkeypatch):
    monkeypatch.delenv(current_authority.ROOT_ENV, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    panel = make_panel(fields=Tripwire(close=frame(1), volume=frame(2)))
    with pytest.raises(HoldoutSealed, match="no current feed is configured"):
        cf.current_panel(panel, "0" * 64)
    assert panel.provenance is None
    assert not (tmp_path / "home").exists()


def test_an_unknown_authorization_refuses_before_reading(store):
    authorize()
    panel = make_panel(fields=Tripwire(close=frame(1), volume=frame(2)))
    with pytest.raises(HoldoutSealed, match="no current authorization"):
        cf.current_panel(panel, "1" * 64)
    assert panel.provenance is None


def test_a_panel_without_provenance_has_no_current_route():
    with pytest.raises(HoldoutError, match="authorized current-feature route"):
        cf.require_current_route(make_panel())
    with pytest.raises(HoldoutError):
        cf.require_current(None, content_hash="0" * 64, start=DATES[0], end=DATES[-1])


def test_an_authorized_panel_carries_verifiable_provenance(store):
    record = authorize()
    current = cf.current_panel(make_panel(), record)
    prov = current.provenance
    assert (prov.dataset_id, prov.as_of, prov.authorization_id) == (
        "current:synthetic-vendor",
        "2026-01-09",
        record,
    )
    assert prov.fields == ("close", "volume")
    assert cf.CurrentFeatureProvenance.from_dict(prov.to_dict()) == prov
    cf.require_current_route(current, active=True)


def test_changed_content_or_source_is_refused(store):
    current = cf.current_panel(make_panel(), authorize())
    edited = current.fields["close"].copy()
    edited.iloc[0, 0] += 1.0
    with pytest.raises(HoldoutSealed, match="content changed"):
        cf.require_current_route(replace(current, fields={**current.fields, "close": edited}))
    with pytest.raises(HoldoutSealed, match="source version changed"):
        cf.require_current_route(replace(current, data_version="synthetic-v2"))
    with pytest.raises(HoldoutSealed, match="content changed"):
        cf.require_current_route(replace(current, universe=current.universe.iloc[:-1]))


def test_fields_outside_the_authorized_coverage_are_refused(store):
    record = authorize(fields=["close"])
    with pytest.raises(HoldoutSealed, match="outside the authorized coverage"):
        cf.current_panel(make_panel(), record)
    assert cf.current_panel(make_panel(), record, fields=["close"]).provenance.fields == ("close",)


def test_a_window_before_the_authorization_is_refused(store):
    record = authorize(start="2026-01-07")
    with pytest.raises(HoldoutSealed, match="outside its authorized window"):
        cf.current_panel(make_panel(), record)


def test_the_sealed_holdout_cannot_be_authorized(store):
    with pytest.raises(HoldoutSealed, match="sealed research holdout"):
        authorize(start="2025-12-29")
    assert not (store / "authorizations").exists()


def test_revocation_stops_new_use_but_not_a_session_before_it(store):
    record = authorize()
    current = cf.current_panel(make_panel(), record)
    current_authority.revoke_current(
        record, revoked_by="A. Signer", reason="test", effective="2026-02-01"
    )
    cf.require_current_route(current)  # frozen use: it covered the panel before the revocation
    with pytest.raises(HoldoutSealed, match="revoked; new current data is refused"):
        cf.require_current_route(current, active=True)
    with pytest.raises(HoldoutSealed, match="revoked; new current data is refused"):
        cf.current_panel(make_panel(), record)


def test_current_fundamentals_have_no_route(store):
    current = cf.current_panel(make_panel(), authorize())
    with pytest.raises(ValueError, match="no qualified route"):
        cf.require_current_route(replace(current, fundamentals=pd.DataFrame({"x": [1]})))


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_id": "current:other"},
        {"as_of": "2026-01-09 00:00"},
        {"fields": ("volume", "close")},
        {"frame_hash": "xyz"},
        {"authorization_id": "short"},
        {"source_version": 3},
    ],
)
def test_malformed_provenance_is_refused(store, changes):
    prov = cf.current_panel(make_panel(), authorize()).provenance
    with pytest.raises(ValueError, match="malformed current provenance"):
        cf.CurrentFeatureProvenance.from_dict(prov.to_dict() | changes)


def test_preflight_names_missing_spreads_and_needs_provenance(store, monkeypatch):
    module = types.ModuleType("capstone.pipeline.program_inputs")
    module.program_input_preflight = lambda spec, source: {
        "blockers": ["feature_missing:volume"],
        "source_preflight_passed": False,
        "hashes": source.qualification["source_hashes"],
    }
    monkeypatch.setitem(sys.modules, "capstone.pipeline.program_inputs", module)
    current = cf.current_panel(make_panel(), authorize())
    report = cf.current_preflight(None, current)
    assert report["blockers"] == ["feature_missing:half_spread", "feature_missing:volume"]
    assert report["hashes"] == {"daily": current.provenance.frame_hash}
    assert report["current_provenance"] == current.provenance.to_dict()
    with pytest.raises(ValueError, match="requires the panel's current provenance"):
        cf.current_preflight(None, make_panel())
