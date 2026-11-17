"""Authorized current-feature provenance and the post-2020 gate.

Closed by default: a panel holding a date after 2020 is admitted (by `freeze_bundle` and
the other current-route checks) only when it carries a `CurrentFeatureProvenance` bound to
a `capstone.current_authority` record. With no authorization supplied, or no store
configured, every path here refuses before reading any feature, and the research holdout
(2021-2025) stays sealed: authorizing a window that overlaps it is refused by the authority.

- `current_panel(panel, record_id)` attaches it; the authorization must be
  ACTIVE and cover [panel start, as_of] and the panel's fields.
- `require_current` (every later use, fresh processes and frozen bundles
  included) needs the recorded authorization to exist, to have covered the
  panel unrevoked at its as_of, and the content to hash to `frame_hash`.

`current_preflight` names each missing feed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import pandas as pd

from capstone import current_authority as _authority_module
from capstone.factors.evaluation import HoldoutError, Panel
from capstone.holdout_authority import HoldoutSealed
from capstone.strategy.contract import CURRENT_ROUTE_MISSING

_HASH = re.compile(r"[0-9a-f]{64}")


def _authority():
    return _authority_module


@dataclass(frozen=True)
class CurrentFeatureProvenance:
    """Where a post-2020 frame came from, bound to the authorization that admitted it."""

    dataset_id: str  # current:<vendor>
    vendor: str
    as_of: str  # the frame's last session
    fields: tuple[str, ...]  # declared coverage; half_spread only when spreads are data
    frame_hash: str
    authorization_id: str  # current_authority record_id
    source_version: str  # the panel's data_version at attachment

    def __post_init__(self):
        if (
            self.dataset_id != _authority().current_dataset(self.vendor)
            or str(pd.Timestamp(self.as_of).date()) != self.as_of
            or not isinstance(self.fields, tuple)
            or list(self.fields) != sorted(set(self.fields))
            or any(not isinstance(name, str) or not name for name in self.fields)
            or not _HASH.fullmatch(str(self.frame_hash))
            or not _HASH.fullmatch(str(self.authorization_id))
            or not isinstance(self.source_version, str)
        ):
            raise ValueError("malformed current provenance")

    def to_dict(self):
        return asdict(self) | {"fields": list(self.fields)}

    @classmethod
    def from_dict(cls, row):
        return cls(**(row | {"fields": tuple(row["fields"])}))


def panel_frames(panel: Panel) -> dict:
    """The labelled frames a frozen bundle stores and a provenance hashes."""
    frames = {
        **{f"field:{n}": f for n, f in panel.fields.items()},
        "universe": panel.universe,
        "half_spread": panel.half_spread,
    }
    if panel.sic is not None:
        frames["sic"] = panel.sic
    if panel.company is not None:
        frames["company"] = panel.company
    for name, available in (panel.availability or {}).items():
        frames[f"availability:{name}"] = available.to_frame("available_at")
    return frames


def panel_hash(panel: Panel) -> str:
    from capstone.execution.bundle import frame_hash  # bundle imports this module
    from capstone.strategy.research import digest

    return digest(
        {
            label: frame_hash(f.to_frame() if isinstance(f, pd.Series) else f)
            for label, f in panel_frames(panel).items()
        }
    )


def _day(value):
    return str(pd.Timestamp(value).date())


def provenance_for(record_id, *, start, as_of, fields, content_hash, source_version):
    """New current data: requires the ACTIVE authorization covering [start, as_of]."""
    authority = _authority()
    row = authority.authorization(record_id)
    fields = tuple(sorted(set(fields)))
    authority.require_active(record_id, vendor=row["vendor"], start=start, end=as_of, fields=fields)
    return CurrentFeatureProvenance(
        row["dataset_id"], row["vendor"], _day(as_of), fields, content_hash, record_id,
        source_version,
    )  # fmt: skip


def require_current(prov, *, content_hash, start, end, source_version=None) -> None:
    """Every use: the recorded authorization covered it unrevoked; content unchanged."""
    if not isinstance(prov, CurrentFeatureProvenance):
        raise HoldoutError(CURRENT_ROUTE_MISSING)
    if source_version is not None and source_version != prov.source_version:
        raise HoldoutSealed("current source version changed since its provenance was taken")
    if prov.frame_hash != content_hash or _day(end) != prov.as_of:
        raise HoldoutSealed("current frame content changed since its provenance was taken")
    _authority().require_valid_at(
        prov.authorization_id, vendor=prov.vendor, start=start, end=end, fields=prov.fields
    )


def require_current_route(panel: Panel, *, active=False) -> None:
    """`target_weights` (frozen check) and `freeze_bundle` (active=True) past 2020."""
    if panel.provenance is not None and panel.fundamentals is not None:
        raise ValueError("current fundamentals have no qualified route")
    dates = pd.DatetimeIndex(panel.universe.index)
    prov = panel.provenance
    require_current(
        prov,
        content_hash=panel_hash(panel) if prov is not None else None,
        start=dates.min(),
        end=dates.max(),
        source_version=panel.data_version,
    )
    if active:
        _authority().require_active(
            prov.authorization_id, vendor=prov.vendor, start=dates.min(), end=dates.max(),
            fields=prov.fields,
        )  # fmt: skip


def current_panel(panel: Panel, record_id, *, fields=None) -> Panel:
    """The panel with its provenance; `fields` declares coverage (default: its fields)."""
    dates = pd.DatetimeIndex(panel.universe.index)
    fields = tuple(panel.fields) if fields is None else fields
    # Refuse before any feature is read: the authorization is checked against the panel's
    # window and declared field names only; the content is hashed afterwards.
    _authority().require_active(
        record_id,
        vendor=_authority().authorization(record_id)["vendor"],
        start=dates.min(),
        end=dates.max(),
        fields=tuple(sorted(set(fields))),
    )
    prov = provenance_for(
        record_id,
        start=dates.min(),
        as_of=dates.max(),
        fields=fields,
        content_hash=panel_hash(panel),
        source_version=panel.data_version,
    )
    current = replace(panel, provenance=prov)
    require_current_route(current, active=True)
    return current


def current_preflight(spec, panel: Panel, *, coverage=None):
    """`program_input_preflight` on a current panel: one blocker per missing feed.

    Only the declared coverage (the provenance's fields) counts as carried.
    Spreads that are not data add `feature_missing:half_spread`.
    """
    # Not yet published: until it is, the preflight refuses (ImportError) rather than guess.
    from capstone.pipeline.program_inputs import program_input_preflight

    prov = panel.provenance
    if coverage is None:
        if not isinstance(prov, CurrentFeatureProvenance):
            raise ValueError("current preflight requires the panel's current provenance")
        coverage = prov.fields
    view = SimpleNamespace(
        data_version=panel.data_version,
        fields={n: f for n, f in panel.fields.items() if n in coverage},
        availability={n: a for n, a in (panel.availability or {}).items() if n in coverage},
    )
    hashes = {"daily": prov.frame_hash} if prov is not None else {}
    report = program_input_preflight(
        spec, SimpleNamespace(panel=view, qualification={"source_hashes": hashes})
    )
    if "half_spread" not in coverage:
        report["blockers"] = sorted({*report["blockers"], "feature_missing:half_spread"})
        report["source_preflight_passed"] = False
    report["current_provenance"] = prov.to_dict() if prov is not None else None
    return report
