"""The drop-folder provider: schema, completeness, staleness, the live-panel trim and the
atomic export. Synthetic data only; no network, no vendor."""

import hashlib
import json

import pytest

from capstone import faults
from capstone.faults import InjectedFault
from capstone.pipeline.autopilot import providers
from capstone.pipeline.autopilot.providers import (
    DROP_SCHEMA,
    DataInvalid,
    DataMissing,
    DataStale,
    DropFolderProvider,
    HoldoutPrices,
    ProviderNotConfigured,
    build,
    check_documents,
    export_from,
    export_session,
    live_start,
    require_trimmed,
    trim_panel,
)

D1, E1 = "2019-03-04", "2019-03-05"
D2, E2 = "2019-03-05", "2019-03-06"
STAMP = "2019-03-10T00:00:00+00:00"


def docs(session, execution, *, events=None, panel=None):
    out = {
        "market": {"session": execution, "prices": {"10001": 11.0, "10002": 21.0}},
        "marks": {"session": session, "prices": {"10001": 10.0, "10002": 20.0}},
    }
    if events is not None:
        out["events"] = events
    if panel is not None:
        out["panel"] = panel
    return out


def frame(days):
    return {"index": list(days), "columns": ["10001"], "values": [[1.0] for _ in days]}


def make_panel(days, session, execution):
    return {
        "data_version": "synthetic",
        "calendar": list(days) + [execution],
        "universe": frame(days),
        "half_spread": frame(days),
        "fields": {"px": frame(days)},
        "availability": {"px": frame(days)},
    }


def at(day):
    return lambda: f"{day}T12:00:00+00:00"


@pytest.fixture
def drop(tmp_path):
    root = tmp_path / "drop"
    root.mkdir()
    export_session(root, "vendor-a", D1, docs(D1, E1), exported_at=STAMP)
    export_session(root, "vendor-a", D2, docs(D2, E2), exported_at=STAMP)
    return root


# ---------------------------------------------------------------------------
# reading a drop folder


def test_a_clean_drop_folder_is_served(drop):
    provider = DropFolderProvider(drop, today=at("2019-03-08"))
    assert provider.sessions(None) == [D1, D2]
    assert provider.sessions(D1) == [D2]
    assert provider.market(D1)["session"] == E1
    assert provider.marks(D2)["prices"] == {"10001": 10.0, "10002": 20.0}
    assert provider.events(D1) is None
    assert provider.panel(D1) is None
    provider.check_fresh()
    assert ("market", D1) in provider.reads


def test_the_manifest_is_the_last_file_and_names_every_file(drop):
    manifest = json.loads((drop / D1 / "manifest.json").read_text())
    assert manifest["schema"] == DROP_SCHEMA
    assert set(manifest["files"]) == {"market.json", "marks.json"}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((drop / D1 / name).read_bytes()).hexdigest() == digest


def test_a_changed_file_fails_its_digest(drop):
    (drop / D1 / "marks.json").write_text(json.dumps(docs(D1, E1)["marks"]) + " ")
    with pytest.raises(DataInvalid, match="digest"):
        DropFolderProvider(drop, today=at("2019-03-08")).marks(D1)


def test_an_unlisted_file_and_a_non_json_file_are_refused(drop):
    (drop / D1 / "events.json").write_text("[]")
    with pytest.raises(DataInvalid, match="not in manifest"):
        DropFolderProvider(drop, today=at("2019-03-08")).market(D1)


def test_an_incomplete_folder_is_not_listed_until_a_later_one_completes(drop):
    pending = drop / "2019-03-06"
    pending.mkdir()
    (pending / "market.json").write_text("{}")
    provider = DropFolderProvider(drop, today=at("2019-03-08"))
    assert provider.sessions(None) == [D1, D2]  # in progress, nothing later is complete
    export_session(
        drop, "vendor-a", "2019-03-07", docs("2019-03-07", "2019-03-08"), exported_at=STAMP
    )
    with pytest.raises(DataMissing, match="no manifest.json"):
        provider.sessions(None)


def test_staging_folders_are_ignored_and_other_names_are_invalid(drop):
    (drop / "_staging").mkdir()
    (drop / ".tmp").mkdir()
    provider = DropFolderProvider(drop, today=at("2019-03-08"))
    assert provider.sessions(None) == [D1, D2]
    (drop / "notes").mkdir()
    with pytest.raises(DataInvalid, match="session folder"):
        provider.sessions(None)


def test_staleness_is_measured_against_the_clock(drop):
    DropFolderProvider(drop, today=at("2019-03-11")).check_fresh()  # 6 days: fine
    with pytest.raises(DataStale) as caught:
        DropFolderProvider(drop, today=at("2019-03-12")).check_fresh()
    assert caught.value.code == "data_stale"
    DropFolderProvider(drop, max_lag_days=30, today=at("2019-03-12")).check_fresh()


def test_an_empty_folder_is_missing_not_stale(tmp_path):
    with pytest.raises(DataMissing):
        DropFolderProvider(tmp_path, today=at("2019-03-12")).check_fresh()


def test_an_execution_session_after_today_is_invalid(drop):
    with pytest.raises(DataInvalid, match="after today"):
        DropFolderProvider(drop, today=at("2019-03-04")).market(D1)


def test_the_execution_session_must_be_the_next_folder(tmp_path):
    export_session(tmp_path, "v", D1, docs(D1, E2), exported_at=STAMP)
    export_session(tmp_path, "v", E1, docs(E1, E2), exported_at=STAMP)
    with pytest.raises(DataInvalid, match="skips"):
        DropFolderProvider(tmp_path, today=at("2019-03-08")).market(D1)
    gap = tmp_path / "gap"
    gap.mkdir()
    export_session(gap, "v", D1, docs(D1, E1), exported_at="2019-03-20")
    export_session(
        gap, "v", "2019-03-08", docs("2019-03-08", "2019-03-11"), exported_at="2019-03-20"
    )
    with pytest.raises(DataMissing, match="has no folder"):
        DropFolderProvider(gap, today=at("2019-03-20")).market(D1)


def test_the_constructor_refuses_a_missing_root_and_a_bad_lag(tmp_path):
    with pytest.raises(ProviderNotConfigured):
        DropFolderProvider(tmp_path / "nope")
    for bad in (0, -1, True, 2.5, "6"):
        with pytest.raises(ProviderNotConfigured):
            DropFolderProvider(tmp_path, max_lag_days=bad)


def test_a_version_one_manifest_is_still_read(drop):
    path = drop / D1 / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = "autopilot-drop-v1"
    path.write_text(json.dumps(manifest))
    assert DropFolderProvider(drop, today=at("2019-03-08")).market(D1)["session"] == E1
    manifest["schema"] = "autopilot-drop-v0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(DataInvalid):
        DropFolderProvider(drop, today=at("2019-03-08")).market(D1)


# ---------------------------------------------------------------------------
# the document schema


def test_check_documents_accepts_a_valid_session():
    check_documents(D1, docs(D1, E1), vendor="v", exported_at=STAMP)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda d: d.pop("market"), DataMissing),
        (lambda d: d.pop("marks"), DataMissing),
        (lambda d: d["market"].update(session=D1), DataInvalid),
        (lambda d: d["market"].update(extra=1), DataInvalid),
        (lambda d: d["market"].update(prices={}), DataInvalid),
        (lambda d: d["market"]["prices"].update({"abc": 1.0}), DataInvalid),
        (lambda d: d["market"]["prices"].update({"10003": float("nan")}), DataInvalid),
        (lambda d: d["market"]["prices"].update({"10003": -2.0}), DataInvalid),
        (lambda d: d["market"]["prices"].update({"10003": True}), DataInvalid),
        (lambda d: d["marks"].update(session=E1), DataInvalid),
        (lambda d: d.update(events={"not": "a list"}), DataInvalid),
        (lambda d: d.update(events=[{"event_id": "e"}]), DataInvalid),
        (
            lambda d: d.update(
                events=[{"event_id": "e", "event_date": E1, "permno": 1, "kind": "k"}]
            ),
            DataInvalid,
        ),
        (
            lambda d: d.update(
                events=[{"event_id": "e", "event_date": D1, "permno": "1", "kind": "k"}]
            ),
            DataInvalid,
        ),
    ],
)
def test_check_documents_refuses_bad_documents(mutate, error):
    document = docs(D1, E1)
    mutate(document)
    with pytest.raises(error):
        check_documents(D1, document)


def test_check_documents_checks_the_manifest_fields():
    with pytest.raises(DataInvalid, match="vendor"):
        check_documents(D1, docs(D1, E1), vendor="")
    with pytest.raises(DataInvalid, match="not a time"):
        check_documents(D1, docs(D1, E1), exported_at="whenever")
    with pytest.raises(DataInvalid, match="before its execution session"):
        check_documents(D1, docs(D1, E1), exported_at="2019-03-04T23:00:00")


def test_holdout_documents_are_a_stop_not_a_blocker():
    document = docs("2020-12-31", "2021-01-04")
    with pytest.raises(HoldoutPrices):
        check_documents("2020-12-31", document)
    with pytest.raises(HoldoutPrices):
        export_session("unused-root", "v", "2020-12-31", document, exported_at="2021-02-01")


# ---------------------------------------------------------------------------
# the live panel trim

LIVE, LIVE_E = "2026-03-03", "2026-03-04"
OLD = ["2019-03-04", "2022-06-01", "2026-03-02", LIVE]


def test_trim_panel_cuts_every_frame_and_the_calendar_without_mutating():
    panel = make_panel(OLD, LIVE, LIVE_E)
    before = json.dumps(panel, sort_keys=True)
    out = trim_panel(panel, "2026-03-02")
    assert out["calendar"] == ["2026-03-02", LIVE, LIVE_E]
    for part in (
        out["universe"],
        out["half_spread"],
        out["fields"]["px"],
        out["availability"]["px"],
    ):
        assert part["index"] == ["2026-03-02", LIVE]
        assert len(part["values"]) == 2
    assert json.dumps(panel, sort_keys=True) == before


def test_require_trimmed_names_the_first_early_row():
    panel = make_panel(OLD, LIVE, LIVE_E)
    with pytest.raises(DataInvalid, match="before the first live session"):
        require_trimmed(panel, "2026-03-02", "where")
    require_trimmed(trim_panel(panel, "2026-03-02"), "2026-03-02", "where")


def test_live_start_is_the_first_complete_live_folder():
    assert live_start("2020-12-30", ["2019-03-04"]) is None
    assert live_start("2025-12-31", []) is None
    assert live_start(LIVE, ["2019-03-04", "2026-03-02", LIVE]) == "2026-03-02"
    assert live_start(LIVE, []) == LIVE


def test_export_trims_a_live_panel_before_writing(tmp_path):
    panel = make_panel(OLD, LIVE, LIVE_E)
    export_session(tmp_path, "v", LIVE, docs(LIVE, LIVE_E, panel=panel), exported_at="2026-03-10")
    written = json.loads((tmp_path / LIVE / "panel.json").read_text())
    assert written["universe"]["index"] == [LIVE]  # the drop's first live session is LIVE
    assert min(written["calendar"]) == LIVE
    provider = DropFolderProvider(tmp_path, today=at("2026-03-10"))
    assert provider.panel(LIVE)["universe"]["index"] == [LIVE]


def test_an_untrimmed_live_panel_is_refused_by_the_checks():
    panel = make_panel(OLD, LIVE, LIVE_E)
    with pytest.raises(DataInvalid, match="first live session"):
        check_documents(LIVE, docs(LIVE, LIVE_E, panel=panel), first_live=LIVE)


def test_a_live_session_needs_a_panel():
    with pytest.raises(DataMissing, match="needs panel.json"):
        check_documents(LIVE, docs(LIVE, LIVE_E))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("fields"),
        lambda p: p.update(fields={}),
        lambda p: p.update(calendar=[LIVE]),
        lambda p: p["universe"].update(index=[]),
        lambda p: p["universe"].update(values=[]),
        lambda p: p["half_spread"].update(index=[LIVE_E]),
        lambda p: p["fields"]["px"].update(values=[[1.0, 2.0]]),
    ],
)
def test_a_malformed_live_panel_is_invalid(mutate):
    panel = make_panel([LIVE], LIVE, LIVE_E)
    mutate(panel)
    with pytest.raises(DataInvalid):
        check_documents(LIVE, docs(LIVE, LIVE_E, panel=panel), first_live=LIVE)


def test_a_well_formed_live_panel_passes():
    panel = make_panel([LIVE], LIVE, LIVE_E)
    check_documents(LIVE, docs(LIVE, LIVE_E, panel=panel), first_live=LIVE)


# ---------------------------------------------------------------------------
# the export


def test_exporting_the_same_documents_twice_is_a_no_op(drop):
    before = (drop / D1 / "manifest.json").read_bytes()
    export_session(drop, "vendor-a", D1, docs(D1, E1), exported_at="2019-03-20T00:00:00")
    assert (drop / D1 / "manifest.json").read_bytes() == before


def test_a_different_re_export_is_refused(drop):
    changed = docs(D1, E1)
    changed["marks"]["prices"]["10001"] = 99.0
    with pytest.raises(DataInvalid, match="other documents"):
        export_session(drop, "vendor-a", D1, changed, exported_at=STAMP)


def test_unknown_documents_are_refused(tmp_path):
    document = docs(D1, E1)
    document["extra"] = {}
    with pytest.raises(DataInvalid, match="unknown documents"):
        export_session(tmp_path, "v", D1, document, exported_at=STAMP)


def test_a_kill_before_the_manifest_leaves_no_session(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_FAULT_INJECTION", "1")
    monkeypatch.setenv("CAPSTONE_FAULT_AT", "export:before_manifest")
    faults._fault_plan.cache_clear()
    faults._fault_hits.clear()
    try:
        with pytest.raises(InjectedFault):
            export_session(tmp_path, "v", D1, docs(D1, E1), exported_at=STAMP)
        assert (tmp_path / D1 / "market.json").is_file()
        assert not (tmp_path / D1 / "manifest.json").exists()
        assert DropFolderProvider(tmp_path, today=at("2019-03-08")).sessions(None) == []
    finally:
        monkeypatch.delenv("CAPSTONE_FAULT_INJECTION")
        monkeypatch.delenv("CAPSTONE_FAULT_AT")
        faults._fault_plan.cache_clear()
        faults._fault_hits.clear()
    export_session(tmp_path, "v", D1, docs(D1, E1), exported_at=STAMP)  # the retry completes
    assert (tmp_path / D1 / "manifest.json").is_file()


class ScriptedAdapter:
    """A fake vendor: answers from memory, counts every call; no I/O of any kind."""

    vendor = "scripted"

    def __init__(self, available):
        self.available = available
        self.calls = []

    def session_documents(self, session):
        self.calls.append(session)
        return self.available.get(session)


def test_export_from_stops_at_the_first_incomplete_session(tmp_path):
    adapter = ScriptedAdapter({D1: docs(D1, E1), D2: None})
    written = export_from(adapter, tmp_path, [D1, D2, "2019-03-06"], exported_at=STAMP)
    assert written == [D1]
    assert adapter.calls == [D1, D2]  # the third session was never asked about
    assert (tmp_path / D1 / "manifest.json").is_file()
    assert not (tmp_path / D2).exists()


# ---------------------------------------------------------------------------
# configuration


def test_build_makes_a_drop_folder_provider(tmp_path):
    section = {"provider": "drop_folder", "provider_config": {"root": str(tmp_path)}}
    provider = build(section, {"clock": at("2019-03-08")})
    assert isinstance(provider, DropFolderProvider)
    assert provider.max_lag_days == providers.MAX_LAG_DAYS
    section["provider_config"]["max_lag_days"] = 3
    assert build(section, {}).max_lag_days == 3


def test_build_refuses_unknown_drop_folder_keys_and_a_missing_root(tmp_path):
    section = {"provider": "drop_folder", "provider_config": {"root": str(tmp_path), "url": "x"}}
    with pytest.raises(ProviderNotConfigured, match="unknown keys"):
        build(section, {})
    with pytest.raises(ProviderNotConfigured, match="root"):
        build({"provider": "drop_folder", "provider_config": {}}, {})
    with pytest.raises(ProviderNotConfigured):
        build({"provider": "drop_folder", "provider_config": {"root": str(tmp_path / "gone")}}, {})
