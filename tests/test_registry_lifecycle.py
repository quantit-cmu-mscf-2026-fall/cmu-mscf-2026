"""Registry lifecycle: state changes only through hash-chained events.

Synthetic registry files in a temp folder. An entry is created through the same
`_commit` that `set_state` uses (a `registered` event, then the entry), or written
by hand as a legacy entry that predates the lifecycle file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from capstone.artifacts import canonical, digest
from capstone.faults import InjectedFault, _fault_hits, _fault_plan
from capstone.strategy.registry import (
    INCUBATION_ONLY,
    LIFECYCLE_FILE,
    MODULE,
    REGISTRY_FILE,
    LifecycleRewritePending,
    RegistryEntry,
    _commit,
    parse_registry,
    read_lifecycle,
    read_registry,
    set_state,
    verify_lifecycle,
)

AT = "2026-11-09T00:00:00Z"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "deploy").mkdir()
    (tmp_path / REGISTRY_FILE).write_bytes(b"# registry\nstrategies: []\n")
    return tmp_path


def registered(root: Path, sid: str = "momo", *, role: str = "deploy", version: str = ""):
    """An entry written the way a registration writes it: its event, then the entry."""
    entry = RegistryEntry(sid, MODULE, "reports/gate.md", role=role, version=version)
    event = {
        "field": "registered",
        "id": sid,
        "role": role,
        "state": "active",
        "module": MODULE,
        "version": version,
        "reviewer": "cal",
        "at": AT,
    }
    return _commit(root, event, entry)


def legacy(root: Path, sid: str = "old") -> None:
    """An entry written by hand, with no event: it predates the lifecycle file."""
    path = root / REGISTRY_FILE
    path.write_bytes(
        path.read_bytes().replace(b"strategies: []\n", b"strategies:\n")
        + (
            f'  - id: "{sid}"\n    module: "{MODULE}"\n'
            '    gate_report: "reports/gate.md"\n    enabled: false\n'
        ).encode()
    )


def lines(root: Path) -> list[str]:
    return (root / LIFECYCLE_FILE).read_text(encoding="utf-8").splitlines(keepends=True)


def test_state_changes_only_via_events(root):
    registered(root)
    path = root / REGISTRY_FILE
    clean = path.read_bytes()
    path.write_bytes(clean + b'    state: "suspended"\n')  # a hand edit, no event behind it
    with pytest.raises(ValueError, match="lifecycle") as refused:
        read_registry(root)
    assert not isinstance(refused.value, LifecycleRewritePending)
    path.write_bytes(clean)
    entry = set_state(root, "momo", "suspended", reason="drawdown review", reviewer="cal", at=AT)
    assert entry.state == "suspended" and read_registry(root) == (entry,)
    first, event = read_lifecycle(root)
    assert (first["field"], first["role"], first["prev_hash"]) == ("registered", "deploy", "")
    assert {k: event[k] for k in ("id", "from", "to", "reason", "reviewer", "at")} == {
        "id": "momo",
        "from": "active",
        "to": "suspended",
        "reason": "drawdown review",
        "reviewer": "cal",
        "at": AT,
    }
    resumed = set_state(root, "momo", "active", reason="reviewed", reviewer="cal", at=AT)
    assert resumed.state == "active" and len(read_lifecycle(root)) == 3
    with pytest.raises(ValueError, match="reason"):
        set_state(root, "momo", "suspended", reason="", reviewer="cal", at=AT)
    with pytest.raises(ValueError, match="already"):
        set_state(root, "momo", "active", reason="x", reviewer="cal", at=AT)
    with pytest.raises(ValueError, match="state must be one of"):
        set_state(root, "momo", "paused", reason="x", reviewer="cal", at=AT)
    with pytest.raises(ValueError, match="reviewer and a timestamp"):
        set_state(root, "momo", "suspended", reason="x", reviewer="", at=AT)
    with pytest.raises(ValueError, match="not registered"):
        set_state(root, "ghost", "suspended", reason="x", reviewer="cal", at=AT)


def test_set_state_never_touches_enabled_or_the_other_entries(root):
    registered(root, "a")
    registered(root, "b")
    path = root / REGISTRY_FILE
    path.write_bytes(path.read_bytes().replace(b"enabled: false", b"enabled: true", 1))
    set_state(root, "a", "suspended", reason="r", reviewer="cal", at=AT)
    a, b = read_registry(root)
    assert (a.enabled, a.state, b.enabled, b.state) == (True, "suspended", False, "active")


def test_retired_is_terminal(root):
    registered(root)
    set_state(root, "momo", "retired", reason="decayed", reviewer="cal", at=AT)
    before = (root / REGISTRY_FILE).read_bytes()
    for state in ("active", "suspended"):
        with pytest.raises(ValueError, match="retired"):
            set_state(root, "momo", state, reason="x", reviewer="cal", at=AT)
    assert (root / REGISTRY_FILE).read_bytes() == before
    assert len(read_lifecycle(root)) == 2  # registered, retired


def test_a_legacy_entry_with_no_events_reads_and_changes_state_through_events(root):
    legacy(root)
    assert read_registry(root)[0].state == "active" and not (root / LIFECYCLE_FILE).exists()
    entry = set_state(root, "old", "suspended", reason="r", reviewer="cal", at=AT)
    assert entry.state == "suspended" and [e["field"] for e in read_lifecycle(root)] == ["state"]
    (root / LIFECYCLE_FILE).unlink()  # a state with no event behind it
    with pytest.raises(ValueError, match="lifecycle"):
        read_registry(root)


def test_the_chain_refuses_a_tampered_log(root):
    registered(root)
    set_state(root, "momo", "suspended", reason="a", reviewer="cal", at=AT)
    set_state(root, "momo", "active", reason="b", reviewer="cal", at=AT)
    genesis, *rest = lines(root)
    entries = read_registry(root)
    tampered = [
        rest[0].replace('"reason":"a"', '"reason":"z"') + rest[1],  # edited event
        rest[1] + rest[0],  # reordered
        rest[1],  # an event deleted
        rest[0].replace(":", ": ", 1) + rest[1],  # no longer canonical
    ]
    for text in tampered:
        with pytest.raises(ValueError, match="lifecycle") as refused:
            verify_lifecycle(genesis + text, entries)
        assert not isinstance(refused.value, LifecycleRewritePending)
    with pytest.raises(ValueError, match="hash chain"):
        verify_lifecycle("".join(rest), entries)  # the registered event deleted
    with pytest.raises(ValueError, match="newline"):
        verify_lifecycle(genesis.rstrip("\n"), entries)
    set_state(root, "momo", "suspended", reason="c", reviewer="cal", at=AT)
    text = (root / LIFECYCLE_FILE).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="lifecycle"):
        verify_lifecycle("".join(text.splitlines(keepends=True)[:3]), read_registry(root))
    assert len(verify_lifecycle(text, read_registry(root))) == 4


def test_events_are_checked_for_shape_and_for_an_impossible_sequence(root):
    registered(root)
    entries = read_registry(root)
    (genesis,) = lines(root)
    with pytest.raises(ValueError, match="not JSON"):
        verify_lifecycle("nope\n", entries)
    with pytest.raises(ValueError, match="registered or state event"):
        verify_lifecycle('{"field":"role"}\n', entries)
    first = json.loads(genesis)
    chained = canonical({**first, "prev_hash": digest(first)}) + chr(10)
    with pytest.raises(ValueError, match="registered after events of its own"):
        verify_lifecycle(genesis + chained, entries)
    set_state(root, "momo", "retired", reason="r", reviewer="cal", at=AT)
    for state in ("suspended", "active"):
        forged = lines(root)[1].replace('"to":"retired"', f'"to":"{state}"')
        with pytest.raises(ValueError, match="lifecycle"):
            verify_lifecycle(genesis + forged, entries)


def test_a_hand_edited_role_or_version_is_refused(root):
    registered(root, role=INCUBATION_ONLY, version="v1")
    path = root / REGISTRY_FILE
    clean = path.read_bytes()
    path.write_bytes(clean.replace(b'    role: "incubation-only"\n', b""))
    with pytest.raises(ValueError, match="role"):
        read_registry(root)
    path.write_bytes(clean.replace(b'version: "v1"', b'version: "v2"'))
    with pytest.raises(ValueError, match="module or version"):
        read_registry(root)
    path.write_bytes(clean)
    assert read_registry(root)[0].role == INCUBATION_ONLY
    path.write_bytes(b"# registry\nstrategies: []\n")  # an event with no entry
    with pytest.raises(LifecycleRewritePending, match="momo"):  # the crash-recoverable shape
        read_registry(root)


@pytest.fixture
def crash(monkeypatch):
    """`crash(True)` arms the injected fault between an event and its entry."""

    def arm(on: bool):
        if on:
            monkeypatch.setenv("CAPSTONE_FAULT_INJECTION", "1")
            monkeypatch.setenv("CAPSTONE_FAULT_AT", "registry:after_event")
        else:
            monkeypatch.delenv("CAPSTONE_FAULT_INJECTION", raising=False)
        _fault_plan.cache_clear()
        _fault_hits.clear()

    yield arm
    arm(False)


def test_a_crash_between_event_and_entry_is_pending_then_completed(root, crash):
    registered(root)
    before = len(lines(root))
    call = {"root": root, "strategy_id": "momo", "state": "suspended"}
    crash(True)
    with pytest.raises(InjectedFault):
        set_state(**call, reason="r", reviewer="cal", at=AT)
    with pytest.raises(LifecycleRewritePending, match="lifecycle_entry_rewrite_pending:momo"):
        read_registry(root)
    other = {**call, "state": "retired"}  # any other call is refused until the same one completes
    with pytest.raises(ValueError, match="lifecycle_entry_rewrite_pending"):
        set_state(**other, reason="other", reviewer="cal", at=AT)
    crash(False)
    entry = set_state(**call, reason="r", reviewer="cal", at=AT)
    assert read_registry(root) == (entry,)
    assert len(lines(root)) == before + 1  # completed without a second event


def test_a_crash_while_registering_is_completed_by_the_same_call(root, crash):
    crash(True)
    with pytest.raises(InjectedFault):
        registered(root)
    with pytest.raises(LifecycleRewritePending):
        read_registry(root)
    crash(False)
    entry = registered(root)
    assert read_registry(root) == (entry,) and len(lines(root)) == 1


def test_a_default_registration_round_trips_with_the_registry_header(root):
    registered(root)
    text = (root / REGISTRY_FILE).read_text(encoding="utf-8")
    assert text.startswith("# registry\nstrategies:\n")
    assert parse_registry(text) == read_registry(root)
