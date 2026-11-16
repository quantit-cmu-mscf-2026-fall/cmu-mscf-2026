"""`register` and `register_version`: a spec's file and entry, and the `registered` event.

Synthetic specs and fake spec modules in a temp folder; nothing is written outside it.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from capstone.faults import InjectedFault, _fault_hits, _fault_plan
from capstone.strategy import registry
from capstone.strategy.registry import (
    INCUBATION_ONLY,
    LIFECYCLE_FILE,
    MODULE,
    PLUMBING_TEST,
    PROGRAM_MODULE,
    REGISTRY_FILE,
    SPEC_DIR,
    LifecycleRewritePending,
    read_lifecycle,
    read_registry,
    register,
    register_version,
    set_state,
)

AT = "2026-11-09T00:00:00Z"
REPORT = "reports/gate.json"


@dataclass(frozen=True)
class ContractSpec:
    """Stands in for a legacy contract spec (its class module is not a program one)."""

    id: str
    gate_report: str = REPORT

    @classmethod
    def from_toml(cls, text):
        return cls(**json.loads(text))


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    anchor: str = "2010-01-01"
    data_version: str = "v1"
    gate_report: str = REPORT

    @property
    def id(self) -> str:
        key = f"{self.name}|{self.anchor}|{self.data_version}"
        return "prog-" + hashlib.sha256(key.encode()).hexdigest()[:8]

    def to_toml(self) -> str:
        return json.dumps({"name": self.name, "anchor": self.anchor, "dv": self.data_version})

    @classmethod
    def from_toml(cls, text):
        raw = json.loads(text)
        return cls(raw["name"], raw["anchor"], raw["dv"])


ProgramSpec.__module__ = PROGRAM_MODULE  # `register` reads the module off the spec's class


def save_spec(spec, path):
    path.write_text(json.dumps({"id": spec.id}), encoding="utf-8")


def load_spec(path):
    return ContractSpec(**json.loads(Path(path).read_text(encoding="utf-8")))


@pytest.fixture
def modules(monkeypatch):
    contract = types.ModuleType(MODULE)
    contract.save_spec, contract.load_spec = save_spec, load_spec
    program = types.ModuleType(PROGRAM_MODULE)
    program.ProgramStrategySpec = ProgramSpec
    gate = types.ModuleType("capstone.program_gate")
    gate.check_schema = lambda report: None
    for module in (contract, program, gate):
        monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def root(tmp_path: Path, modules) -> Path:
    (tmp_path / "deploy").mkdir()
    (tmp_path / REGISTRY_FILE).write_bytes(b"# registry\nstrategies: []\n")
    (tmp_path / "reports").mkdir()
    write_report(tmp_path, ["whatever"])
    return tmp_path


def write_report(root: Path, ids, **over) -> None:
    report = {
        "schema": "program-gate-report-v1",
        "procedure_id": "proc",
        "strategy_ids": ids,
        "holdout_opened": False,
        **over,
    }
    (root / REPORT).write_text(json.dumps(report), encoding="utf-8")


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_register_writes_spec_entry_and_event_and_is_inert(root):
    entry = register(ContractSpec("momo"), root, reviewer="reviewer-a", at=AT)
    assert (entry.id, entry.module, entry.enabled, entry.role) == ("momo", MODULE, False, "deploy")
    assert read_registry(root) == (entry,)
    assert (root / SPEC_DIR / "momo.toml").exists()
    (event,) = read_lifecycle(root)
    assert event["field"] == "registered"
    assert (event["role"], event["state"], event["reviewer"], event["at"]) == (
        "deploy",
        "active",
        "reviewer-a",
        AT,
    )
    # everything written is under the given root, in the deploy folder
    assert {p.split("/")[0] for p in snapshot(root)} <= {"deploy", "reports"}


def test_registering_twice_changes_no_byte_and_keeps_enabled_and_role(root):
    register(ContractSpec("momo"), root, role=INCUBATION_ONLY, at=AT)
    path = root / REGISTRY_FILE
    path.write_bytes(path.read_bytes().replace(b"enabled: false", b"enabled: true"))
    before = snapshot(root)
    again = register(ContractSpec("momo"), root, at="later")  # default role, no flip
    assert (again.enabled, again.role) == (True, INCUBATION_ONLY)
    assert snapshot(root) == before


def test_a_gate_report_is_required_and_must_agree(root):
    with pytest.raises(ValueError, match="no gate report"):
        register(ContractSpec("a", gate_report=""), root)
    with pytest.raises(ValueError, match="names gate report"):
        register(ContractSpec("a"), root, gate_report="other.json")
    assert register(ContractSpec("a", gate_report=""), root, gate_report="x.json").gate_report == (
        "x.json"
    )


def test_a_suspended_or_retired_id_is_refused(root):
    register(ContractSpec("a"), root, at=AT)
    set_state(root, "a", "suspended", reason="r", reviewer="x", at=AT)
    with pytest.raises(ValueError, match="registering never changes a state"):
        register(ContractSpec("a"), root)
    set_state(root, "a", "retired", reason="r", reviewer="x", at=AT)
    with pytest.raises(ValueError, match="retired"):
        register(ContractSpec("a"), root)


def test_a_spec_file_that_holds_another_id_is_refused(root):
    register(ContractSpec("a"), root)
    (root / SPEC_DIR / "a.toml").write_text(json.dumps({"id": "b"}), encoding="utf-8")
    with pytest.raises(ValueError, match="holds spec id b"):
        register(ContractSpec("a"), root)


def test_non_string_reviewer_or_at_is_refused(root):
    with pytest.raises(ValueError, match="must be strings"):
        register(ContractSpec("a"), root, at=None)  # type: ignore[arg-type]


def test_a_program_needs_at_and_a_gate_report_naming_it(root):
    spec = ProgramSpec("alpha")
    with pytest.raises(ValueError, match="needs a timestamp"):
        register(spec, root)
    with pytest.raises(ValueError, match="is not a JSON object"):
        register(replace(spec, gate_report=""), root, gate_report="reports/missing.json", at=AT)
    with pytest.raises(ValueError, match="does not name"):
        register(spec, root, at=AT)  # the report names "whatever"
    write_report(root, [spec.id], schema="wrong")
    with pytest.raises(ValueError, match="has schema 'wrong'"):
        register(spec, root, at=AT)
    write_report(root, [spec.id])
    entry = register(spec, root, at=AT, reviewer="reviewer-a")
    assert (entry.module, entry.enabled) == (PROGRAM_MODULE, False)
    assert (root / SPEC_DIR / f"{spec.id}.toml").read_text(encoding="utf-8") == spec.to_toml()
    assert read_lifecycle(root)[0]["module"] == PROGRAM_MODULE


def test_a_plumbing_test_program_needs_no_gate_report(root):
    spec = ProgramSpec("plumb", gate_report="")
    entry = register(spec, root, role=PLUMBING_TEST, gate_report="none.json", at=AT)
    assert entry.role == PLUMBING_TEST


def test_a_program_entry_without_a_registered_event_is_refused(root):
    spec = ProgramSpec("alpha")
    write_report(root, [spec.id])
    register(spec, root, at=AT)
    (root / LIFECYCLE_FILE).unlink()
    with pytest.raises(ValueError, match="no registered event"):
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


def test_a_crash_after_the_event_is_completed_by_the_same_register_call(root, crash):
    crash(True)
    with pytest.raises(InjectedFault):
        register(ContractSpec("a"), root, at=AT)
    with pytest.raises(LifecycleRewritePending):
        read_registry(root)
    crash(False)
    entry = register(ContractSpec("a"), root, at=AT)
    assert read_registry(root) == (entry,)
    assert len(read_lifecycle(root)) == 1


def parent_and_child(root):
    parent = ProgramSpec("alpha")
    write_report(root, [parent.id])
    register(parent, root, at=AT)
    return parent, replace(parent, anchor="2015-01-01", data_version="v2")


def test_register_version_adds_an_incubation_only_child_of_the_parent(root):
    parent, child = parent_and_child(root)
    entry = register_version(child, root, parent_id=parent.id, at=AT, reviewer="reviewer-a")
    assert entry.id == child.id
    assert entry.role == INCUBATION_ONLY
    assert entry.version == f"reanchor-of:{parent.id}"
    assert entry.gate_report == REPORT
    assert entry.enabled is False
    assert registry.lineage_parent(entry.version) == parent.id
    # idempotent: a second call changes no byte
    before = snapshot(root)
    assert register_version(child, root, parent_id=parent.id, at="later", reviewer="r") == entry
    assert snapshot(root) == before


def test_register_version_refuses_a_non_parent_or_a_non_reanchoring(root):
    parent, child = parent_and_child(root)
    with pytest.raises(ValueError, match="has no parent"):
        register_version(child, root, parent_id="nope", at=AT, reviewer="r")
    with pytest.raises(ValueError, match="not a new program version"):
        register_version(ContractSpec("c"), root, parent_id=parent.id, at=AT, reviewer="r")
    with pytest.raises(ValueError, match="not a re-anchoring"):
        register_version(
            ProgramSpec("beta", anchor="2015-01-01"), root, parent_id=parent.id, at=AT, reviewer="r"
        )
    set_state(root, parent.id, "suspended", reason="r", reviewer="x", at=AT)
    with pytest.raises(ValueError, match="needs an active parent"):
        register_version(child, root, parent_id=parent.id, at=AT, reviewer="r")


def test_register_version_refuses_an_id_registered_otherwise(root):
    parent, child = parent_and_child(root)
    write_report(root, [child.id])
    register(child, root, at=AT)  # as a plain program, not as a version
    with pytest.raises(ValueError, match="but not as"):
        register_version(child, root, parent_id=parent.id, at=AT, reviewer="r")
