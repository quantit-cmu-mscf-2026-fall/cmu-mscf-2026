"""Registry-driven retire / suspend / promote: synthetic registries and fake specs only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from capstone.pipeline import lifecycle_actions as la
from capstone.strategy.registry import (
    LIFECYCLE_FILE,
    MODULE,
    REGISTRY_FILE,
    SAVED_PORTFOLIO_MODULE,
    SPEC_DIR,
    read_registry,
)


@dataclass(frozen=True)
class FakeMember:
    id: str


@dataclass(frozen=True)
class FakePortfolio:
    id: str
    specs: tuple
    membership: tuple = ()
    schedule: tuple = ()
    chronological: bool = True

    def to_toml(self):
        return f"portfolio {self.id} {json.dumps(list(self.membership))}"


def registry_text(entries):
    body = "".join(
        f'  - id: "{sid}"\n    module: "{module}"\n    gate_report: "r.json"\n'
        "    enabled: false\n" + (f'    role: "{role}"\n' if role else "")
        for sid, module, role in entries
    )
    return "strategies:\n" + body


def make_root(tmp_path, entries=(("alpha", MODULE, ""), ("book", SAVED_PORTFOLIO_MODULE, ""))):
    (tmp_path / REGISTRY_FILE).parent.mkdir(parents=True)
    (tmp_path / REGISTRY_FILE).write_text(registry_text(entries), encoding="utf-8")
    (tmp_path / SPEC_DIR).mkdir(parents=True)
    return tmp_path


def args(root, sid="alpha", **kwargs):
    base = {
        "registry": str(root),
        "id": sid,
        "reason": "synthetic",
        "reviewer": "reviewer-a",
        "at": "2020-03-02",
        "run": None,
        "resume": False,
        "confirm_record": None,
    }
    return SimpleNamespace(**(base | kwargs))


@pytest.fixture
def portfolios(monkeypatch):
    held = {"book": FakePortfolio("book", (FakeMember("alpha"), FakeMember("beta")))}
    monkeypatch.setattr(la, "registry_spec", lambda root, sid, module: held[sid])
    return held


def state_of(root, sid):
    return next(e.state for e in read_registry(root) if e.id == sid)


def test_suspend_then_resume_keeps_history(tmp_path):
    root = make_root(tmp_path)
    assert la.suspend(args(root))["state"] == "suspended"
    assert state_of(root, "alpha") == "suspended"
    result = la.suspend(args(root, resume=True, at="2020-03-03"))
    assert (result["command"], result["state"]) == ("resume", "active")
    events = (root / LIFECYCLE_FILE).read_text(encoding="utf-8").splitlines()
    assert len(events) == 2
    assert result["lifecycle_head"]


def test_an_unregistered_strategy_is_refused_and_nothing_is_written(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(la.Refused, match="not registered"):
        la.suspend(args(root, sid="ghost"))
    assert not (root / LIFECYCLE_FILE).exists()


def test_retire_records_a_departure_in_every_holding_portfolio(tmp_path, portfolios):
    root = make_root(tmp_path)
    result = la.retire(args(root))
    assert result["departures"] == ["book"]
    assert state_of(root, "alpha") == "retired"
    written = (root / SPEC_DIR / "book.toml").read_text(encoding="utf-8")
    assert '"departure"' in written and '"registry_retired"' in written
    assert '"2020-03-02"' in written


def test_retire_is_terminal_but_completes_unwritten_departures(tmp_path, portfolios):
    root = make_root(tmp_path)
    la.retire(args(root))
    # the first call's departure is treated as lost (a crash), so repeating completes it
    assert la.retire(args(root))["departures"] == ["book"]
    portfolios["book"] = FakePortfolio(
        "book",
        portfolios["book"].specs,
        membership=({"kind": "departure", "strategy_id": "alpha"},),
    )
    with pytest.raises(la.Refused, match="retired is terminal"):
        la.retire(args(root))


def test_a_fixed_pool_portfolio_is_refused_before_any_write(tmp_path, portfolios):
    root = make_root(tmp_path)
    portfolios["book"] = FakePortfolio("book", portfolios["book"].specs, chronological=False)
    with pytest.raises(la.Refused, match="fixed pool"):
        la.retire(args(root))
    assert state_of(root, "alpha") == "active"
    assert not (root / LIFECYCLE_FILE).exists()


def test_a_departure_cannot_precede_a_recorded_decision(tmp_path, portfolios):
    root = make_root(tmp_path)
    portfolios["book"] = FakePortfolio(
        "book", portfolios["book"].specs, schedule=({"decision_session": "2021-01-04"},)
    )
    with pytest.raises(la.Refused, match="precedes"):
        la.retire(args(root))
    assert state_of(root, "alpha") == "active"


def test_a_bad_timestamp_is_refused(tmp_path, portfolios):
    root = make_root(tmp_path)
    with pytest.raises(la.Refused, match="not a timestamp"):
        la.retire(args(root, at="not-a-date"))


def test_actions_are_logged_under_run(tmp_path):
    root = make_root(tmp_path)
    run = tmp_path / "run"
    la.suspend(args(root, run=str(run)))
    rows = (run / la.ACTIONS).read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[0])["command"] == "suspend"


def test_promote_needs_an_explicit_confirm_record(tmp_path):
    root = make_root(tmp_path, (("inc", MODULE, "incubation-only"),))
    with pytest.raises(la.Refused, match="needs a confirm record"):
        la.promote_entry(args(root, sid="inc"))
    with pytest.raises(la.Refused, match="missing confirm record"):
        la.promote_entry(args(root, sid="inc", confirm_record=tmp_path / "none.json"))
    assert read_registry(root)[0].role == "incubation-only"


def test_promote_reads_a_monitor_entry_and_hands_it_to_the_registry(tmp_path, monkeypatch):
    root = make_root(tmp_path, (("inc", MODULE, "incubation-only"),))
    store = tmp_path / "run" / la.MONITOR
    store.mkdir(parents=True)
    entry = {"strategy_id": "inc", "role": "incubation-only", "state": {"e": 1}}
    (store / "inc.json").write_text(json.dumps(entry), encoding="utf-8")
    seen = {}

    def fake_promote(root, sid, *, confirm_record, reviewer, at):
        seen.update(confirm_record=confirm_record, reviewer=reviewer)
        return SimpleNamespace(role="deploy")

    monkeypatch.setattr(la, "promote", fake_promote)
    result = la.promote_entry(args(root, sid="inc", run=str(tmp_path / "run")))
    assert seen["confirm_record"] == {"strategy_id": "inc", "confirm_state": {"e": 1}}
    assert result["role"] == "deploy"


def test_only_an_incubation_monitor_promotes(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps({"strategy_id": "x", "role": "deploy", "state": {}}), encoding="utf-8"
    )
    with pytest.raises(la.Refused, match="only an incubation-only"):
        la.confirm_record(path, "x")


def test_recommendations_are_listed_never_acted_on(tmp_path):
    store = tmp_path / la.MONITOR
    store.mkdir()
    for sid, decision in (("a", "retire"), ("b", "promote"), ("c", "hold")):
        document = {"strategy_id": sid, "role": "deploy", "state": {"decision": decision}}
        (store / f"{sid}.json").write_text(json.dumps(document), encoding="utf-8")
    rows = la.recommendations(tmp_path)
    assert [(r["strategy_id"], r["decision"]) for r in rows] == [("a", "retire"), ("b", "promote")]
    assert "--id a" in rows[0]["action"]
    assert la.recommendations(tmp_path / "absent") == []


def test_close_states_lists_non_active_strategies_of_a_bundle(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    la.suspend(args(root))
    monkeypatch.setattr(la, "SavedProgramPortfolioSpec", FakePortfolio)
    book = FakePortfolio("book", (FakeMember("alpha"),))
    monkeypatch.setattr(la, "load_bundle", lambda path: (None, None, [book], None))
    assert la.close_states(root, "bundle") == {"alpha": "suspended"}
    monkeypatch.setattr(la, "load_bundle", lambda path: (None, None, [FakeMember("zzz")], None))
    with pytest.raises(la.Refused, match="no retired or suspended"):
        la.close_states(root, "bundle")


def test_nothing_acts_on_import_or_without_args():
    assert "def main" not in Path(la.__file__).read_text(encoding="utf-8")
    with pytest.raises(AttributeError):
        la.retire(SimpleNamespace())
