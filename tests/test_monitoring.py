"""Journal-attributed realized returns and the durable monitor store.

Synthetic only: each journal is a few hand-written plans and fills in a temporary SQLite
file (`verify_plan` is replaced by the identity, so the plans need only the fields
monitoring reads). No data, no model, no broker, no financial claim.
"""

import json
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from capstone import faults
from capstone.execution.bundle import ACCOUNT_EVENTS_VERSION
from capstone.pipeline import monitoring

DAYS = ("2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07")


@pytest.fixture(autouse=True)
def plain_plans(monkeypatch):
    monkeypatch.setattr(monitoring, "verify_plan", lambda plan: plan)


def plan(plan_id, session, positions, cash, prices, attribution, **extra):
    return {
        "plan_id": plan_id,
        "decision_session": session,
        "account_positions": {str(n): q for n, q in positions.items()},
        "account_cash": cash,
        "account_receivables": [],
        "reference_prices": {str(n): p for n, p in prices.items()},
        "attribution": {"strategies": attribution},
        "allocation": {"weights": {"a": 0.5, "b": 0.5}},
    } | extra


def journal(folder, entries, identity="offline-simulator/x"):
    """execution.sqlite under `folder` with (plan, [(permno, shares, price, commission)])."""
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / monitoring.JOURNAL)
    db.execute("CREATE TABLE identity (name TEXT)")
    db.execute("CREATE TABLE plans (id TEXT PRIMARY KEY, document TEXT)")
    db.execute("CREATE TABLE instructions (id INTEGER PRIMARY KEY, plan TEXT, payload TEXT)")
    db.execute(
        "CREATE TABLE executions (id INTEGER PRIMARY KEY, instruction INTEGER, "
        "shares REAL, price REAL, commission REAL)"
    )
    db.execute("INSERT INTO identity VALUES (?)", (identity,))
    for one, fills in entries:
        db.execute("INSERT INTO plans VALUES (?, ?)", (one["plan_id"], json.dumps(one)))
        for permno, shares, price, commission in fills:
            cursor = db.execute(
                "INSERT INTO instructions (plan, payload) VALUES (?, ?)",
                (one["plan_id"], json.dumps({"permno": permno})),
            )
            db.execute(
                "INSERT INTO executions (instruction, shares, price, commission) "
                "VALUES (?, ?, ?, ?)",
                (cursor.lastrowid, shares, price, commission),
            )
    db.commit()
    db.close()
    return folder


def two_names(root, third=None):
    """Plan p1 buys 10 of name 1 and 5 of name 2 (fee 1 each); p2 holds them at new marks."""
    books = {"a": {"1": 0.5}, "b": {"2": 0.5}}
    first = plan("p1", DAYS[0], {}, 1000.0, {1: 10.0, 2: 20.0}, books)
    second = plan("p2", DAYS[1], {1: 10, 2: 5}, 798.0, {1: 11.0, 2: 19.0}, books)
    entries = [(first, [(1, 10, 10.0, 1.0), (2, 5, 20.0, 1.0)]), (second, [])]
    return journal(root, entries + ([third] if third else []))


def test_attribution_sums_to_account_return(tmp_path):
    records, pending = monitoring.session_returns(two_names(tmp_path))
    assert pending == "p2" and len(records) == 1
    (record,) = records
    assert record["tags"] == ["synthetic"] and record["session"] == DAYS[1]
    assert math.isclose(record["account_return"], 3 / 1000)
    parts = record["strategies"]
    assert math.isclose(parts["a"]["contribution"], 9 / 1000)  # +10 on name 1, less 1 of fees
    assert math.isclose(parts["b"]["contribution"], -6 / 1000)  # -5 on name 2, less 1 of fees
    assert math.isclose(sum(p["contribution"] for p in parts.values()), record["account_return"])
    assert math.isclose(parts["a"]["return"], 0.018) and math.isclose(parts["b"]["return"], -0.012)
    assert record["fees"] == 2.0 and record["fallback_names"] == []


def test_a_journal_that_is_not_the_simulator_is_tagged_real(tmp_path):
    books = {"a": {"1": 1.0}}
    only = plan("p1", DAYS[0], {}, 100.0, {1: 10.0}, books)
    after = plan("p2", DAYS[1], {}, 100.0, {1: 10.0}, books)
    journal(tmp_path, [(only, []), (after, [])], identity="recorded-journal")
    (record,), _ = monitoring.session_returns(tmp_path)
    assert record["tags"] == ["real"]


def test_unexplained_cash_and_positions_are_refused(tmp_path):
    books = {"a": {"1": 1.0}}
    first = plan("p1", DAYS[0], {}, 1000.0, {1: 10.0}, books)
    off = plan("p2", DAYS[1], {1: 10}, 899.0, {1: 10.0}, books)  # 1000 - 100 is 900
    journal(tmp_path / "cash", [(first, [(1, 10, 10.0, 0.0)]), (off, [])])
    with pytest.raises(ValueError, match=r"journal does not chain: .* cash differs by \$-1\.00"):
        monitoring.session_returns(tmp_path / "cash")
    short = plan("p2", DAYS[1], {1: 9}, 900.0, {1: 10.0}, books)
    journal(tmp_path / "shares", [(first, [(1, 10, 10.0, 0.0)]), (short, [])])
    with pytest.raises(ValueError, match="positions plus its fills differ"):
        monitoring.session_returns(tmp_path / "shares")


def test_receivables_settle_into_cash_and_accrue_as_income(tmp_path):
    books = {"a": {"1": 1.0}}
    due = {"event_id": "dividend-1", "amount": 12.5}
    first = plan("p1", DAYS[0], {1: 10}, 0.0, {1: 10.0}, books, account_receivables=[])
    second = plan("p2", DAYS[1], {1: 10}, 0.0, {1: 10.0}, books, account_receivables=[due])
    third = plan("p3", DAYS[2], {1: 10}, 12.5, {1: 10.0}, books)
    journal(tmp_path, [(first, []), (second, []), (third, [])])
    first_record, second_record = monitoring.session_returns(tmp_path)[0]
    assert first_record["cash_events"] == {"settled": {}, "accrued": {"dividend-1": 12.5}}
    assert second_record["cash_events"] == {"settled": {"dividend-1": 12.5}, "accrued": {}}
    # Accrued income is a return; the later settlement only moves it into cash.
    assert math.isclose(first_record["account_return"], 12.5 / 100)
    assert second_record["account_return"] == pytest.approx(0.0, abs=1e-12)
    assert first_record["fallback_names"] == ["receivable:dividend-1"]  # listed, as income
    assert second_record["fallback_names"] == []
    changed = plan(
        "p3", DAYS[2], {1: 10}, 0.0, {1: 10.0}, books, account_receivables=[due | {"amount": 9.0}]
    )
    journal(tmp_path / "changed", [(first, []), (second, []), (changed, [])])
    with pytest.raises(ValueError, match="receivable amounts changed"):
        monitoring.session_returns(tmp_path / "changed")


def test_a_split_gives_whole_shares_to_the_holders_and_the_marks_follow(tmp_path):
    books = {"a": {"1": 1.0}}
    split = {
        "version": ACCOUNT_EVENTS_VERSION,
        "events": [
            {
                "event_id": "split-1",
                "permno": 1,
                "kind": "split",
                "ratio": 2.0,
                "cash": 0.0,
                "effective_session": DAYS[2],
            }
        ],
    }
    first = plan("p1", DAYS[0], {1: 10}, 0.0, {1: 10.0}, books)
    second = plan("p2", DAYS[1], {1: 10}, 0.0, {1: 10.0}, books)
    third = plan("p3", DAYS[2], {1: 20}, 0.0, {1: 5.0}, books, account_corporate_events=split)
    journal(tmp_path, [(first, []), (second, []), (third, [])])
    first_record, second_record = monitoring.session_returns(tmp_path)[0]
    assert "corporate_events" not in first_record and first_record["account_return"] == 0
    assert second_record["corporate_events"] == split["events"]
    assert second_record["account_return"] == pytest.approx(0.0, abs=1e-12)  # 20 x 5 = 10 x 10
    late = split["events"][0] | {"effective_session": "2021-02-01"}
    bad = third | {"account_corporate_events": {**split, "events": [late]}}
    journal(tmp_path / "late", [(first, []), (second, []), (bad, [])])
    with pytest.raises(ValueError, match="effective after its plan"):
        monitoring.session_returns(tmp_path / "late")
    with pytest.raises(ValueError, match="unknown corporate event record"):
        monitoring.corporate_events(
            second, {**third, "account_corporate_events": {**split, "version": "v0"}}
        )


def test_two_plans_in_one_session_are_refused(tmp_path):
    books = {"a": {"1": 1.0}}
    same = [plan(f"p{i}", DAYS[0], {}, 1.0, {1: 1.0}, books) for i in (1, 2)]
    journal(tmp_path, [(one, []) for one in same])
    with pytest.raises(ValueError, match="share a decision session"):
        monitoring.session_returns(tmp_path)
    journal(tmp_path / "again", [(same[0], [])])
    journal(tmp_path / "again" / "copy", [(same[0], [])])
    with pytest.raises(ValueError, match="journaled more than once"):
        monitoring.read_journals(tmp_path / "again")


def toy(plan_id, session, positions, prices, attribution, cash=1000.0):
    return plan(plan_id, session, positions, cash, prices, attribution)


def test_carried_names_use_earlier_attribution_then_flagged_equal_split():
    earlier = toy("p0", "2021-01-04", {}, {7: 10.0}, {"a": {"7": 0.4}, "b": {"8": 0.4}})
    held, prices = {7: 10, 8: 10, 9: 10}, {7: 10.0, 8: 10.0, 9: 10.0}
    current = toy("p1", "2021-01-05", held, prices, {"a": {"10": 0.3}, "b": {"10": 0.2}})
    after = toy("p2", "2021-01-06", held, {7: 11.0, 8: 12.0, 9: 13.0}, {})
    record = monitoring.attribute(current, [], after, [earlier], "synthetic")
    parts = {s: v["contribution"] * record["equity"] for s, v in record["strategies"].items()}
    # 7 last attributed to a (+10), 8 to b (+20); 9 never attributed: equal split of +30.
    assert parts == pytest.approx({"a": 10 + 15, "b": 20 + 15})
    assert record["fallback_names"] == ["9"]
    (event,) = monitoring.observations([record], "a")
    assert event["attribution_fallback"] is True and event["fallback_names"] == ["9"]
    only = {"account_positions": {"7": 10}}
    clean = monitoring.attribute(current | only, [], after | only, [earlier], "synthetic")
    assert clean["fallback_names"] == [] and set(clean["strategies"]) == {"a", "b"}
    assert clean["strategies"]["a"]["contribution"] * clean["equity"] == pytest.approx(10)
    (event,) = monitoring.observations([clean], "a")
    assert event["attribution_fallback"] is False


def test_a_name_nobody_ever_attributed_has_no_owner():
    with pytest.raises(ValueError, match="no strategy to attribute"):
        monitoring.split(5, [{}, {}])


def planted(values, start="2021-01-04"):
    dates = pd.bdate_range(start, periods=len(values))
    return [
        {"plan_id": f"p{i}", "session": str(d.date()), "return": float(r), "tags": ["synthetic"]}
        for i, (d, r) in enumerate(zip(dates, values, strict=True))
    ]


def outcome(store, strategy, role, backtest, live, horizon):
    assumptions, state = monitoring.start(
        pd.Series(backtest), role=role, alpha=0.05, horizon=horizon, source="planted"
    )
    events = monitoring.record(store, strategy, role, assumptions, state, planted(live))
    return events[-1]


def test_role_specific_outcomes(tmp_path):
    rng = np.random.default_rng(11)
    store = tmp_path / "store"
    edge = rng.normal(0.001, 0.01, 500)
    null = rng.normal(0.0, 0.01, 500)
    wobble = [0.005, -0.005] * 5
    # Deploy: a planted decay retires; an intact edge to the horizon is kept; never promotes.
    decayed = outcome(store, "deploy-decay", "deploy", edge, [-0.02] * 60, 60)
    assert decayed["decision"] == "retire" and decayed["n_obs"] < 60
    kept = outcome(store, "deploy-null", "deploy", edge, [0.001 + w for w in wobble], 10)
    assert kept["decision"] == "keep" and kept["n_obs"] == 10
    # Incubation-only: a planted edge is promoted (reported only); no edge retires at horizon.
    promoted = outcome(store, "inc-edge", "incubation-only", null, [0.01] * 80, 80)
    assert promoted["decision"] == "promote" and promoted["reason"] == "confirmed"
    assert promoted["confirm_e_value"] >= 20 and promoted["n_obs"] < 80
    ended = outcome(store, "inc-null", "incubation-only", null, wobble, 10)
    assert ended["decision"] == "retire" and ended["reason"] == "not confirmed"
    lines = (store / monitoring.EVENTS).read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["strategy_id"] for line in lines} == {
        "deploy-decay",
        "deploy-null",
        "inc-edge",
        "inc-null",
    }
    # A final decision is sticky: later sessions are not consumed.
    again = monitoring.record(
        store,
        "deploy-decay",
        "deploy",
        *monitoring.start(pd.Series(edge), role="deploy", alpha=0.05, horizon=60, source="planted"),
        planted([0.0] * 3, start="2022-01-03"),
    )
    assert again == []


def test_start_refuses_an_unknown_role():
    with pytest.raises(ValueError, match="role must be one of"):
        monitoring.start(pd.Series([0.01, -0.01] * 50), role="plumbing-test", **ARGS)


ARGS = {"alpha": 0.05, "horizon": 10, "source": "x"}


def store_files(store):
    return {p.name: p.read_bytes() for p in sorted(store.iterdir())}


def returns_frame(seed=3):
    values = np.random.default_rng(seed).normal(0.0005, 0.01, (300, 2))
    return pd.DataFrame(values, index=pd.bdate_range("2019-01-01", periods=300), columns=["a", "b"])


def test_run_updates_the_store_once_per_session_and_refuses_changed_assumptions(tmp_path):
    third = (plan("p3", DAYS[2], {1: 10, 2: 5}, 798.0, {1: 12.0, 2: 19.0}, {"a": {"1": 0.5}}), [])
    root = two_names(tmp_path / "run", third)
    frame, store = returns_frame(), tmp_path / "store"
    summary = monitoring.run(root, store, frame, horizon=20)
    assert summary["new_events"] == 4 and summary["pending_plan"] == "p3"
    assert set(summary["strategies"]) == {"a", "b"}
    assert summary["strategies"]["a"]["n_obs"] == 2 and summary["strategies"]["b"]["n_obs"] == 2
    before = store_files(store)
    assert monitoring.run(root, store, frame, horizon=20)["new_events"] == 0
    assert store_files(store) == before
    with pytest.raises(ValueError, match="monitor assumptions changed"):
        monitoring.run(root, store, frame, horizon=20, alpha=0.1)
    with pytest.raises(ValueError, match="no base-scenario funded returns for strategy b"):
        monitoring.run(root, tmp_path / "other", frame[["a"]])
    # Events are session-major: both strategies' first session, then both strategies' second.
    keys = [
        (e["session"], e["strategy_id"])
        for e in map(json.loads, (store / monitoring.EVENTS).read_text().splitlines())
    ]
    assert keys == [(DAYS[1], "a"), (DAYS[1], "b"), (DAYS[2], "a"), (DAYS[2], "b")]


@pytest.fixture
def fault(monkeypatch):
    def arm(point):
        monkeypatch.setenv("CAPSTONE_FAULT_INJECTION", "1")
        monkeypatch.setenv("CAPSTONE_FAULT_AT", point)
        faults._fault_plan.cache_clear()
        faults._fault_hits.clear()

    yield arm
    monkeypatch.delenv("CAPSTONE_FAULT_INJECTION", raising=False)
    faults._fault_plan.cache_clear()
    faults._fault_hits.clear()


def test_crash_between_event_append_and_state_write_resumes_exactly(tmp_path, fault):
    third = (plan("p3", DAYS[2], {1: 10, 2: 5}, 798.0, {1: 12.0, 2: 19.0}, {"a": {"1": 0.5}}), [])
    root = two_names(tmp_path / "run", third)
    frame = returns_frame()
    monitoring.run(root, tmp_path / "continuous", frame)
    fault("monitor:after_event:2")
    with pytest.raises(faults.InjectedFault, match="monitor:after_event"):
        monitoring.run(root, tmp_path / "crashed", frame)
    store = tmp_path / "crashed"
    lines = (store / monitoring.EVENTS).read_text(encoding="utf-8").splitlines()
    state = json.loads((store / "a.json").read_text(encoding="utf-8"))
    assert len(lines) == 2 and len(state["applied"]) == 1  # the crash window
    fault("")  # disarm
    monitoring.run(root, store, frame)
    assert store_files(store) == store_files(tmp_path / "continuous")


def test_a_torn_last_event_line_is_cut(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    whole = json.dumps({"strategy_id": "a", "plan_id": "p1", "session": DAYS[1]})
    (store / monitoring.EVENTS).write_bytes(f'{whole}\n{{"strategy_id": "a", "pla'.encode())
    assert monitoring.logged(store) == {("a", "p1", DAYS[1])}
    assert (store / monitoring.EVENTS).read_bytes() == f"{whole}\n".encode()
    assert monitoring.logged(tmp_path / "nowhere") == set()


def test_out_of_order_sessions_are_refused(tmp_path):
    assumptions, state = monitoring.start(
        pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 200)), role="deploy", **ARGS
    )
    store = tmp_path / "store"
    monitoring.record(store, "a", "deploy", assumptions, state, planted([0.001], "2021-01-05"))
    with pytest.raises(ValueError, match="out of order"):
        monitoring.record(store, "a", "deploy", assumptions, state, planted([0.001], "2021-01-04"))


def test_the_frame_digest_follows_values_and_gaps():
    frame = returns_frame()
    assert monitoring.frame_digest(frame) == monitoring.frame_digest(frame.copy())
    assert monitoring.frame_digest(frame) != monitoring.frame_digest(frame * 1.0001)
    gap = frame.copy()
    gap.iloc[0, 0] = np.nan
    zero = frame.copy()
    zero.iloc[0, 0] = 0.0
    assert monitoring.frame_digest(gap) != monitoring.frame_digest(zero)
