"""Journal-attributed realized returns and the durable incubation monitor.

Returns. Every ``execution.sqlite`` under a run root holds frozen plans and their fills.
Plans chain by decision session; session k is marked at plan k+1's reference prices.
Account value = cash + receivables + shares x price; per name, P&L = shares after x mark -
shares before x reference price - cash paid for fills + the name's corporate-event cash.
Receivables (``account_receivables``): gone from plan k+1 = settled (cash, no P&L); new in
plan k+1 = accrued (income). Corporate events (``account_corporate_events``, written by
`ActionBook.account_events`) effective after plan k's decision session apply in recorded
order, before plan k's fills if effective by its execution session: splits/conversions
give whole shares x ratio (of ``new_permno``), event cash adds to cash, delisting cash
retires the name. Carried positions and cash must equal plan k+1's (cash to $0.01), else
``journal does not chain`` with the unexplained amount.

Attribution. A name's P&L is split over the plan's ``attribution`` strategies (program
book, or coefficient x member book) by signed weight. Event cash and converted shares (at
the new permno's mark) belong to the source permno's holders, whom an unattributed new
permno follows. A carried, unattributed name goes to the strategies that last attributed
it, else equally; accrued income is split like the held book's value; both are listed
(``fallback_names``). Applied corporate events are listed with their ``basis``. Fees go by
attributed traded notional. Contributions over plan k's account value sum to its return;
the monitored return is contribution / allocation coefficient.

Monitor and store. `start` sets the monitor as `capstone.incubate.incubate` does; the
decisions are reported only. ``<store>/<strategy_id>.json`` is replaced atomically and
``<store>/events.jsonl`` appended first (once per strategy, plan_id, session), so a crash
between the two re-derives the same state and appends nothing twice.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import pandas as pd

from capstone.artifacts import atomic_json
from capstone.execution.bundle import ACCOUNT_EVENTS_VERSION
from capstone.execution.frozen import verify_plan
from capstone.faults import maybe_fault
from capstone.incubate import IncubationSettings, incubate
from capstone.incubation.confirm import ConfirmState
from capstone.incubation.monitor import MonitorConfig, MonitorState
from capstone.pipeline.actions import entitlement
from capstone.strategy.registry import INCUBATION_ONLY
from capstone.strategy.research import canonical, digest

JOURNAL = "execution.sqlite"
EVENTS = "events.jsonl"
SIMULATOR = "offline-simulator/"
ROLES = ("deploy", INCUBATION_ONLY)


def read_journals(root):
    """{plan_id: (plan, fills [(permno, shares, price, commission)], tag)} from every journal."""
    found = {}
    for path in sorted(Path(root).rglob(JOURNAL)):
        with closing(sqlite3.connect(path)) as db:
            identities = [row[0] for row in db.execute("SELECT name FROM identity")]
            tag = "synthetic" if all(i.startswith(SIMULATOR) for i in identities) else "real"
            for plan_id, document in db.execute("SELECT id, document FROM plans ORDER BY id"):
                if plan_id in found:
                    raise ValueError(f"plan {plan_id} is journaled more than once under {root}")
                fills = [
                    (json.loads(payload)["permno"], shares, price, commission)
                    for payload, shares, price, commission in db.execute(
                        "SELECT i.payload,e.shares,e.price,e.commission FROM executions e "
                        "JOIN instructions i ON i.id=e.instruction WHERE i.plan=? ORDER BY e.id",
                        (plan_id,),
                    )
                ]
                found[plan_id] = (verify_plan(json.loads(document)), fills, tag)
    return found


def split(name, attributions):
    """(fractions of one name's P&L per strategy, fell back to an equal split).

    `attributions` runs newest first: the plan's own, then earlier plans', so a name the
    plan carries without attributing it goes to the strategies that last attributed it.
    """
    for strategies in attributions:
        weights = {s: float(book.get(str(name), 0.0)) for s, book in strategies.items()}
        total = sum(weights.values())
        if abs(total) > 1e-12:
            return {s: w / total for s, w in weights.items() if w}, False
    owners = next((sorted(a) for a in attributions if a), None)
    if not owners:
        raise ValueError(f"no strategy to attribute P&L of {name} to")
    return {s: 1 / len(owners) for s in owners}, True


def receivables(plan):
    return {row["event_id"]: float(row["amount"]) for row in plan.get("account_receivables", [])}


def corporate_events(plan, after):
    """Plan k+1's account events effective after plan k's decision session, as recorded."""
    record = after.get("account_corporate_events")
    if not record:
        return []
    if record["version"] != ACCOUNT_EVENTS_VERSION:
        raise ValueError(f"unknown corporate event record {record['version']}")
    for late in (e for e in record["events"] if e["effective_session"] > after["decision_session"]):
        raise ValueError(f"corporate event {late['event_id']} is effective after its plan")
    return [e for e in record["events"] if e["effective_session"] > plan["decision_session"]]


def transform(events, position, received, moved):
    """Whole shares x ratio (fractions come as cash-in-lieu); delisting cash retires n."""
    for e in events:
        n = e["permno"]
        if e["kind"] == "delisting_cash":
            position.pop(n, None)
        if e["kind"] in ("split", "conversion"):
            whole = entitlement(position.pop(n, 0), e["ratio"])[0]
            m = e.get("new_permno", n)
            position[m] = position.get(m, 0) + whole
            moved += [(n, m, whole)] if m != n and whole else []
        else:
            received[n] = received.get(n, 0.0) + e["cash"]


def attribute(plan, fills, after, earlier, tag):
    """One session's return and contributions (module doc); `earlier`: newest first."""
    held = {int(n): q for n, q in plan["account_positions"].items()}
    reference = {int(n): float(p) for n, p in plan["reference_prices"].items()}
    mark = {int(n): float(p) for n, p in after["reference_prices"].items()}
    position, spent, traded, fees = dict(held), defaultdict(float), defaultdict(float), 0.0
    # Events on or before plan k's execution session precede its fills, as in the replay.
    events, received, moved = corporate_events(plan, after), {}, []
    execution = plan.get("execution_session", plan["decision_session"])
    transform([e for e in events if e["effective_session"] <= execution], position, received, moved)
    for name, shares, price, commission in fills:
        position[name] = position.get(name, 0) + shares
        spent[name] += shares * price
        traded[name] += abs(shares * price)
        fees += commission
    transform([e for e in events if e["effective_session"] > execution], position, received, moved)
    position = {n: q for n, q in position.items() if q}
    owed, due = receivables(plan), receivables(after)
    changed = sorted(e for e in set(owed) & set(due) if owed[e] != due[e])
    if changed:
        raise ValueError(f"journal does not chain: receivable amounts changed {changed}")
    settled = {e: a for e, a in owed.items() if e not in due}
    accrued = {e: a for e, a in due.items() if e not in owed}
    cash = plan["account_cash"] - sum(spent.values()) - fees + sum(settled.values())
    cash += sum(received.values())
    following = {int(n): q for n, q in after["account_positions"].items()}
    if position != following:
        raise ValueError(
            f"journal does not chain: plan {plan['plan_id']} positions plus its fills differ "
            f"from the account in plan {after['plan_id']}"
        )
    if not math.isclose(cash, after["account_cash"], abs_tol=0.01):
        raise ValueError(
            f"journal does not chain: plan {after['plan_id']} cash differs by "
            f"${after['account_cash'] - cash:.2f} unexplained by fills, commissions or "
            f"settled receivables{' or corporate events' if events else ''} of plan "
            f"{plan['plan_id']}"
        )
    names = sorted(set(held) | set(position) | set(received))
    pnl = {
        n: (position[n] * mark[n] if n in position else 0)
        - (held[n] * reference[n] if n in held else 0)
        - spent[n]
        + received.get(n, 0)
        for n in names
    }
    for n, m, whole in moved:  # converted shares, at their mark, go to the source's holders
        pnl[n], pnl[m] = pnl[n] + whole * mark[m], pnl[m] - whole * mark[m]
    alias = {m: n for n, m, _ in moved}
    equity = plan["account_cash"] + sum(owed.values())
    equity += sum(q * reference[n] for n, q in held.items())
    attributions = [p["attribution"]["strategies"] for p in [plan, *earlier]]
    fractions, fallbacks = {}, []
    for n in names:
        fractions[n], fell_back = split(n, attributions)
        if fell_back and n in alias:  # a converted-into name nobody attributes
            fractions[n], fell_back = split(alias[n], attributions)
        fallbacks += [str(n)] if fell_back else []
    # A receivable carries no permno: accrued income is split like the held book's value.
    exposure = {n: abs(q * reference[n]) for n, q in held.items()}
    income = defaultdict(float)
    for event, amount in sorted(accrued.items()):
        if not sum(exposure.values()):
            raise ValueError(f"receivable {event} accrued with no position held at plan k")
        for n, value in exposure.items():
            for s, f in fractions[n].items():
                income[s] += amount * f * value / sum(exposure.values())
        fallbacks.append(f"receivable:{event}")
    owners = sorted(set(attributions[0]) | {s for f in fractions.values() for s in f})
    notional = {s: sum(abs(fractions[n].get(s, 0)) * traded[n] for n in names) for s in owners}
    turnover = sum(notional.values())
    weights = plan["allocation"]["weights"]
    strategies = {}
    for s in owners:
        gross = sum(fractions[n].get(s, 0) * pnl[n] for n in names) + income[s]
        contribution = (gross - (fees * notional[s] / turnover if turnover else 0.0)) / equity
        coefficient = weights.get(s)
        held_return = contribution / coefficient if coefficient else None
        strategies[s] = {
            "coefficient": coefficient,
            "contribution": contribution,
            "return": held_return,
        }
    value = after["account_cash"] + sum(due.values())
    value += sum(q * mark[n] for n, q in position.items())
    return {
        "plan_id": plan["plan_id"],
        "decision_session": plan["decision_session"],
        "session": after["decision_session"],
        "tags": [tag],
        "equity": equity,
        "fees": fees,
        "cash_events": {"settled": settled, "accrued": accrued},
        "fallback_names": fallbacks,
        "account_return": (value - equity) / equity,
        "strategies": strategies,
    } | ({"corporate_events": events} if events else {})


def session_returns(root, earlier=()):
    """(attributed sessions in order, plan_id still waiting for its marks or None);
    `earlier`: attributions of plans before the root's first, newest first."""
    journals = read_journals(root)
    chain = sorted(journals.values(), key=lambda j: j[0]["decision_session"])
    sessions = [plan["decision_session"] for plan, _, _ in chain]
    if len(set(sessions)) != len(sessions):
        raise ValueError("two journaled plans share a decision session; keep one per session")
    records, earlier = [], list(earlier)
    for (plan, fills, tag), (after, _, _) in zip(chain, chain[1:], strict=False):
        records.append(attribute(plan, fills, after, earlier, tag))
        earlier = [plan, *earlier]
    return records, chain[-1][0]["plan_id"] if chain else None


def observations(records, strategy):
    """The monitor inputs of one strategy, in session order."""
    return [
        {
            "plan_id": r["plan_id"],
            "decision_session": r["decision_session"],
            "session": r["session"],
            "return": r["strategies"][strategy]["return"],
            "contribution": r["strategies"][strategy]["contribution"],
            "account_return": r["account_return"],
            "attribution_fallback": bool(r["fallback_names"]),
            "fallback_names": r["fallback_names"],
            "tags": r["tags"],
        }
        | ({"corporate_events": r["corporate_events"]} if "corporate_events" in r else {})
        for r in records
        if strategy in r["strategies"]
    ]


def frame_digest(frame):
    return digest(
        {
            "index": [str(pd.Timestamp(d).date()) for d in frame.index],
            "columns": [str(c) for c in frame.columns],
            "values": frame.astype(float).fillna(0.0).to_numpy().tolist(),
            "missing": frame.isna().to_numpy().tolist(),
        }
    )


def start(backtest, *, role, alpha, horizon, source):
    """(assumptions, fresh state), set as `capstone.incubate.incubate` sets them."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    base = MonitorConfig.from_backtest(backtest)
    settings = IncubationSettings(
        alpha=alpha, horizon=horizon, confirm_alpha=alpha, confirm_horizon=horizon
    )
    started = incubate(
        settings,
        family=str(getattr(backtest, "name", "")),
        expected_mean=base.expected_mean,
        scale=base.scale,
        search_returns=pd.Series(backtest),
        live=[],
        confirming=role == INCUBATION_ONLY,
    )
    assumptions = {
        "expected_mean": started.config.expected_mean,
        "expected_mean_basis": "MonitorConfig.from_backtest(base-scenario funded returns)",
        "backtest_sd": base.scale,
        "inflation_factor": started.lr_factor,
        "scale": started.config.scale,
        "alpha": alpha,
        "horizon": started.config.horizon,
        "backtest_periods": int(pd.Series(backtest).notna().sum()),
        "source_digest": source,
    }
    return assumptions, started.state if role == "deploy" else started.confirm


def restore(role, data):
    return (MonitorState if role == "deploy" else ConfirmState).from_dict(data)


def logged(store):
    """(strategy, plan_id, session) keys already in events.jsonl; a torn last line is cut."""
    path = Path(store) / EVENTS
    if not path.exists():
        return set()
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1]
        with path.open("r+b") as stream:
            stream.truncate(len(raw))
    lines = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    return {(e["strategy_id"], e["plan_id"], e["session"]) for e in lines}


def append(store, event):
    with (Path(store) / EVENTS).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(event) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def record(store, strategy, role, assumptions, state, inputs):
    """Apply new sessions to the strategy's stored state; returns the new events."""
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{strategy}.json"
    applied = []
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["role"] != role or saved["assumptions"] != assumptions:
            raise ValueError(
                f"monitor assumptions changed for {strategy}; use a new store "
                f"(stored role {saved['role']}, source {saved['assumptions']['source_digest']})"
            )
        state, applied = restore(role, saved["state"]), saved["applied"]
    done, events = logged(store), []
    for item in inputs:
        key = [item["plan_id"], item["session"]]
        if key in applied:
            continue
        if applied and item["session"] <= applied[-1][1]:
            raise ValueError(f"session {item['session']} for {strategy} is out of order")
        if state.decision != "continue":
            break  # final decisions are sticky
        value = item["return"]
        state = state.update(math.nan if value is None else value)
        decay = state if role == "deploy" else state.decay
        event = {"strategy_id": strategy, "role": role} | item
        event |= {"e_value": decay.e_value, "n_obs": decay.n_obs, "decision": state.decision}
        if role != "deploy":
            event |= {"confirm_e_value": state.e_value, "reason": state.reason}
        if (strategy, *key) not in done:
            append(store, event)
        maybe_fault("monitor:after_event")
        applied = [*applied, key]
        document = {"strategy_id": strategy, "role": role, "assumptions": assumptions}
        atomic_json(path, document | {"state": state.to_dict(), "applied": applied})
        events.append(event)
    return events


def run(root, store, returns, *, role="deploy", alpha=0.05, horizon=None, earlier=()):
    """Attribute every journaled session under `root` and update the store."""
    records, pending = session_returns(root, earlier)
    horizon = horizon or (
        IncubationSettings().horizon if role == "deploy" else IncubationSettings().confirm_horizon
    )
    source = frame_digest(returns)
    strategies = sorted({s for r in records for s in r["strategies"]})
    started = {}
    for strategy in strategies:
        if strategy not in returns.columns:
            raise ValueError(f"no base-scenario funded returns for strategy {strategy}")
        started[strategy] = start(
            returns[strategy], role=role, alpha=alpha, horizon=horizon, source=source
        )
    new = 0
    # Session-major, so events.jsonl has one order however the sessions were split.
    for item in records:
        for strategy in sorted(item["strategies"]):
            inputs = observations([item], strategy)
            new += len(record(store, strategy, role, *started[strategy], inputs))
    summary = {}
    for strategy in strategies:
        saved = json.loads((Path(store) / f"{strategy}.json").read_text(encoding="utf-8"))
        current = restore(role, saved["state"])
        decay = current if role == "deploy" else current.decay
        summary[strategy] = {"decision": current.decision, "e_value": decay.e_value}
        summary[strategy]["n_obs"] = decay.n_obs
    return {"store": str(store), "new_events": new, "pending_plan": pending, "strategies": summary}
