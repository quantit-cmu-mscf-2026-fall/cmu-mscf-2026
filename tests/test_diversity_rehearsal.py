"""Archived flat-account rehearsal of a candidate set, on a synthetic panel only."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from capstone.factors.evaluation import Panel
from capstone.pipeline.diversity_rehearsal import event_blockers, rehearsals
from capstone.strategy.reference import ReferenceStrategySpec

CLAIM = "716bdaf7915c65be"
AT = pd.Timestamp("2019-11-29")


class Spec(ReferenceStrategySpec):
    """A reference profile that names its construction method, as candidate sets do."""

    method = "monthly_momentum"


def synthetic_inputs(events=None):
    dates = pd.bdate_range("2018-01-01", "2020-01-31")
    calendar = pd.bdate_range("2018-01-01", "2020-03-02")
    names = pd.Index(range(1, 41), name="permno")
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0, 0.008, (len(dates), len(names))), dates, names)
    close = 100 * (1 + returns).cumprod()
    fields = {
        "returns": returns,
        "close": close,
        "open": close / (1 + returns),
        "shares": close * 0 + 1_000_000,
        "cap": close * 1_000_000,
        "volume": close * 0 + 300_000,
    }
    panel = Panel(
        fields,
        close.notna(),
        close * 0 + 0.0005,
        data_version="synthetic-rehearsal",
        calendar=calendar,
        availability={name: pd.Series(dates, index=dates) for name in fields},
    )
    symbols = pd.DataFrame({n: f"S{n}" for n in names}, index=dates)
    if events is None:
        events = pd.DataFrame(columns=["event_id", "permno", "event_date", "kind"])
    return SimpleNamespace(
        panel=panel, closes=close, opens=fields["open"], symbols=symbols, events=events
    )


def test_pending_actions_block_an_instruction_on_the_execution_session():
    events = pd.DataFrame(
        {
            "event_id": ["held", "other", "dividend"],
            "permno": [1, 2, 1],
            "event_date": pd.to_datetime(["2000-01-04", "2000-01-04", "2000-01-04"]),
            "kind": ["unsupported", "split", "cash_dividend"],
        }
    )
    plan = {"projected_shares": {"1": 10}, "execution_session": "2000-01-04"}
    assert event_blockers(events, plan) == ["held"]  # other permno and other kinds do not block
    plan["execution_session"] = "2000-01-05"
    assert not event_blockers(events, plan)


def test_a_candidate_is_rehearsed_twice_with_no_extra_submissions(tmp_path):
    inputs = synthetic_inputs()
    spec = Spec(CLAIM, "synthetic-rehearsal", "2018-12-31")
    rehearsals(tmp_path, inputs, [spec])
    (row,) = json.loads((tmp_path / "rehearsals.json").read_text(encoding="utf-8"))
    assert row["id"] == spec.id and row["duplicate_run_no_extra_submissions"] is True
    assert row["result"]["fresh_process_identical"] and row["result"]["archived_only"]
    assert row["result"]["real_broker_verified"] is False
    assert "unqualified contracts and admission remain blocked" in row["scope"]


def test_a_pending_event_blocks_the_candidate_instead_of_rehearsing_it(tmp_path):
    calendar = synthetic_inputs().panel.calendar
    execution = calendar[calendar.get_loc(AT) + 1]
    events = pd.DataFrame(
        {
            "event_id": [f"held-{n}" for n in range(1, 41)],
            "permno": list(range(1, 41)),
            "event_date": [execution] * 40,
            "kind": ["unsupported"] * 40,
        }
    )
    spec = Spec(CLAIM, "synthetic-rehearsal", "2018-12-31")
    rehearsals(tmp_path, synthetic_inputs(events), [spec])
    (row,) = json.loads((tmp_path / "rehearsals.json").read_text(encoding="utf-8"))
    assert row["status"] == "blocked" and row["event_ids"]
    assert set(row["event_ids"]) <= set(events.event_id)
    assert "result" not in row


def test_a_failure_to_freeze_is_recorded_as_blocked_not_raised(tmp_path, monkeypatch):
    inputs = synthetic_inputs()
    spec = Spec(CLAIM, "synthetic-rehearsal", "2018-12-31")

    def refuse(*args, **kwargs):
        raise ValueError("synthetic refusal")

    monkeypatch.setattr("capstone.execution.bundle.freeze_bundle", refuse)
    rehearsals(tmp_path, inputs, [spec])
    (row,) = json.loads((tmp_path / "rehearsals.json").read_text(encoding="utf-8"))
    assert row == {"id": spec.id, "status": "blocked", "reason": "synthetic refusal"}
