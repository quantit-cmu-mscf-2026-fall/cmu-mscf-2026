"""Replay evidence: fallback cap, declared-rule reports and the written artifact folder.

Everything is synthetic: histories, orders and corporate actions are built in memory.
"""

import json

import numpy as np
import pandas as pd
import pytest

from capstone.execution.contracts import CASH_IN_LIEU_VERSIONS, FALLBACK_VERSIONS
from capstone.pipeline.actions import CASH_IN_LIEU_DEFAULT
from capstone.pipeline.crsp_actions import HELD_VALUE_RULE
from capstone.pipeline.evidence import (
    cash_in_lieu_defaults,
    eligibility_exclusions,
    fallback_report,
    replay_evidence,
    unqualified,
    write_json,
)

CAP = {"position_day_share": 0.01, "absolute_pnl_share": 0.01}


def _history(sessions, names, pnl, capital=1e6):
    dates = pd.bdate_range("2001-01-01", periods=sessions)
    equity = capital + pnl * np.arange(1, sessions + 1)
    previous = np.r_[capital, equity[:-1]]
    return pd.DataFrame(
        {
            "equity": equity,
            "net": equity / previous - 1,
            "positions": [{n: 1 for n in range(names)}] * sessions,
        },
        index=dates,
    )


def _exit(history, back, shares, price, cost=0.0):
    session = history.index[-1]
    return {
        "date": str(session.date()),
        "fallback_exit": {
            "session": str(session.date()),
            "last_price_date": str(history.index[-1 - back].date()),
            "shares": shares,
            "price": price,
            "cost": cost,
        },
    }


def test_cap_position_day_trigger_alone_marks_the_run_unqualified():
    history = _history(200, 1, 100.0)
    report = fallback_report(history, [_exit(history, 3, 1, 1.0, 0.01)], CAP)
    assert report["count"] == 1
    assert report["position_days"] == 201 and report["fallback_position_days"] == 3
    assert report["position_day_share"] == pytest.approx(3 / 201)
    assert report["absolute_pnl"] == pytest.approx(20000)
    assert report["absolute_pnl_share"] == pytest.approx(1.01 / 20000)
    assert report["qualified"] is False
    assert report["exceeded"] == ["position_day_share"]


def test_cap_absolute_pnl_trigger_alone_marks_the_run_unqualified():
    history = _history(1000, 5, 10.0)
    report = fallback_report(history, [_exit(history, 1, 50, 10.0)], CAP)
    assert report["position_day_share"] == pytest.approx(1 / 5001)
    assert report["fallback_pnl"] == pytest.approx(500)
    assert report["absolute_pnl_share"] == pytest.approx(500 / 10000)
    assert report["qualified"] is False and report["exceeded"] == ["absolute_pnl_share"]


def test_cap_within_both_limits_is_qualified_and_no_exit_is_trivially_qualified():
    history = _history(1000, 5, 10.0)
    report = fallback_report(history, [_exit(history, 1, 5, 10.0)], CAP)
    assert report["qualified"] is True and report["exceeded"] == []
    empty = fallback_report(history, [], CAP)
    assert empty["count"] == 0 and empty["qualified"] is True
    assert empty["position_day_share"] == 0 and empty["absolute_pnl_share"] == 0


def test_an_exit_with_no_pnl_to_divide_by_is_unbounded_and_unqualified():
    history = _history(10, 1, 0.0)
    report = fallback_report(history, [_exit(history, 1, 1, 5.0)], CAP)
    assert report["absolute_pnl_share"] == float("inf")
    assert "absolute_pnl_share" in report["exceeded"]


def test_unqualified_keeps_identity_and_counts_but_never_returns():
    assert unqualified({"held_fallbacks": {"qualified": True}}) is None
    assert unqualified({}) is None
    report = {
        "held_fallbacks": {"qualified": False, "count": 2},
        "artifacts": "replays/s/base",
        "execution_report_id": "abc",
        "sharpe": 1.5,
        "annual_mean": 0.1,
    }
    out = unqualified(report)
    assert out["status"] == "unqualified"
    assert out["completed"] is False and out["eligible_for_training"] is False
    assert out["formal_admission"] is False and out["replay_completed"] is True
    assert out["held_fallbacks"]["count"] == 2 and out["execution_report_id"] == "abc"
    assert not {"sharpe", "annual_mean"} & set(out)


def test_cash_in_lieu_defaults_counts_amounts_and_marks_only_non_default_rules():
    rows = [
        {
            "event_id": "a",
            "date": "2000-01-04",
            "cash_in_lieu": {"rule": CASH_IN_LIEU_DEFAULT, "amount": 20.0},
        },
        {
            "event_id": "b",
            "date": "2000-01-05",
            "cash_in_lieu": {"rule": "other_rule", "amount": 5.0},
        },
        {"event_id": "c", "date": "2000-01-06", "kind": "split"},
    ]
    out = cash_in_lieu_defaults(rows)
    assert out["rule"] == CASH_IN_LIEU_DEFAULT
    assert out["count"] == 2 and out["amount"] == 25.0
    assert out["events"] == [
        {"event_id": "a", "date": "2000-01-04", "amount": 20.0},
        {"event_id": "b", "date": "2000-01-05", "amount": 5.0, "rule": "other_rule"},
    ]
    assert cash_in_lieu_defaults([])["count"] == 0


def test_eligibility_exclusions_dedupes_skips_empty_and_sorts():
    a = {"formation_session": "2020-02-03", "exclusions": ["x"]}
    b = {"formation": "2020-01-02", "excluded": ["y"]}
    empty = {"formation_session": "2020-03-02", "exclusions": []}
    out = eligibility_exclusions([a, dict(a), b, empty])
    assert out == [
        {"formation": "2020-01-02", "exclusions": ["y"]},
        {"formation": "2020-02-03", "exclusions": ["x"]},
    ]


def test_write_json_is_sorted_and_rejects_nonfinite(tmp_path):
    path = tmp_path / "x.json"
    write_json(path, {"b": 1, "a": 2})
    text = path.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"b"')
    with pytest.raises(ValueError):
        write_json(path, {"a": float("nan")})


def _result(version, orders=None, actions=None):
    dates = pd.bdate_range("2020-01-01", periods=30)
    rng = np.random.default_rng(0)
    net = pd.Series(rng.normal(0.001, 0.01, len(dates)), index=dates)
    history = pd.DataFrame(
        {
            "net": net,
            "equity": 1000 * (1 + net).cumprod(),
            "cost": 0.5,
            "gross_exposure": 1.0,
            "net_exposure": 0.5,
            "positions": [{"1": 1}] * len(dates),
            "market_values": [{"1": 10.0}] * len(dates),
        },
        index=dates,
    )
    return {
        "strategy_id": "s1",
        "history": history,
        "orders": orders or [],
        "corporate_actions": actions or [],
        "positions": {"1": 1},
        "cash": 5.0,
        "account_hash": "h",
        "execution_contract": {"version": version, "fallback_cap": CAP},
    }


def test_replay_evidence_writes_the_artifact_folder_and_summarises(tmp_path):
    orders = [{"status": "blocked_x"}, {"status": "expired_corporate_action"}]
    report = replay_evidence(tmp_path, _result("legacy-version", orders), "base")
    folder = tmp_path / "replays" / "s1" / "base"
    names = {p.name for p in folder.iterdir()}
    assert {"daily.parquet", "holdings.json", "orders.json"} <= names
    assert {"corporate-actions.json", "account.json"} <= names
    assert report["artifacts"] == "replays/s1/base"
    assert report["days"] == 30 and report["blocked_executions"] == 1
    assert report["expired_plans"] == 1 and report["total_cost"] == pytest.approx(15.0)
    assert report["mean_gross_exposure"] == 1.0 and report["mean_net_exposure"] == 0.5
    assert report["sharpe"] is not None and report["maximum_drawdown"] <= 0
    assert not {"held_fallbacks", "held_carry", "cash_in_lieu_defaults"} & set(report)
    daily = pd.read_parquet(folder / "daily.parquet")
    assert "positions" not in daily.columns and "market_values" not in daily.columns
    saved = json.loads((folder / "account.json").read_text(encoding="utf-8"))
    assert saved["account_hash"] == "h"
    with pytest.raises(FileExistsError):
        replay_evidence(tmp_path, _result("legacy-version"), "base")


def test_replay_evidence_declares_the_sections_its_version_registers(tmp_path):
    version = FALLBACK_VERSIONS[0]
    assert version in CASH_IN_LIEU_VERSIONS
    actions = [
        {
            "event_id": "e1",
            "date": "2020-01-10",
            "kind": "delist",
            "amount": 7.0,
            "basis": HELD_VALUE_RULE,
        },
        {
            "event_id": "e2",
            "date": "2020-01-11",
            "cash_in_lieu": {"rule": CASH_IN_LIEU_DEFAULT, "amount": 2.0},
        },
    ]
    orders = [{"status": "carried_at_last_valid_mark"}]
    report = replay_evidence(tmp_path, _result(version, orders, actions), "base")
    assert report["held_carry"] == {"carried_sessions": 1, "fallback_exits": 0}
    assert report["cash_in_lieu_defaults"]["count"] == 1
    assert report["held_action_values"]["count"] == 1
    assert report["held_action_values"]["amount"] == 7.0
    assert report["held_fallbacks"]["qualified"] is True


def test_replay_evidence_has_no_sharpe_for_a_flat_series(tmp_path):
    result = _result("legacy-version")
    result["history"]["net"] = 0.0
    report = replay_evidence(tmp_path, result, "flat")
    assert report["sharpe"] is None and report["hac_pvalue"] is None
