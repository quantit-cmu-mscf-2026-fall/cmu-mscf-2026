"""Portable replay reporting, separate from daily pipeline orchestration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from capstone.backtest import sharpe_ratio
from capstone.evaluate import sharpe_test
from capstone.execution.contracts import (
    CASH_IN_LIEU_VERSIONS,
    FALLBACK_VERSIONS,
    HELD_ACTION_VERSIONS,
    HELD_CARRY_VERSIONS,
)
from capstone.factors.research_contracts import digest
from capstone.pipeline.actions import CASH_IN_LIEU_DEFAULT
from capstone.pipeline.crsp_actions import HELD_VALUE_RULE


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _metrics(series):
    clean = series.dropna()
    equity = (1 + clean).cumprod()
    path = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)], ignore_index=True)
    if len(clean) < 3 or clean.std() == 0:
        sharpe, pvalue = None, None
    else:
        test = sharpe_test(clean, method="hac")
        sharpe, pvalue = float(sharpe_ratio(clean, 252)), float(test.pvalue_greater)
    return {
        "days": len(clean),
        "annual_mean": float(clean.mean() * 252),
        "annual_volatility": float(clean.std() * np.sqrt(252)),
        "sharpe": sharpe,
        "hac_pvalue": pvalue,
        "maximum_drawdown": float((path / path.cummax() - 1).min()),
    }


def replay_evidence(root, result, scenario):
    relative = Path("replays") / result["strategy_id"] / scenario
    folder = root / relative
    folder.mkdir(parents=True)
    history = result["history"]
    history.drop(
        columns=[
            "positions",
            "market_values",
            "formation_eligibility",
            "construction_evidence",
            "source_intent",
        ],
        errors="ignore",
    ).to_parquet(folder / "daily.parquet")
    write_json(
        folder / "holdings.json",
        [
            {"date": str(day.date()), "positions": positions}
            for day, positions in history["positions"].items()
        ],
    )
    write_json(folder / "orders.json", result["orders"])
    write_json(folder / "corporate-actions.json", result["corporate_actions"])
    eligibility = {}
    recorded = history.get("formation_eligibility", pd.Series(dtype=object)).dropna().tolist()
    pending = result.get("pending") or {}
    recorded.append(pending.get("formation_eligibility", {}))
    sources = history.get("source_intent", pd.Series(dtype=object)).dropna().tolist()
    if "source_intent" in pending:
        sources.append(pending["source_intent"])
    if sources:
        unique = {row["source_intent_id"]: row for row in sources}
        write_json(folder / "construction-sources.json", [unique[k] for k in sorted(unique)])
        recorded.extend(
            row.get("construction_evidence", {}).get("formation_eligibility", {}) for row in sources
        )
    for formation in recorded:
        eligibility.update({row["id"]: row for row in formation.values()})
    if eligibility:
        write_json(
            folder / "formation-eligibility.json", [eligibility[k] for k in sorted(eligibility)]
        )
    write_json(
        folder / "account.json",
        {
            "positions": result["positions"],
            "cash": result["cash"],
            "receivables": result.get("receivables", []),
            "equity": float(history.iloc[-1].equity),
            "account_hash": result["account_hash"],
        },
    )
    version = result["execution_contract"]["version"]
    declared = {}
    if version in HELD_CARRY_VERSIONS:
        declared["held_carry"] = {
            "carried_sessions": sum(
                o.get("status") == "carried_at_last_valid_mark" for o in result["orders"]
            ),
            "fallback_exits": sum("fallback_cost" in o for o in result["orders"]),
        }
    if version in CASH_IN_LIEU_VERSIONS:
        declared["cash_in_lieu_defaults"] = cash_in_lieu_defaults(result["corporate_actions"])
    if version in HELD_ACTION_VERSIONS:
        rows = [r for r in result["corporate_actions"] if r.get("basis") == HELD_VALUE_RULE]
        declared["held_action_values"] = {
            "rule": HELD_VALUE_RULE,
            "count": len(rows),
            "amount": sum(r["amount"] for r in rows),
            "events": [{k: r[k] for k in ("event_id", "date", "kind", "amount")} for r in rows],
        }
    if version in FALLBACK_VERSIONS:
        cap = result["execution_contract"]["fallback_cap"]
        declared["held_fallbacks"] = fallback_report(history, result["orders"], cap)
    return {
        **declared,
        **_metrics(history["net"]),
        "artifacts": relative.as_posix(),
        "mean_gross_exposure": float(history.gross_exposure.mean()),
        "mean_net_exposure": float(history.net_exposure.mean()),
        "total_cost": float(history.cost.sum()),
        "blocked_executions": sum(
            o.get("status", "").startswith("blocked") for o in result["orders"]
        ),
        "expired_plans": sum(
            o.get("status") == "expired_corporate_action" for o in result["orders"]
        ),
    }


def eligibility_exclusions(evidence):
    """Per-formation exclusions from program (v2/v3) and benchmark formation audits."""
    unique = {digest(row): row for row in evidence}
    rows = [
        {
            "formation": row.get("formation_session", row.get("formation")),
            "exclusions": row.get("exclusions", row.get("excluded")),
        }
        for row in unique.values()
        if row.get("exclusions") or row.get("excluded")
    ]
    return sorted(rows, key=lambda row: (str(row["formation"]), digest(row)))


def cash_in_lieu_defaults(corporate_actions):
    """Count and amounts of declared cash-in-lieu defaults (program-events-v4)."""
    rows = [row for row in corporate_actions if "cash_in_lieu" in row]
    return {
        "rule": CASH_IN_LIEU_DEFAULT,
        "count": len(rows),
        "amount": sum(row["cash_in_lieu"]["amount"] for row in rows),
        "events": [
            {
                "event_id": row["event_id"],
                "date": row["date"],
                "amount": row["cash_in_lieu"]["amount"],
                # v5/v7 close fallback only; v4 rows keep their exact shape.
                **({} if rule == CASH_IN_LIEU_DEFAULT else {"rule": rule}),
            }
            for row in rows
            for rule in [row["cash_in_lieu"]["rule"]]
        ],
    }


def fallback_report(history, orders, cap):
    """Fallback cap. Position-days: held names at each session's close, plus one per
    fallback exit for its exit session. A fallback exit's position-days: the sessions
    after its last observed close through the exit session (at least 1), i.e. every
    session the position was valued without a qualified observation. Absolute P&L: the
    sum of |daily equity change|. P&L attributed to fallback exits: each exit's whole
    value at the assumed price plus its cost (|shares| * price + cost), since the true
    outcome is unobserved. Exceeding either registered share makes the run unqualified."""
    exits = [o["fallback_exit"] for o in orders if "fallback_exit" in o]
    dates = history.index
    position_days = int(history["positions"].map(len).sum()) + len(exits)
    affected = sum(
        max(
            1,
            int(
                (
                    (dates > pd.Timestamp(row["last_price_date"]))
                    & (dates <= pd.Timestamp(row["session"]))
                ).sum()
            ),
        )
        for row in exits
    )
    pnl = float((history.equity - history.equity / (1 + history.net)).abs().sum())
    at_stake = sum(abs(row["shares"]) * row["price"] + row["cost"] for row in exits)
    shares = {
        "position_day_share": affected / position_days if position_days else 0.0,
        "absolute_pnl_share": at_stake / pnl if pnl else (0.0 if not exits else float("inf")),
    }
    exceeded = [
        key for key in ("position_day_share", "absolute_pnl_share") if shares[key] > cap[key]
    ]
    return {
        "rule": "exit_at_last_observed_close_capped_cost_v1",
        "count": len(exits),
        "position_days": position_days,
        "fallback_position_days": affected,
        "absolute_pnl": pnl,
        "fallback_pnl": at_stake,
        **shares,
        "cap": dict(cap),
        "exceeded": exceeded,
        "qualified": not exceeded,
        "exits": exits,
    }


# What an unqualified scenario keeps: identity and the fallback counts, never returns.
KEPT = (
    "artifacts",
    "held_fallbacks",
    "execution_report_id",
    "execution_inputs_hash",
    "execution_contract",
    "initialization",
    "initialization_qualifying",
)


def unqualified(report):
    """A scenario over its registered fallback cap is `unqualified`; its returns are
    not evidence (not reported, not a training or funnel input). None when within cap."""
    if report.get("held_fallbacks", {}).get("qualified", True):
        return None
    return {
        "status": "unqualified",
        "completed": False,
        "replay_completed": True,
        "formal_admission": False,
        "eligible_for_training": False,
        **{key: report[key] for key in KEPT if key in report},
    }
