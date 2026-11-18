"""Archived flat-account rehearsal using the existing frozen planner and simulator."""

import pandas as pd

from capstone.artifacts import atomic_json


def event_blockers(events, plan):
    names = {int(n) for n in plan["projected_shares"]}
    relevant = events[
        (pd.to_datetime(events.event_date) == pd.Timestamp(plan["execution_session"]))
        & events.permno.isin(names)
        & events.kind.isin(["split", "conversion", "settlement", "unsupported"])
    ]
    return relevant.event_id.tolist()


def rehearsals(root, inputs, strategy_specs):
    from capstone.execution.bundle import freeze_bundle
    from capstone.execution.frozen import build_plan
    from capstone.pipeline.rehearsal import rehearse_bundle

    rows = []
    for s in strategy_specs:
        at = pd.Timestamp("2019-12-27" if s.method == "weekly_reversal" else "2019-11-29")
        dest = root / "rehearsals" / s.id
        try:
            calendar = inputs.panel.calendar
            position = calendar.get_indexer([at])[0] + 1  # 0 when `at` is not a session
            if position == 0 or position >= len(calendar):
                rows.append(
                    {
                        "id": s.id,
                        "status": "blocked",
                        "reason": f"{at.date()} is not followed by a session on the calendar",
                    }
                )
                continue
            execution = calendar[position]
            bundle = freeze_bundle(
                dest / "inputs",
                inputs.panel,
                [s],
                at,
                prices=inputs.closes.loc[at].dropna().to_dict(),
                instruments={
                    n: {
                        "symbol": str(inputs.symbols.at[at, n]),
                        "currency": "USD",
                        "sec_type": "STK",
                        "qualified": False,
                    }
                    for n in inputs.panel.universe.columns
                },
                account={
                    "positions": {},
                    "open_orders": [],
                    "cash": 100000.0,
                    "capital": 100000.0,
                },
                execution_cost_bps=6.0,
                limits={
                    "max_gross": 1.0,
                    "max_weight": 1.0,
                    "max_order_notional": 1e20,
                },
            )
            pending_events = event_blockers(inputs.events, build_plan(bundle))
            if pending_events:
                rows.append(
                    {
                        "id": s.id,
                        "status": "blocked",
                        "event_ids": pending_events,
                        "reason": "pending instructions require corporate-action reconciliation",
                    }
                )
                continue
            market = {
                "session": str(execution.date()),
                "prices": inputs.opens.loc[execution].dropna().to_dict(),
                "commission_bps": 1.0,
                "execution_adjustment_bps": 5.0,
            }
            result = rehearse_bundle(bundle, dest / "run", market)
            repeated = rehearse_bundle(bundle, dest / "run", market)
            rows.append(
                {
                    "id": s.id,
                    "result": result,
                    "duplicate_run_no_extra_submissions": result["submissions"]
                    == repeated["submissions"],
                    "scope": (
                        "archived flat-account engineering rehearsal; assumed costs; "
                        "unqualified contracts and admission remain blocked"
                    ),
                }
            )
        except ValueError as error:  # the freeze/plan/rehearse refusals; bugs still raise
            rows.append({"id": s.id, "status": "blocked", "reason": str(error)})
    atomic_json(root / "rehearsals.json", rows)
