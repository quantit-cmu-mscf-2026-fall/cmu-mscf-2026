"""Gross cohort diagnostics, including January at ages 1/13/25/37/49 months.

Tracking a cohort beyond a strategy's exit does not keep that trade open.
These descriptive paper checks never admit a strategy or open the holdout.
"""

from __future__ import annotations

import pandas as pd

from capstone import runlog
from capstone.factors.reference import field, january_diagnostic, month_ends
from capstone.strategy.reference import ReferenceStrategySpec, _legs

DIAGNOSTICS = {
    "reversal_36m": ["1f6143c812601140", "2c119c2ef01bbb47"],
    "volume_momentum_6m": ["c661c36c0d7e5514"],
    "momentum_6m": ["295eb52f7e2e2d40"],
    "high_52week": ["5ce22233201426b8"],
}
EXECUTABLE = {
    "reversal_36m": "533148edf022a26e",
    "volume_momentum_6m": "8b44455fa35a48b1",
    "momentum_6m": "716bdaf7915c65be",
    "high_52week": "3bcdbdb5d717313b",
}


def cohort_diagnostics(panel, *, anchor, start, end):
    if pd.Timestamp(end) > pd.Timestamp("2020-12-31"):
        raise ValueError("cohort diagnostics cannot open the sealed holdout")
    runlog.log_run(
        "reference-cohort-diagnostics",
        params={
            "data_version": panel.data_version,
            "anchor": anchor,
            "start": start,
            "end": end,
            "tracking_months": 60,
        },
        tags=["reference-v2", "diagnostic"],
        notes="Registered before gross cohort outcomes exist.",
    )
    daily = field(panel, "returns", end)
    monthly = (1 + daily).groupby(daily.index.to_period("M")).prod(min_count=1) - 1
    complete = daily.notna().groupby(daily.index.to_period("M")).all()
    monthly = monthly.where(complete)
    ends = month_ends(panel.calendar, end)
    rows, blocked = [], []
    for method, claim in EXECUTABLE.items():
        spec = ReferenceStrategySpec(claim, panel.data_version, anchor)
        for formation in ends:
            if formation < pd.Timestamp(anchor) or (
                method == "reversal_36m" and formation.month != 12
            ):
                continue
            try:
                long, short = _legs(spec, panel, formation, panel.calendar)
            except (ValueError, KeyError) as error:
                blocked.append(
                    {
                        "method": method,
                        "formation": str(formation.date()),
                        "status": "construction_failed",
                        "reason": str(error),
                    }
                )
                continue
            for date in ends:
                age = (date.to_period("M") - formation.to_period("M")).n
                if not 1 <= age <= 60 or not pd.Timestamp(start) <= date <= pd.Timestamp(end):
                    continue
                returns = monthly.loc[date.to_period("M")]
                long_returns = returns.reindex(long.index)
                short_returns = returns.reindex(short.index)
                measured = bool(long_returns.notna().all() and short_returns.notna().all())
                rows.append(
                    {
                        "method": method,
                        "date": str(date.date()),
                        "formation": str(formation.date()),
                        "age_months": age,
                        "measured": measured,
                        "long_return": float(long_returns @ long) if measured else None,
                        "short_return": float(short_returns @ short) if measured else None,
                    }
                )
    frame = pd.DataFrame(
        rows,
        columns=[
            "method",
            "date",
            "formation",
            "age_months",
            "measured",
            "long_return",
            "short_return",
        ],
    )
    january = january_diagnostic(frame[frame.method == "reversal_36m"]).copy()
    for name in ("date", "formation"):
        january[name] = january[name].dt.date.astype(str)
    return {
        "gross_diagnostic_only": True,
        "hypothetical_cohorts_not_executed_positions": True,
        "claims": DIAGNOSTICS,
        "blocked_formations": blocked,
        "cohorts": rows,
        "january": january.astype(object).where(january.notna(), None).to_dict("records"),
    }
