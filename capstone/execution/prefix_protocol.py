"""Opt-in observed-prefix identity for historical Program account continuation.

`canonical_inputs` and `formation_panel` read `capstone.pipeline.prefix_inputs` lazily: a
build without that module fails with an import error rather than continuing on a
different closure.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

EXECUTION_VERSION = "program-observed-prefix-v6"
# Registered benchmark controls on the same completed-prefix route (v6 benchmark arithmetic).
BENCHMARK_EXECUTION_VERSION = "benchmark-observed-prefix-v1"
# The same routes under the held-action rules (program-events-v5 / benchmark v7: valued held
# actions, deferred-entry requalification, close cash-in-lieu). New registrations use these;
# v6 / benchmark v1 keep their behaviour.
HELD_EXECUTION_VERSION = "program-observed-prefix-v7"
HELD_BENCHMARK_EXECUTION_VERSION = "benchmark-observed-prefix-v2"
# Plus capped fallback exits and as-of held-action classification.
FALLBACK_EXECUTION_VERSION = "program-observed-prefix-v8"
FALLBACK_BENCHMARK_EXECUTION_VERSION = "benchmark-observed-prefix-v3"
PROGRAM_VERSIONS = (EXECUTION_VERSION, HELD_EXECUTION_VERSION, FALLBACK_EXECUTION_VERSION)
BENCHMARK_PREFIX_VERSIONS = (
    BENCHMARK_EXECUTION_VERSION,
    HELD_BENCHMARK_EXECUTION_VERSION,
    FALLBACK_BENCHMARK_EXECUTION_VERSION,
)


def observed_events(events):
    """Later event kinds cannot leak an all-null future schema into old prefixes."""
    required = {"event_id", "event_date", "permno", "kind"}
    columns = [name for name in events if name in required or events[name].notna().any()]
    result = events[columns].copy()
    for name in result:
        if name in {"event_date", "pay_date", "observed_pay_date"}:
            result[name] = pd.to_datetime(result[name]).astype("datetime64[ns]")
        elif name == "permno":
            result[name] = result[name].astype("int64")
        elif pd.api.types.is_numeric_dtype(result[name]):
            result[name] = result[name].astype("float64")
        else:
            result[name] = result[name].astype(object)
    return result


def canonical_inputs(panel, opens, closes, events, cost_schedule, through):
    """Canonicalize observations, without asserting their source qualification.

    Future-only universe columns and values not available by the completed
    cutoff are absent from this closure. Known conversion destinations remain.
    The original event payment metadata is retained, so later revision of an
    earlier entitlement cannot masquerade as an unchanged observed prefix.
    """
    from capstone.pipeline.historical import HistoricalInputs
    from capstone.pipeline.market_costs import HistoricalCostSchedule
    from capstone.pipeline.prefix_inputs import input_prefix

    if events is None:
        raise ValueError("observed-prefix execution requires supplied events")
    spreads = panel.half_spread * 10000 if cost_schedule is None else cost_schedule.spreads
    inputs = HistoricalInputs(
        panel,
        opens,
        closes,
        events,
        panel.fields.get("returns", pd.DataFrame(index=panel.universe.index)),
        spreads,
        {},
    )
    view = input_prefix(inputs, through)
    panel = replace(
        view.panel,
        fields={
            name: values.rename(name) if isinstance(values, pd.Series) else values
            for name, values in view.panel.fields.items()
        },
        availability={
            name: values.rename("known_at") for name, values in view.panel.availability.items()
        },
    )
    schedule = None
    if cost_schedule is not None:
        if type(cost_schedule) is not HistoricalCostSchedule:
            raise ValueError("observed-prefix costs require the qualified shared schedule")
        schedule = HistoricalCostSchedule(
            view.half_spread_bps,
            cost_schedule.costs,
            multiplier=cost_schedule.multiplier,
        )
    return panel, view.opens, view.closes, observed_events(view.events), schedule


def formation_panel(panel, through):
    """The exact dated universe/availability closure for pending-source checks."""
    empty = pd.DataFrame(columns=["event_id", "event_date", "permno", "kind"])
    values = panel.fields.get("close", panel.universe.astype(float))
    observed, _, _, _, _ = canonical_inputs(panel, values, values, empty, None, through)
    return replace(observed, target_cache=None)
