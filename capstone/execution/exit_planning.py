"""Program-only between-formation closures using the existing fixed-share planner."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import pandas as pd

from capstone.execution.planning import share_plan
from capstone.strategy.program import ProgramStrategySpec, program_weights
from capstone.strategy.schedule_dispatch import decision_due


def program_exit_plan(
    specs, panel, as_of, current, prices, capital, *, min_notional=100.0, fee_bps=0.0, carry=None
):
    """Close names absent from every active program while preserving surviving shares.

    The caller supplies only currently allocated programs and applies shared risk,
    cash, qualification and reconciliation checks to the complete projected book.
    Formation and allocation rebalances use the existing ordinary planning path.
    """
    specs = tuple(specs)
    if not specs or any(not isinstance(spec, ProgramStrategySpec) for spec in specs):
        raise ValueError("exit-only planning requires nonempty all-schema6 program specifications")
    if len({spec.side for spec in specs}) != 1:
        raise ValueError("exit-only planning requires exactly one portfolio profile")
    if len({spec.id for spec in specs}) != len(specs):
        raise ValueError("exit-only planning requires distinct active program identities")
    formation = [decision_due(spec, panel, as_of) for spec in specs]
    if any(formation):
        raise ValueError("formation decisions require the ordinary rebalance planning path")
    if not isinstance(current, Mapping):
        raise ValueError("current positions must be an instrument-to-whole-shares mapping")
    held = {}
    for name, quantity in current.items():
        if isinstance(name, bool) or not isinstance(name, Integral) or name <= 0:
            raise ValueError("position identifiers must be positive integer instruments")
        if isinstance(quantity, bool) or not isinstance(quantity, Integral):
            raise ValueError("current positions must be whole-share integers")
        if quantity:
            held[int(name)] = int(quantity)
    allowed, eligibility = set(), {}
    for spec in specs:
        book = program_weights(spec, panel, as_of)
        eligibility.update(book.attrs.get("formation_eligibility", {}))
        for name, weight in book.items():
            if not weight:
                continue
            if isinstance(name, bool) or not isinstance(name, Integral) or name <= 0:
                raise ValueError("program target identifiers must be positive integer instruments")
            allowed.add(int(name))
    exiting = {name: quantity for name, quantity in held.items() if name not in allowed}
    plan = share_plan(
        pd.Series(dtype=float),
        exiting,
        prices,
        capital,
        min_notional=min_notional,
        side=specs[0].side,
        fee_bps=fee_bps,
        carry=carry,
    )
    survivors = {name: quantity for name, quantity in held.items() if name in allowed}
    for key in ("targets", "projected"):
        plan[key] = dict(sorted((survivors | plan[key]).items()))
    if eligibility:
        plan["formation_eligibility"] = eligibility
    return plan
