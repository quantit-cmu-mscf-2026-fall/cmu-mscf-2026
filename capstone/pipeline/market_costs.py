"""Prior-session, stock-specific historical cost assumptions shared with the planner.

A `HistoricalCostSchedule` turns a table of half spreads (bps, dates x permno) and a
cost assumption (`commission_bps`, `slippage_bps`, `short_borrow_bps_annual`,
`financing_bps_annual`) into the all-in per-instrument rate for a decision date. The
rate comes from the previous completed session only: a later row never changes it.
Rates feed `capstone.execution.planning.cost_rate` and `share_plan`, which refuse an
instrument with no qualified quote rather than trade it for free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from capstone import shared_data as sd
from capstone.strategy.research import digest


class HistoricalCostSchedule:
    def __init__(self, half_spread_bps, costs, *, multiplier=1.0):
        if multiplier not in {0, 1, 2}:
            raise ValueError("cost scenario is outside the frozen study")
        if half_spread_bps.index.has_duplicates or half_spread_bps.columns.has_duplicates:
            raise ValueError("cost observations must be unique")
        self.spreads, self.costs, self.multiplier = half_spread_bps.copy(), costs, multiplier
        self.id = digest(
            {
                "source": half_spread_bps.sort_index()
                .sort_index(axis=1)
                .to_json(orient="split", date_format="iso", double_precision=15),
                "commission": costs.commission_bps,
                "slippage": costs.slippage_bps,
                "borrow": costs.short_borrow_bps_annual,
                "financing": costs.financing_bps_annual,
                "multiplier": multiplier,
                "timing": "previous_completed_session",
            }
        )

    def components(self, decision):
        table = self._components(decision)
        return {n: dict(parts) for n, parts in table.items()}

    def _components(self, decision):
        if pd.Timestamp(decision) not in self.spreads.index:
            raise ValueError("completed-session cost observation is missing")
        if self.multiplier == 0:
            return {
                int(n): {"commission": 0.0, "spread": 0.0, "slippage": 0.0}
                for n in self.spreads.columns
            }
        spread = self.spreads.loc[pd.Timestamp(decision)]
        good = spread.notna() & np.isfinite(spread) & spread.ge(0)
        return {
            int(n): {
                "commission": self.costs.commission_bps * self.multiplier,
                "spread": float(p) * self.multiplier,
                "slippage": self.costs.slippage_bps * self.multiplier,
            }
            for n, p in spread[good].items()
        }

    def rates(self, decision):
        return self._rates(decision)

    def _rates(self, decision):
        return {n: sum(parts.values()) for n, parts in self._components(decision).items()}

    def fallback_components(self):
        """Held-exit fallback (v3/v4 contracts): the capped half-spread sd.MAX_SPREAD / 2."""
        return {
            "commission": self.costs.commission_bps * self.multiplier,
            "spread": sd.MAX_SPREAD / 2 * 1e4 * self.multiplier,
            "slippage": self.costs.slippage_bps * self.multiplier,
        }

    def fallback_bps(self):
        return sum(self.fallback_components().values())

    def carry_rates(self):
        return (
            self.costs.short_borrow_bps_annual * self.multiplier,
            self.costs.financing_bps_annual * self.multiplier,
        )
