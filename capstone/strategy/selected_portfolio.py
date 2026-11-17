"""A chronological research-selection comparison through the existing portfolio engine."""

from dataclasses import dataclass

import pandas as pd

from capstone.strategy.portfolio import PortfolioSpec, allocation, portfolio_target
from capstone.strategy.research import digest


@dataclass(frozen=True)
class SelectedPortfolioSpec(PortfolioSpec):
    schedule: tuple = ()

    @property
    def id(self):
        return digest({"base": super().id, "schedule": self.schedule, "version": 1})[:16]

    def members(self, as_of):
        known = [
            f for f in self.schedule if pd.Timestamp(f["decision_session"]) <= pd.Timestamp(as_of)
        ]
        if not known:
            return ()
        chosen = max(known, key=lambda f: f["decision_session"])
        if pd.Timestamp(chosen["decision_session"]) >= pd.Timestamp(chosen["start"]):
            raise ValueError("selection must precede its evaluation period")
        ids = set(chosen["selected"])
        available = {s.id for s in self.specs}
        if ids - available:
            raise ValueError("selection contains an unknown strategy")
        return tuple(s for s in self.specs if s.id in ids)

    def weights(self, panel, as_of):
        members = self.members(as_of)
        if not members:
            return pd.Series(dtype=float, name="weight")
        allocated = allocation(members, self.returns, as_of, mode=self.mode)
        return portfolio_target(members, panel, as_of, allocated)[0]
