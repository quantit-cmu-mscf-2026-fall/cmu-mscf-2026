"""Versioned, prior-only allocation; correlations describe redundancy, not admission."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone.execution.data import FrozenSnapshotSource, MarketDataSource
from capstone.execution.deploy import Refused
from capstone.strategy.contract import target_weights
from capstone.strategy.research import digest


def combined_book(
    specs, data: MarketDataSource, as_of: pd.Timestamp, shares=None
) -> tuple[pd.Series, dict]:
    """The strategies' books combined, and each book: averaged (equal capital
    each) or, with `shares` (one per spec), weighted by them."""
    books, data_versions = {}, set()
    for spec in specs:
        panel = data.panel(spec.fields, as_of)
        dates = pd.DatetimeIndex(panel.universe.index)
        if len(dates) and dates.max() > as_of:
            raise Refused(f"the data source returned dates after {as_of.date()}")
        books[spec.id] = target_weights(spec, panel, as_of)
        data_versions.add(panel.data_version)
    if shares is None:
        target = pd.concat(list(books.values())).groupby(level=0).sum() / len(books)
    else:
        scaled = [books[s.id] * w for s, w in zip(specs, shares, strict=True)]
        target = pd.concat(scaled).groupby(level=0).sum()
    target = target[target != 0].sort_index().astype("float64").rename("weight")
    target.index = target.index.astype("int64").rename("permno")
    eligibility = {}
    for book in books.values():
        eligibility.update(book.attrs.get("formation_eligibility", {}))
    if eligibility:
        target.attrs["formation_eligibility"] = eligibility
    return target, {"books": books, "data_versions": sorted(data_versions)}


@dataclass(frozen=True)
class PortfolioSpec:
    specs: tuple
    returns: pd.DataFrame
    mode: str = "equal_family"

    @property
    def side(self):
        if len({s.side for s in self.specs}) != 1:
            raise ValueError("portfolio profiles must remain separate")
        return self.specs[0].side

    @property
    def id(self):
        return digest(
            {
                "strategies": sorted(s.id for s in self.specs),
                "mode": self.mode,
                "side": self.side,
                "version": 2,
            }
        )[:16]


def allocation(
    specs,
    returns: pd.DataFrame,
    as_of,
    *,
    mode="equal_family",
    return_basis=None,
    policy=None,
    previous=None,
):
    ordered = sorted(specs, key=lambda s: s.id)
    if not ordered or len({s.id for s in ordered}) != len(ordered):
        raise ValueError("allocation requires unique strategies")
    if len({s.side for s in ordered}) != 1:
        raise ValueError("long-only and long-short profiles must remain separate")
    if mode in {"equal_weight", "shrunk_max_sharpe", "median_window_sharpe"}:
        from capstone.strategy.robust_allocation import optimize_allocation

        if return_basis != "self_financing_excess":
            raise ValueError("robust allocation requires explicit self_financing_excess returns")
        if set(returns.columns) != {s.id for s in ordered}:
            raise ValueError("return columns must identify the candidate pool exactly")
        result = optimize_allocation(returns, as_of, mode=mode, policy=policy, previous=previous)
        result["side"] = ordered[0].side
        result["id"] = digest(result)
        return result
    if mode not in {"equal_family", "correlation_groups"}:
        raise ValueError("unknown allocation mode")
    if returns.index.has_duplicates or returns.columns.has_duplicates:
        raise ValueError("duplicate return dates or strategy ids")
    past = returns.loc[returns.index < pd.Timestamp(as_of)].sort_index()
    if np.isinf(past.to_numpy()).any():
        raise ValueError("infinite strategy return")
    families = {}
    for spec in ordered:
        families.setdefault(spec.hypothesis_id, []).append(spec.id)
    family_returns = {}
    for name, ids in families.items():
        if all(i in past.columns for i in ids):
            family_returns[name] = past[ids].mean(axis=1, skipna=False)
    frame = pd.DataFrame(family_returns)
    roots = {name: name for name in families}

    def root(name):
        while roots[name] != name:
            name = roots[name]
        return name

    pairs = []
    names = sorted(families)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = frame[[a, b]].dropna() if a in frame and b in frame else pd.DataFrame()
            rho = float(overlap[a].corr(overlap[b])) if len(overlap) >= 252 else float("nan")
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "observations": len(overlap),
                    "correlation": rho if np.isfinite(rho) else None,
                }
            )
            if mode == "correlation_groups" and np.isfinite(rho) and rho >= 0.8:
                ra, rb = root(a), root(b)
                roots[max(ra, rb)] = min(ra, rb)
    groups = {}
    for name in names:
        groups.setdefault(root(name), []).append(name)
    weights = {}
    for members in groups.values():
        for name in members:
            for strategy_id in families[name]:
                weights[strategy_id] = 1 / len(groups) / len(members) / len(families[name])
    result = {
        "schema_version": 2,
        "mode": mode,
        "side": ordered[0].side,
        "as_of": str(pd.Timestamp(as_of).date()),
        "threshold": 0.8,
        "minimum_overlap": 252,
        "weights": weights,
        "groups": list(groups.values()),
        "pairs": pairs,
        "history_end": None if past.empty else str(past.index[-1].date()),
    }
    result["id"] = digest(result)
    return result


def portfolio_target(specs, panel, as_of, allocated):
    ordered = sorted(specs, key=lambda s: s.id)
    if set(allocated["weights"]) != {s.id for s in ordered}:
        raise ValueError("allocation does not identify exactly the selected strategies")
    if allocated["side"] != ordered[0].side or any(s.side != allocated["side"] for s in ordered):
        raise ValueError("portfolio profile mismatch")
    shares = [allocated["weights"][s.id] for s in ordered]
    total = sum(shares)
    valid_total = (
        total <= 1 + 1e-8 if allocated.get("schema_version") in {3, 4} else np.isclose(total, 1)
    )
    if any(not np.isfinite(w) or w < 0 for w in shares) or not valid_total:
        raise ValueError("allocation must be finite, nonnegative and respect capital")
    book, details = combined_book(ordered, FrozenSnapshotSource(panel), pd.Timestamp(as_of), shares)
    contributions = {
        s.id: details["books"][s.id].mul(w).to_dict() for s, w in zip(ordered, shares, strict=True)
    }
    details["attribution"] = {
        "kind": "target-weight attribution, before integer rounding",
        "strategies": contributions,
    }
    return book, details
