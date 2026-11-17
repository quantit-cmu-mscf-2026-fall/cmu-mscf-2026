from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from capstone.strategy.portfolio import allocation


@dataclass
class Spec:
    id: str
    hypothesis_id: str
    side: str = "long_short"


def test_prior_only_connected_groups_and_family_budgets():
    dates = pd.bdate_range("2000-01-03", periods=270)
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=(2, len(dates)))
    returns = pd.DataFrame({"a": x, "b": x, "c": y}, index=dates)
    specs = [Spec(n, n) for n in "abc"]
    result = allocation(specs, returns, dates[-10], mode="correlation_groups")
    assert result["weights"] == {"a": 0.25, "b": 0.25, "c": 0.5}
    changed = returns.copy()
    changed.loc[changed.index >= dates[-10]] = 1000
    assert allocation(specs[::-1], changed, dates[-10], mode="correlation_groups") == result
    assert allocation(specs, returns, dates[-10])["weights"] == dict.fromkeys("abc", 1 / 3)


def test_short_overlap_is_unknown_and_opposite_returns_do_not_collapse():
    dates = pd.bdate_range("2000-01-03", periods=260)
    r = np.random.default_rng(2).normal(size=len(dates))
    frame = pd.DataFrame({"a": r, "b": -r}, index=dates)
    specs = [Spec("a", "A"), Spec("b", "B")]
    assert len(allocation(specs, frame, dates[-1], mode="correlation_groups")["groups"]) == 2
    assert allocation(specs, frame, dates[100])["pairs"][0]["correlation"] is None
    with pytest.raises(ValueError, match="separate"):
        allocation([Spec("a", "A"), Spec("b", "B", "long_only")], frame, dates[-1])
