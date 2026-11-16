from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capstone.pipeline.allocation_diagnosis import frozen_provider
from capstone.pipeline.reference_allocation import replay
from capstone.strategy.portfolio import portfolio_target
from capstone.strategy.robust_allocation import AllocationPolicy


def panel():
    index = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    x = np.random.default_rng(404).normal(0.005, 0.025, (len(index), 5))
    return pd.DataFrame(x, index=index, columns=["HML", "RMW", "CMA", "Mom", "BAB"]).assign(
        RF=0.001, **{"Mkt-RF": 0.004}
    )


def policy(**costs):
    return AllocationPolicy(
        periods=12, lookback=120, minimum=60, window=36, step=12, holding_periods=3, **costs
    )


def test_frozen_schedule_reproduces_and_is_not_reoptimized():
    frame = panel()
    original, decisions = replay(frame, "equal_weight", policy(switch_bps=6, ongoing_bps=10))
    by_date = {pd.Timestamp(d["as_of"]): d for d in decisions}
    # Invalid optimizer mode proves the saved schedule is used without a fit.
    frozen, saved = replay(
        frame,
        "never-call-optimizer",
        policy(switch_bps=6, ongoing_bps=10),
        decision_provider=lambda at, _: by_date[at],
    )
    pd.testing.assert_frame_equal(original, frozen)
    assert decisions == saved
    gross, _ = replay(
        frame, "never-call-optimizer", policy(), decision_provider=lambda at, _: by_date[at]
    )
    assert gross.equity.iloc[-1] > frozen.equity.iloc[-1]
    assert gross.switch_cost.sum() == 0
    assert gross.ongoing_cost.sum() == 0


@pytest.mark.parametrize(
    "weights", [{"HML": 1}, dict.fromkeys(["HML", "RMW", "CMA", "Mom", "BAB"], -0.1)]
)
def test_untrusted_schedule_rejected(weights):
    with pytest.raises(ValueError, match="frozen"):
        replay(
            panel(),
            "frozen",
            policy(),
            decision_provider=lambda at, _: {"as_of": str(at.date()), "weights": weights},
        )


def test_risk_overlay_uses_only_prior_data_and_never_levers():
    frame = panel()
    at = pd.Timestamp("2000-01-01")
    schedule = [
        {
            "as_of": str(at.date()),
            "weights": dict.fromkeys(["HML", "RMW", "CMA", "Mom", "BAB"], 0.2),
            "id": "source",
        }
    ]
    expected = frozen_provider(schedule, frame, risk_limit=0.01)(at, None)
    changed = frame.copy()
    changed.loc[at:] = 0.5
    actual = frozen_provider(schedule, changed.iloc[::-1], risk_limit=0.01)(at, None)
    assert actual == expected
    assert 0 < sum(expected["weights"].values()) < 1
    assert expected["reoptimization"] is False


@pytest.mark.parametrize("weight", [0.0, 0.25])
def test_shared_portfolio_target_preserves_v2_cash(monkeypatch, weight):
    import capstone.strategy.portfolio as module

    spec = SimpleNamespace(id="a", side="long_only")
    book = pd.Series({101: 1.0})
    monkeypatch.setattr(
        module,
        "combined_book",
        lambda specs, data, at, shares: (book * shares[0], {"books": {"a": book}}),
    )
    allocation = {"schema_version": 4, "side": "long_only", "weights": {"a": weight}}
    target, _ = portfolio_target([spec], None, "2000-01-01", allocation)
    assert target.iloc[0] == weight
    allocation["schema_version"] = 2
    with pytest.raises(ValueError, match="capital"):
        portfolio_target([spec], None, "2000-01-01", allocation)
