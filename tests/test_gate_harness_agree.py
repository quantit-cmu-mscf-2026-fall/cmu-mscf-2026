from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from gate_helpers import N_BOOT, N_HOLDOUT, N_SEARCH, Market, gate_once, market

from capstone import calibration
from capstone.funnel import Thresholds
from capstone.gate import DecisionRule

BASELINE_Q = 0.05
Q_GRID = (0.01, 0.05, 0.10, 0.20, 0.50)
TH = Thresholds(stage1="bootstrap")
WORLDS = [
    pytest.param(dict(seed=0), id="null"),
    pytest.param(dict(seed=1, share_real=0.3), id="planted"),
    pytest.param(dict(seed=2, share_real=0.3, sharpe=1.0, rho=0.2, ar1=0.1), id="planted-rho-ar1"),
    pytest.param(dict(seed=3, rho=0.2, ar1=0.1), id="null-rho-ar1"),
]


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


def harness_statistics(m: Market) -> pd.DataFrame:
    fs = calibration.FamilySet(m.returns, m.family, m.truth, N_SEARCH, N_HOLDOUT)
    stats = calibration.family_statistics(
        fs, stage2_years=1.0, n_boot=N_BOOT, seed=0, q=TH.q1, designs=("reuse",)
    )
    return stats["reuse"]


def gate_card(m: Market) -> pd.DataFrame:
    return gate_once(m, decision_rule=DecisionRule("baseline", q=BASELINE_Q)).scorecard


def harness_funnel(stats: pd.DataFrame) -> pd.Series:
    return calibration.run_funnel(stats, TH)["stage4"]


def assert_same(gate_column: pd.Series, harness: pd.Series) -> None:
    pd.testing.assert_series_equal(
        gate_column, harness.reindex(gate_column.index).astype(bool), check_names=False
    )


@pytest.mark.parametrize("world", WORLDS)
def test_the_gate_decides_what_the_harness_measured(ledger, world):
    m = market(**world)
    card = gate_card(m)
    stats = harness_statistics(m)

    pd.testing.assert_series_equal(
        card["p1"], stats["p1_bootstrap"].reindex(card.index), check_names=False
    )
    assert_same(card["accepted"], calibration.run_baseline(stats, q=BASELINE_Q, stage1="bootstrap"))
    for q in Q_GRID:
        assert_same(card["s1_stat"] <= q, calibration.run_baseline(stats, q=q, stage1="bootstrap"))
    assert_same(card["funnel_pass"], harness_funnel(stats))


def test_the_planted_world_has_something_to_agree_on():
    stats = harness_statistics(market(seed=1, share_real=0.3))
    assert calibration.run_baseline(stats, q=BASELINE_Q, stage1="bootstrap").any()
    assert harness_funnel(stats).any()


def test_a_thinly_covered_holdout_fails_in_the_gate_but_not_in_the_harness(ledger):
    m = market(seed=1, share_real=0.3)
    stats = harness_statistics(m)
    passer = harness_funnel(stats).idxmax()
    variant = stats.loc[passer, "best"]
    returns = m.returns.copy()
    holdout_days = returns.index[N_SEARCH:]
    idle = [d for i, d in enumerate(holdout_days) if i % 5 < 3]
    returns.loc[idle, variant] = float("nan")
    gappy = dataclasses.replace(m, returns=returns)

    card = gate_card(gappy)
    gappy_stats = harness_statistics(gappy)

    assert harness_funnel(gappy_stats)[passer]
    assert not card.loc[passer, "funnel_pass"]
    assert card.loc[passer, "ho_n_obs"] < 0.5 * N_HOLDOUT
    unaffected = card.index != passer
    assert_same(card["funnel_pass"][unaffected], harness_funnel(gappy_stats))
    assert_same(
        card["accepted"], calibration.run_baseline(gappy_stats, q=BASELINE_Q, stage1="bootstrap")
    )
