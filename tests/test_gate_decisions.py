"""Tests for the registered gate's decisions on known truth, and its report.

- on null markets (independent, correlated, autocorrelated) nothing is
  accepted; on a planted one the planted families are;
- the scorecard has one row per stage-1 family with every stage's statistic,
  threshold and pass flag, graded scores and a batch header, and the report
  round-trips;
- each accepted family hands incubation an expected mean that is not the raw
  in-sample mean.

Synthetic markets from `gate_helpers` only (they end before 2021); a
temporary ledger per test. No real data is read.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
from gate_helpers import (
    N_FAMILIES,
    N_HOLDOUT,
    N_SEARCH,
    TRIAL_NAME,
    K,
    Market,
    config_for,
    gate_once,
    market,
)

from capstone.funnel import Thresholds
from capstone.gate import TrialScope
from capstone.scorecard import (
    HEADER_FIELDS,
    SCORECARD_COLUMNS,
    GateHeader,
    GateResult,
    read_report,
    write_report,
)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


@pytest.fixture(scope="module")
def null_market() -> Market:
    return market(seed=2)


@pytest.fixture(scope="module")
def planted_market() -> Market:
    return market(seed=1, share_real=0.3)


# ---------------------------------------------------------------------------
# Decisions on known truth
# ---------------------------------------------------------------------------


class TestDecisions:
    @pytest.mark.parametrize(
        "defects",
        [{}, {"rho": 0.3}, {"ar1": 0.2}],
        ids=["independent", "rho", "ar1"],
    )
    @pytest.mark.parametrize("seed", range(5))
    def test_null_market_accepts_nothing(self, ledger, seed, defects):
        result = gate_once(market(seed=100 + seed, **defects), seed=seed)
        assert result.accepted == ()
        assert not result.scorecard["accepted"].any()

    @pytest.mark.parametrize("defects", [{}, {"rho": 0.3}, {"ar1": 0.2}], ids=["iid", "rho", "ar1"])
    @pytest.mark.parametrize("seed", range(2))
    def test_planted_market_accepts_the_planted_families(self, ledger, seed, defects):
        m = market(seed=200 + seed, share_real=0.3, **defects)
        result = gate_once(m, seed=seed)
        planted = set(m.truth[m.truth].index)
        accepted = set(result.accepted)
        assert len(accepted & planted) >= len(planted) - 1  # most of them
        assert len(accepted - planted) <= 1

    def test_accepted_is_every_gate_passed(self, ledger, planted_market):
        card = gate_once(planted_market).scorecard
        every = card[["s1_pass", "s2_pass", "s3_pass", "s4_pass"]].all(axis=1)
        pd.testing.assert_series_equal(card["accepted"], every, check_names=False)

    def test_accepted_list_matches_the_scorecard(self, ledger, planted_market):
        result = gate_once(planted_market)
        assert result.accepted == tuple(result.scorecard.index[result.scorecard["accepted"]])


# ---------------------------------------------------------------------------
# Scorecard, header, report, expected mean
# ---------------------------------------------------------------------------


class TestScorecard:
    def test_one_row_per_family_with_the_documented_columns(self, ledger, planted_market):
        result = gate_once(planted_market, generator="agent-v1")
        card = result.scorecard
        assert isinstance(result, GateResult)
        assert tuple(card.columns) == SCORECARD_COLUMNS
        assert card.index.name == "family"
        assert set(card.index) == set(planted_market.family)
        assert (card["k"] == K).all()
        assert (card["generator"] == "agent-v1").all()
        assert card["best"].map(planted_market.family).eq(card.index.to_series()).all()
        th = Thresholds(stage1="bootstrap")
        assert (card["s1_threshold"] == th.q1).all()
        assert (card["s2_threshold"] == th.alpha2).all()
        assert (card["s4_threshold"] == th.q4).all()
        assert not card["s3_measured"].any()
        assert card["s3_stat"].isna().all()  # stage 3 is not measured yet
        assert (card["s3_pass"] == card["s2_pass"]).all()
        assert ((card["s1_stat"] <= th.q1) == card["s1_pass"]).all()
        reached = card["s3_pass"]
        assert card.loc[reached, "s4_stat"].notna().all()
        assert card.loc[~reached, "s4_stat"].isna().all()
        assert not card.loc[~reached, "s4_pass"].any()
        assert card.loc[reached, ["ho_sharpe", "ho_p", "ho_psr"]].notna().all().all()
        assert card.loc[~reached, ["ho_sharpe", "ho_p", "ho_psr", "ho_mean"]].isna().all().all()
        assert result.loadings.empty

    def test_header(self, ledger, planted_market):
        result = gate_once(planted_market, generator="agent-v1")
        h = result.header
        assert isinstance(h, GateHeader)
        assert tuple(f.name for f in dataclasses.fields(GateHeader)) == HEADER_FIELDS
        config = config_for(planted_market, generator="agent-v1")
        assert h.config_hash == config.config_hash
        assert h.generator == "agent-v1"
        assert h.thresholds == Thresholds(stage1="bootstrap")
        assert h.trial_scope == TrialScope(name=TRIAL_NAME)
        assert h.trial_count == planted_market.returns.shape[1]
        assert h.n_families == N_FAMILIES and h.n_variants == N_FAMILIES * K
        assert h.n_search_obs == N_SEARCH
        assert h.n_holdout_obs == N_HOLDOUT
        assert 0.0 <= h.pbo <= 1.0
        assert -1.0 <= h.avg_correlation <= 1.0
        assert 0.0 <= h.search_fdr_estimate <= 1.0  # something passed stage 1
        assert h.data_version == "synthetic"

    def test_search_fdr_estimate_is_nan_when_nothing_passes(self, ledger, null_market):
        result = gate_once(null_market)
        assert not result.scorecard["s1_pass"].any()
        assert math.isnan(result.header.search_fdr_estimate)
        assert result.header.n_holdout_obs == 0

    @pytest.mark.parametrize("which", ["planted", "null"])
    def test_report_round_trips(self, ledger, planted_market, null_market, tmp_path, which):
        result = gate_once(planted_market if which == "planted" else null_market)
        path = tmp_path / "reports" / "gate_report.json"
        write_report(result, path)
        back = read_report(path)
        pd.testing.assert_frame_equal(back.scorecard, result.scorecard)
        assert back.accepted == result.accepted


class TestExpectedMean:
    def test_expected_mean_is_conservative(self, ledger, planted_market):
        result = gate_once(planted_market)
        card = result.scorecard
        assert result.accepted
        acc = card.loc[list(result.accepted)]
        assert np.isfinite(acc["expected_mean"]).all()
        assert (acc["expected_mean"] <= acc["is_mean"]).all()
        assert (acc["expected_mean"] <= acc["ho_mean"]).all()
        assert (acc["expected_mean"] < acc["is_mean"]).all()  # the haircut bites
        assert card.loc[~card["accepted"], "expected_mean"].isna().all()
        assert result.expected_means == dict(acc["expected_mean"])
