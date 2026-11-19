"""Tests for capstone.gate.run_gate: the registered, single-look, blind gate.

The contract (`docs/validation/framework.md`, "The scorecard"):

- a gate config is registered on the ledger before any stage runs; an
  unregistered or changed config raises;
- the look at the holdout needs a named authorizer, is logged before the
  holdout provider is asked for anything, and happens once per config;
- stages 1-3 never see a holdout date;
- graded scores never change a decision.

Every market is synthetic and ends years before any real holdout (the
`test_markets_end_before_the_real_holdout` guard), and every test points
`CAPSTONE_LEDGER_DIR` at a temporary directory. No real data is read.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
from gate_helpers import (
    AUTHORIZER,
    STAGE_1_TO_3,
    Holdout,
    Market,
    config_for,
    gate_once,
    log_trials,
    looks,
    market,
    run,
    search_of,
)

from capstone import gate, grading, runlog
from capstone.funnel import Thresholds
from capstone.gate import (
    GateConfig,
    GateNotRegistered,
    HoldoutAlreadyOpened,
    register,
    run_gate,
)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A fresh, empty ledger directory; returns a function that makes another."""
    counter = iter(range(10_000))

    def fresh():
        path = tmp_path / f"ledger{next(counter)}"
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(path))
        return path

    fresh()
    return fresh


@pytest.fixture(scope="module")
def null_market() -> Market:
    return market(seed=2)


@pytest.fixture(scope="module")
def planted_market() -> Market:
    return market(seed=1, share_real=0.3)


def test_markets_end_before_the_real_holdout(null_market, planted_market):
    # Nothing here may reach the project's real holdout dates (2021 onwards).
    for m in (null_market, planted_market):
        assert m.holdout_end < pd.Timestamp("2021-01-01")


def test_a_gate_config_defaults_to_the_bootstrap_family_maximum(null_market):
    config = GateConfig(
        search_end=null_market.search_end,
        holdout_start=null_market.holdout_start,
        holdout_end=null_market.holdout_end,
        data_version="synthetic",
        generator="test-gen",
    )
    assert config.thresholds.stage1 == "bootstrap"


# ---------------------------------------------------------------------------
# Registration and authorisation
# ---------------------------------------------------------------------------


class TestRegistrationRequired:
    def test_unregistered_config_raises(self, ledger, null_market):
        log_trials(null_market.returns.shape[1])
        config = config_for(null_market)
        holdout = Holdout(null_market.returns)
        with pytest.raises(GateNotRegistered):
            run(config, null_market, holdout=holdout)
        assert looks(config.config_hash) == []
        assert holdout.calls == []

    def test_changed_config_raises(self, ledger, null_market):
        log_trials(null_market.returns.shape[1])
        register(config_for(null_market))
        changed = config_for(null_market, thresholds=Thresholds(q1=0.2, stage1="bootstrap"))
        with pytest.raises(GateNotRegistered):
            run(changed, null_market)
        assert looks(changed.config_hash) == []

    def test_gate_entries_are_not_trials(self, ledger, null_market):
        result = gate_once(null_market)
        assert result.header.trial_count == null_market.returns.shape[1]

    def test_a_registered_factor_frame_is_refused_until_stage_3_exists(self, ledger, null_market):
        log_trials(null_market.returns.shape[1])
        config = config_for(null_market, factors_digest="0" * 64)
        register(config)
        with pytest.raises(ValueError, match="stage-3"):
            run(config, null_market)
        assert looks(config.config_hash) == []


class TestAuthorizer:
    @pytest.mark.parametrize("bad", [None, "", "   ", 7, ["gate-owner"]])
    def test_a_named_authorizer_is_required_and_spends_nothing(self, ledger, null_market, bad):
        log_trials(null_market.returns.shape[1])
        config = config_for(null_market)
        register(config)
        holdout = Holdout(null_market.returns)
        with pytest.raises(ValueError, match="authorizer"):
            run(config, null_market, holdout=holdout, authorizer=bad)
        assert looks(config.config_hash) == [] and holdout.calls == []
        run(config, null_market)  # the look is still available
        assert len(looks(config.config_hash)) == 1

    def test_there_is_no_default_authorizer(self, ledger, null_market):
        log_trials(null_market.returns.shape[1])
        config = config_for(null_market)
        register(config)
        with pytest.raises(TypeError):
            run_gate(
                config,
                search_of(null_market),
                null_market.family,
                holdout=lambda *a: None,
            )

    def test_the_authorizer_is_recorded_with_the_look(self, ledger, planted_market):
        config = config_for(planted_market)
        log_trials(planted_market.returns.shape[1])
        register(config)
        run(config, planted_market, authorizer="  gate-owner  ")
        (look,) = looks(config.config_hash)
        assert look["params"]["authorizer"] == "gate-owner"
        assert look["params"]["data_version"] == "synthetic"
        assert set(look["tags"]) == {gate.GATE_TAG, gate.SYNTHETIC_TAG}


# ---------------------------------------------------------------------------
# One look at the holdout
# ---------------------------------------------------------------------------


class TestSingleLook:
    def test_second_look_raises(self, ledger, planted_market):
        config = config_for(planted_market)
        gate_once(planted_market)
        with pytest.raises(HoldoutAlreadyOpened):
            run(config, planted_market)
        assert len(looks(config.config_hash)) == 1

    def test_second_look_raises_even_with_other_data_or_authorizer(self, ledger, planted_market):
        config = config_for(planted_market)
        gate_once(planted_market)
        doubled = dataclasses.replace(planted_market, returns=planted_market.returns * 2.0)
        with pytest.raises(HoldoutAlreadyOpened):
            run(config, doubled, authorizer="someone-else")

    def test_a_correction_cannot_undo_a_look(self, ledger, planted_market):
        config = config_for(planted_market)
        gate_once(planted_market)
        (look,) = looks(config.config_hash)
        runlog.log_run(
            runlog.CORRECTION_NAME,
            params={
                "ledger_name": gate.LOOK_NAME,
                "excludes": [{"ts_utc": look["ts_utc"]}],
                "reason": "try again",
            },
        )
        with pytest.raises(HoldoutAlreadyOpened):
            run(config, planted_market)

    def test_look_is_logged_before_the_provider_is_called(self, ledger, planted_market):
        config = config_for(planted_market)
        holdout = Holdout(planted_market.returns, config.config_hash)
        gate_once(planted_market, holdout=holdout)
        assert len(holdout.calls) == 1
        ids, start, end, n_looks = holdout.calls[0]
        assert n_looks == 1
        assert (start, end) == (planted_market.holdout_start, planted_market.holdout_end)

    def test_provider_is_asked_once_for_exactly_the_survivors(self, ledger, planted_market):
        holdout = Holdout(planted_market.returns)
        result = gate_once(planted_market, holdout=holdout)
        ((ids, *_),) = holdout.calls
        survivors = result.scorecard.loc[result.scorecard["s3_pass"], "best"]
        assert set(ids) == set(survivors) and len(ids) == len(survivors) > 0

    def test_look_spends_the_config_even_when_nothing_reaches_stage_4(self, ledger, null_market):
        config = config_for(null_market)
        holdout = Holdout(null_market.returns)
        result = gate_once(null_market, holdout=holdout)
        assert not result.scorecard["s3_pass"].any()
        assert holdout.calls == []  # never asked: nothing to test
        assert len(looks(config.config_hash)) == 1
        with pytest.raises(HoldoutAlreadyOpened):
            run(config, null_market)

    def test_failure_before_stage_4_does_not_spend_the_look(self, ledger, planted_market):
        config = config_for(planted_market)
        register(config)
        log_trials(3)  # fewer trials on the ledger than variants: an unlogged search
        with pytest.raises(ValueError, match="trial"):
            run(config, planted_market)
        assert looks(config.config_hash) == []
        log_trials(planted_market.returns.shape[1])
        run(config, planted_market)
        assert len(looks(config.config_hash)) == 1

    def test_returns_past_search_end_are_refused_before_the_look(self, ledger, planted_market):
        log_trials(planted_market.returns.shape[1])
        config = config_for(planted_market)
        register(config)
        with pytest.raises(ValueError, match="search_end"):
            run_gate(
                config,
                planted_market.returns,  # includes the holdout dates
                planted_market.family,
                holdout=Holdout(planted_market.returns),
                authorizer=AUTHORIZER,
            )
        assert looks(config.config_hash) == []


class TestStagesOneToThreeAreBlind:
    def test_poisoned_holdout_leaves_stages_1_to_3_unchanged(self, ledger, planted_market):
        clean = gate_once(planted_market)
        poisoned_returns = planted_market.returns.copy()
        holdout = poisoned_returns.index >= planted_market.holdout_start
        poisoned_returns.loc[holdout] = -10.0 * poisoned_returns.loc[holdout] - 0.05
        poisoned = dataclasses.replace(planted_market, returns=poisoned_returns)
        ledger()
        dirty = gate_once(poisoned)
        pd.testing.assert_frame_equal(clean.scorecard[STAGE_1_TO_3], dirty.scorecard[STAGE_1_TO_3])
        for name in ("pbo", "avg_correlation", "search_fdr_estimate", "n_search_obs"):
            a, b = getattr(clean.header, name), getattr(dirty.header, name)
            assert a == b or (math.isnan(a) and math.isnan(b)), name
        # The poison reached stage 4, which is the point of the holdout.
        assert clean.accepted and not dirty.accepted

    def test_stages_1_to_3_receive_no_holdout_dates(self, ledger, planted_market, monkeypatch):
        received = []
        real = gate.funnel.family_statistics

        def spy(returns, *args, **kwargs):
            received.append(returns.index.max())
            return real(returns, *args, **kwargs)

        monkeypatch.setattr(gate.funnel, "family_statistics", spy)
        gate_once(planted_market)
        assert received and max(received) <= planted_market.search_end

    def test_the_provider_is_not_called_until_stages_1_to_3_are_done(
        self, ledger, planted_market, monkeypatch
    ):
        order = []
        real = gate.funnel.screen

        def spy(*args, **kwargs):
            order.append("screen")
            return real(*args, **kwargs)

        monkeypatch.setattr(gate.funnel, "screen", spy)

        def provider(ids, start, end):
            order.append("holdout")
            return planted_market.returns.loc[start:end, list(ids)]

        gate_once(planted_market, holdout=provider)
        assert order == ["screen", "holdout"]


class TestGradedScoresNeverDecide:
    def test_garbage_graded_scores_change_nothing(self, ledger, planted_market, monkeypatch):
        baseline = gate_once(planted_market)
        rng = np.random.default_rng(0)
        real = grading.graded_scores

        def garbage(*args, **kwargs):
            frame = real(*args, **kwargs)
            noise = rng.normal(size=frame.shape) * 1e6
            return pd.DataFrame(noise, index=frame.index, columns=frame.columns)

        monkeypatch.setattr(grading, "graded_scores", garbage)
        ledger()
        perturbed = gate_once(planted_market)
        flags = ["s1_pass", "s2_pass", "s3_pass", "s4_pass", "accepted"]
        pd.testing.assert_frame_equal(baseline.scorecard[flags], perturbed.scorecard[flags])
        assert baseline.accepted == perturbed.accepted
        assert not np.allclose(baseline.scorecard["holm"], perturbed.scorecard["holm"])

    def test_trial_count_moves_the_haircut_not_the_decision(self, ledger, planted_market):
        few = gate_once(planted_market)
        ledger()
        many = gate_once(planted_market, n_trials=1_000)
        assert many.header.trial_count == 1_000
        assert few.accepted == many.accepted
        hc_few = few.scorecard.loc[list(few.accepted), "haircut_sharpe"]
        hc_many = many.scorecard.loc[list(many.accepted), "haircut_sharpe"]
        assert (hc_many < hc_few).all()
