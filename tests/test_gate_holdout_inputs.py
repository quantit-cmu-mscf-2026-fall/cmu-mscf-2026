"""Tests for the gate's holdout inputs: checked on arrival, asked for on demand.

- the holdout provider is called only when a survivor needs it, and only
  after every check that can fail before the look has passed;
- its answer must be exactly the requested variants, on registered holdout
  dates only, finite (NaN means missing); a bad answer leaves the look spent;
- a survivor with no holdout returns fails stage 4 rather than passing or
  crashing the run.

Synthetic markets from `gate_helpers` only (they end before 2021); a
temporary ledger per test. No real data is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gate_helpers import (
    Holdout,
    Market,
    config_for,
    gate_once,
    log_trials,
    looks,
    market,
    run,
)

from capstone.gate import GateNotRegistered, HoldoutAlreadyOpened, register


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
def planted_market() -> Market:
    return market(seed=1, share_real=0.3)


def serving(m: Market, change=None):
    """A provider over `m`'s returns whose answer `change(frame, ids)` may spoil."""
    base = Holdout(m.returns)

    def provider(ids, start, end):
        frame = base(ids, start, end)
        return frame if change is None else change(frame, tuple(ids))

    provider.calls = base.calls
    return provider


def never_called(*args, **kwargs):
    raise AssertionError("the holdout provider was called")


class TestProviderIsAskedOnDemand:
    def test_not_called_when_the_config_is_not_registered(self, ledger, planted_market):
        log_trials(planted_market.returns.shape[1])
        with pytest.raises(GateNotRegistered):
            run(config_for(planted_market), planted_market, holdout=never_called)

    def test_not_called_without_an_authorizer(self, ledger, planted_market):
        log_trials(planted_market.returns.shape[1])
        config = config_for(planted_market)
        register(config)
        with pytest.raises(ValueError, match="authorizer"):
            run(config, planted_market, holdout=never_called, authorizer="")

    def test_not_called_when_a_check_before_the_look_fails(self, ledger, planted_market):
        config = config_for(planted_market, n_boot=5)  # too few draws for 10 families
        log_trials(planted_market.returns.shape[1])
        register(config)
        with pytest.raises(ValueError, match="n_boot"):
            run(config, planted_market, holdout=never_called)
        assert looks(config.config_hash) == []

    def test_not_called_twice_and_not_after_a_look(self, ledger, planted_market):
        holdout = serving(planted_market)
        config = config_for(planted_market)
        gate_once(planted_market, holdout=holdout)
        assert len(holdout.calls) == 1
        with pytest.raises(HoldoutAlreadyOpened):
            run(config, planted_market, holdout=never_called)

    def test_a_provider_that_fails_still_spends_the_look(self, ledger, planted_market):
        config = config_for(planted_market)

        def broken(ids, start, end):
            raise RuntimeError("the data source is down")

        log_trials(planted_market.returns.shape[1])
        register(config)
        with pytest.raises(RuntimeError, match="down"):
            run(config, planted_market, holdout=broken)
        assert len(looks(config.config_hash)) == 1
        with pytest.raises(HoldoutAlreadyOpened):
            run(config, planted_market)


class TestTheAnswerIsChecked:
    def refused(self, ledger, m: Market, change, error, match):
        config = config_for(m)
        log_trials(m.returns.shape[1])
        register(config)
        with pytest.raises(error, match=match):
            run(config, m, holdout=serving(m, change))
        assert len(looks(config.config_hash)) == 1  # the look is spent
        with pytest.raises(HoldoutAlreadyOpened):
            run(config, m)

    def test_an_extra_variant_is_refused(self, ledger, planted_market):
        def add(frame, ids):
            extra = next(c for c in planted_market.returns.columns if c not in ids)
            frame = frame.copy()
            frame[extra] = planted_market.returns.loc[frame.index, extra]
            return frame

        self.refused(ledger, planted_market, add, ValueError, "exactly the requested columns")

    def test_a_missing_variant_is_refused(self, ledger, planted_market):
        self.refused(
            ledger,
            planted_market,
            lambda frame, ids: frame.drop(columns=[ids[0]]),
            ValueError,
            "exactly the requested columns",
        )

    def test_duplicate_columns_are_refused(self, ledger, planted_market):
        self.refused(
            ledger,
            planted_market,
            lambda frame, ids: pd.concat([frame, frame[[ids[0]]]], axis=1),
            ValueError,
            "exactly the requested columns",
        )

    def test_search_period_dates_are_refused(self, ledger, planted_market):
        # A provider that hands back the search period's returns as well.
        self.refused(
            ledger,
            planted_market,
            lambda frame, ids: planted_market.returns[list(ids)],
            ValueError,
            "outside the holdout",
        )

    def test_dates_after_holdout_end_are_refused(self, ledger, planted_market):
        def late(frame, ids):
            later = frame.iloc[[-1]].set_axis(frame.index[[-1]] + pd.offsets.BDay(5))
            return pd.concat([frame, later])

        self.refused(ledger, planted_market, late, ValueError, "outside the holdout")

    def test_unordered_or_repeated_dates_are_refused(self, ledger, planted_market):
        self.refused(
            ledger,
            planted_market,
            lambda frame, ids: pd.concat([frame, frame.iloc[[0]]]),
            ValueError,
            "strictly increasing",
        )

    def test_infinite_returns_are_refused(self, ledger, planted_market):
        def spoil(frame, ids):
            frame = frame.copy()
            frame.iloc[3, 0] = np.inf
            return frame

        self.refused(ledger, planted_market, spoil, ValueError, "finite")

    @pytest.mark.parametrize(
        "answer",
        [None, [1.0, 2.0], pd.DataFrame({"a": [1.0]}, index=[0])],
        ids=["none", "list", "no-date-index"],
    )
    def test_something_that_is_not_a_dated_frame_is_refused(self, ledger, planted_market, answer):
        self.refused(
            ledger, planted_market, lambda frame, ids: answer, TypeError, "indexed by date"
        )

    def test_a_reordered_answer_is_accepted(self, ledger, planted_market):
        ordered = gate_once(planted_market)
        ledger()

        def reverse(frame, ids):
            return frame[list(ids)[::-1]]

        shuffled = gate_once(planted_market, holdout=serving(planted_market, reverse))
        pd.testing.assert_frame_equal(ordered.scorecard, shuffled.scorecard)


class TestMissingHoldout:
    """A survivor with no holdout returns fails stage 4; the others are unaffected."""

    def blanked(self, m: Market, which: int = 0):
        seen = {}

        def change(frame, ids):
            frame = frame.copy()
            seen["variant"] = ids[which]
            frame[ids[which]] = np.nan
            return frame

        return change, seen

    def test_a_survivor_with_no_holdout_fails_stage_4(self, ledger, planted_market):
        baseline = gate_once(planted_market)
        assert len(baseline.accepted) >= 2
        ledger()
        change, seen = self.blanked(planted_market)
        result = gate_once(planted_market, holdout=serving(planted_market, change))
        card = result.scorecard
        (family,) = card.index[card["best"] == seen["variant"]]
        assert card.loc[family, "s3_pass"]  # it survived the screen
        assert not card.loc[family, "s4_pass"] and not card.loc[family, "accepted"]
        assert np.isnan(card.loc[family, "s4_stat"])
        assert np.isnan(card.loc[family, ["ho_sharpe", "ho_p", "ho_mean"]]).all()
        others = [f for f in baseline.accepted if f != family]
        assert set(result.accepted) >= set(others) - {family}
        assert family not in result.accepted

    def test_every_survivor_without_holdout_accepts_nothing(self, ledger, planted_market):
        def empty(frame, ids):
            return frame.iloc[:0]

        result = gate_once(planted_market, holdout=serving(planted_market, empty))
        assert result.accepted == ()
        assert result.scorecard.loc[result.scorecard["s3_pass"], "s4_pass"].eq(False).all()
        assert result.header.n_holdout_obs == 0
        # The look was still logged and spent.
        assert len(looks(config_for(planted_market).config_hash)) == 1
