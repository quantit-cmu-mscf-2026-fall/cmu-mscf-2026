"""Incubation as the second filter: the opt-in `incubate` outcome of stage 4.

`Thresholds(promote="incubate", q4_incubate=...)` labels a stage-3 survivor that fails
stage 4 at `q4` but passes `q4_incubate` on the same statistic (the BH-adjusted holdout
p) as `incubate`: never accepted. The default, `promote="accept"`, has no such outcome and
must register, hash and decide exactly as before. Synthetic scorecards only.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from capstone import promotion
from capstone.funnel import Thresholds, thresholds_dict
from capstone.gate import GateConfig

INCUBATE = Thresholds(promote="incubate", q4_incubate=0.5)


def content_hash(th: Thresholds) -> str:
    return hashlib.sha256(json.dumps(thresholds_dict(th), sort_keys=True).encode()).hexdigest()


def scorecard() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "s1_pass": [True, True, True, True, False],
            "s2_pass": [True, True, True, True, False],
            "s3_pass": [True, True, True, False, False],
            "s4_pass": [True, False, False, False, False],
            "s4_stat": [0.1, 0.4, 0.7, np.nan, np.nan],
            "is_mean": [0.002, 0.002, 0.002, 0.002, 0.002],
        },
        index=list("abcde"),
    )


class TestThresholds:
    def test_defaults_are_accept_with_no_incubation_level(self):
        th = Thresholds()
        assert th.promote == "accept" and th.q4_incubate is None

    def test_incubate_needs_a_level_looser_than_q4_and_below_1(self):
        assert Thresholds(promote="incubate", q4_incubate=0.5).q4_incubate == 0.5
        for bad in (None, 0.30, 0.2, 1.0, 1.2, 0.0, True, "0.5"):
            with pytest.raises(ValueError, match="q4_incubate"):
                Thresholds(promote="incubate", q4_incubate=bad)
        with pytest.raises(ValueError, match="q4_incubate"):
            Thresholds(promote="incubate", q4=0.6, q4_incubate=0.5)

    def test_the_level_is_stored_as_a_float_so_equal_settings_hash_alike(self):
        one = Thresholds(promote="incubate", q4_incubate=np.float32(0.5))
        other = Thresholds(promote="incubate", q4_incubate=0.5)
        assert content_hash(one) == content_hash(other)
        assert type(one.q4_incubate) is float

    def test_q4_incubate_is_refused_without_promote_incubate(self):
        # A level nothing reads would change the hash without changing a decision.
        with pytest.raises(ValueError, match="q4_incubate"):
            Thresholds(q4_incubate=0.5)

    def test_unknown_promote_is_refused(self):
        with pytest.raises(ValueError, match="promote"):
            Thresholds(promote="maybe", q4_incubate=0.5)

    def test_incubate_needs_the_bh_stage_4(self):
        # q4_incubate is a level on BH's adjusted p; the DSR gate has no such statistic.
        with pytest.raises(ValueError, match="bh"):
            Thresholds(promote="incubate", q4_incubate=0.5, final="dsr")


class TestRegisteredContent:
    def test_the_default_registers_neither_field(self):
        content = thresholds_dict(Thresholds())
        assert "promote" not in content and "q4_incubate" not in content
        assert set(content) == {"q1", "stage1", "alpha2", "final", "d4", "q4", "final_trials"}
        # Spelled out, the default is the same setting.
        spelled = Thresholds(promote="accept", q4_incubate=None)
        assert content_hash(spelled) == content_hash(Thresholds())

    def test_promote_and_q4_incubate_enter_the_content(self):
        base = Thresholds()
        incubate = Thresholds(promote="incubate", q4_incubate=0.5)
        looser = Thresholds(promote="incubate", q4_incubate=0.6)
        assert len({content_hash(base), content_hash(incubate), content_hash(looser)}) == 3
        content = thresholds_dict(incubate)
        assert content["promote"] == "incubate" and content["q4_incubate"] == 0.5

    def test_the_content_reads_back_to_the_same_setting(self):
        for th in (Thresholds(), INCUBATE):
            assert Thresholds(**json.loads(json.dumps(thresholds_dict(th)))) == th


def decided(card: pd.DataFrame) -> pd.DataFrame:
    """The funnel's rule: accepted = s1-s4, as a gate with decision_rule "funnel" sets it."""
    card = card.copy()
    card["accepted"] = card["s1_pass"] & card["s2_pass"] & card["s3_pass"] & card["s4_pass"]
    return card


def config(**th) -> GateConfig:
    return GateConfig(
        search_end="2020-12-31",
        holdout_start="2021-01-01",
        holdout_end="2025-12-31",
        thresholds=Thresholds(stage1="bootstrap", **th),
        data_version="v",
        generator="g",
    )


BH = config()


class TestOutcomes:
    def test_the_promotion_module_holds_the_rule(self):
        incubate = promotion.outcomes(decided(scorecard()), BH, 0.5)
        # b is between q4 (0.30) and q4_incubate (0.5); c is above it; d never reached stage 4.
        assert list(incubate) == [False, True, False, False, False]

    def test_a_looser_level_incubates_more(self):
        card = decided(scorecard())
        at_half = promotion.outcomes(card, BH, 0.5)
        at_eight = promotion.outcomes(card, BH, 0.8)
        assert list(at_eight) == [False, True, True, False, False]
        assert at_half.sum() <= at_eight.sum()

    def test_never_a_stage_3_failure(self):
        # Row f failed stage 3 but carries a small s4_stat; only the s3_pass check excludes it.
        card = decided(scorecard())
        card.loc["f"] = [True, True, False, False, 0.35, 0.002, False]
        out = promotion.outcomes(card, BH, 0.8)
        assert not out["f"]
        assert not out[~card["s3_pass"]].any()

    def test_a_row_exactly_at_the_level_incubates(self):
        card = decided(scorecard())
        card.loc["g"] = [True, True, True, False, 0.5, 0.002, False]
        out = promotion.outcomes(card, BH, 0.5)
        assert out["g"]
        assert not promotion.outcomes(card, BH, 0.49)["g"]

    def test_decide_keeps_the_gates_accepted_and_adds_incubate(self):
        card = promotion.decide(decided(scorecard()), BH, 0.5)
        assert list(card["accepted"]) == [True, False, False, False, False]
        assert list(card["incubate"]) == [False, True, False, False, False]
        assert not (card["accepted"] & card["incubate"]).any()

    def test_decide_does_not_touch_expected_mean(self):
        card = decided(scorecard())
        card["expected_mean"] = card["is_mean"].where(card["accepted"])
        before = card["expected_mean"].copy()
        out = promotion.decide(card, BH, 0.5)
        pd.testing.assert_series_equal(out["expected_mean"], before)
        assert out["expected_mean"].notna().tolist() == out["accepted"].tolist()

    def test_a_card_the_gate_has_not_decided_is_refused(self):
        with pytest.raises(ValueError, match="accepted"):
            promotion.decide(scorecard(), BH, 0.5)

    def test_baseline_accepted_rows_stay_accepted(self):
        # Decision rule "baseline" accepts on s1 alone (BH on the bootstrap p), so a
        # family that fails s4 (A: s4_stat 0.35) or stops at s1 (B) is still accepted.
        card = pd.DataFrame(
            {
                "s1_pass": [True, True, True],
                "s2_pass": [True, False, True],
                "s3_pass": [True, False, True],
                "s4_pass": [False, False, False],
                "s4_stat": [0.35, np.nan, 0.45],
                "accepted": [True, True, False],
            },
            index=list("ABC"),
        )
        out = promotion.decide(card, BH, 0.5)
        assert list(out["accepted"]) == [True, True, False]
        # A is accepted, so it cannot also incubate; only the unaccepted near-miss does.
        assert list(out["incubate"]) == [False, False, True]


class TestRegistration:
    def test_a_dsr_card_is_refused(self):
        # Under final="dsr" s4_stat is a deflated Sharpe, higher is better: the 0.02 row is
        # the worst, and a "<= level" test would label it incubate.
        card = pd.DataFrame(
            {
                "s3_pass": [True, True],
                "s4_stat": [0.02, 0.90],
                "accepted": [False, False],
            },
            index=["low", "high"],
        )
        dsr = config(final="dsr")
        with pytest.raises(ValueError, match="bh"):
            promotion.decide(card, dsr, 0.5)
        assert "incubate" not in card

    def test_the_level_must_sit_between_the_cards_q4_and_1(self):
        card = decided(scorecard())
        for bad in (0.30, 0.2, 1.0, 1.2, True, "0.5", None):
            with pytest.raises(ValueError, match="q4_incubate"):
                promotion.outcomes(card, BH, bad)
        with pytest.raises(ValueError, match="q4_incubate"):
            promotion.outcomes(card, config(q4=0.6), 0.5)


class TestGateRefusesWhatItDoesNotRun:
    def test_a_gate_config_refuses_incubate_until_the_gate_emits_it(self):
        from capstone.gate import GateConfig

        gate_fields = dict(
            search_end="2020-12-31",
            holdout_start="2021-01-01",
            holdout_end="2025-12-31",
            thresholds=Thresholds(stage1="bootstrap", promote="incubate", q4_incubate=0.5),
            data_version="v",
            generator="g",
        )
        with pytest.raises(ValueError, match="incubate"):
            GateConfig(**gate_fields)
