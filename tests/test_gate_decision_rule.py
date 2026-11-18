"""Tests for the gate's decision rule: which column decides `accepted`.

`docs/validation/framework.md` (Q5, "Registered thresholds: crsp-v1") makes
the baseline, BH on the families' bootstrap p-values at the registered q, the
deciding rule, with the funnel "reported alongside". So:

- the rule is a hashed field of `GateConfig` and `accepted` follows it;
- under the baseline, `accepted` is `bh_adjusted(p1) <= q` and nothing else;
- the funnel's verdict stays on the scorecard as `funnel_pass`;
- the rule used is recorded in the look entry, the header and the report;
- `load_registered` reads the registered file's config and rule.

Synthetic markets only; a temporary ledger per test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from gate_helpers import (
    FUNNEL,
    Market,
    config_for,
    gate_once,
    log_trials,
    looks,
    market,
    run,
)

from capstone import gate
from capstone.evaluate import bh_adjusted
from capstone.funnel import Thresholds
from capstone.gate import DecisionRule, register
from capstone.scorecard import read_report, write_report

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


@pytest.fixture(scope="module")
def planted_market() -> Market:
    return market(seed=1, share_real=0.3)


def baseline(q: float) -> DecisionRule:
    return DecisionRule("baseline", q=q)


class TestTheRuleDecides:
    def test_the_baseline_accepts_exactly_what_bh_on_the_family_p_values_accepts(
        self, ledger, planted_market
    ):
        card = gate_once(planted_market, decision_rule=baseline(0.05)).scorecard
        expected = bh_adjusted(card["p1"]) <= 0.05
        pd.testing.assert_series_equal(card["accepted"], expected, check_names=False)
        assert card["accepted"].any()

    def test_the_baseline_ignores_the_funnels_verdict(self, ledger, planted_market):
        # A level nothing meets: the funnel (q1 = 0.10, q4 = 0.30) still accepts.
        result = gate_once(planted_market, decision_rule=baseline(1e-12))
        assert result.accepted == ()
        assert result.scorecard["funnel_pass"].any()
        assert not result.scorecard["accepted"].any()

    def test_the_baseline_accepts_what_the_funnel_rejects(self, ledger, planted_market):
        loose = gate_once(planted_market, decision_rule=baseline(0.9)).scorecard
        assert (loose["accepted"] & ~loose["funnel_pass"]).any()
        assert loose["accepted"].sum() > loose["funnel_pass"].sum()

    def test_the_funnel_rule_accepts_what_the_funnel_passes(self, ledger, planted_market):
        card = gate_once(planted_market, decision_rule=FUNNEL).scorecard
        pd.testing.assert_series_equal(card["accepted"], card["funnel_pass"], check_names=False)
        every = card[["s1_pass", "s2_pass", "s3_pass", "s4_pass"]].all(axis=1)
        pd.testing.assert_series_equal(card["funnel_pass"], every, check_names=False)

    def test_the_funnels_column_is_the_same_whichever_rule_decides(
        self, ledger, planted_market, tmp_path, monkeypatch
    ):
        a = gate_once(planted_market, decision_rule=FUNNEL).scorecard
        monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "second"))
        b = gate_once(planted_market, decision_rule=baseline(0.05)).scorecard
        pd.testing.assert_series_equal(a["funnel_pass"], b["funnel_pass"])
        pd.testing.assert_frame_equal(
            a.drop(columns=["accepted", "expected_mean"]),
            b.drop(columns=["accepted", "expected_mean"]),
        )

    def test_expected_means_follow_the_deciding_rule(self, ledger, planted_market):
        result = gate_once(planted_market, decision_rule=baseline(0.9))
        card = result.scorecard
        assert set(result.expected_means) == set(result.accepted)
        assert card.loc[card["accepted"], "expected_mean"].notna().all()
        assert card.loc[~card["accepted"], "expected_mean"].isna().all()


class TestTheRuleIsHashedAndRecorded:
    def test_changing_the_rule_or_its_q_changes_the_hash(self, planted_market):
        funnel = config_for(planted_market, decision_rule=FUNNEL)
        base = config_for(planted_market, decision_rule=baseline(0.05))
        other_q = config_for(planted_market, decision_rule=baseline(0.10))
        assert len({funnel.config_hash, base.config_hash, other_q.config_hash}) == 3
        assert base.to_dict()["decision_rule"] == {"rule": "baseline", "q": 0.05, "method": "bh"}
        assert gate.GateConfig.from_dict(base.to_dict()) == base

    def test_a_rule_change_is_a_different_registration_and_look(self, ledger, planted_market):
        gate_once(planted_market, decision_rule=baseline(0.05))
        # Same everything but the rule: a new hash, so not covered by the first registration.
        other = config_for(planted_market, decision_rule=baseline(0.10))
        assert gate._entries_for(gate.REGISTRATION_NAME, other.config_hash) == []
        with pytest.raises(gate.GateNotRegistered):
            run(other, planted_market)

    @pytest.mark.parametrize("bad", [None, "baseline", {"rule": "baseline", "q": 0.05}])
    def test_a_rule_that_is_not_a_decision_rule_is_refused_at_construction(
        self, planted_market, bad
    ):
        with pytest.raises(TypeError, match="DecisionRule"):
            config_for(planted_market, decision_rule=bad)

    def test_the_baseline_needs_the_bootstrap_p_values(self, ledger, planted_market):
        log_trials(planted_market.returns.shape[1])
        config = config_for(
            planted_market, thresholds=Thresholds(stage1="rho"), decision_rule=baseline(0.05)
        )
        register(config)
        with pytest.raises(ValueError, match="bootstrap"):
            run(config, planted_market)
        assert looks(config.config_hash) == []

    def test_the_look_the_header_and_the_report_record_the_rule(
        self, ledger, planted_market, tmp_path
    ):
        result = gate_once(planted_market, decision_rule=baseline(0.05))
        assert result.header.decision_rule == baseline(0.05)
        (look,) = looks(config_for(planted_market, decision_rule=baseline(0.05)).config_hash)
        assert look["params"]["decision_rule"] == {"rule": "baseline", "q": 0.05, "method": "bh"}
        path, _ = write_report(result, tmp_path / "gate.json")
        back = read_report(path)
        assert back.header.decision_rule == baseline(0.05)
        assert back.accepted == result.accepted
        assert "funnel_pass" in back.scorecard.columns

    def test_a_report_written_before_the_rule_was_recorded_still_reads(
        self, ledger, planted_market, tmp_path
    ):
        import json

        result = gate_once(planted_market, decision_rule=FUNNEL)
        path, _ = write_report(result, tmp_path / "gate.json")
        report = json.loads(path.read_text())
        del report["header"]["decision_rule"]
        path.write_text(json.dumps(report))
        assert read_report(path).header.decision_rule is None


class TestDecisionRule:
    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"rule": "other"}, ValueError),
            ({"rule": "baseline"}, TypeError),  # no q
            ({"rule": "baseline", "q": 0.0}, ValueError),
            ({"rule": "baseline", "q": 1.0}, ValueError),
            ({"rule": "baseline", "q": True}, TypeError),
            ({"rule": "baseline", "q": 0.05, "method": "holm"}, ValueError),
            ({"rule": "funnel", "q": 0.05}, ValueError),
        ],
    )
    def test_bad_rules_are_refused(self, kwargs, error):
        with pytest.raises(error):
            DecisionRule(**kwargs)

    def test_the_registered_file_rule_loads(self):
        rule = DecisionRule.from_dict(
            {"rule": "baseline", "method": "bh", "q": 0.05, "on": "a free-text note"}
        )
        assert rule == baseline(0.05)


class TestRegisteredFile:
    def test_crsp_v1_loads_with_its_baseline_rule(self):
        config = gate.load_registered(ROOT / "scripts" / "gate_crsp_v1.json")
        assert config.decision_rule == baseline(0.05)
        assert config.thresholds.stage1 == "bootstrap" and config.n_boot == 1999

    def test_the_docs_say_which_column_decides(self):
        text = (ROOT / "docs" / "handoff" / "acceptance-gate.md").read_text(encoding="utf-8")
        assert "`accepted`" in text and "`funnel_pass`" in text and "decision_rule" in text
