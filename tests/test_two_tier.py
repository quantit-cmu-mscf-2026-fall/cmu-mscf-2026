"""Two-tier rules: the vectorized statistic, survivors, a sample drawn from ids only, the
rank-agreement check (and that it switches the screen off), and one ledger trial per
tier-1 variant, logged once.

Synthetic only; no model calls, no network. The procedure-identity tests are in
`test_procedure_contracts.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_procedure_contracts import h, spec, spec_record

from capstone import runlog
from capstone.factors.procedure_contracts import ProcedureSpec
from capstone.pipeline import two_tier

TIER1_ID = h("tier1")


def rule(**override):
    values = {
        "tier1_id": TIER1_ID,
        "survivors_per_family": 1,
        "agreement_sample": 4,
        "min_rank_agreement": 0.5,
        "calibration_tier1_error": 1.0,
    }
    return two_tier.section(**(values | override))


def test_a_version_one_world_bypasses_the_screen():
    events = [{"arm": "research", "status": "evaluated", "column": "c0"}]
    kept, screens = two_tier.calibration_events(spec(), None, None, events, 0, 0)
    assert kept is events and screens is None


def test_the_section_builder_returns_a_row_the_procedure_accepts():
    row = rule()
    procedure = ProcedureSpec.freeze(**spec_record(), evaluation=row)
    assert procedure.two_tier == row and procedure.identity != spec().identity
    with pytest.raises(ValueError, match="at least one"):
        rule(survivors_per_family=0)


# --- the rules --------------------------------------------------------------------------------


def test_the_statistic_is_the_vectorized_annualized_sharpe():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(0.001, 0.01, (200, 5)), columns=list("abcde"))
    frame["flat"] = 0.002
    frame.loc[:150, "e"] = np.nan
    stat = two_tier.statistic(frame)
    expected = frame.mean() / frame.std() * np.sqrt(252)
    np.testing.assert_allclose(stat[list("abcde")], expected[list("abcde")], rtol=1e-12)
    assert np.isnan(stat["flat"])


def test_survivors_are_the_best_per_family_and_unscored_variants_always_survive():
    scores = pd.Series({"a1": 1.0, "a2": 3.0, "a3": 2.0, "b1": -1.0, "b2": np.nan, "b3": 0.5})
    family = pd.Series({v: v[0] for v in scores.index})
    assert two_tier.survivors(scores, family, 1) == ["a2", "b2", "b3"]
    assert two_tier.survivors(scores, family, 2) == ["a2", "a3", "b1", "b2", "b3"]
    assert two_tier.survivors(scores.drop("b2"), family, 1) == ["a2", "b3"]
    assert two_tier.survivors(scores, family, 9) == list(scores.index)


def test_the_sample_reads_ids_only():
    ids = [f"v{i}" for i in range(20)]
    first = two_tier.sample(ids, 5, 0)
    assert len(first) == 5 and first == two_tier.sample(ids, 5, 0)
    assert two_tier.sample(ids, 5, 1) != first
    scores = pd.Series(np.arange(20.0), index=ids)
    family = pd.Series("f", index=ids)
    flipped = two_tier.decide(-scores, family, rule(agreement_sample=5, agreement_seed=0))
    plain = two_tier.decide(scores, family, rule(agreement_sample=5, agreement_seed=0))
    assert flipped["sample"] == plain["sample"] == first
    assert plain["replay"] == [v for v in ids if v in set(first) | {"v19"}]


def test_the_rank_agreement_check_switches_the_screen_off():
    tier2 = pd.Series([0.1, 0.5, 0.9, 1.3, 1.7], index=list("abcde"))
    same = two_tier.agreement(tier2 * 2 + 1, tier2, rule(min_rank_agreement=0.8))
    assert same["screen_on"] and same["spearman"] == pytest.approx(1.0) and same["pairs"] == 5
    reversed_ = two_tier.agreement(-tier2, tier2, rule(min_rank_agreement=0.8))
    assert not reversed_["screen_on"] and reversed_["spearman"] == pytest.approx(-1.0)
    partial = pd.Series([1.0, 2.0, 3.0, 5.0, 4.0], index=list("abcde"))
    check = two_tier.agreement(partial, tier2, rule(min_rank_agreement=0.95))
    assert check["spearman"] == pytest.approx(0.9) and not check["screen_on"]
    assert two_tier.agreement(partial, tier2, rule(min_rank_agreement=0.9))["screen_on"]
    # fewer than three pairs (blocked exact replays) cannot pass
    few = two_tier.agreement(tier2, tier2.where(tier2 > 1.0), rule(min_rank_agreement=-1.0))
    assert few["pairs"] == 2 and few["spearman"] is None and not few["screen_on"]


def test_calibration_screen_keeps_everything_when_the_check_fails():
    rng = np.random.default_rng(0)
    columns = [f"c{i}" for i in range(12)]
    returns = pd.DataFrame(rng.normal(0, 0.01, (300, 12)), columns=columns)
    returns.iloc[:, :6] += np.linspace(0.0, 0.003, 6)  # distinct true Sharpes
    family = pd.Series({c: "f0" if i < 6 else "f1" for i, c in enumerate(columns)})
    on = rule(survivors_per_family=1, agreement_sample=6, calibration_tier1_error=0.0)
    kept, record = two_tier.calibration_columns(returns, columns, family, 250, 7, 0, on)
    assert record["agreement"]["screen_on"] and record["agreement"]["spearman"] == 1.0
    assert record["screened"] == 12 and record["replayed"] == len(kept) < 12
    off = rule(
        survivors_per_family=1,
        agreement_sample=6,
        calibration_tier1_error=50.0,
        min_rank_agreement=0.99,
    )
    kept, record = two_tier.calibration_columns(returns, columns, family, 250, 7, 0, off)
    assert not record["agreement"]["screen_on"] and kept == columns
    assert record["replayed"] == record["screened"] == 12


def test_every_tier1_variant_is_one_ledger_trial_logged_once(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    ids = [f"variant-{i}" for i in range(7)]
    args = {"rule": rule(), "data_version": "synthetic", "seed": 0}
    assert two_tier.log_screen("p" * 64, "m" * 64, ids, **args) == 7
    assert runlog.trial_count(name=two_tier.SCREEN_STAGE) == 7
    assert two_tier.log_screen("p" * 64, "m" * 64, ids, **args) == 0  # resumed: logs nothing
    assert two_tier.log_screen("p" * 64, "n" * 64, ids[:2], **args) == 2  # another screen
    assert runlog.trial_count(name=two_tier.SCREEN_STAGE) == 9
    rows = [e for e in runlog.read_entries() if e["name"] == two_tier.SCREEN_STAGE]
    assert [r["params"]["variant_id"] for r in rows[:7]] == ids
    assert {r["params"]["tier1_id"] for r in rows} == {TIER1_ID}
