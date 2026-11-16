"""Discovery evidence for reference returns: synthetic returns, logged before computed."""

import numpy as np
import pandas as pd
import pytest

from capstone import runlog
from capstone.pipeline import reference_validation
from capstone.pipeline.reference_validation import discovery_evidence
from capstone.reference import generator_name
from capstone.strategy.reference import ReferenceStrategySpec
from capstone.strategy.research import contracts

DATA_VERSION = "synthetic-validation-fixture"


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


def specs():
    claims = [c["id"] for c in contracts()["claims"] if c["status"] == "executable"]
    return [
        ReferenceStrategySpec(claim, DATA_VERSION, "2000-12-29", side)
        for claim in claims[:2]
        for side in ("long_only", "long_short")
    ]


def returns(items, end="2005-12-30", seed=7):
    index = pd.bdate_range(end=end, periods=300)
    noise = np.random.default_rng(seed).normal(0, 0.01, (len(index), len(items)))
    return pd.DataFrame(noise, index=index, columns=[s.id for s in items])


def test_signal_free_returns_give_one_row_per_family_and_no_admission():
    items = specs()
    result = discovery_evidence(returns(items), items)
    assert result["generator"] == generator_name()
    assert result["formally_admitted"] is False and result["holdout_opened"] is False
    assert result["selection_policy_qualified"] is False
    assert {row["family"] for row in result["families"]} == {s.claim_id for s in items}
    assert all(0 < row["pvalue"] <= 1 for row in result["families"])
    assert all(row["bh_adjusted"] >= row["pvalue"] - 1e-12 for row in result["families"])
    assert all(row["pvalue"] > 0.01 for row in result["families"])


def test_a_planted_signal_is_found_in_its_family_only():
    items = specs()
    data = returns(items)
    data[items[0].id] += 0.01
    rows = {r["family"]: r for r in discovery_evidence(data, items)["families"]}
    assert rows[items[0].claim_id]["pvalue"] < 0.01
    assert rows[items[2].claim_id]["pvalue"] > 0.01


def test_the_call_is_logged_before_any_statistic_is_computed(monkeypatch):
    items = specs()

    def inspect_ledger(*args, **kwargs):
        assert [e["name"] for e in runlog.read_entries()] == ["reference-discovery-screen"]
        raise RuntimeError("stop after the ledger check")

    monkeypatch.setattr(reference_validation, "bootstrap_family_pvalues", inspect_ledger)
    with pytest.raises(RuntimeError, match="stop after"):
        discovery_evidence(returns(items), items)
    entry = runlog.read_entries()[0]
    assert entry["params"]["data_version"] == DATA_VERSION
    assert entry["params"]["bootstrap_draws"] == 1999


def test_returns_after_the_discovery_period_are_refused_without_a_ledger_entry():
    items = specs()
    with pytest.raises(ValueError, match="sealed holdout"):
        discovery_evidence(returns(items, end="2021-03-31"), items)
    assert runlog.read_entries() == []


def test_a_column_without_a_family_is_refused():
    items = specs()
    data = returns(items)
    data["unknown"] = 0.0
    with pytest.raises(ValueError, match="research family"):
        discovery_evidence(data, items)
