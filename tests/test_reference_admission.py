import json

import pandas as pd
import pytest

from capstone import reference
from capstone.strategy.recorded import RecordedExecutionBlocked, RecordedStrategySpec
from capstone.strategy.reference import ReferenceStrategySpec

CLAIM = "533148edf022a26e"


def spec():
    return ReferenceStrategySpec(CLAIM, "synthetic-v1", "2000-01-31")


def write_report(tmp_path, spec, **overrides):
    header = {
        "generator": reference.generator_name(),
        "data_version": spec.data_version,
        "n_holdout_obs": 12,
    }
    report = {"header": header, "accepted": [spec.claim_id], "scorecard_csv": "card.csv"}
    for key, value in overrides.items():
        if key in header:
            header[key] = value
        else:
            report[key] = value
    pd.DataFrame({"best": [spec.variant_id]}, index=[spec.claim_id]).to_csv(tmp_path / "card.csv")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_generator_name_is_versioned_and_stable():
    name = reference.generator_name()
    assert name.startswith("reference-v") and name == reference.generator_name()


def test_missing_report_never_admits():
    result = reference.admission(spec())
    assert result["admitted"] is False
    assert result["blockers"] == ["reviewed_gate_report_missing"]
    assert result["report_hash"] is None


def test_matching_report_admits(tmp_path):
    s = spec()
    result = reference.admission(s, write_report(tmp_path, s))
    assert result["admitted"] is True and result["blockers"] == []
    assert result["report_hash"]


@pytest.mark.parametrize(
    "override, blocker",
    [
        ({"generator": "reference-v0/other"}, "superseded_or_unrelated_validation_generator"),
        ({"data_version": "other"}, "validation_data_version_mismatch"),
        ({"accepted": []}, "family_not_accepted"),
        ({"n_holdout_obs": 0}, "formal_validation_incomplete"),
        ({"scorecard_csv": "../card.csv"}, "invalid_scorecard_reference"),
    ],
)
def test_each_unmet_condition_blocks(tmp_path, override, blocker):
    s = spec()
    result = reference.admission(s, write_report(tmp_path, s, **override))
    assert result["admitted"] is False
    assert blocker in result["blockers"]


def test_chosen_variant_must_match_scorecard(tmp_path):
    s = spec()
    path = write_report(tmp_path, s)
    pd.DataFrame({"best": ["not-the-variant"]}, index=[s.claim_id]).to_csv(tmp_path / "card.csv")
    assert "chosen_variant_mismatch" in reference.admission(s, path)["blockers"]


def test_recorded_specs_cannot_be_admitted():
    class Recorded(RecordedStrategySpec):
        def __init__(self):
            pass

    with pytest.raises(RecordedExecutionBlocked):
        reference.admission(Recorded())
