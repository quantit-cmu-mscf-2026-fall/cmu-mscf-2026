"""Tests for capstone.scorecard: the gate's scorecard shape and its report.

The gate itself comes later in this stack, so these build a `GateResult` by
hand: the documented columns and header, and a report that reads back to
the same result as plain JSON beside a CSV.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from capstone.funnel import Thresholds, thresholds_dict
from capstone.scorecard import (
    HEADER_FIELDS,
    SCORECARD_COLUMNS,
    GateHeader,
    GateResult,
    TrialScope,
    read_report,
    write_report,
)

FAMILIES = ["h0", "h1", "h2"]


def _card(accepted: tuple) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    card = pd.DataFrame(
        rng.normal(size=(len(FAMILIES), len(SCORECARD_COLUMNS))),
        index=pd.Index(FAMILIES, name="family"),
        columns=list(SCORECARD_COLUMNS),
    )
    card["best"] = [f"{f}_v1" for f in FAMILIES]
    card["k"] = 5
    card["generator"] = "test-gen"
    for column in ("s1_pass", "s2_pass", "s3_pass", "s4_pass"):
        card[column] = [True, True, False]
    card["s3_measured"] = False
    card[["s3_stat", "s3_threshold", "s3_alpha", "s3_t"]] = np.nan
    card.loc["h2", ["s4_stat", "ho_sharpe", "ho_p", "ho_psr", "ho_mean", "expected_mean"]] = np.nan
    card["lfdr"] = np.nan
    card["accepted"] = card.index.isin(accepted)
    return card


def _result(accepted=("h0",), loadings=None, **header) -> GateResult:
    fields = dict(
        config_hash="a" * 64,
        registered_utc="2026-11-15T00:00:00+00:00",
        generator="test-gen",
        data_version="synthetic",
        factors_digest=None,
        thresholds=Thresholds(stage1="bootstrap"),
        n_boot=199,
        seed=0,
        trial_scope=TrialScope(name="evaluate-factor", include_tags=["synthetic"]),
        search_end="2020-12-31",
        holdout_start="2021-01-04",
        holdout_end="2025-12-31",
        trial_count=15,
        n_families=len(FAMILIES),
        n_variants=15,
        n_search_obs=7812,
        n_holdout_obs=1260,
        search_fdr_estimate=0.25,
        pbo=float("nan"),
        avg_correlation=0.12,
    )
    fields.update(header)
    loadings = pd.DataFrame() if loadings is None else loadings
    return GateResult(_card(accepted), GateHeader(**fields), tuple(accepted), loadings)


def _same(a, b) -> bool:
    """Equal, with NaN equal to NaN."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name}")


class TestShape:
    def test_every_stage_has_a_statistic_threshold_and_pass_flag(self):
        assert len(set(SCORECARD_COLUMNS)) == len(SCORECARD_COLUMNS)
        for stage in (1, 2, 3, 4):
            for part in ("stat", "threshold", "pass"):
                assert f"s{stage}_{part}" in SCORECARD_COLUMNS
        assert {"best", "k", "generator", "accepted"} <= set(SCORECARD_COLUMNS)

    def test_header_fields_are_the_dataclass_fields(self):
        assert HEADER_FIELDS == tuple(f.name for f in dataclasses.fields(GateHeader))
        assert {"trial_count", "search_fdr_estimate", "pbo", "avg_correlation"} <= set(
            HEADER_FIELDS
        )

    def test_tag_filters_are_lists_not_strings(self):
        with pytest.raises(TypeError):
            TrialScope(include_tags="synthetic")
        with pytest.raises(TypeError):
            TrialScope(exclude_tags="smoke")
        scope = TrialScope(include_tags=["a", "b"], exclude_tags=["c"])
        assert scope.include_tags == ("a", "b") and scope.exclude_tags == ("c",)

    def test_expected_means_are_the_accepted_families(self):
        result = _result(accepted=("h0", "h1"))
        card = result.scorecard
        assert result.expected_means == {
            "h0": card.loc["h0", "expected_mean"],
            "h1": card.loc["h1", "expected_mean"],
        }
        assert _result(accepted=()).expected_means == {}


class TestReport:
    def test_writes_plain_json_beside_a_csv(self, tmp_path):
        result = _result()
        path = tmp_path / "reports" / "gate_report.json"
        assert write_report(result, path) == (path, path.with_suffix(".csv"))
        assert path.with_suffix(".csv").exists()
        text = path.read_text(encoding="utf-8")
        report = json.loads(text, parse_constant=_reject_constant)  # NaN written as null
        assert report["scorecard_csv"] == "gate_report.csv"
        assert report["header"]["thresholds"] == thresholds_dict(result.header.thresholds)
        assert report["header"]["pbo"] is None
        assert report["accepted"] == ["h0"]

    @pytest.mark.parametrize(
        "accepted, loadings",
        [
            (("h0",), None),
            ((), None),
            (("h0", "h1"), pd.DataFrame({"Mkt-RF": [0.9, 0.1], "SMB": [np.nan, -0.2]})),
        ],
    )
    def test_report_round_trips(self, tmp_path, accepted, loadings):
        if loadings is not None:
            loadings.index = pd.Index(["h0", "h1"], name="family")
        result = _result(accepted=accepted, loadings=loadings)
        path = tmp_path / "gate_report.json"
        write_report(result, path)
        back = read_report(path)
        pd.testing.assert_frame_equal(back.scorecard, result.scorecard)
        assert back.accepted == result.accepted
        assert _same(dataclasses.asdict(back.header), dataclasses.asdict(result.header))
        assert back.header.thresholds == result.header.thresholds
        assert back.header.trial_scope == result.header.trial_scope
        pd.testing.assert_frame_equal(back.loadings, result.loadings)

    @pytest.mark.parametrize(
        "families, best",
        [
            (["NA", "EU", "JP"], None),  # "NA" is a default missing marker
            (["001", "002", "003"], None),  # would be read as 1, 2, 3
            (FAMILIES, ["7", "8", "9"]),  # best would be read as int
        ],
    )
    def test_labels_read_back_as_written(self, tmp_path, families, best):
        result = _result(accepted=("h0", "h1"))
        card = result.scorecard.rename(index=dict(zip(FAMILIES, families, strict=True)))
        if best is not None:
            card["best"] = best
        card["generator"] = "000"
        result = dataclasses.replace(result, scorecard=card, accepted=tuple(families[:2]))
        path = tmp_path / "gate_report.json"
        write_report(result, path)
        back = read_report(path)
        pd.testing.assert_frame_equal(back.scorecard, result.scorecard)
        assert back.accepted == result.accepted

    @pytest.mark.parametrize("accepted", [("h1", "h0"), ("h0", "h2")])
    def test_a_result_whose_accepted_disagrees_with_its_scorecard_is_refused(self, accepted):
        # Caught when the result is built, not when the report is read at deploy.
        good = _result(accepted=("h0", "h1"))
        with pytest.raises(ValueError, match="accepted"):
            dataclasses.replace(good, accepted=accepted)

    def test_an_accepted_list_that_disagrees_with_the_scorecard_is_refused(self, tmp_path):
        path = tmp_path / "gate_report.json"
        write_report(_result(accepted=("h0",)), path)
        report = json.loads(path.read_text(encoding="utf-8"))
        report["accepted"] = ["h0", "h1"]
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(ValueError, match="accepted"):
            read_report(path)

    @pytest.mark.parametrize("column", ["s1_pass", "s3_measured", "s4_pass"])
    @pytest.mark.parametrize("cell", ["", "maybe", "1", "yes", "true"])
    def test_a_malformed_flag_cell_is_refused_not_read_as_true(self, tmp_path, column, cell):
        # `astype(bool)` turned an empty cell (NaN) into True: a family that
        # never reached stage 4 could read back as having passed it.
        path = tmp_path / "gate_report.json"
        _, csv_path = write_report(_result(accepted=("h0",)), path)
        card = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        card.loc[2, column] = cell
        card.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match=column):
            read_report(path)
