"""Evaluation as the deepen move's scorer (QUANTIT-114): `EvaluationScorer`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from test_factor_evaluation import fake_rows

from capstone.factors import evaluation as ev
from capstone.factors.evaluation import HoldoutError, build_panel, factor_returns
from capstone.factors.llm import load_config
from capstone.factors.scoring import EvaluationScorer, crsp_settings
from capstone.factors.tree import parse


def test_the_scorer_gives_the_screen_z_and_the_variants_net_returns(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    panel = build_panel(fake_rows(n_days=500, planted=0.004), data_version="v-test")
    spec = ev.TradingSpec(hold=5)
    scorer = EvaluationScorer(panel, spec)
    good, bad = scorer(parse("volume")), scorer(parse("-volume"))
    net = factor_returns(parse("volume"), panel, spec).net
    assert good.quality == pytest.approx(stats.norm.isf(ev.screen_pvalue(net)))
    assert good.quality > 2.3 and bad.quality < -2.3  # the sign is kept
    np.testing.assert_array_equal(good.values, net.to_numpy())
    assert good.params["data_version"] == "v-test" and good.params["spec"]["hold"] == 5
    assert good.metrics["pvalue_screen"] == pytest.approx(ev.screen_pvalue(net))
    assert not (tmp_path / "runs.jsonl").exists()  # the scorer never logs; deepen does


def test_the_scorer_returns_none_for_a_factor_it_cannot_compute(monkeypatch):
    def failing(tree, fields, factors, **options):
        raise ValueError("cannot compute")

    monkeypatch.setattr(ev, "evaluate_pit", failing)
    scorer = EvaluationScorer(build_panel(fake_rows(n_stocks=9, n_days=40)), ev.TradingSpec())
    assert scorer(parse("volume")) is None


def test_the_scorer_logs_the_panels_own_period():
    # A synthetic panel is not 1990-2020; the trial's params say what it scored.
    rows = fake_rows(n_stocks=9, n_days=60, start="2016-03-01")
    scored = EvaluationScorer(build_panel(rows), ev.TradingSpec())(parse("volume"))
    dates = pd.DatetimeIndex(rows["date"])
    expected = [str(dates.min().date()), str(dates.max().date())]
    assert expected == ["2016-03-01", "2016-05-23"]
    assert scored.params["period"] == expected


def test_the_scorer_refuses_a_panel_that_holds_the_holdout():
    import dataclasses

    panel = build_panel(fake_rows(n_stocks=6, n_days=40))
    late = panel.universe.copy()
    late.index = late.index[:-1].append(pd.DatetimeIndex(["2021-02-01"]))
    with pytest.raises(HoldoutError):
        EvaluationScorer(dataclasses.replace(panel, universe=late), ev.TradingSpec())


def test_crsp_settings_put_the_memory_on_the_z_scale():
    config, spec = crsp_settings(load_config())
    assert spec == ev.TradingSpec(hold=21, neutral="none")
    assert config.memory_signed_quality and config.memory_quality_edges == (0.0, 1.0, 1.645)
    assert config.memory_min_quality == 0.0 and config.memory_high_quality == 1.645
    assert load_config().memory_quality_edges == (0.05, 0.10, 0.20)  # the default is untouched


def test_crsp_settings_refuse_unknown_sections_and_keys(tmp_path):
    bad_section = tmp_path / "a.toml"
    bad_section.write_text("[surprise]\nx = 1\n")
    with pytest.raises(ValueError, match="unknown sections"):
        crsp_settings(load_config(), bad_section)
    bad_key = tmp_path / "b.toml"
    bad_key.write_text("[memory]\nmemory_no_such_key = 1\n")
    with pytest.raises(ValueError, match="unknown .memory. keys"):
        crsp_settings(load_config(), bad_key)
