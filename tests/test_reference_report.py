"""capstone.pipeline.reference_report: renders figures and RESULTS.html from a results folder.

The folder is synthetic (made here from a seeded generator); no published data is read and
nothing leaves `tmp_path`.
"""

from __future__ import annotations

import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from capstone.pipeline import reference_report

FACTORS = ["HML", "RMW", "CMA", "Mom", "BAB"]
PERIOD = {
    "cagr": 0.0512,
    "sharpe": 0.41,
    "annual_volatility": 0.0987,
    "maximum_drawdown": -0.1834,
    "mean_invested": 0.62,
    "active_range": "2-5 sleeves",
}


SETTINGS = {
    "capital": 100000,
    "period": {"full": "2000–2002", "later": "2002", "retained": "2000–2002"},
    "bootstrap": {"draws": 777, "block_months": 6, "seed": 31},
    "costs": {"invested_bp": 7, "switching_bp": 3, "doubled_multiple": 2},
    "shrinkage": 0.35,
    "volatility_ceiling": 0.08,
}


def write_results(root, sharpe=0.41, settings=None, factors=FACTORS, period=PERIOD, names=None):
    names = names or factors
    rng = np.random.default_rng(0)
    months = pd.date_range("2000-01-31", periods=36, freq="ME")
    for mode in reference_report.LABELS:
        folder = root / "assumed_base" / mode
        folder.mkdir(parents=True)
        weights = rng.dirichlet(np.ones(len(factors)), size=len(months)) * 0.9
        frame = pd.DataFrame(weights, columns=[f"weight_{n}" for n in factors], index=months)
        frame.insert(0, "equity", 100000 * np.cumprod(1 + rng.normal(0.004, 0.02, len(months))))
        frame.index.name = "month"
        frame.to_csv(folder / "monthly.csv")
    size = len(factors)
    correlation = pd.DataFrame(np.eye(size) * 0.6 + 0.4, index=factors, columns=factors)
    correlation.to_csv(root / "correlations.csv")
    period = {**period, "sharpe": sharpe}
    allocations = {}
    for scenario in ("assumed_base", "assumed_doubled"):
        for mode in reference_report.LABELS:
            item = {"full_1996_2020": period, "later_2016_2020": {**period, "sharpe": None}}
            if scenario == "assumed_base":
                item["paired_uncertainty_vs_equal"] = {
                    "sharpe_difference_95_interval": [-0.12, 0.31]
                }
                item["contribution"] = {
                    n: {"mean_coefficient": 0.18, "factor_pnl_dollars_before_proxy_costs": 12345.6}
                    for n in names
                }
            allocations[f"{scenario}/{mode}"] = item
    results = {"allocations": allocations, "standalone_gross": {n: PERIOD for n in names}}
    (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"git_commit": "abc123", "settings": settings or copy.deepcopy(SETTINGS)}),
        encoding="utf-8",
    )


def test_report_writes_figures_and_html(tmp_path):
    write_results(tmp_path)
    reference_report.report(tmp_path)
    for name in ("portfolio-results.png", "relationships-and-weights.png", "RESULTS.html"):
        assert (tmp_path / name).stat().st_size > 0
    html = (tmp_path / "RESULTS.html").read_text(encoding="utf-8")
    assert "5.12%" in html and "0.41" in html and "-18.34%" in html
    assert "undefined" in html
    assert "<code>abc123</code>" in html and "Shrunk maximum Sharpe" in html
    assert "[-0.12, 0.31]" in html and "$12,346" in html


def test_report_is_self_contained(tmp_path):
    write_results(tmp_path)
    reference_report.report(tmp_path)
    html = (tmp_path / "RESULTS.html").read_text(encoding="utf-8")
    assert str(tmp_path) not in html
    assert not re.search(r"(?<![\w/:.])(?:[A-Za-z]:[\/]|/(?:Users|home|tmp|var|private)/)", html)
    assert 'src="portfolio-results.png"' in html and 'src="relationships-and-weights.png"' in html


def test_missing_inputs_raise_rather_than_render_a_partial_report(tmp_path):
    write_results(tmp_path)
    (tmp_path / "correlations.csv").unlink()
    with pytest.raises(FileNotFoundError):
        reference_report.report(tmp_path)
    assert not (tmp_path / "RESULTS.html").exists()


def render(root, **kwargs):
    write_results(root, **kwargs)
    reference_report.report(root)
    return (root / "RESULTS.html").read_text(encoding="utf-8")


def with_setting(path, value):
    settings = copy.deepcopy(SETTINGS)
    node = settings
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return settings


def test_page_states_the_settings_from_the_manifest(tmp_path):
    html = render(tmp_path)
    for text in ("2000–2002", "777 draws of 6-month blocks", "seed 31", "7bp", "3bp", "14bp"):
        assert text in html
    assert "35%" in html and "8%" in html and "$100,000" in html
    assert "1996" not in html and "seed 404" not in html and "1,999" not in html


CHANGES = [
    (("capital",), 25000, "$25,000"),
    (("period", "full"), "1990–1999", "1990–1999"),
    (("period", "later"), "1995", "Later slice: 1995"),
    (("period", "retained"), "1980–2010", "Only 1980–2010"),
    (("bootstrap", "draws"), 4321, "4,321 draws"),
    (("bootstrap", "block_months"), 9, "9-month blocks"),
    (("bootstrap", "seed"), 99, "seed 99"),
    (("costs", "invested_bp"), 11, "11bp"),
    (("costs", "switching_bp"), 5, "5bp"),
    (("costs", "doubled_multiple"), 3, "21bp"),
    (("shrinkage",), 0.6, "60% shrinkage"),
    (("volatility_ceiling",), 0.12, "12% estimated"),
]


@pytest.mark.parametrize(("path", "value", "expected"), CHANGES)
def test_changing_one_setting_changes_the_page(tmp_path, path, value, expected):
    base = render(tmp_path / "a", settings=None) if (tmp_path / "a").mkdir() is None else ""
    (tmp_path / "b").mkdir()
    changed = render(tmp_path / "b", settings=with_setting(path, value))
    assert expected in changed and expected not in base
    assert changed != base


MISSING = [("capital",), ("period",), ("bootstrap",), ("costs",), ("shrinkage",)]
MISSING += [
    ("volatility_ceiling",),
    ("period", "full"),
    ("bootstrap", "seed"),
    ("costs", "invested_bp"),
]


@pytest.mark.parametrize("path", MISSING)
def test_missing_setting_raises_and_leaves_no_page(tmp_path, path):
    settings = copy.deepcopy(SETTINGS)
    node = settings
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    write_results(tmp_path, settings=settings)
    with pytest.raises(ValueError, match=path[-1]):
        reference_report.report(tmp_path)
    assert not (tmp_path / "RESULTS.html").exists()


def test_drawdown_matches_the_series_for_any_starting_capital():
    equity = np.array([10500.0, 9975.0, 10200.0])
    expected = np.array([0.0, 9975 / 10500 - 1, 10200 / 10500 - 1])
    np.testing.assert_allclose(reference_report.drawdown(equity, 10000), expected)
    assert reference_report.drawdown(equity, 10000).min() > -0.06
    assert reference_report.drawdown(equity, 100000).min() < -0.9


def test_report_plots_drawdown_from_the_manifest_capital(tmp_path, monkeypatch):
    seen = []
    real = reference_report.drawdown
    monkeypatch.setattr(
        reference_report,
        "drawdown",
        lambda equity, capital: seen.append(capital) or real(equity, capital),
    )
    render(tmp_path, settings=with_setting(("capital",), 10000))
    assert seen == [10000] * len(reference_report.LABELS)


def test_failed_rerun_removes_the_stale_page_and_draws_nothing(tmp_path):
    render(tmp_path)
    (tmp_path / "portfolio-results.png").unlink()
    (tmp_path / "correlations.csv").unlink()
    with pytest.raises(FileNotFoundError):
        reference_report.report(tmp_path)
    assert not (tmp_path / "RESULTS.html").exists()
    assert not (tmp_path / "portfolio-results.png").exists()


@pytest.mark.parametrize("count", [4, 6])
def test_correlation_size_follows_the_data(tmp_path, count):
    factors = [f"F{i}" for i in range(count)]
    html = render(tmp_path, factors=factors)
    assert "F3" in html


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_render_as_undefined(tmp_path, bad):
    html = render(tmp_path, period={**PERIOD, "cagr": bad, "maximum_drawdown": bad})
    assert not re.search(r"(?<![a-z])-?(nan|inf)(?![a-z])", html.lower())
    assert html.count("undefined") > 10


def test_names_and_active_range_are_escaped(tmp_path):
    html = render(
        tmp_path,
        names=["<b>X</b>", "A&B"],
        period={**PERIOD, "active_range": "<script>alert(1)</script>"},
    )
    assert "<b>X</b>" not in html and "&lt;b&gt;X&lt;/b&gt;" in html and "A&amp;B" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
