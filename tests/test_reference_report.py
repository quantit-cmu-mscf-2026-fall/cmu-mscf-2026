"""capstone.pipeline.reference_report: renders figures and RESULTS.html from a results folder.

The folder is synthetic (made here from a seeded generator); no published data is read and
nothing leaves `tmp_path`.
"""

from __future__ import annotations

import json

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


def write_results(root, sharpe=0.41):
    rng = np.random.default_rng(0)
    months = pd.date_range("2000-01-31", periods=36, freq="ME")
    for mode in reference_report.LABELS:
        folder = root / "assumed_base" / mode
        folder.mkdir(parents=True)
        weights = rng.dirichlet(np.ones(len(FACTORS)), size=len(months)) * 0.9
        frame = pd.DataFrame(weights, columns=[f"weight_{n}" for n in FACTORS], index=months)
        frame.insert(0, "equity", 100000 * np.cumprod(1 + rng.normal(0.004, 0.02, len(months))))
        frame.index.name = "month"
        frame.to_csv(folder / "monthly.csv")
    correlation = pd.DataFrame(np.eye(5) * 0.6 + 0.4, index=FACTORS, columns=FACTORS)
    correlation.to_csv(root / "correlations.csv")
    period = {**PERIOD, "sharpe": sharpe}
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
                    for n in FACTORS
                }
            allocations[f"{scenario}/{mode}"] = item
    results = {"allocations": allocations, "standalone_gross": {n: PERIOD for n in FACTORS}}
    (root / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"git_commit": "abc123"}), encoding="utf-8")


def test_report_writes_figures_and_html(tmp_path):
    write_results(tmp_path)
    reference_report.report(tmp_path)
    for name in ("portfolio-results.png", "relationships-and-weights.png", "RESULTS.html"):
        assert (tmp_path / name).stat().st_size > 0
    html = (tmp_path / "RESULTS.html").read_text(encoding="utf-8")
    assert "5.12%" in html and "0.41" in html and "-18.34%" in html
    assert "undefined" in html  # a null Sharpe is shown, not dropped or crashed on
    assert "<code>abc123</code>" in html and "Shrunk maximum Sharpe" in html
    assert "[-0.12, 0.31]" in html and "$12,346" in html


def test_report_is_self_contained(tmp_path):
    write_results(tmp_path)
    reference_report.report(tmp_path)
    html = (tmp_path / "RESULTS.html").read_text(encoding="utf-8")
    assert str(tmp_path) not in html
    assert 'src="portfolio-results.png"' in html and 'src="relationships-and-weights.png"' in html


def test_missing_inputs_raise_rather_than_render_a_partial_report(tmp_path):
    write_results(tmp_path)
    (tmp_path / "correlations.csv").unlink()
    with pytest.raises(FileNotFoundError):
        reference_report.report(tmp_path)
    assert not (tmp_path / "RESULTS.html").exists()
