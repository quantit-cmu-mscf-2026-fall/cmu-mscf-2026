"""The report renders injected, synthetic frames; no licensed data is read."""

import json

import pandas as pd

from capstone.pipeline.qualification_report import _html, _number, render


def _summary(**over):
    base = {
        "study_hash": "fixture",
        "synthetic_only": False,
        "historical_observations_loaded": True,
        "evaluated_candidates": 0,
        "release": {"commit": "fixture"},
    }
    return {**base, **over}


def test_padded_short_returns_do_not_generate_empty_correlation_charts(tmp_path):
    dates = pd.bdate_range("2000-01-03", periods=300)
    matrix = pd.DataFrame(float("nan"), index=dates, columns=["a", "b"])
    matrix.iloc[:10] = 0.01
    matrix.to_parquet(tmp_path / "candidate-returns.parquet")
    (tmp_path / "correlations.json").write_text(
        json.dumps(
            {"rolling_252_month_ends": [{"as_of": "2000-12-29", "median": None, "p90": None}]}
        )
    )
    render(tmp_path, _summary(), [], [], [], {"sources": {}})
    assert not (tmp_path / "correlations.png").exists()
    assert not (tmp_path / "rolling-correlations.png").exists()
    assert "252 overlapping observed daily returns" in (tmp_path / "REPORT.md").read_text()


def test_missing_performance_is_na_never_zero():
    assert _number(None) == "N/A"
    assert _number(None, percent=True) == "N/A"
    assert _number(0.0) == "0.00"
    assert _number(0.1234, percent=True) == "12.34%"


def test_html_escapes_markup_and_renders_tables():
    out = _html("# T\n\n| a | b |\n|---|---|\n| <script> | **x** |\n\n[l](f.json)")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<h1>T</h1>" in out
    assert "<strong>x</strong>" in out
    assert "<a href='f.json'>l</a>" in out
    assert out.count("<tr>") == 2


def test_blocked_report_lists_candidates_and_flags(tmp_path):
    candidates = [
        {
            "construction": "reference",
            "method": "m",
            "side": "long_only",
            "candidate_id": "c1",
            "status": "blocked",
        }
    ]
    preflight = {
        "sources": {"prices": {"qualified": False, "blockers": ["missing"]}},
        "descriptor_present": False,
        "global_blockers": ["no sessions"],
    }
    render(tmp_path, _summary(historical_observations_loaded=False), candidates, [], [], preflight)
    text = (tmp_path / "REPORT.md").read_text()
    assert "Historical evaluation is blocked" in text
    assert "| prices | False | missing |" in text
    assert "| m | long_only | reference" in text
    assert "N/A" in text
    assert "Global blockers: no sessions." in text
    assert "Holdout opened: **false**" in text
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "coverage.png").exists()
    assert (tmp_path / "plot-runtime.json").exists()


def test_completed_portfolio_is_plotted_from_daily_artifact(tmp_path):
    (tmp_path / "p1").mkdir()
    dates = pd.bdate_range("2001-01-02", periods=20)
    pd.DataFrame({"net": 0.001}, index=dates).to_parquet(tmp_path / "p1" / "daily.parquet")
    portfolios = [
        {
            "completed": True,
            "allocation": "equal_family",
            "scenario": "base",
            "profile": "long_only",
            "kind": "basket",
            "artifacts": "p1",
            "status": "evaluated",
        }
    ]
    render(tmp_path, _summary(synthetic_only=True), [], portfolios, [], {"sources": {}})
    assert (tmp_path / "portfolio-results.png").exists()
    text = (tmp_path / "REPORT.md").read_text()
    assert "![Portfolio comparisons](portfolio-results.png)" in text
    assert "Synthetic prices cannot establish investment performance." in text
