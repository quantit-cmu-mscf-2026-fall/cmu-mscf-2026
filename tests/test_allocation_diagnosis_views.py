import numpy as np
import pandas as pd
import pytest

from capstone.pipeline.allocation_diagnosis_views import (
    LABELS,
    SCENARIOS,
    _figures,
    _number,
    _table,
    diagnostic_tables,
    pool_tables,
)
from capstone.pipeline.reference_allocation import statistics
from capstone.pipeline.reference_inputs import POOL


def history_frame():
    dates = pd.date_range("1996-01-01", periods=300, freq="MS")
    returns = pd.Series(0.004 + 0.02 * np.sin(np.arange(300)), index=dates)
    return pd.DataFrame(
        {
            "return": returns,
            "excess": returns - 0.001,
            "equity": 100000 * (1 + returns).cumprod(),
            "invested": 0.8,
            "active": 4,
            "ongoing_cost": 10.0,
            "switch_cost": 1.0,
            "turnover": 0.1,
        }
    )


def diagnostic(solver):
    return {
        "forecast": {
            "scored_quarters": 7,
            "quarter_arithmetic_rmse": {"raw": 0.03, "half_shrunk": 0.02, "zero_mean": 0.01},
            "forecasts": [
                {
                    "status": "forecast_created",
                    "sample_covariance_annual_volatility": 0.1,
                    "shrunk_covariance_annual_volatility": 0.09,
                },
                {"status": "skipped"},
            ],
            "following_quarter_scores": [
                {"status": "scored", "actual_three_month_annualized_scatter": 0.11}
            ],
        },
        "concentration": {
            "mean_normalized_max_weight": 0.5,
            "maximum_normalized_max_weight": 1.0,
            "mean_effective_number": 2.0,
            "BAB_signed_pnl_fraction": 1.25,
            "cash_only_months": 3,
        },
        "risk_sharpe_identity": {
            "annual_excess_mean_difference": 0.001,
            "volatility_term": 0.003,
            "sharpe_term": -0.002,
        },
        "solver": solver,
    }


def test_number_formats_and_marks_missing_values_undefined():
    assert _number(0.1234, "percent") == "12.34%"
    assert _number(0.001, "pp") == "+0.100 pp"
    assert _number(1234567.4, "money") == "$1,234,567"
    assert _number(2 / 3) == "0.667"
    assert _number(None) == _number(float("nan")) == _number(float("inf")) == "undefined"


def test_table_escapes_headers_and_caption_but_not_prerendered_cells():
    html = _table("a<b", ["h&1"], [["<b>x</b>"]])
    assert "<caption>a&lt;b</caption>" in html and "h&amp;1" in html
    assert "<td><b>x</b></td>" in html


def test_diagnostic_tables_report_each_rule_and_skip_empty_solver_rows():
    solver = [
        {
            "lattice_points_evaluated": 1001,
            "local_successes": 4,
            "local_attempts": 5,
            "best_lattice_objective": 0.15,
            "best_local_objective": 0.12,
            "feasible_prior_included": True,
        },
        {"lattice_points_evaluated": 1001, "best_lattice_objective": None},
    ]
    results = {"diagnostics": {}}
    for mode in LABELS:
        results["diagnostics"]["legacy/" + mode] = diagnostic([])
        results["diagnostics"]["v2/" + mode] = diagnostic(solver)
    forecast, risk, concentration, identity, solver_table = diagnostic_tables(results)
    assert forecast.count("<tr>") == 7 and "V2: Median-window Sharpe" in forecast
    assert "10.00%" in risk and "9.00%" in risk and "11.00%" in risk
    assert "125.00%" in concentration and "<td>3</td>" in concentration
    assert "+0.100 pp" in identity and "-0.200 pp" in identity
    assert solver_table.count("<tr>") == 4  # header plus the three V2 rules only
    assert "[1001]" in solver_table and "4/5" in solver_table and "0.030" in solver_table


def test_pool_tables_list_every_factor_and_every_five_year_slice():
    factor = {
        "annual_excess_mean": 0.02,
        "annual_volatility": 0.1,
        "sharpe": 0.2,
        "annual_excess_mean_95_marginal_interval": [-0.01, 0.05],
    }
    pool = {
        "overall": dict.fromkeys(POOL, factor),
        "five_year_periods": {
            f"{y}-{y + 4}": dict.fromkeys(POOL, factor) for y in range(1996, 2021, 5)
        },
    }
    overall, periods = pool_tables(pool)
    assert overall.count("<tr>") == len(POOL) + 1
    assert periods.count("<tr>") == 5 * len(POOL) + 1
    assert "-1.00% to 5.00%" in overall


def test_figures_write_both_png_files(tmp_path):
    history = history_frame()
    item = {"full_1996_2020": statistics(history)}
    legacy = {"assumed_base/" + mode: history for mode in LABELS}
    corrected = dict.fromkeys(LABELS, history)
    results = {
        "frozen_target_costs": {mode: dict.fromkeys(SCENARIOS, item) for mode in LABELS},
        "prior_risk_overlay": dict.fromkeys(LABELS, item),
    }
    _figures(tmp_path, results, legacy, corrected)
    for name in ("version-comparison.png", "cost-and-risk-controls.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_figures_fail_loudly_when_a_rule_is_missing(tmp_path):
    with pytest.raises(KeyError):
        _figures(tmp_path, {}, {}, {})
