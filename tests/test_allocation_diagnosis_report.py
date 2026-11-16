import json
import re

import numpy as np
import pandas as pd
import pytest

from capstone.pipeline.allocation_diagnosis_report import LABELS, SCENARIOS, report
from capstone.pipeline.reference_allocation import statistics
from capstone.pipeline.reference_inputs import POOL


def report_fixture(root):
    dates = pd.date_range("1996-01-01", periods=300, freq="MS")
    returns = pd.Series(0.004 + 0.02 * np.sin(np.arange(300)), index=dates)
    history = pd.DataFrame(
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
    history.index.name = "month"
    item = {
        "full_1996_2020": statistics(history),
        "paired_vs_old_equal": {"sharpe_difference_95_interval": [None, None]},
    }
    results = {
        "legacy_equivalence": {},
        "corrected_v2": {},
        "frozen_target_costs": {},
        "prior_risk_overlay": {},
        "hindsight_fixed_mix": {},
        "diagnostics": {},
    }
    for scenario in SCENARIOS:
        for mode in LABELS:
            key = scenario + "/" + mode
            results["legacy_equivalence"][key] = True
            results["corrected_v2"][key] = item
            for version in ("legacy-reproduction", "corrected-v2"):
                target = root / version / key
                target.mkdir(parents=True)
                history.to_csv(target / "monthly.csv")
                (target / "decisions.json").write_text("[]")
    f = {
        "scored_quarters": 1,
        "quarter_arithmetic_rmse": {"raw": 0.03, "half_shrunk": 0.02, "zero_mean": 0.01},
        "forecasts": [
            {
                "status": "forecast_created",
                "sample_covariance_annual_volatility": 0.1,
                "shrunk_covariance_annual_volatility": 0.09,
            }
        ],
        "following_quarter_scores": [
            {"status": "scored", "actual_three_month_annualized_scatter": 0.11}
        ],
    }
    c = {
        "mean_normalized_max_weight": 0.5,
        "maximum_normalized_max_weight": 1.0,
        "mean_effective_number": 2.0,
        "BAB_signed_pnl_fraction": 1.25,
        "cash_only_months": 0,
    }
    identity = {
        "annual_excess_mean_difference": 0.001,
        "volatility_term": 0.003,
        "sharpe_term": -0.002,
    }
    solver = [
        {
            "lattice_points_evaluated": 1001,
            "local_successes": 4,
            "local_attempts": 5,
            "best_lattice_objective": 0.15,
            "best_local_objective": 0.16,
            "feasible_prior_included": True,
        }
    ]
    for mode in LABELS:
        results["frozen_target_costs"][mode] = dict.fromkeys(SCENARIOS, item)
        results["prior_risk_overlay"][mode] = item
        for version in ("legacy", "v2"):
            results["diagnostics"][version + "/" + mode] = {
                "forecast": f,
                "concentration": c,
                "risk_sharpe_identity": identity,
                "solver": solver if version == "v2" else [],
            }
    for mode in list(LABELS)[1:]:
        results["hindsight_fixed_mix"][mode] = {
            **item,
            "fit": {"weights": dict.fromkeys(POOL, 0.2)},
        }
    factor = {
        "annual_excess_mean": 0.02,
        "annual_volatility": 0.1,
        "sharpe": 0.2,
        "annual_excess_mean_95_marginal_interval": [-0.01, 0.05],
    }
    results["pool_evidence"] = {
        "overall": dict.fromkeys(POOL, factor),
        "five_year_periods": {
            f"{y}-{y + 4}": dict.fromkeys(POOL, factor) for y in range(1996, 2021, 5)
        },
    }
    manifest = {
        "git_commit": "fixture<unsafe>",
        "registration": {
            "trial_accounting": {"v1_reproduction_comparisons": 9, "v2_policy_cost_comparisons": 9}
        },
    }
    (root / "results.json").write_text(json.dumps(results))
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "trial-accounting.json").write_text("{}")
    (root / "comparisons.jsonl").write_text("{}\n")


def test_report_renders_every_control_preserves_evidence_and_local_links(tmp_path):
    report_fixture(tmp_path)
    preserved = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report(tmp_path)
    html = (tmp_path / "RESULTS.html").read_text(encoding="utf-8")
    assert html.count("NONDEPLOYABLE: ") == 2
    assert html.count('href="corrected-v2/') == 18
    assert "9 of 9 original comparisons reproduced exactly" in html
    assert "not mathematical upper bounds" in html
    assert "not equal realized-risk" in " ".join(html.split())
    assert "Zero-mean RMSE" in html
    assert "125.00%" in html
    assert "fixture&lt;unsafe&gt;" in html and "fixture<unsafe>" not in html
    for name in ("version-comparison.png", "cost-and-risk-controls.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for link in re.findall(r'(?:href|src)="([^"]+)"', html):
        assert (tmp_path / link).is_file(), link
    assert not re.search(r'(?:href|src)="https?://', html)
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_report_marks_an_undefined_interval_and_refuses_missing_evidence(tmp_path):
    report_fixture(tmp_path)
    results = json.loads((tmp_path / "results.json").read_text())
    for item in results["corrected_v2"].values():
        item["paired_vs_old_equal"]["sharpe_difference_95_interval"] = None
    (tmp_path / "results.json").write_text(json.dumps(results))
    report(tmp_path)
    assert "Equal weight: undefined</li>" in (tmp_path / "RESULTS.html").read_text(encoding="utf-8")
    (tmp_path / "corrected-v2/assumed_base/equal_weight/decisions.json").unlink()
    with pytest.raises(FileNotFoundError):
        report(tmp_path)
