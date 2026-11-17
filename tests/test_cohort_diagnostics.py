import json

import pandas as pd
import pytest

from capstone.factors.evaluation import Panel
from capstone.pipeline import diagnostics


def panel():
    dates = pd.bdate_range("1999-12-01", "2000-04-03")
    returns = pd.DataFrame({1: 0.001, 2: -0.001}, index=dates)
    return Panel(
        {"returns": returns},
        returns.notna(),
        returns * 0,
        data_version="synthetic-diagnostics",
        calendar=dates,
        availability={"returns": pd.Series(dates, index=dates)},
    )


def test_invalid_diagnostic_formations_are_reported_without_aborting_other_methods(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

    def legs(spec, *args):
        if spec.method == "volume_momentum_6m":
            raise ValueError("turnover-conditioned portfolio has an empty leg")
        return pd.Series({1: 1.0}), pd.Series({2: 1.0})

    monkeypatch.setattr(diagnostics, "_legs", legs)
    result = diagnostics.cohort_diagnostics(
        panel(), anchor="1999-12-31", start="2000-01-01", end="2000-03-31"
    )
    assert result["blocked_formations"]
    assert {r["method"] for r in result["blocked_formations"]} == {"volume_momentum_6m"}
    assert result["cohorts"] and result["january"]
    assert all(r["method"] != "volume_momentum_6m" for r in result["cohorts"])


def test_every_diagnostic_formation_can_fail_without_inventing_a_result(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

    def unavailable(*args):
        raise ValueError("required history unavailable")

    monkeypatch.setattr(diagnostics, "_legs", unavailable)
    result = diagnostics.cohort_diagnostics(
        panel(), anchor="1999-12-31", start="2000-01-01", end="2000-03-31"
    )
    assert result["cohorts"] == [] and result["january"] == []
    assert result["blocked_formations"]


def test_missing_january_observations_remain_null_in_strict_json(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    sample = panel()
    sample.fields["returns"].loc["2000-01-03", 1] = float("nan")
    monkeypatch.setattr(
        diagnostics, "_legs", lambda *args: (pd.Series({1: 1.0}), pd.Series({2: 1.0}))
    )
    result = diagnostics.cohort_diagnostics(
        sample, anchor="1999-12-31", start="2000-01-01", end="2000-03-31"
    )
    assert result["january"]
    assert all(not r["measured"] and r["long_return"] is None for r in result["january"])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_window_reaching_the_sealed_holdout_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="sealed holdout"):
        diagnostics.cohort_diagnostics(
            panel(), anchor="1999-12-31", start="2000-01-01", end="2021-01-01"
        )
    assert not list(tmp_path.iterdir())
