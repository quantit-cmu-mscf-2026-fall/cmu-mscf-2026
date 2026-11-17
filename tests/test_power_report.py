"""The gate's power report: in the header, never in a decision.

`capstone.power` simulates a gate configuration; `power.for_gate` turns a
registered `GateConfig` and its search period into the header's minimum
detectable Sharpe per stage, and the scorecard header carries it.
`scripts/power_analysis.py` reports the same analysis for a registered bar.

- the adapter reads the config's own thresholds, bootstrap, family sizes,
  search length, holdout business days and the search period's correlations;
- reporting power changes no config hash and writes nothing to the ledger;
- the header round-trips through the report, and a report written before the
  field existed still reads back (no power);
- the script is deterministic for its seed and logs one ledger entry before
  printing.

Synthetic data only (every date is before 2021); temporary ledgers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_scorecard import _result

from capstone import calibration, power, runlog
from capstone.funnel import Thresholds
from capstone.gate import GateConfig
from capstone.power import MinDetectableSharpe, PowerBudget
from capstone.scorecard import read_report, write_report

N_FAMILIES = 4
K = 3
CHEAP = PowerBudget(n_sims=2, sharpe_grid=(0.0, 1.0, 3.0), seed=0)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "power_analysis.py"
_spec = importlib.util.spec_from_file_location("power_analysis_script", SCRIPT)
script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = script
_spec.loader.exec_module(script)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


@pytest.fixture(scope="module")
def world():
    fs = calibration.make_family_set(
        N_FAMILIES, K, 0.25, 1.5, seed=1, n_search=4 * 252, n_holdout=2 * 252
    )
    dates = pd.bdate_range("1990-01-01", periods=len(fs.returns))
    search = fs.returns.iloc[: fs.n_search].set_axis(dates[: fs.n_search])
    return search, fs.family, dates[fs.n_search], dates[-1]


def config_for(world, **overrides) -> GateConfig:
    search, _, holdout_start, holdout_end = world
    kwargs = dict(
        search_end=search.index[-1],
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        data_version="synthetic",
        generator="test-gen",
        thresholds=Thresholds(stage1="bootstrap"),
        n_boot=99,  # n_boot + 1 >= 20 x 4 families at q = 0.05
    )
    kwargs.update(overrides)
    return GateConfig(**kwargs)


class TestForGate:
    def test_header_reports_each_stage(self, world):
        search, family, *_ = world
        mds = power.for_gate(config_for(world), search, family, 12, CHEAP)
        assert [m.stage for m in mds] == ["stage1", "stage2", "stage3", "stage4", "end_to_end"]
        assert all(isinstance(m, MinDetectableSharpe) for m in mds)
        by_stage = {m.stage: m for m in mds}
        assert not by_stage["stage2"].gated  # alpha2 = 1.0: no stage-2 gate
        assert not by_stage["stage3"].gated  # stage 3 is not modelled
        assert by_stage["stage2"].power_50 is None and by_stage["stage2"].power_80 is None
        for name in ("stage1", "stage4", "end_to_end"):
            assert by_stage[name].gated
            m = by_stage[name]
            assert m.power_50 is None or 0.0 <= m.power_50 <= 3.0

    def test_the_spec_is_the_gates_configuration(self, world):
        search, family, holdout_start, holdout_end = world
        config = config_for(world)
        spec = power.spec_for_gate(
            thresholds=config.thresholds,
            n_boot=config.n_boot,
            search=search,
            family=family,
            holdout_periods=len(pd.bdate_range(holdout_start, holdout_end)),
            n_trials=12,
        )
        assert spec.thresholds == config.thresholds and spec.n_boot == 99
        assert spec.variants == (K,) * N_FAMILIES
        assert spec.search_periods == len(search) == 4 * 252
        assert spec.holdout_periods == len(pd.bdate_range(holdout_start, holdout_end))
        assert spec.n_trials == 12
        assert 0.0 <= spec.across_corr <= spec.within_corr <= 1.0
        assert spec.within_corr == pytest.approx(0.9, abs=0.05)  # make_family_set's default

    def test_for_gate_equals_the_power_analysis_of_that_spec(self, world):
        search, family, holdout_start, holdout_end = world
        config = config_for(world)
        spec = power.spec_for_gate(
            thresholds=config.thresholds,
            n_boot=config.n_boot,
            search=search,
            family=family,
            holdout_periods=len(pd.bdate_range(holdout_start, holdout_end)),
            n_trials=12,
        )
        direct = power.power_analysis(spec, CHEAP).min_detectable_sharpe()
        assert power.for_gate(config, search, family, 12, CHEAP) == direct

    def test_deterministic_for_a_seed(self, world):
        search, family, *_ = world
        config = config_for(world)
        a = power.for_gate(config, search, family, 12, CHEAP)
        assert a == power.for_gate(config, search, family, 12, CHEAP)

    def test_single_variant_families_are_refused_by_the_spec(self, world):
        search, family, *_ = world
        singles = family.copy()
        singles.iloc[:] = [f"s{i}" for i in range(len(singles))]
        with pytest.raises(ValueError, match="at least 2 variants"):
            power.for_gate(config_for(world), search, singles, 12, CHEAP)

    def test_reporting_power_writes_nothing_and_changes_no_hash(self, world, ledger):
        search, family, *_ = world
        config = config_for(world)
        before = config.config_hash
        power.for_gate(config, search, family, 12, CHEAP)
        assert config.config_hash == before
        assert not (ledger / "runs.jsonl").exists()
        assert "power" not in json.dumps(config.to_dict())
        assert "min_detectable" not in json.dumps(config.to_dict())


class TestHeader:
    def test_a_header_without_power_reports_none(self):
        header = _result().header
        assert header.min_detectable_sharpe is None and header.power_sims == 0

    def test_the_header_round_trips_with_power(self, tmp_path):
        mds = (
            MinDetectableSharpe("stage1", True, 0.9, 1.4),
            MinDetectableSharpe("stage2", False, None, None),
            MinDetectableSharpe("stage3", False, None, None),
            MinDetectableSharpe("stage4", True, 1.1, None),
            MinDetectableSharpe("end_to_end", True, 1.3, 2.0),
        )
        result = _result(min_detectable_sharpe=mds, power_sims=200)
        path, _ = write_report(result, tmp_path / "gate.json")
        back = read_report(path)
        assert back.header.min_detectable_sharpe == mds
        assert back.header.power_sims == 200
        report = json.loads(path.read_text())
        assert report["header"]["min_detectable_sharpe"][0] == {
            "stage": "stage1",
            "gated": True,
            "power_50": 0.9,
            "power_80": 1.4,
        }

    def test_a_report_written_before_the_field_existed_still_reads(self, tmp_path):
        path, _ = write_report(_result(), tmp_path / "gate.json")
        report = json.loads(path.read_text())
        del report["header"]["min_detectable_sharpe"], report["header"]["power_sims"]
        path.write_text(json.dumps(report))
        back = read_report(path)
        assert back.header.min_detectable_sharpe is None and back.header.power_sims == 0

    def test_power_is_not_part_of_what_a_decision_reads(self):
        a = _result(power_sims=0)
        b = _result(
            min_detectable_sharpe=(MinDetectableSharpe("stage1", True, 0.5, 0.9),), power_sims=7
        )
        pd.testing.assert_frame_equal(a.scorecard, b.scorecard)
        assert a.accepted == b.accepted


# ---------------------------------------------------------------------------
# The script


PARAMS = {
    "description": "test shape",
    "search_years": 4,
    "holdout_years": 2,
    "variants": [3, 3, 3, 3],
    "within_corr": 0.8,
    "across_corr": 0.1,
    "n_trials": 12,
    "n_sims": 2,
    "sharpe_grid": [0.0, 1.0, 3.0],
    "seed": 0,
}


def run_script(tmp_path, monkeypatch, capsys, *, params=PARAMS, argv=("--quick",)):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))
    monkeypatch.setattr(sys, "argv", ["power_analysis.py", *argv, "--params", str(path)])
    script.main()
    return capsys.readouterr().out


class TestScript:
    def test_the_shipped_params_describe_a_valid_spec(self):
        p = json.loads(script.PARAMS.read_text())
        assert p["seed"] == 0 and p["n_sims"] >= 100
        assert list(p["sharpe_grid"]) == sorted(p["sharpe_grid"])
        assert p["n_trials"] >= sum(p["variants"])
        assert p["within_corr"] >= p["across_corr"]

    def test_it_logs_one_entry_before_printing_and_prints_each_stage(
        self, tmp_path, ledger, monkeypatch, capsys
    ):
        out = run_script(tmp_path, monkeypatch, capsys)
        (entry,) = runlog.read_entries()
        assert entry["name"] == "validation-power"
        assert set(entry["tags"]) == {"synthetic", "power"}
        assert entry["params"]["data_version"] == "synthetic"
        assert entry["params"]["n_sims"] == 10  # --quick
        assert entry["seed"] == 0
        for stage in ("stage1", "stage2", "stage3", "stage4", "end_to_end"):
            assert stage in out
            assert f"{stage}_mds50" in entry["metrics"]
            assert f"{stage}_null_rate" in entry["metrics"]
        assert "MDS 50%" in out and "power by true Sharpe" in out

    def test_it_is_deterministic_for_its_seed(self, tmp_path, ledger, monkeypatch, capsys):
        run_script(tmp_path, monkeypatch, capsys)
        run_script(tmp_path, monkeypatch, capsys)
        first, second = runlog.read_entries()
        a = {k: v for k, v in first["metrics"].items() if k != "seconds"}
        b = {k: v for k, v in second["metrics"].items() if k != "seconds"}
        assert a == b

    def test_the_gates_own_bootstrap_floor_sets_the_draws(
        self, tmp_path, ledger, monkeypatch, capsys
    ):
        run_script(tmp_path, monkeypatch, capsys)
        (entry,) = runlog.read_entries()
        assert entry["params"]["n_boot"] == 4 * 20 - 1  # families / 0.05, minus 1

    def test_the_curve_rises_with_the_true_sharpe(self, tmp_path, ledger, monkeypatch, capsys):
        out = run_script(
            tmp_path,
            monkeypatch,
            capsys,
            params={**PARAMS, "sharpe_grid": [0.0, 4.0]},
            argv=(),
        )
        rows = [line.split() for line in out.splitlines() if line[:6].strip() in {"0.00", "4.00"}]
        by_sharpe = {float(r[0]): [float(x) for x in r[1:]] for r in rows}
        assert np.mean(by_sharpe[4.0]) >= np.mean(by_sharpe[0.0])
