import hashlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from capstone import runlog
from capstone.pipeline.reference_allocation import replay, run, statistics, uncertainty
from capstone.pipeline.reference_inputs import POOL, bab_months, french_months, validate_reference
from capstone.strategy.robust_allocation import AllocationPolicy

REGISTRATION = Path(__file__).resolve().parents[1] / "config/reference_allocation.json"


def fixture_frame():
    index = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    x = np.random.default_rng(404).normal(0.003, 0.02, size=(len(index), len(POOL)))
    return pd.DataFrame(x, index=index, columns=POOL).assign(RF=0.001, **{"Mkt-RF": 0.004})


def test_explicit_study_ledger_preserves_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "other-study"))
    runlog.log_run("reference-fixture", ledger_dir=tmp_path / "this-study")
    assert (tmp_path / "this-study/runs.jsonl").exists()
    assert not (tmp_path / "other-study").exists()


def test_public_input_units_cutoff_and_missing_rejection():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("factors.csv", ",HML,RF\n199001,1.5,0.2\n202101,secret,ignored\n")
    french = french_months(stream.getvalue())
    assert french.HML.iloc[0] == 0.015 and french.RF.iloc[0] == 0.002
    book = Workbook()
    sheet = book.active
    sheet.title = "BAB Factors"
    sheet.append(["DATE", "USA"])
    sheet.append(["01/31/1990", 0.015])
    sheet.append(["01/31/2021", "not_a_number"])
    stream = io.BytesIO()
    book.save(stream)
    bab = bab_months(stream.getvalue())
    assert bab.iloc[0] == 0.015 and len(bab) == 1
    with pytest.raises(ValueError, match="calendar"):
        validate_reference(fixture_frame().iloc[:-1])
    frame = fixture_frame()
    frame.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_reference(frame)


def test_replay_cash_interest_drift_switch_costs_and_reconciliation():
    frame = fixture_frame()
    policy = AllocationPolicy(
        periods=12,
        lookback=120,
        minimum=60,
        window=36,
        step=12,
        holding_periods=3,
        switch_bps=6,
        ongoing_bps=10,
    )
    history, decisions = replay(frame, "equal_weight", policy)
    assert len(history) == 300 and len(decisions) == 100
    assert all(d["history_end"] < d["as_of"] for d in decisions)
    first = history.iloc[0]
    invested = first.invested
    fee_fraction = (0.0006 * invested) / (1 + 0.0006 * invested)
    assert first.switch_cost == pytest.approx(100000 * fee_fraction)
    assert first.cash_interest_pnl == pytest.approx((100000 - first.switch_cost) * 0.001)
    assert history.switch_cost.iloc[1] == 0
    # Ongoing drag is 10bp a month on the invested fraction of the funded account.
    assert first.ongoing_cost == pytest.approx((100000 - first.switch_cost) * 0.001 * invested)
    assert not np.allclose(
        history.filter(like="weight_").iloc[0], history.filter(like="weight_").iloc[1]
    )
    net_pnl = (
        history.factor_pnl + history.cash_interest_pnl - history.ongoing_cost - history.switch_cost
    )
    np.testing.assert_allclose(
        net_pnl, history.equity.diff().fillna(first.equity - 100000), atol=1e-7
    )
    np.testing.assert_allclose(history.excess, history["return"] - history.RF)
    np.testing.assert_allclose(history.filter(like="factor_pnl_").sum(axis=1), history.factor_pnl)
    assert uncertainty(history, history)["sharpe_difference_95_interval"] == [0, 0]
    assert statistics(history)["months"] == 300


def test_duplicate_rows_rejected_and_row_permutation_preserves_replay():
    frame = fixture_frame()
    with pytest.raises(ValueError, match="calendar"):
        validate_reference(pd.concat([frame, frame.iloc[:1]]))
    policy = AllocationPolicy(
        periods=12, lookback=120, minimum=60, window=36, step=12, holding_periods=3
    )
    left, ld = replay(frame, "equal_weight", policy)
    right, rd = replay(frame.iloc[::-1, ::-1], "equal_weight", policy)
    pd.testing.assert_frame_equal(left, right)
    assert ld == rd


def _write_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    raw = inputs / "monthly.csv"
    fixture_frame().to_csv(raw, index_label="month", float_format="%.17g", lineterminator="\n")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    (inputs / "sources.json").write_text(json.dumps({"monthly_sha256": digest}))
    return inputs, digest


def test_run_rejects_a_registration_that_differs_from_the_frozen_one(tmp_path):
    inputs, _ = _write_inputs(tmp_path)
    registration = tmp_path / "registration.json"
    config = json.loads(REGISTRATION.read_text())
    registration.write_text(json.dumps({**config, "purpose": "something else"}))
    with pytest.raises(ValueError, match="registration"):
        run(inputs, registration, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_rejects_inputs_whose_hash_changed(tmp_path):
    inputs, _ = _write_inputs(tmp_path)
    (inputs / "sources.json").write_text(json.dumps({"monthly_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="hash"):
        run(inputs, REGISTRATION, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_freezes_the_manifest_and_logs_one_entry_per_comparison(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "default-ledger"))
    stub = types.SimpleNamespace(report=lambda output: None)
    monkeypatch.setitem(sys.modules, "capstone.pipeline.reference_report", stub)
    inputs, digest = _write_inputs(tmp_path)
    out = tmp_path / "out"
    run(inputs, REGISTRATION, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert "config/reference_allocation.json" in manifest["code"]
    accounting = json.loads((out / "trial-accounting.json").read_text())
    assert accounting["planned_policy_cost_comparisons"] == 9
    assert accounting["selection_adjusted_admission"] is False
    entries = [json.loads(x) for x in (out / "ledger/runs.jsonl").read_text().splitlines()]
    assert len(entries) == 9 and {e["params"]["input_sha256"] for e in entries} == {digest}
    assert all("not-admission" in e["tags"] for e in entries)
    assert not (tmp_path / "default-ledger").exists()  # the shared ledger is untouched
    results = json.loads((out / "results.json").read_text())
    assert len(results["allocations"]) == 9 and set(results["standalone_gross"]) == set(POOL)
    paired = results["allocations"]["assumed_base/median_window_sharpe"][
        "paired_uncertainty_vs_equal"
    ]
    assert paired["draws"] == 1999 and paired["selection_adjusted"] is False
