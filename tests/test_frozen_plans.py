import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from reference_fakes import reference_panel

from capstone.execution.bundle import freeze_bundle, load_bundle
from capstone.execution.frozen import build_plan, read_plan, write_plan
from capstone.strategy.reference import ReferenceStrategySpec


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    path = tmp_path_factory.mktemp("frozen") / "bundle"
    panel = reference_panel()
    spec = ReferenceStrategySpec("716bdaf7915c65be", panel.data_version, "2000-12-29")
    prices = panel.fields["close"].loc["2005-12-30"].to_dict()
    instruments = {
        n: {"symbol": f"S{n}", "currency": "USD", "sec_type": "STK", "exchange": "XSYN"}
        for n in prices
    }
    freeze_bundle(
        path,
        panel,
        [spec],
        "2005-12-30",
        prices=prices,
        instruments=instruments,
        execution_cost_bps=11.0,
        account={"positions": {}, "open_orders": [], "cash": 100000.0, "capital": 100000.0},
    )
    return path


def test_frozen_plan_is_repeatable_but_not_implicitly_admitted(frozen, tmp_path):
    first = build_plan(frozen)
    assert first == build_plan(frozen)
    assert first["construction_verified"] and not first["transmittable"]
    assert "reviewed_gate_report_missing" in first["validation_blockers"]
    assert "contracts_unqualified" in first["broker_blockers"]
    assert (
        first["execution_session"] == "2006-01-02"
    )  # synthetic session calendar, not a real exchange calendar
    assert all(o["tif"] == "DAY" and o["outside_rth"] is False for o in first["orders"])
    path = write_plan(first, tmp_path / "plan.json")
    assert read_plan(path) == json.loads(path.read_text())
    assert path.with_suffix(".md").exists()
    preview = path.with_suffix(".md").read_text()
    assert "Current shares | Target shares | After orders | Net trade" in preview
    assert first["input_hash"] in preview and "Broker qualified: False" in preview
    assert first["orders"][0]["instruction_id"] in preview


def test_fresh_process_has_the_same_instruction_hash(frozen):
    code = (
        "from capstone.execution.frozen import build_plan; import sys; "
        "print(build_plan(sys.argv[1])['document_hash'])"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, "-c", code, str(frozen)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == build_plan(frozen)["document_hash"]


def test_modified_snapshot_is_refused(frozen, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(frozen, copy)
    path = copy / "inputs.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(copy)


def test_input_bundle_cannot_be_overwritten(frozen):
    with pytest.raises(FileExistsError):
        freeze_bundle(frozen, None, None, None, prices=None, instruments=None, account=None)


@pytest.mark.parametrize(
    "extra",
    [
        {"accountId": "SYNTHETIC_PRIVATE_VALUE"},
        {"metadata": {"credentials": "SYNTHETIC_PRIVATE_VALUE"}},
        {"positions": {"SYNTHETIC_PRIVATE_VALUE": 1}},
        {"open_orders": [{"account": "SYNTHETIC_PRIVATE_VALUE"}]},
        {"paper_account_verified": "SYNTHETIC_PRIVATE_VALUE"},
        {"cash": "SYNTHETIC_PRIVATE_VALUE"},
    ],
)
def test_private_account_metadata_is_rejected_before_writing(tmp_path, extra):
    account = {"positions": {}, "open_orders": [], "cash": 1000.0, "capital": 1000.0}
    account.update(extra)
    root = tmp_path / "private"
    with pytest.raises(ValueError) as error:
        freeze_bundle(root, None, None, None, prices=None, instruments=None, account=account)
    assert "SYNTHETIC_PRIVATE_VALUE" not in str(error.value)
    assert not root.exists()


def test_row_and_instrument_permutation_cannot_change_instructions(frozen, tmp_path):
    data, panel, specs, returns = load_bundle(frozen)
    shuffled = replace(
        panel,
        fields={n: f.iloc[::-1, ::-1] for n, f in panel.fields.items()},
        universe=panel.universe.iloc[::-1, ::-1],
        half_spread=panel.half_spread.iloc[::-1, ::-1],
        availability={n: a.iloc[::-1] for n, a in panel.availability.items()},
    )
    path = freeze_bundle(
        tmp_path / "shuffled",
        shuffled,
        specs,
        data["as_of"],
        prices={int(n): p for n, p in data["prices"].items()},
        instruments={int(n): p for n, p in data["instruments"].items()},
        account=data["account"],
        execution_cost_bps=data["execution_cost_bps"],
        returns=returns,
    )
    assert build_plan(path) == build_plan(frozen)


def test_zero_position_does_not_require_a_dead_instrument_mapping(frozen, tmp_path):
    data, panel, specs, returns = load_bundle(frozen)
    account = dict(data["account"], positions={999999: 0})
    path = freeze_bundle(
        tmp_path / "zero-position",
        panel,
        specs,
        data["as_of"],
        prices={int(n): p for n, p in data["prices"].items()},
        instruments={int(n): p for n, p in data["instruments"].items()},
        account=account,
        execution_cost_bps=data["execution_cost_bps"],
        returns=returns,
    )
    assert build_plan(path)["account_positions"] == {}
