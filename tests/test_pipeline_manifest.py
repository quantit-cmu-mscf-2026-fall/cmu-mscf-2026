"""The search's gate registration and manifest (`pipeline.e2e.gate_config`, `pipeline.manifest`).

A scripted model client and synthetic rows; a temporary ledger. No model, no network,
no market data.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_factor_evaluation import fake_rows
from test_pipeline_search_steps import _config, _run

from capstone import gate, runlog
from capstone.factors.evaluation import build_panel
from capstone.funnel import Thresholds
from capstone.pipeline import e2e, manifest
from capstone.pipeline.config import REPO, load_config
from capstone.pipeline.search_identity import config_digest
from capstone.scorecard import TrialScope


@pytest.fixture(scope="module")
def panel():
    return build_panel(fake_rows(n_stocks=12, n_days=150), "fixture")


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.mark.parametrize(
    ("families", "n_boot"),
    [(1, 1999), (50, 1999), (200, 3999), (201, 4499)],
)
def test_sized_n_boot_rounds_up_to_the_step(families, n_boot):
    assert e2e.sized_n_boot(1999, families) == n_boot
    assert (e2e.sized_n_boot(1999, families) + 1) % e2e.N_BOOT_STEP == 0


def test_sized_n_boot_keeps_a_larger_configured_floor():
    assert e2e.sized_n_boot(9999, 3) == 9999


def _registered(tmp_path, panel, **changes):
    config = _config(tmp_path, **changes)
    found, workdir = _run(tmp_path, panel, config)
    return config, found, workdir


def _register(config, found, workdir, **kwargs):
    return manifest.register_search(
        config, found, data_version="v1", model_calls=7, workdir=workdir, **kwargs
    )


def test_registering_a_search_writes_the_gate_config_and_manifest(tmp_path, panel, ledger):
    config, found, workdir = _registered(tmp_path, panel)
    record = _register(config, found, workdir)

    registered = gate.GateConfig.from_dict(record.gate_config)
    assert registered.trial_scope == TrialScope(
        include_tags=tuple(sorted((config.run_tag, config.data_tag)))
    )
    assert registered.factors_digest is None and record.stage3 == "unmeasured"
    assert registered.decision_rule == gate.DecisionRule("funnel")
    assert registered.min_holdout_coverage == gate.DEFAULT_MIN_HOLDOUT_COVERAGE
    assert (
        registered.n_boot == record.n_boot == e2e.sized_n_boot(config.n_boot, len(record.families))
    )
    assert record.n_variants == found.matrix.shape[1]
    assert record.trial_count >= record.n_variants
    assert record.run_trial_count == record.trial_count  # one run on this data
    assert (record.model_calls, record.config_digest) == (7, config_digest(config))
    regs = [e for e in runlog.read_entries() if e["name"] == gate.REGISTRATION_NAME]
    assert [e["params"]["config_hash"] for e in regs] == [record.gate_config_hash]
    assert e2e.report_path(workdir, registered).name == f"gate-{record.gate_config_hash[:12]}.json"


def test_the_manifest_round_trips_and_its_frames_are_read_back(tmp_path, panel, ledger):
    config, found, workdir = _registered(tmp_path, panel)
    record = _register(config, found, workdir)

    assert manifest.read_manifest(workdir / manifest.MANIFEST_FILE) == record
    matrix, summary = manifest.read_frames(workdir, record)
    assert list(matrix.columns) == list(found.matrix.columns)
    assert list(summary.index) == list(found.summary.index)


def test_the_registered_hash_is_sensitive_to_what_the_manifest_records(tmp_path):
    config = _config(tmp_path)
    base = e2e.gate_config(config, "v1", 5)
    for changed in (
        e2e.gate_config(replace(config, seed=1), "v1", 5),
        e2e.gate_config(config, "v2", 5),
        e2e.gate_config(config, "v1", 400),
        e2e.gate_config(replace(config, deepen_selector="agent"), "v1", 5),
        e2e.gate_config(replace(config, run_id="other"), "v1", 5),
        e2e.gate_config(replace(config, search_end="2005-12-29"), "v1", 5),
    ):
        assert changed.config_hash != base.config_hash
    assert e2e.gate_config(config, "v1", 5).config_hash == base.config_hash


def test_an_unknown_key_or_an_edited_gate_config_is_refused(tmp_path, panel, ledger):
    config, found, workdir = _registered(tmp_path, panel)
    _register(config, found, workdir)
    path = workdir / manifest.MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))

    path.write_text(json.dumps(data | {"extra": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown manifest keys"):
        manifest.read_manifest(path)
    edited = json.loads(json.dumps(data))
    edited["gate_config"]["seed"] += 1
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="edited"):
        manifest.read_manifest(path)


def test_an_edited_parquet_is_detected(tmp_path, panel, ledger):
    config, found, workdir = _registered(tmp_path, panel)
    record = _register(config, found, workdir)
    found.matrix.mul(2).to_parquet(workdir / record.matrix_file)
    with pytest.raises(ValueError, match="changed since the search"):
        manifest.read_frames(workdir, record)


def test_a_second_registration_in_the_workdir_is_refused(tmp_path, panel, ledger):
    config, found, workdir = _registered(tmp_path, panel)
    _register(config, found, workdir)
    with pytest.raises(FileExistsError):
        _register(config, found, workdir)


def test_fewer_trials_than_variants_is_refused_before_registering(tmp_path, panel, ledger):
    config, found, workdir = _registered(tmp_path, panel)
    other = replace(config, run_id="unlogged", data_label="synthetic-other")
    with pytest.raises(ValueError, match="trials in scope"):
        _register(other, found, workdir)
    assert not [e for e in runlog.read_entries() if e["name"] == gate.REGISTRATION_NAME]
    assert not (workdir / manifest.MANIFEST_FILE).exists()


def test_the_trials_count_matches_the_gates_own(tmp_path, panel, ledger):
    config, found, workdir = _registered(tmp_path, panel)
    scope = TrialScope(include_tags=(config.run_tag, config.data_tag))
    assert manifest.trials_in_scope(scope) == gate._trial_count(scope) == len(runlog.read_entries())
    _register(config, found, workdir)
    assert manifest.trials_in_scope(scope) == gate._trial_count(
        scope
    )  # the gate's entry is not one


def test_a_real_data_config_with_empty_thresholds_is_refused_unless_allowed(tmp_path):
    real = replace(_config(tmp_path), data_label="crsp-x", thresholds_source="code-defaults")
    with pytest.raises(e2e.ThresholdsNotRegistered):
        e2e.gate_config(real, "v1", 5)
    allowed = e2e.gate_config(real, "v1", 5, allow_default_thresholds=True)
    assert allowed.thresholds == real.thresholds
    stated = replace(real, thresholds_source="config")
    assert e2e.gate_config(stated, "v1", 5).config_hash == allowed.config_hash
    synthetic = replace(real, data_label="synthetic-crsp")
    assert e2e.gate_config(synthetic, "v1", 5).config_hash != allowed.config_hash


def test_load_config_records_where_the_thresholds_came_from(tmp_path):
    real = load_config(REPO / "config" / "e2e_crsp.toml")
    assert real.thresholds_source == "config"
    text = (REPO / "config" / "e2e_crsp.toml").read_text(encoding="utf-8")
    head, _, tail = text.partition("[gate.thresholds]")
    bare = head + "[stage3]" + tail.partition("[stage3]")[2]
    path = tmp_path / "bare.toml"
    path.write_text(bare, encoding="utf-8")
    empty = load_config(path)
    assert empty.thresholds_source == "code-defaults" and empty.thresholds == Thresholds()
    assert config_digest(real) == config_digest(replace(real, thresholds_source="code-defaults"))
    e2e.check_thresholds(real)
    with pytest.raises(e2e.ThresholdsNotRegistered):
        e2e.check_thresholds(empty)
