import os

import pytest

from capstone import runlog
from capstone.pipeline.accounting import import_opportunities, ledger_at, reconcile_sources
from capstone.pipeline.study import file_digest, freeze


def fixture(tmp_path, columns=("v1",)):
    original = tmp_path / "original"
    with ledger_at(original):
        for name in ("evaluate-factor", "deepen"):
            runlog.log_run(name, params={"variant_id": "v1"}, tags=["e2e-run:crsp-v1"])
    basket = tmp_path / "basket"
    with ledger_at(basket):
        for suffix in ("registration", "result"):
            runlog.log_run("crsp-v1-basket-exploration-" + suffix)
    matrix = tmp_path / "matrix.bin"
    matrix.write_bytes(b"synthetic matrix stand-in")
    source = original / "runs.jsonl"
    with ledger_at(tmp_path / "registration-ledger"):
        registration = freeze(
            tmp_path / "registration.json",
            sources={"original_ledger": source, "original_return_matrix": matrix},
        )
    seen = []

    def loader(path):
        seen.append(path)
        return ["date", *columns]

    result = reconcile_sources(
        registration,
        source,
        matrix,
        basket / "runs.jsonl",
        tmp_path / "accounting.json",
        matrix_columns=loader,
    )
    return registration, result, seen


def test_accounting_preserves_adaptive_trials_and_hashes(tmp_path):
    _, result, seen = fixture(tmp_path)
    assert seen == [tmp_path / "matrix.bin"]
    assert result["reconciled"] and result["source_entries"] == 4
    assert result["return_columns"] == 1 and result["conservative_trial_entries"] == 3
    assert result["source_summary"]["deepen_entries"] == 1
    before = file_digest(tmp_path / "original" / "runs.jsonl")
    with ledger_at(tmp_path / "new-ledger"):
        assert import_opportunities(result) == 3
        assert runlog.trial_count(include_tags=["eligible-research-trial"]) == 3
    assert before == file_digest(tmp_path / "original" / "runs.jsonl")
    with pytest.raises(FileExistsError):
        fixture(tmp_path)


def test_unmapped_column_blocks_import(tmp_path):
    _, result, _ = fixture(tmp_path, columns=("v1", "orphan"))
    assert not result["reconciled"]
    with pytest.raises(ValueError, match="unresolved"):
        import_opportunities(result)


def test_changed_source_is_refused(tmp_path):
    registration, _, _ = fixture(tmp_path)
    (tmp_path / "matrix.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        reconcile_sources(
            registration,
            tmp_path / "original" / "runs.jsonl",
            tmp_path / "matrix.bin",
            tmp_path / "basket" / "runs.jsonl",
            tmp_path / "again.json",
            matrix_columns=lambda p: ["v1"],
        )


def test_ledger_at_restores_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPSTONE_LEDGER_DIR", raising=False)
    with ledger_at(tmp_path):
        assert os.environ["CAPSTONE_LEDGER_DIR"] == str(tmp_path)
    assert "CAPSTONE_LEDGER_DIR" not in os.environ
