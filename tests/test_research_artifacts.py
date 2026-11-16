"""Research artifacts: manifest identity, ledger registration, checkpoints, the pilot report."""

import json
import os
from dataclasses import dataclass

import pytest

from capstone import runlog
from capstone.artifacts import digest
from capstone.factors import research_artifacts as ra


@dataclass(frozen=True)
class Cfg:
    model: str = "m"
    depth: int = 2


@dataclass(frozen=True)
class Policy:
    limit: int = 3


BENCH = {"call_limits": {"total": 120}, "packets": []}


def make_manifest(**overrides):
    args = {
        "benchmark": BENCH,
        "configs": {"A": Cfg()},
        "checks": {"ok": True},
        "policy": Policy(),
    } | overrides
    return ra.manifest(**args)


def test_manifest_records_identity_and_inert_defaults():
    m = make_manifest()
    assert m["call_limits"] == {"total": 120}
    assert m["research_policy"] == {"limit": 3}
    assert m["configs"] == {"A": {"model": "m", "depth": 2}}
    assert m["seed"] == 404
    assert m["financial_trials"] == 0
    assert m["holdout_access"] is False
    assert m["broker_access"] is False
    assert m["controller_default"] is False
    assert set(m["release"]) >= {"commit", "dirty", "merged_main"}
    assert "python" in m["runtime"] and "anthropic" in m["runtime"]
    assert "capstone/factors/research_artifacts.py" in m["code"]
    assert "config/factors.toml" in m["code"]
    assert all(len(h) == 64 for h in m["code"].values())
    json.dumps(m)  # portable


def test_manifest_digest_follows_its_inputs():
    assert digest(make_manifest()) == digest(make_manifest())
    assert digest(make_manifest()) != digest(make_manifest(configs={"A": Cfg(depth=3)}))
    assert digest(make_manifest()) != digest(make_manifest(policy=Policy(limit=4)))


def test_register_logs_once_and_restores_the_ledger_dir(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(outer))
    frozen = make_manifest()
    ra.register(tmp_path / "root", frozen)
    ra.register(tmp_path / "root", frozen)  # same manifest: no second entry
    assert os.environ["CAPSTONE_LEDGER_DIR"] == str(outer)
    assert not (outer / runlog.LEDGER_FILENAME).exists()  # the caller's ledger is untouched
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "root" / "ledger"))
    entries = runlog.read_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["params"]["manifest_digest"] == digest(frozen)
    assert entry["params"]["financial_trials"] == 0
    assert "nonfinancial" in entry["tags"]
    assert entry["seed"] == 404


def test_register_restores_unset_ledger_dir_even_on_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPSTONE_LEDGER_DIR", raising=False)
    with pytest.raises(ValueError):
        ra.register(tmp_path, {"call_limits": float("nan")})
    assert "CAPSTONE_LEDGER_DIR" not in os.environ


def test_register_distinguishes_manifests(tmp_path, monkeypatch):
    ra.register(tmp_path, make_manifest())
    ra.register(tmp_path, make_manifest(configs={"A": Cfg(depth=9)}))
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    assert len(runlog.read_entries()) == 2


def test_checkpoint_round_trip_and_atomic_write(tmp_path):
    path = tmp_path / "deep" / "c.json"
    ra.checkpoint(path, {"x": [1, 2]}, "m1")
    assert ra.read_checkpoint(path, "m1") == {"x": [1, 2]}
    assert not list(tmp_path.rglob("*.tmp"))


def test_checkpoint_refuses_another_manifest_or_edited_contents(tmp_path):
    path = tmp_path / "c.json"
    ra.checkpoint(path, {"x": 1}, "m1")
    with pytest.raises(ValueError, match="checkpoint identity"):
        ra.read_checkpoint(path, "m2")
    document = json.loads(path.read_text())
    document["payload"]["x"] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checkpoint identity"):
        ra.read_checkpoint(path, "m1")


def row(**over):
    return {
        "completed_packets": 1,
        "reviewed_packets": 1,
        "grounded_families": ["f1", "f2"],
        "verified_method_artifacts": 1,
        "unsupported_claims": 0,
        "outbound_reservations": 4,
        "complete_specifications": 1,
        "data_ready": 1,
        "verified_constructions": 0,
        "retained_data_blocked_specs": 0,
        "unsafe_readiness_claims": 0,
    } | over


def make_report_inputs(tmp_path, version=1):
    frozen = {
        "benchmark": {
            "packets": [
                {
                    "id": "p1",
                    "title": "Momentum <script>",
                    "required_inputs": ["daily_prices", "float_shares"],
                    "source_url": 'https://example.com/a?x="1"',
                    "full_methods": True,
                },
                {
                    "id": "p2",
                    "title": "Value",
                    "required_inputs": [],
                    "source_url": "https://example.com/b",
                    "full_methods": False,
                },
            ]
        }
    }
    summary = {
        "decision": "keep_going",
        "mode": "exploratory",
        "criteria": {"c": 1},
        "review_disagreements": [],
        "metrics": {"A": row(), "B": row(grounded_families=["f1"]), "C": row(grounded_families=[])},
    }
    if version == 2:
        summary["review_version"] = 2
    accounting = {"outbound_reservations": 12}
    key = digest(frozen)
    ra.checkpoint(tmp_path / "cases/A/p1.json", {"artifact": {"status": "grounded_ok"}}, key)
    ra.checkpoint(
        tmp_path / "cases/B/p1.json",
        {"artifact": {"status": "interpreted", "interpretation": {"method": {"k": "<v>"}}}},
        key,
    )
    return frozen, summary, accounting


def test_report_writes_files_and_escapes_untrusted_text(tmp_path):
    frozen, summary, accounting = make_report_inputs(tmp_path)
    ra.write_pilot_report(tmp_path, frozen, summary, accounting, stop_reason="budget <hit>")
    html = (tmp_path / "REPORT.html").read_text(encoding="utf-8")
    assert "Momentum &lt;script&gt;" in html and "<script>" not in html
    assert "float shares" in html and "daily prices" in html
    assert "budget &lt;hit&gt;" in html
    assert "keep going" in html
    assert "12 / 120" in html
    assert "methods available" in html and "methods unavailable" in html
    assert "&lt;v&gt;" in html and "construction unverified" in html
    assert 'href="cases/A/p1.json"' in html
    assert "C: pending" in html  # no case file yet
    assert "Evaluator v2" not in html
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert json.loads((tmp_path / "accounting.json").read_text()) == accounting
    svg = (tmp_path / "coverage.svg").read_text(encoding="utf-8")
    assert "2 / 6" in svg and "1 / 6" in svg and "0 / 6" in svg
    assert 'width="200"' in svg and 'width="100"' in svg


def test_report_v2_adds_dimensions_and_renames_the_column(tmp_path):
    frozen, summary, accounting = make_report_inputs(tmp_path, version=2)
    ra.write_pilot_report(tmp_path, frozen, summary, accounting)
    html = (tmp_path / "REPORT.html").read_text(encoding="utf-8")
    assert "Evaluator v2 dimensions" in html
    assert "unsafe_readiness_claims" in html
    assert "Source-faithful complete specs" in html and "Verified methods" not in html
    assert "Stopped:" not in html


def test_report_refuses_a_case_file_from_another_study(tmp_path):
    frozen, summary, accounting = make_report_inputs(tmp_path)
    ra.checkpoint(tmp_path / "cases/C/p2.json", {"artifact": {"status": "x"}}, "other")
    with pytest.raises(ValueError, match="checkpoint identity"):
        ra.write_pilot_report(tmp_path, frozen, summary, accounting)


def test_atomic_json_is_the_shared_helper():
    from capstone import artifacts

    assert ra.atomic_json is artifacts.atomic_json
