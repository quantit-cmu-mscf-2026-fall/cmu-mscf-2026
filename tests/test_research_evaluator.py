import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_fresh_process_cli_evidence_and_registration_are_deterministic(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo), OPENBLAS_NUM_THREADS="1")
    for folder in ("first", "second", "first"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "capstone.pipeline.research_evaluator",
                "verify",
                "--output",
                str(tmp_path / folder),
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "first/proof.json").read_bytes() == (
        tmp_path / "second/proof.json"
    ).read_bytes()
    assert len((tmp_path / "first/ledger/runs.jsonl").read_text().splitlines()) == 1
    assert "scripted" in (tmp_path / "first/REPORT.html").read_text()
    manifest = json.loads((tmp_path / "first/manifest.json").read_text())
    assert manifest["review_version"] == 2 and manifest["paid_request_limit"] == 0
