import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from capstone.factors.llm import FactorConfig
from capstone.factors.research_requests import RequestJournal
from capstone.factors.research_review import FLAGS
from capstone.pipeline.research_experiment import execute

REPO = Path(__file__).resolve().parents[1]
CONFIG = FactorConfig(model="fixture", alignment_model="fixture-judge")


class FakeAgent:
    """Scripted fixture output is never real-agent or financial evidence."""

    max_retries = 0

    def __init__(self, fail_first=False):
        self.messages = self
        self.calls = 0
        self.fail_first = fail_first

    def create(self, **kwargs):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            error = RuntimeError("fixture")
            error.status_code = 400
            error.body = {"error": {"message": "credit balance too low"}}
            raise error
        tool = kwargs["tools"][0]["name"]
        if tool == "record_hypotheses":
            answer = {"hypotheses": []}
        elif tool == "record_research_method":
            paper = json.loads(kwargs["messages"][0]["content"])
            data = json.loads(paper["methods"])
            answer = {
                "observation": "scripted fixture",
                "knowledge": "fixture",
                "justification": "fixture",
                "specification": "fixture",
                "falsification_condition": "independent fixture comparison",
                **{
                    k: data[k]
                    for k in ("source_url", "method_location", "universe", "required_inputs")
                },
                "method": {
                    n: "unverified fixture"
                    for n in (
                        "direction",
                        "selection",
                        "weighting",
                        "formation",
                        "holding",
                        "exits",
                        "availability",
                    )
                },
                "variant": "reference",
                "profile": "long_short",
                "deviations": [],
            }
        elif tool == "review_methods":
            context = json.loads(kwargs["messages"][0]["content"])
            answer = {
                "judgments": [
                    {
                        "blind_id": row["blind_id"],
                        **dict.fromkeys(FLAGS, False),
                        "correct_blocking": True,
                        "reason": "scripted exploratory fixture",
                    }
                    for row in context["artifacts"]
                ]
            }
        else:
            raise AssertionError("unexpected model operation: " + tool)
        return NS(
            content=[NS(type="tool_use", name=tool, input=answer)],
            usage=NS(input_tokens=3, output_tokens=4),
        )


@pytest.mark.slow
def test_complete_fixture_pilot_replays_without_duplicate_calls_or_registration(tmp_path):
    client = FakeAgent()
    result = execute(tmp_path, client=client, offline_fixture=True, config=CONFIG)
    assert result["complete_matched_study"] and result["decision"] == "inconclusive"
    assert result["mode"] == "offline_fixture" and not result["formal_admission"]
    accounting = json.loads((tmp_path / "accounting.json").read_text())
    assert accounting["outbound_reservations"] == client.calls == 39
    before = (tmp_path / "summary.json").read_bytes()
    ledger = (tmp_path / "ledger/runs.jsonl").read_bytes()
    second = execute(tmp_path, resume=True, client=client, offline_fixture=True, config=CONFIG)
    assert second == result and client.calls == 39
    assert (tmp_path / "summary.json").read_bytes() == before
    assert (tmp_path / "ledger/runs.jsonl").read_bytes() == ledger
    assert (tmp_path / "relationships.svg").exists()
    assert (tmp_path / "coverage.svg").exists()
    assert "Evidence mode: offline_fixture" in (tmp_path / "REPORT.html").read_text()
    assert "offline_fixture" in (tmp_path / "manifest.json").read_text()


@pytest.mark.slow
def test_credit_failure_preserves_request_and_resumes_only_after_resolution(tmp_path):
    client = FakeAgent(fail_first=True)
    first = execute(tmp_path, client=client, offline_fixture=True, config=CONFIG)
    assert not first["complete_matched_study"]
    assert "credits_exhausted" in first["stop_reason"] and client.calls == 1
    assert (tmp_path / "REPORT.html").exists()
    frozen = json.loads((tmp_path / "manifest.json").read_text())
    with RequestJournal(tmp_path / "requests.db", frozen) as journal:
        request = journal.report()["requests"][0]
        assert request["state"] == "paused"
        journal.resolve(request["id"], "retry", "fixture operator restored credits")
    second = execute(tmp_path, resume=True, client=client, offline_fixture=True, config=CONFIG)
    assert second["complete_matched_study"] and client.calls == 40
    assert len(list((tmp_path / "stores/A").glob("lehmann-attempt-*.db"))) == 2
    assert len((tmp_path / "ledger/runs.jsonl").read_text().splitlines()) == 1


@pytest.mark.slow
def test_changed_configuration_cannot_reuse_live_study_identity(tmp_path):
    client = FakeAgent()
    execute(tmp_path, client=client, offline_fixture=True, config=CONFIG)
    from dataclasses import replace

    with pytest.raises(ValueError, match="manifest changed"):
        execute(
            tmp_path,
            resume=True,
            client=client,
            offline_fixture=True,
            config=replace(CONFIG, model="different-fixture"),
        )
    assert client.calls == 39


@pytest.mark.slow
def test_fresh_process_offline_verification_has_identical_canonical_evidence(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS="1")
    for folder in ("first", "second"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "capstone.pipeline.research_experiment",
                "verify",
                "--output",
                str(tmp_path / folder),
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "first/conformance.json").read_bytes() == (
        tmp_path / "second/conformance.json"
    ).read_bytes()
