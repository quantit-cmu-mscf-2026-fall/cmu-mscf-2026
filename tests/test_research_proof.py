"""The offline proof fixtures: small independently calculated examples, no model calls."""

import pytest

from capstone.factors import research_proof
from capstone.factors.research_proof import (
    ScriptedProofClient,
    construction_handoff_checks,
    development_checks,
    recovery_checks,
)


def test_six_development_cases_use_shared_construction_and_independent_examples():
    checks = development_checks()
    assert len(checks) == 8
    assert all(checks.values())


def test_known_construction_handoff_blocks_stale_financial_calibration(tmp_path):
    assert all(construction_handoff_checks(tmp_path).values())
    # A second pass over the same journal resumes at the saved state and agrees.
    assert all(construction_handoff_checks(tmp_path).values())


def test_saved_responses_replay_and_an_uncertain_send_is_never_retried(tmp_path):
    checks = recovery_checks(tmp_path)
    assert checks == {
        "saved_response_replayed": True,
        "uncertain_request_not_retried": True,
        "fixture_model_calls_only": True,
    }


def test_the_scripted_client_is_a_local_fixture_that_counts_its_calls():
    client = ScriptedProofClient()
    assert client.messages.create(model="fixture").content[0].input == {"saved": True}
    assert client.calls == 1
    with pytest.raises(TimeoutError):
        client.create(model="timeout-fixture")
    assert client.calls == 2


def test_the_proof_module_makes_no_model_or_licensed_data_call():
    source = open(research_proof.__file__, encoding="utf-8").read()
    for forbidden in ("anthropic", "shared_data", "wrds", "claude-"):
        assert forbidden not in source
