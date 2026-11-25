"""`python -m capstone.pipeline.real_run`: argument parsing and the refusals that come
before anything is loaded or sent.

A fake client factory and a stubbed run; never the network, a model or market data.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from capstone.pipeline import placebo, real_run
from capstone.pipeline.config import REPO
from capstone.pipeline.manifest import MANIFEST_FILE

CRSP = REPO / "config" / "e2e_crsp.toml"


class Boom(AssertionError):
    pass


def _forbid(*args, **kwargs):
    raise Boom("loaded or called something")


@pytest.fixture
def untouched(monkeypatch):
    """Anything that would load data or reach a model fails the test."""
    monkeypatch.setattr(real_run, "run_real_search", _forbid)
    monkeypatch.setattr(real_run, "_client", _forbid)


def _run(tmp_path, *extra):
    return real_run.main(["--workdir", str(tmp_path / "run"), *extra], client_factory=_forbid)


def test_it_refuses_without_an_api_key(untouched, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    assert _run(tmp_path) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_it_refuses_without_an_explicit_ledger(untouched, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-key")
    monkeypatch.delenv("CAPSTONE_LEDGER_DIR", raising=False)
    assert _run(tmp_path) == 2
    assert "CAPSTONE_LEDGER_DIR" in capsys.readouterr().err


def test_it_never_prints_the_key(untouched, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    monkeypatch.delenv("CAPSTONE_LEDGER_DIR", raising=False)
    _run(tmp_path)
    out = capsys.readouterr()
    assert "sk-secret-value" not in out.out + out.err


def test_it_refuses_a_config_without_on_budget_stop(untouched, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-key")
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    synthetic = REPO / "config" / "e2e_synthetic.toml"
    assert _run(tmp_path, "--config", str(synthetic)) == 2
    assert "on_budget" in capsys.readouterr().err


def test_the_placebo_ledger_rule_is_checked_before_the_key(
    untouched, monkeypatch, tmp_path, capsys
):
    text = CRSP.read_text(encoding="utf-8").replace(
        "[run]", '[run]\nplacebo = "cross_section_permutation"\nplacebo_seed = 1', 1
    )
    config = tmp_path / "placebo.toml"
    config.write_text(text.replace('papers = "config/', f'papers = "{REPO.as_posix()}/config/'))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))

    def refuse(environ=None):
        raise placebo.PlaceboRefused("placebo ledger")

    monkeypatch.setattr(placebo, "check_ledger", refuse)
    assert _run(tmp_path, "--config", str(config)) == 2
    assert "placebo ledger" in capsys.readouterr().err


def test_a_workdir_that_holds_a_search_is_refused_before_the_client_is_used(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-key")
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / MANIFEST_FILE).write_text("{}", encoding="utf-8")
    made = []

    def factory():
        made.append(1)
        return _forbid

    assert real_run.main(["--workdir", str(tmp_path / "run")], client_factory=factory) == 2
    assert "already holds a search" in capsys.readouterr().err
    assert made == []


def test_a_run_gets_the_factory_client_and_prints_the_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-key")
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    seen = {}
    summary = dict(
        model_calls=7,
        usage={"usd": 1.5, "unpriced": ["odd-model"]},
        trial_count=40,
        run_tag="e2e-run:crsp-v1",
        deepen={"rounds": 3, "stopped": ""},
        n_variants=12,
        families=["a", "b"],
        gate_config_hash="ab" * 32,
        stage3="unmeasured",
    )

    def fake_run(config, *, client, workdir):
        seen.update(config=config, client=client, workdir=workdir)
        return summary

    monkeypatch.setattr(real_run, "run_real_search", fake_run)
    client = object()
    assert real_run.main(["--workdir", str(tmp_path / "run")], client_factory=lambda: client) == 0
    assert seen["client"] is client and seen["workdir"] == tmp_path / "run"
    assert seen["config"].run_tag == "e2e-run:crsp-v1"
    out = capsys.readouterr().out
    assert "7 of" in out and "$1.50" in out and "odd-model" in out
    assert "holdout not opened" in out and "stage 3      unmeasured" in out
    assert "discovery (gate stages 1-3) is a separate step" in out


def test_the_default_client_is_built_with_no_sdk_retries_and_prompt_caching(monkeypatch):
    built = {}

    class Fake:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.max_retries = kwargs.get("max_retries", 2)

    # CI installs without the `agents` extra, so `anthropic` may not exist: insert a fake
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Fake))
    client = real_run._client()
    assert isinstance(client, real_run.PromptCaching)
    assert built == {"max_retries": 0} and client.max_retries == 0


def test_summarize_only_needs_no_key_and_calls_nothing(untouched, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CAPSTONE_LEDGER_DIR", raising=False)
    written = []
    monkeypatch.setattr(real_run, "write_summary", written.append)
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / MANIFEST_FILE).write_text(json.dumps({}), encoding="utf-8")
    assert real_run.main(["--workdir", str(workdir), "--summarize-only"]) == 0
    assert written == [workdir]
    assert "run_summary.json" in capsys.readouterr().out


def test_summarize_only_refuses_a_workdir_without_a_search(untouched, tmp_path):
    assert _run(tmp_path, "--summarize-only") == 2


def test_the_runner_has_no_holdout_or_discovery_path():
    source = Path(real_run.__file__).read_text(encoding="utf-8")
    for word in ("run_holdout", "holdout_loader", "run_gate", "run_discovery", "authorize"):
        assert word not in source
