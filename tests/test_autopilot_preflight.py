"""Preflight rows for the sources inbox and the model key: SET/UNSET only, never the value.

Fixtures only; nothing connects and no model is called.
"""

from __future__ import annotations

import json

from test_autopilot_init import by_check, ready

from capstone.pipeline import model_key
from capstone.pipeline.autopilot import __main__ as cli
from capstone.pipeline.autopilot import preflight
from capstone.pipeline.autopilot.config import Key

REAL = [
    ("real_model_calls = false", "real_model_calls = true"),
    ("accounting_roots = []", 'accounting_roots = []\nruns_dir = "runs"'),
    ("run_ceiling = 0", "run_ceiling = 4"),
    ("global_ceiling = 0", "global_ceiling = 8"),
]
SOURCES = {"sources": {"inbox": Key("path", None, "the folder sources are dropped into")}}


def test_key_state_is_set_only_for_a_nonblank_value():
    assert model_key.key_state({model_key.API_KEY_ENV: "x"}) == "SET"
    assert model_key.key_state({model_key.API_KEY_ENV: "   "}) == "UNSET"
    assert model_key.key_state({model_key.API_KEY_ENV: None}) == "UNSET"
    assert model_key.key_state({}) == "UNSET"


def test_the_key_is_not_looked_at_while_model_calls_are_off(tmp_path):
    rows, _ = preflight.run(ready(tmp_path), env={})
    key = by_check(rows)["api_key"][0]
    assert (key.status, key.code, key.required) == ("skipped", "not_applicable", False)
    assert not [row for row in rows if row.failed]


def test_real_model_calls_report_the_key_set_or_unset_never_its_value(tmp_path):
    config = ready(tmp_path, edits=REAL)
    secret = "fake-value-for-the-test-only"
    rows, _ = preflight.run(config, env={model_key.API_KEY_ENV: secret})
    key = by_check(rows)["api_key"][0]
    assert (key.status, key.detail) == ("ok", "the key variable is SET")
    assert secret not in json.dumps(preflight.document(rows))
    rows, _ = preflight.run(config, env={})
    key = by_check(rows)["api_key"][0]
    assert (key.status, key.code, key.failed) == ("missing", "api_key_unset", True)
    assert "set the model API key" in key.fix and "UNSET" in key.detail


def test_the_default_environment_is_the_process_environment(tmp_path, monkeypatch):
    config = ready(tmp_path, edits=REAL)
    monkeypatch.delenv(model_key.API_KEY_ENV, raising=False)
    assert by_check(preflight.run(config)[0])["api_key"][0].code == "api_key_unset"
    monkeypatch.setenv(model_key.API_KEY_ENV, "fake-value-for-the-test-only")
    assert by_check(preflight.run(config)[0])["api_key"][0].status == "ok"


def test_no_sources_section_means_no_inbox_to_check(tmp_path):
    row = by_check(preflight.run(ready(tmp_path))[0])["inbox"][0]
    assert (row.status, row.required) == ("skipped", False)


def test_a_configured_inbox_must_be_an_existing_folder(tmp_path):
    config = ready(tmp_path)
    with config.open("a", encoding="utf-8") as stream:
        stream.write('\n[sources]\ninbox = "inbox"\n')
    rows, loaded = preflight.run(config, extra=SOURCES)
    assert loaded is not None
    row = by_check(rows)["inbox"][0]
    assert (row.status, row.code) == ("missing", "inbox_missing") and row.failed
    (config.parent / "inbox").mkdir()
    row = by_check(preflight.run(config, extra=SOURCES)[0])["inbox"][0]
    assert (row.status, row.code) == ("ok", "ok")
    with config.open("a", encoding="utf-8") as stream:
        stream.write("")
    unset = tmp_path / "unset"
    unset_config = ready(unset)
    unset_config.write_text(
        unset_config.read_text(encoding="utf-8") + "\n[sources]\n", encoding="utf-8"
    )
    row = by_check(preflight.run(unset_config, extra=SOURCES)[0])["inbox"][0]
    assert row.code == "inbox_missing" and row.detail == "None"


def test_the_key_value_is_never_printed(tmp_path, monkeypatch, capsys):
    config = ready(tmp_path, edits=REAL)
    secret = "fake-value-for-the-test-only"
    monkeypatch.setenv(model_key.API_KEY_ENV, secret)
    assert cli.main(["preflight", "--config", str(config)]) == 0
    shown = capsys.readouterr()
    assert secret not in shown.out + shown.err
    assert "SET" in shown.out
    written = (config.parent / "preflight.json").read_text(encoding="utf-8")
    assert secret not in written
