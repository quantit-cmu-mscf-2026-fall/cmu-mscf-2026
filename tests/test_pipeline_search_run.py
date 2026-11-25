"""`pipeline.search.run_search`: load the search rows, search, register, never open the holdout.

A scripted model client, synthetic rows and a recording loader; a temporary ledger.
No model, no network, no market data.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from test_deepen import ScriptedEditor
from test_factor_evaluation import fake_rows
from test_pipeline_search_steps import EDITS, ScriptedClient, _config

from capstone import gate
from capstone import shared_data as sd
from capstone.factors.evaluation import DISCOVERY_END, HoldoutError
from capstone.pipeline import manifest, placebo, search
from capstone.pipeline.e2e import ThresholdsNotRegistered
from capstone.pipeline.rows import rows_digest, sorted_rows

PERIOD = {
    "search_start": "2015-01-01",
    "search_end": "2015-12-31",
    "holdout_start": "2016-01-04",
    "holdout_end": "2016-12-30",
}


class Loader:
    """Fake rows that record every (start, end) they are asked for."""

    def __init__(self, rows):
        self.rows, self.asked = rows, []

    def __call__(self, start, end):
        self.asked.append((start, end))
        return self.rows.copy()


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


@pytest.fixture(scope="module")
def rows():
    return fake_rows(n_stocks=12, n_days=150)


def _go(tmp_path, config, loader, *, client=None, name="run", **kwargs):
    client = client or ScriptedClient()
    run = search.run_search(
        config,
        client=client,
        editor=ScriptedEditor(*EDITS),
        loader=loader,
        workdir=tmp_path / name,
        **kwargs,
    )
    return run, client


def _config_for(tmp_path, **changes):
    return _config(tmp_path, **PERIOD, **changes)


def test_run_search_asks_for_the_search_period_only_and_registers(tmp_path, rows, ledger):
    config = _config_for(tmp_path)
    loader = Loader(rows)
    run, client = _go(tmp_path, config, loader, source_version="fx")

    assert loader.asked == [("2015-01-01", "2015-12-31")]
    record = manifest.read_manifest(run.path)
    assert record == run.manifest
    digest = rows_digest(sorted_rows(rows))
    assert record.data_version == f"{config.data_label}:fx:{digest}"
    assert record.model_calls == client.calls > 0
    assert gate.GateConfig.from_dict(record.gate_config).search_end == "2015-12-31"
    assert json.loads(run.path.read_text(encoding="utf-8"))["n_variants"] == record.n_variants


def test_the_default_loader_reads_the_shared_file_and_joins_its_data_version(
    tmp_path, rows, ledger, monkeypatch
):
    asked = []

    class Cached:
        def __init__(self, name, columns):
            asked.append((name, columns))

        def __call__(self, start, end):
            asked.append((start, end))
            return rows.copy()

    monkeypatch.setattr(search, "CachedRows", Cached)
    monkeypatch.setattr(sd, "data_version", lambda: "shared-v7")
    config = _config_for(tmp_path)
    run, _ = _go(tmp_path, config, None)

    assert asked[0][0] == search.DAILY_FILE and asked[1] == ("2015-01-01", "2015-12-31")
    assert ":shared-v7:" in run.manifest.data_version


def test_a_second_search_in_the_workdir_is_refused_before_any_model_call(tmp_path, rows, ledger):
    config = _config_for(tmp_path)
    _go(tmp_path, config, Loader(rows))
    loader, client = Loader(rows), ScriptedClient()
    with pytest.raises(FileExistsError, match="already holds a search"):
        _go(tmp_path, config, loader, client=client)
    assert client.calls == 0 and loader.asked == []


def test_a_workdir_with_only_a_factor_store_is_refused(tmp_path, rows, ledger):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "factors.db").write_bytes(b"")
    client = ScriptedClient()
    with pytest.raises(FileExistsError):
        _go(tmp_path, _config_for(tmp_path), Loader(rows), client=client)
    assert client.calls == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda r: r[r["date"] > "2015-03-01"].assign(
                date=lambda d: d["date"] + pd.Timedelta(days=400)
            ),
            "past search_end",
        ),
        (lambda r: r.assign(date=r["date"] - pd.Timedelta(days=400)), "before 2015-01-01"),
        (lambda r: r.iloc[0:0], "no rows"),
    ],
)
def test_bad_rows_are_refused_before_any_model_call(tmp_path, rows, ledger, change, message):
    client = ScriptedClient()
    with pytest.raises(ValueError, match=message):
        _go(tmp_path, _config_for(tmp_path), Loader(change(rows)), client=client)
    assert client.calls == 0 and not (tmp_path / "run").exists()


def test_a_search_end_after_the_discovery_period_is_refused_before_loading(tmp_path, rows, ledger):
    config = _config(
        tmp_path,
        search_start="2015-01-01",
        search_end="2021-06-30",
        holdout_start="2021-07-01",
        holdout_end="2021-12-31",
    )
    loader, client = Loader(rows), ScriptedClient()
    with pytest.raises(HoldoutError, match=DISCOVERY_END):
        _go(tmp_path, config, loader, client=client)
    assert loader.asked == [] and client.calls == 0


def test_a_placebo_config_is_refused_before_loading_or_any_model_call(
    tmp_path, rows, ledger, monkeypatch
):
    monkeypatch.setattr(placebo, "check_ledger", lambda environ=None: None)
    config = _config_for(tmp_path, placebo="cross_section_permutation", placebo_seed=1)
    loader, client = Loader(rows), ScriptedClient()
    with pytest.raises(placebo.PlaceboRefused, match="not supported"):
        _go(tmp_path, config, loader, client=client)
    assert loader.asked == [] and client.calls == 0


def test_the_placebo_ledger_rule_is_checked_first(tmp_path, rows, monkeypatch):
    def refuse(environ=None):
        raise placebo.PlaceboRefused("ledger")

    monkeypatch.setattr(placebo, "check_ledger", refuse)
    config = _config_for(tmp_path, placebo="cross_section_permutation", placebo_seed=1)
    with pytest.raises(placebo.PlaceboRefused, match="^ledger$"):
        _go(tmp_path, config, Loader(rows))


def test_default_thresholds_on_real_data_are_refused_before_loading(tmp_path, rows, ledger):
    config = replace(
        _config_for(tmp_path), data_label="crsp-fixture", thresholds_source="code-defaults"
    )
    loader, client = Loader(rows), ScriptedClient()
    with pytest.raises(ThresholdsNotRegistered):
        _go(tmp_path, config, loader, client=client)
    assert loader.asked == [] and client.calls == 0
