"""The pipeline view: places runs from progress files and the ledger, builds its page.

Fixtures only: a temp progress folder and a temp ledger; no data, no server.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "mscf-plugin"
SCRIPTS = PLUGIN / "skills" / "pipeline-view" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_scene  # noqa: E402
import mscf_progress  # noqa: E402
import pipeline_view  # noqa: E402

# --- the lab's state --------------------------------------------------------------------------


STATIONS = json.loads((SCRIPTS / "stations.json").read_text())["stations"]


def write_run(
    pdir: Path,
    run_id: str,
    name: str,
    *,
    started: float,
    heartbeat: float,
    status="running",
    ended=None,
    evs=(),
) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "name": name,
        "user": "analyst",
        "started": started,
        "heartbeat": heartbeat,
        "status": status,
        "exit_code": None if status == "running" else 0,
        "ended": ended,
    }
    (pdir / f"{run_id}.json").write_text(json.dumps(meta))
    (pdir / f"{run_id}.events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))


def ledger_entry(name: str, when: float, *tags: str) -> dict:
    return {
        "name": name,
        "ts_utc": datetime.fromtimestamp(when, UTC).isoformat(),
        "tags": list(tags),
    }


def test_line_places_runs_from_progress_events_and_the_ledger(tmp_path):
    now = time.time()
    p = tmp_path / "progress"
    write_run(
        p,
        "a",
        "momentum",
        started=now - 60,
        heartbeat=now - 1,
        evs=[{"station": "backtest", "pct": 0.4}],
    )
    write_run(
        p,
        "b",
        "value",
        started=now - 60,
        heartbeat=now - 1,
        status="done",
        ended=now - 5,
        evs=[{"station": "ledger"}],
    )
    write_run(p, "c", "quality", started=now - 60, heartbeat=now - 40, evs=[{"station": "signal"}])
    write_run(
        p,
        "d",
        "waiting",
        started=now - 60,
        heartbeat=now - 1,
        evs=[{"station": "backtest"}, {"waiting": True}],
    )
    write_run(
        p, "e", "old", started=now - 9000, heartbeat=now - 8000, status="done", ended=now - 7200
    )
    write_run(
        p,
        "f",
        "shipped",
        started=now - 60,
        heartbeat=now - 1,
        status="done",
        ended=now - 2,
        evs=[{"station": "incubation"}],
    )
    entries = [
        ledger_entry("value", now - 10, "stage:screen", "verdict:fail"),
        ledger_entry(
            "value", now - 9000, "stage:robustness", "verdict:pass"
        ),  # before the run started
        ledger_entry("shipped", now - 3, "stage:incubation", "verdict:pass"),
    ]
    jobs = {j["name"]: j for j in pipeline_view.build_line(p, entries, STATIONS, now)["jobs"]}
    assert set(jobs) == {"momentum", "value", "quality", "waiting", "shipped"}
    assert jobs["momentum"]["station"] == "backtest" and jobs["momentum"]["pct"] == 0.4
    assert jobs["momentum"]["status"] == "running" and not jobs["momentum"]["waiting"]
    assert jobs["value"]["station"] == "screen" and jobs["value"]["status"] == "rejected"
    assert jobs["quality"]["status"] == "stalled"
    assert jobs["waiting"]["waiting"] is True
    assert jobs["shipped"]["status"] == "passed" and jobs["shipped"]["index"] == 9


def test_reject_counts_use_a_rolling_window_and_keep_the_total():
    now = time.time()
    entries = [
        ledger_entry("a", now - 3600, "stage:screen", "verdict:fail"),
        ledger_entry("b", now - 3 * 86400, "stage:screen", "verdict:fail"),
        ledger_entry("c", now - 60, "stage:robustness", "verdict:pass"),
        ledger_entry("d", now - 60, "stage:robustness", "verdict:fail"),
    ]
    day = pipeline_view.reject_counts(entries, STATIONS, now, 24)
    assert day["screen"] == {"window": 1, "total": 2, "hours": 24}
    assert day["robustness"]["window"] == 1 and day["final"]["total"] == 0
    assert pipeline_view.reject_counts(entries, STATIONS, now, 24 * 7)["screen"]["window"] == 2


def test_snapshot_embeds_scene_and_escaped_state_without_the_session_id(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    entry = {
        "name": "</script><b>x",
        "ts_utc": datetime.now(UTC).isoformat(),
        "session_id": "secret-session",
        "tags": ["stage:screen", "verdict:fail"],
        "params": {"end": "2020-12-31"},
    }
    (ledger / "runs.jsonl").write_text(json.dumps(entry) + "\n")
    (tmp_path / "repo").mkdir()
    out = tmp_path / "view.html"
    pipeline_view.main(
        [
            "--repo",
            str(tmp_path / "repo"),
            "--ledger-dir",
            str(ledger),
            "--progress-dir",
            str(tmp_path / "p"),
            "--snapshot",
            str(out),
        ]
    )
    html = out.read_text(encoding="utf-8")
    assert '<svg id="lab"' in html and "@font-face" not in html  # no bundled fonts: system faces
    assert "/*FONTS*/" not in html and "<!--SCENE-->" not in html
    assert "</script><b>" not in html and "secret-session" not in html
    embedded = html.split("const SNAPSHOT = ", 1)[1].split(";\n", 1)[0]
    state = json.loads(embedded)
    assert state["trials"] == 1 and state["rejects"]["screen"]["window"] == 1
    assert str(tmp_path) not in html  # file names only: no home-directory paths
    assert state["ledger"] == "runs.jsonl" and state["line"]["progress_dir"] == "p"


def test_every_less_than_in_the_embedded_state_is_escaped(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    entry = {"name": "<!--<script>", "ts_utc": datetime.now(UTC).isoformat(), "tags": []}
    (ledger / "runs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    state = pipeline_view.build_state(ledger / "runs.jsonl", ROOT, tmp_path / "p", 24.0)
    html = pipeline_view.render(state)
    embedded = html.split("const SNAPSHOT = ", 1)[1].split(";\n", 1)[0]
    assert "<" not in embedded
    assert json.loads(embedded)["top_names"][0]["name"] == "<!--<script>"


def test_scene_file_is_in_sync_with_its_generator_and_has_every_hook_the_page_uses():
    svg = (
        (SCRIPTS / "lab_scene.svg").read_text(encoding="utf-8").replace(chr(13) + chr(10), chr(10))
    )
    assert svg == build_scene.scene(), "run build_scene.py"
    page = (SCRIPTS / "view.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([\w-]+)"', svg))
    for i in range(10):
        assert {
            f"inst-{i}",
            f"lamp-{i}",
            f"basket-{i}",
            f"basket-label-{i}",
            f"basket-tubes-{i}",
        } <= ids
        assert (f"lcd-{i}" in ids) == (i not in (7, 8))
    used = set(re.findall(r'getElementById\("([\w-]+)"\)', page))
    assert used <= ids | {"tab-line", "tab-ledger", "tab-line-btn"}, used - ids
    assert [s["id"] for s in STATIONS][5:] == [
        "screen",
        "robustness",
        "incremental",
        "final",
        "incubation",
    ]
    assert len(STATIONS) == len(build_scene.NAMES) == 10


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_script_parses(tmp_path):
    html = pipeline_view.render(None)
    script = html[html.index("<script>") + len("<script>") : html.rindex("</script>")]
    (tmp_path / "page.js").write_text(script, encoding="utf-8")
    assert subprocess.run(["node", "--check", str(tmp_path / "page.js")]).returncode == 0


def test_progress_files_stay_outside_the_repository(monkeypatch):
    monkeypatch.delenv("MSCF_PROGRESS_DIR", raising=False)
    assert ROOT not in mscf_progress.progress_dir().parents
    assert timedelta(minutes=30).total_seconds() == pipeline_view.KEEP_FINISHED_SECONDS


@pytest.mark.parametrize(
    "flag,where",
    [
        ("--ledger-dir", "outputs/registered-slice-01/ledger"),
        ("--progress-dir", "data_cache/wrds"),
        ("--snapshot", "outputs/view.html"),
        ("--snapshot", "pipeline-lab.html"),
        ("--snapshot", "docs/nested/pipeline-lab.html"),
    ],
)
def test_viewer_refuses_run_outputs_and_market_data(tmp_path, flag, where):
    with pytest.raises(SystemExit) as stop:
        pipeline_view.main(["--repo", str(tmp_path), flag, str(tmp_path / where), "--json"])
    assert stop.value.code == 2


def test_plugin_scripts_never_name_run_outputs_or_market_data():
    """Only the guard may mention them, and only to deny them."""
    for path in SCRIPTS.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        code = text.split("def off_limits", 1)[0] + text.split("return None", 1)[-1]
        assert not re.search(r"outputs/|\.parquet|data_cache|sd\.load|shared_data", code), path.name


def test_fonts_are_embedded_only_when_the_woff2_files_are_present(tmp_path, monkeypatch):
    assert pipeline_view.fonts_css() == ""  # the repository ships no binary assets
    (tmp_path / "overpass-latin-600-normal.woff2").write_bytes(b"fake")
    monkeypatch.setattr(pipeline_view, "ASSETS", tmp_path)
    css = pipeline_view.fonts_css()
    assert css.count("@font-face") == 1 and "base64,ZmFrZQ==" in css
    assert not list(SCRIPTS.rglob("*.woff2"))


def test_trials_come_from_runlog_so_a_ledger_correction_is_not_a_trial(tmp_path, monkeypatch):
    from capstone import runlog

    ledger = tmp_path / "ledger"
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(ledger))
    tags = ["stage:screen"]
    first = runlog.log_run("mom", params={"lb": 1}, seed=1, tags=tags)
    runlog.log_run("mom", params={"lb": 2}, seed=1, tags=tags)
    runlog.log_run(
        runlog.CORRECTION_NAME,
        params={
            "ledger_name": "mom",
            "excludes": [{"ts_utc": first["ts_utc"], "lb": 1}],
            "reason": "duplicate of a crashed run",
        },
    )
    assert runlog.trial_count() == 1
    state = pipeline_view.build_state(ledger / "runs.jsonl", ROOT, tmp_path / "p", 24.0)
    assert state["trials"] == runlog.trial_count() == 1
    assert state["names"] == 1 and state["ledger_error"] is None
    assert [n["name"] for n in state["top_names"]] == ["mom"]
    assert next(x for x in state["stages"] if x["stage"] == "screen")["runs"] == 1
    assert state["hygiene"].get("no data_version") == 1  # the correction is not a trial run


def test_a_line_that_is_json_but_not_an_object_is_skipped_with_an_error(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    good = json.dumps({"name": "mom", "ts_utc": "2026-01-01T00:00:00+00:00", "tags": []})
    (ledger / "runs.jsonl").write_text(good + "\n[1, 2]\n", encoding="utf-8")
    state = pipeline_view.build_state(ledger / "runs.jsonl", ROOT, tmp_path / "p", 24.0)
    assert state["names"] == 1 and state["trials"] == 0
    assert "line 2 is not a JSON object" in state["ledger_error"]
    assert str(tmp_path) not in json.dumps(state)


def test_a_ledger_error_does_not_carry_the_folder_path(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "runs.jsonl").write_text("{not json\n", encoding="utf-8")
    state = pipeline_view.build_state(ledger / "runs.jsonl", ROOT, tmp_path / "p", 24.0)
    assert "runs.jsonl:1" in state["ledger_error"] and str(tmp_path) not in state["ledger_error"]
    page = (SCRIPTS / "view.html").read_text(encoding="utf-8")
    assert "st.ledger_error" in page.split("function render", 1)[1]  # the tile shows it


@pytest.mark.parametrize(
    "entry,reads",
    [
        ({"params": {"holdout_start": "2021-01-01"}}, True),
        ({"tags": ["stage:final"]}, True),
        ({"tags": ["holdout"]}, True),
        ({"params": {"end": "2024-12-31"}}, True),
        ({"params": {"end": "2020-12-31"}, "tags": ["stage:screen"]}, False),
        ({"tags": ["stage:incubation"]}, False),
    ],
)
def test_the_markers_of_a_holdout_read(entry, reads):
    assert pipeline_view.reads_holdout(entry) is reads


def test_the_server_answers_only_a_loopback_host(tmp_path):
    import http.client
    import threading

    server = pipeline_view.make_server(tmp_path / "runs.jsonl", ROOT, 0, tmp_path / "p", 24.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]

        def status(host):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.putrequest("GET", "/state.json", skip_host=True)
            conn.putheader("Host", host)
            conn.endheaders()
            code = conn.getresponse().status
            conn.close()
            return code

        assert status(f"127.0.0.1:{port}") == 200 and status(f"localhost:{port}") == 200
        assert status(f"evil.example:{port}") == 403 and status("rebind.example") == 403
    finally:
        server.shutdown()
        server.server_close()


def test_an_unreadable_ledger_line_is_an_error_not_a_trial(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    good = json.dumps({"name": "mom", "ts_utc": "2026-01-01T00:00:00+00:00", "tags": []})
    (ledger / "runs.jsonl").write_text(good + "\n{not json\n", encoding="utf-8")
    state = pipeline_view.build_state(ledger / "runs.jsonl", ROOT, tmp_path / "p", 24.0)
    assert state["trials"] == 0 and "not valid JSON" in state["ledger_error"]
    assert "<unreadable line>" not in json.dumps(state["top_names"])
