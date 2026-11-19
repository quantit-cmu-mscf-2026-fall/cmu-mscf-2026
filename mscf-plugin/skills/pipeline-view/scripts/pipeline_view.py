"""Live view of the cmu-mscf-2026 pipeline, built from the run ledger (stdlib only).

    python pipeline_view.py                       # serve http://127.0.0.1:8765, refreshes every 2 s
    python pipeline_view.py --port 9000 --ledger-dir .ledger-scratch
    python pipeline_view.py --snapshot view.html  # one self-contained page, no server

Two tabs: "Production line" (experiments in progress, from mscf_progress / pipeline_run.py
heartbeats in $MSCF_PROGRESS_DIR) and "Ledger" (everything logged). The page reads only those
files, the ledger (runs.jsonl) and git metadata. It never writes to the
repository, never reads market data, and binds to 127.0.0.1 only. A snapshot carries
user names and metrics: keep it out of the public repository.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STAGES = ["screen", "robustness", "incremental", "final", "incubation"]
SEARCH_END = "2020-12-31"
HERE = Path(__file__).resolve().parent
PAGE = HERE / "view.html"
STATIONS = HERE / "stations.json"
SCENE = HERE / "lab_scene.svg"
ASSETS = HERE / "assets"
FONT_FILES = [
    ("Overpass", 600, "overpass-latin-600-normal.woff2"),
    ("Overpass", 800, "overpass-latin-800-normal.woff2"),
    ("Plex Mono", 500, "ibm-plex-mono-latin-500-normal.woff2"),
]
REJECT_WINDOW_HOURS = 24
STALE_SECONDS = 15
KEEP_FINISHED_SECONDS = 30 * 60

sys.path.insert(0, str(HERE))
from mscf_progress import progress_dir  # noqa: E402


def off_limits(path: Path, repo: Path) -> str | None:
    """The viewer reads progress files and the ledger only: never run outputs or market data."""
    path = path.resolve()
    if path == repo / "outputs" or repo / "outputs" in path.parents:
        return "outputs/ holds immutable run artifacts"
    if "data_cache" in path.parts or path.suffix == ".parquet":
        return "shared market data is read only through capstone.shared_data"
    return None


def inside_repo(path: Path, repo: Path) -> bool:
    path = path.resolve()
    return path == repo or repo in path.parents


def counted_entries(path: Path, repo: Path) -> tuple[list[dict], int, str | None]:
    """(entries that are trials, the trial count, a ledger error or None).

    The count is `capstone.runlog`'s own: ledger corrections applied, correction entries never
    trials, an unreadable line an error (not a trial). Without `capstone` importable the
    viewer cannot apply corrections, so it reports the error instead of a raw line count."""
    entries = read_entries(path)
    corrupt = [e for e in entries if "corrupt" in (e.get("tags") or [])]
    plain = [e for e in entries if e not in corrupt]
    try:
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from capstone import runlog

        previous = os.environ.get("CAPSTONE_LEDGER_DIR")
        os.environ["CAPSTONE_LEDGER_DIR"] = str(path.parent)
        try:
            trials = runlog._Corrections(runlog._read_numbered()).trials
        finally:
            if previous is None:
                os.environ.pop("CAPSTONE_LEDGER_DIR", None)
            else:
                os.environ["CAPSTONE_LEDGER_DIR"] = previous
    except ImportError:
        kept = [e for e in plain if e.get("name") != "ledger-correction"]
        return kept, len(kept), "capstone.runlog not importable: corrections not applied"
    except ValueError as exc:
        return plain, 0, str(exc)
    return trials, len(trials), None


def ledger_path(ledger_dir: str | None, repo: Path) -> Path:
    root = ledger_dir or os.environ.get("CAPSTONE_LEDGER_DIR")
    return (Path(root) if root else repo / "experiments") / "runs.jsonl"


def read_entries(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append({"name": "<unreadable line>", "tags": ["corrupt"]})
    return entries


def tag_values(entry: dict, prefix: str) -> list[str]:
    return [t.split(":", 1)[1] for t in entry.get("tags") or [] if t.startswith(prefix + ":")]


def reads_holdout(entry: dict) -> bool:
    params = entry.get("params") or {}
    end = str(params.get("end") or params.get("holdout_end") or "")
    return "holdout" in (entry.get("tags") or []) or end[:10] > SEARCH_END


def git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def build_state(
    path: Path, repo: Path, pdir: Path | None = None, reject_hours: float = REJECT_WINDOW_HOURS
) -> dict:
    entries, trial_total, ledger_error = counted_entries(path, repo)
    stations = (read_json(STATIONS) or {}).get("stations", [])
    now = time.time()
    line = build_line(pdir or progress_dir(), entries, stations, now)
    stage_runs, stage_pass = Counter(), Counter()
    by_name, by_user = Counter(), Counter()
    families = defaultdict(set)
    hygiene = Counter()
    timeline = []
    for i, e in enumerate(entries, 1):
        name = e.get("name", "?")
        by_name[name] += 1
        by_user[e.get("user", "unknown")] += 1
        stages = tag_values(e, "stage") or ["untagged"]
        for s in stages:
            stage_runs[s] += 1
            if "pass" in tag_values(e, "verdict"):
                stage_pass[s] += 1
        for fam in tag_values(e, "family"):
            families[fam].add(name)
        params = e.get("params") or {}
        if e.get("seed") is None:
            hygiene["no seed"] += 1
        if "data_version" not in params:
            hygiene["no data_version"] += 1
        if not e.get("metrics"):
            hygiene["no metrics"] += 1
        timeline.append({"ts": e.get("ts_utc"), "n": i})
    holdout = [
        {"ts": e.get("ts_utc"), "name": e.get("name"), "user": e.get("user")}
        for e in entries
        if reads_holdout(e)
    ]
    recent = [
        {
            "ts": e.get("ts_utc"),
            "user": e.get("user"),
            "name": e.get("name"),
            "stage": ",".join(tag_values(e, "stage")) or "-",
            "verdict": ",".join(tag_values(e, "verdict")) or "-",
            "seed": e.get("seed"),
            "metrics": e.get("metrics") or {},
            "sha": (e.get("git_sha") or "")[:7],
        }
        for e in entries[-25:][::-1]
    ]
    last_dv = next(
        (
            (e.get("params") or {}).get("data_version")
            for e in reversed(entries)
            if (e.get("params") or {}).get("data_version")
        ),
        None,
    )
    return {
        "ledger": str(path),
        "ledger_exists": path.is_file(),
        "mtime": path.stat().st_mtime if path.is_file() else None,
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "sha": git(repo, "rev-parse", "--short", "HEAD"),
        "trials": trial_total,
        "ledger_error": ledger_error,
        "names": len(by_name),
        "users": len(by_user),
        "families": len(families),
        "data_version": last_dv,
        "stages": [
            {"stage": s, "runs": stage_runs.get(s, 0), "passed": stage_pass.get(s, 0)}
            for s in STAGES + (["untagged"] if stage_runs.get("untagged") else [])
        ],
        "top_names": [{"name": n, "runs": c} for n, c in by_name.most_common(10)],
        "timeline": timeline[-2000:],
        "holdout_reads": holdout,
        "hygiene": dict(hygiene),
        "recent": recent,
        "line": line,
        "rejects": reject_counts(entries, stations, now, reject_hours),
        "shipped_total": sum(
            1
            for e in entries
            if "incubation" in tag_values(e, "stage") and "pass" in tag_values(e, "verdict")
        ),
        "now": now,
    }


def epoch(ts: str | None) -> float | None:
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except ValueError:
        return None


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_line(pdir: Path, entries: list[dict], stations: list[dict], now: float) -> dict:
    ids = [s["id"] for s in stations]
    by_stage = {s["ledger_stage"]: s["id"] for s in stations if s.get("ledger_stage")}
    jobs = []
    for meta_path in sorted(pdir.glob("*.json")) if pdir.is_dir() else []:
        meta = read_json(meta_path)
        if not meta or "run_id" not in meta:
            continue
        ended = meta.get("ended")
        if ended and now - ended > KEEP_FINISHED_SECONDS:
            continue
        station, pct, note, waiting = ids[0], None, "", False
        events = pdir / f"{meta['run_id']}.events.jsonl"
        for ev in read_entries(events):
            if ev.get("station") in ids:
                station, pct = ev["station"], ev.get("pct")
            elif "pct" in ev:
                pct = ev["pct"]
            if "waiting" in ev:
                waiting = bool(ev["waiting"])
            if ev.get("note"):
                note = ev["note"]
        started = meta.get("started") or now
        verdict, logged = None, []
        for e in entries:
            t = epoch(e.get("ts_utc"))
            if e.get("name") == meta.get("name") and t is not None and t >= started - 1:
                logged.append(e)
        for e in logged:
            stage = next((by_stage[s] for s in tag_values(e, "stage") if s in by_stage), None)
            target = stage or "ledger"
            if ids.index(target) >= ids.index(station):
                station, pct = target, 1.0
            verdicts = tag_values(e, "verdict")
            verdict = verdicts[-1] if verdicts else verdict
        status = meta.get("status", "running")
        if status == "running" and now - (meta.get("heartbeat") or 0) > STALE_SECONDS:
            status = "stalled"
        if verdict == "fail":
            outcome = "rejected"
        elif status == "done":
            outcome, pct = ("passed" if verdict == "pass" else "done"), 1.0
        else:
            outcome = status
        jobs.append(
            {
                "run_id": meta["run_id"],
                "name": meta.get("name", "?"),
                "user": meta.get("user"),
                "station": station,
                "index": ids.index(station),
                "pct": pct,
                "note": note,
                "status": outcome,
                "waiting": waiting and outcome == "running",
                "exit_code": meta.get("exit_code"),
                "logged": len(logged),
                "elapsed": round((ended or now) - started, 1),
                "started": started,
            }
        )
    jobs.sort(key=lambda j: j["started"])
    return {"stations": stations, "jobs": jobs, "progress_dir": str(pdir)}


def fonts_css() -> str:
    import base64

    rules = []
    for family, weight, name in FONT_FILES:
        path = ASSETS / name
        if path.is_file():
            data = base64.b64encode(path.read_bytes()).decode()
            rules.append(
                f'@font-face{{font-family:"{family}";font-weight:{weight};font-display:swap;'
                f'src:url(data:font/woff2;base64,{data}) format("woff2")}}'
            )
    return "\n".join(rules)


def render(state: dict | None) -> str:
    page = PAGE.read_text(encoding="utf-8")
    scene = (
        SCENE.read_text(encoding="utf-8")
        if SCENE.is_file()
        else "<p>lab_scene.svg missing: run build_scene.py</p>"
    )
    embedded = "null" if state is None else json.dumps(state).replace("</", "<\\/")
    return (
        page.replace("/*FONTS*/", fonts_css())
        .replace("<!--SCENE-->", scene)
        .replace("/*__STATE__*/null", embedded)
    )


def reject_counts(entries: list[dict], stations: list[dict], now: float, hours: float) -> dict:
    """verdict:fail per gate from the ledger: inside the rolling window and all time."""
    by_stage = {s["ledger_stage"]: s["id"] for s in stations if s.get("ledger_stage")}
    out = {sid: {"window": 0, "total": 0, "hours": hours} for sid in by_stage.values()}
    for e in entries:
        if "fail" not in tag_values(e, "verdict"):
            continue
        for stage in tag_values(e, "stage"):
            sid = by_stage.get(stage)
            if sid is None:
                continue
            out[sid]["total"] += 1
            t = epoch(e.get("ts_utc"))
            if t is not None and now - t <= hours * 3600:
                out[sid]["window"] += 1
    return out


def serve(path: Path, repo: Path, port: int, pdir: Path, reject_hours: float) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path.split("?")[0] == "/state.json":
                body, kind = (
                    json.dumps(build_state(path, repo, pdir, reject_hours)).encode(),
                    "application/json",
                )
            elif self.path in ("/", "/index.html"):
                body, kind = render(None).encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"pipeline view: http://127.0.0.1:{port}  (ledger {path}; Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="repository root (default: cwd)")
    parser.add_argument(
        "--ledger-dir",
        help="ledger directory (default: $CAPSTONE_LEDGER_DIR or <repo>/experiments)",
    )
    parser.add_argument(
        "--progress-dir",
        help="live-run directory (default: $MSCF_PROGRESS_DIR or ~/.cache/mscf-pipeline/progress)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reject-window-hours",
        type=float,
        default=24.0,
        help="rolling window for the reject counts on the baskets (default 24)",
    )
    parser.add_argument("--snapshot", help="write one self-contained HTML page here and exit")
    parser.add_argument("--json", action="store_true", help="print the state as JSON and exit")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    path = ledger_path(args.ledger_dir, repo)
    pdir = Path(args.progress_dir) if args.progress_dir else progress_dir()
    for label, target in (("ledger", path), ("progress", pdir), ("snapshot", args.snapshot)):
        reason = target and off_limits(Path(target), repo)
        if reason:
            parser.error(f"--{label} path refused: {reason}")
    if args.snapshot and inside_repo(Path(args.snapshot), repo):
        parser.error(
            "--snapshot path refused: a snapshot carries private ledger content and must not "
            "be written inside the repository working tree"
        )
    if args.json:
        print(json.dumps(build_state(path, repo, pdir, args.reject_window_hours), indent=2))
        return 0
    if args.snapshot:
        Path(args.snapshot).write_text(
            render(build_state(path, repo, pdir, args.reject_window_hours)), encoding="utf-8"
        )
        print(f"wrote {args.snapshot}")
        return 0
    serve(path, repo, args.port, pdir, args.reject_window_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
