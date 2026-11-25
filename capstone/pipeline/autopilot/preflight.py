"""`python -m capstone.pipeline.autopilot preflight --config <autopilot.toml>`: the read-only
checklist before a first run.

It changes nothing a run reads, never sends an order and never calls a model. Every
check is one row: a status (`ok`, `missing`, `wrong` or `skipped`), a typed code
(`CODES`), what it found, and one line on how to fix it. A row is required or a warning;
the exit code is 0 only when every required row is `ok`, else 2. The rows go to
`<run root>/ops/preflight.json`, or, before the run root exists, to
`<config folder>/preflight.json`: preflight never creates the run root.

Offline checks (the only ones in this module): the config through the same loader as
`tick` (paths resolved against the config's folder); every `TODO(person)` marker still in
the config and the files it names (`init` writes them); every configured path; the
registry checkout (a separate git clone holding `deploy/strategies.yml`, never the code
checkout); the input archive's folder; the sources inbox (when a `[sources]` section is
configured); `[spend]` against `real_model_calls`; and, only with real model calls on,
whether the model API key variable is SET or UNSET (its value is never read into a row).

Checks that need a broker, a data provider or a model are not part of this module; a
later change adds them behind their own flags and imports them lazily, so the offline
core never reaches anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from capstone.pipeline import model_key
from capstone.pipeline.autopilot import config as config_module

STATUSES = ("ok", "missing", "wrong", "skipped")
TODO = "TODO(person):"
SCHEMA = "autopilot-preflight-v1"
REPORT = "preflight.json"
REGISTRY_FILE = Path("deploy") / "strategies.yml"
CREATED = {("autopilot", "run_root")}
OWN_ROWS = {("autopilot", "checkout"), ("autopilot", "input_archive")}

CODES = {
    "ok": "nothing to do",
    "config_unreadable": "write the file from `autopilot init`, or fix the TOML syntax",
    "config_invalid": "fix each listed key; the same loader runs on every tick",
    "todo_open": "choose the value, then delete the TODO(person) marker",
    "path_missing": "create it, or fix the path (relative paths resolve against the config)",
    "path_parent_missing": "create the parent folder; the run creates the folder itself",
    "registry_not_set": "set [autopilot] checkout to the registry clone",
    "registry_is_code_checkout": "point [autopilot] checkout at a separate registry clone",
    "registry_not_git": "clone the registry repository there (git clone)",
    "registry_file_missing": "the registry clone needs deploy/strategies.yml",
    "archive_not_set": "set [autopilot] input_archive (obtained out of band)",
    "inbox_missing": "create the [sources] inbox folder and drop the sources in",
    "spend_inconsistent": "fix [spend] as the detail says",
    "api_key_unset": "set the model API key in the run's environment (real model calls only)",
    "not_applicable": "nothing to do",
}


@dataclass
class Row:
    check: str
    status: str
    code: str
    detail: str = ""
    required: bool = True

    def __post_init__(self):
        if self.status not in STATUSES or self.code not in CODES:
            raise ValueError(f"preflight row {self.check}: {self.status} {self.code}")

    @property
    def fix(self):
        return CODES[self.code]

    @property
    def failed(self):
        return self.required and self.status != "ok"

    def line(self):
        level = "" if self.required or self.status in ("ok", "skipped") else " (warning)"
        text = f"{self.status:<8}{self.code:<26}{self.check}: {self.detail}{level}"
        return text if self.status in ("ok", "skipped") else f"{text} | fix: {self.fix}"

    def document(self):
        return asdict(self) | {"fix": self.fix}


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, check, status, code, detail="", *, required=True):
        if status == "skipped":
            required = False
        self.rows.append(Row(check, status, code, str(detail), required))

    def ok(self, check, detail=""):
        self.add(check, "ok", "ok", detail)

    def skip(self, check, detail):
        self.add(check, "skipped", "not_applicable", detail)


def code_checkout():
    """The checkout this package runs from: the registry clone must not be inside it."""
    return Path(__file__).resolve().parents[3]


def _config_folder(path):
    return Path(path).resolve().parent


def _todos(checks, files):
    found = 0
    for path in files:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, 1):
            if TODO in line:
                found += 1
                text = line.strip()[:160]
                checks.add("todo", "missing", "todo_open", f"{Path(path).name}:{number}: {text}")
    if not found:
        checks.ok("todo", f"no {TODO} marker in {len(files)} file(s)")


def _named_files(config):
    """The config and every file a `template` key of it names."""
    files = [config.path]
    for values in config.sections.values():
        value = values.get("template")
        if value:
            files.append(Path(value))
    return files


def _paths(checks, config, schemas):
    for section, schema in schemas.items():
        for key, spec in schema.items():
            value = (config.sections.get(section) or {}).get(key)
            if spec.kind != "path" or value is None or (section, key) in OWN_ROWS:
                continue
            path, name = Path(value), f"path [{section}] {key}"
            if path.exists():
                checks.ok(name, str(path))
            elif (section, key) in CREATED:
                if path.parent.is_dir():
                    checks.ok(name, f"{path} (created by the run)")
                else:
                    checks.add(name, "missing", "path_parent_missing", str(path.parent))
            else:
                checks.add(name, "missing", "path_missing", str(path))


def _registry(checks, config):
    root = config.core.get("checkout")
    if root is None:
        checks.add("registry", "missing", "registry_not_set", "[autopilot] checkout is not set")
        return
    root, code = Path(root).resolve(), code_checkout()
    if root == code or code in root.parents:
        checks.add("registry", "wrong", "registry_is_code_checkout", f"{root} is in {code}")
    elif not (root / ".git").exists():
        checks.add("registry", "wrong", "registry_not_git", f"{root} is not a git clone")
    elif not (root / REGISTRY_FILE).is_file():
        checks.add("registry", "missing", "registry_file_missing", str(root))
    else:
        checks.ok("registry", f"{root} holds {REGISTRY_FILE.as_posix()}")


def _archive(checks, config):
    archive = config.core.get("input_archive")
    if archive is None:
        checks.add("input_archive", "missing", "archive_not_set", "[autopilot] input_archive")
    elif Path(archive).is_dir():
        checks.ok("input_archive", archive)
    else:
        checks.add("input_archive", "missing", "path_missing", archive)


def _inbox(checks, config):
    section = config.sections.get("sources")
    if section is None:
        checks.skip("inbox", "no [sources] section is configured")
        return
    inbox = section.get("inbox")
    if inbox and Path(inbox).is_dir():
        checks.ok("inbox", inbox)
    else:
        checks.add("inbox", "missing", "inbox_missing", str(inbox))


def _spend(checks, config, env):
    section, real = config.sections["spend"], config.core["real_model_calls"]
    problems = []
    if section["run_ceiling"] > section["global_ceiling"]:
        problems.append("run_ceiling is above global_ceiling")
    if real:
        if section["run_ceiling"] <= 0:
            problems.append("real_model_calls = true with run_ceiling 0: nothing may call")
        if section["runs_dir"] is None or not Path(section["runs_dir"]).is_dir():
            problems.append("runs_dir must be an existing folder holding every run root")
    if problems:
        checks.add("spend", "wrong", "spend_inconsistent", "; ".join(problems))
    else:
        mode = "real model calls" if real else "no model calls (real_model_calls = false)"
        checks.ok("spend", f"{mode}; run_ceiling {section['run_ceiling']}")
    if not real:
        checks.skip("api_key", "model calls are off (real_model_calls = false)")
    elif model_key.key_state(env) == "SET":
        checks.ok("api_key", "the key variable is SET")
    else:
        checks.add("api_key", "missing", "api_key_unset", "the key variable is UNSET")


def report_path(config_path, config=None):
    """`<run root>/ops/preflight.json` once the run root exists, else beside the config
    (preflight never creates the run root)."""
    if config is not None and config.run_root.is_dir():
        return config.run_root / "ops" / REPORT
    return _config_folder(config_path) / REPORT


def run(config_path, *, stages=(), extra=None, env=None):
    """(rows, config or None) of every offline check; nothing is written. `stages` and
    `extra` are the stage list and extra config sections the config is loaded against;
    `env` is the environment the key check reads (default: the process's)."""
    env = os.environ if env is None else env
    checks = Checks()
    try:
        config = config_module.load(config_path, list(stages), extra=extra)
    except config_module.ConfigError as error:
        code = "config_invalid" if Path(config_path).is_file() else "config_unreadable"
        status = "wrong" if code == "config_invalid" else "missing"
        checks.add("config", status, code, "; ".join(str(error).splitlines()))
        if Path(config_path).is_file():
            _todos(checks, [Path(config_path)])
        return checks.rows, None
    checks.ok("config", f"{config.path} validates (the loader `tick` uses)")
    _todos(checks, _named_files(config))
    _paths(checks, config, config_module.schemas(list(stages), extra))
    _registry(checks, config)
    _archive(checks, config)
    _inbox(checks, config)
    _spend(checks, config, env)
    return checks.rows, config


def document(rows):
    failed = [row.code for row in rows if row.failed]
    return {
        "schema": SCHEMA,
        "ready": not failed,
        "failed": failed,
        "rows": [row.document() for row in rows],
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(args, stages=(), extra=None):
    """The `preflight` subcommand (`__main__`): print the rows, write the report."""
    rows, config = run(args.config, stages=stages, extra=extra)
    for row in rows:
        print(row.line())
    report = document(rows)
    target = report_path(args.config, config)
    if config is not None or target.parent.is_dir():
        _write_json(target, report)
    failed = report["failed"]
    summary = f"preflight: {len(rows)} checks, {len(failed)} required not ok; report {target}"
    print(summary, file=sys.stderr if failed else sys.stdout)
    return 2 if failed else 0


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--config", required=True)
