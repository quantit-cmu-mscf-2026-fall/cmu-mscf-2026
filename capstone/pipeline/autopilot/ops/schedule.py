"""Render the scheduler entries that run `autopilot tick` once a day (`[schedule]`).

    python -m capstone.pipeline.autopilot.ops.schedule --config <autopilot.toml>
        [--python <venv python>] [--checkout <checkout>] [--out <folder>]

fills `deploy/autopilot/task.xml.template` (Windows Task Scheduler, UTF-16 as
`schtasks /Create /XML` expects) and `deploy/autopilot/autopilot.cron.template` with
`[schedule] task_name` and `time`, the interpreter, the checkout and the config, and
writes `task.xml`, `autopilot.cron` and `schedule.json` to `--out` (default
`<run>/autopilot/ops/schedule/`). It registers nothing and installs nothing: a person
imports the task or the cron line. The task runs `deploy/autopilot/run-tick.ps1`, which
logs to `<run>/autopilot/ops/tick.log`. A bad `time` or `task_name` is one line on
stderr, exit 2.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shlex
import sys
from pathlib import Path
from string import Template
from xml.sax.saxutils import escape

from capstone.pipeline.autopilot.config import Key
from capstone.pipeline.autopilot.ops.common import ops_dir, write_json

CHECKOUT = Path(__file__).resolve().parents[4]
TEMPLATES = ("deploy", "autopilot")
TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
TASK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,99}$")

SECTIONS = {
    "schedule": {
        "task_name": Key("str", "capstone-autopilot", "Task Scheduler / cron entry name"),
        "time": Key("str", "18:30", "local time of the daily tick, HH:MM"),
    },
}


class ScheduleError(ValueError):
    pass


def check(section):
    """(hour, minute) of a valid `[schedule]`; ScheduleError otherwise."""
    match = TIME.match(section["time"])
    if match is None:
        raise ScheduleError(f"[schedule] time {section['time']!r} is not HH:MM (24 h)")
    if TASK_NAME.match(section["task_name"]) is None:
        raise ScheduleError(
            f"[schedule] task_name {section['task_name']!r}: letters, digits, space . _ - only"
        )
    return int(match.group(1)), int(match.group(2))


def _ps_quote(value):
    return '"' + str(value).replace('"', '`"') + '"'


def render(config, *, python, checkout=CHECKOUT, out=None):
    """Write the rendered entries; returns the `schedule.json` document."""
    section = config.sections["schedule"]
    hour, minute = check(section)
    checkout = Path(checkout).resolve()
    templates = checkout.joinpath(*TEMPLATES)
    if not (templates / "task.xml.template").is_file():
        templates = CHECKOUT.joinpath(*TEMPLATES)  # an export without deploy/: this module's
    out = Path(out) if out else ops_dir(config.run_root) / "schedule"
    log = ops_dir(config.run_root) / "tick.log"
    config_path = Path(config.path).resolve()
    script = checkout / "deploy" / "autopilot" / "run-tick.ps1"
    flags = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]
    values = [script, "-Checkout", checkout, "-Python", python, "-Config", config_path]
    command = [*flags, *map(str, values), "-Log", str(log)]
    arguments = " ".join(a if a.startswith("-") or a == "Bypass" else _ps_quote(a) for a in command)
    task_name = section["task_name"]
    xml = Template((templates / "task.xml.template").read_text(encoding="utf-8")).substitute(
        TASK_NAME=escape(task_name),
        TIME=f"{hour:02d}:{minute:02d}",
        DESCRIPTION=escape(f"capstone autopilot tick: {config_path}"),
        ARGUMENTS=escape(arguments),
        CHECKOUT=escape(str(checkout)),
    )
    posix = lambda p: shlex.quote(Path(p).as_posix())  # noqa: E731
    cron = Template((templates / "autopilot.cron.template").read_text(encoding="utf-8"))
    cron_text = cron.substitute(
        MINUTE=minute,
        HOUR=hour,
        CHECKOUT=posix(checkout),
        PYTHON=posix(python),
        CONFIG=posix(config_path),
        LOG=posix(log),
        TASK_NAME=task_name,
    )
    out.mkdir(parents=True, exist_ok=True)
    task_xml = out / "task.xml"
    with task_xml.open("w", encoding="utf-16", newline="\r\n") as stream:
        stream.write(xml)
    (out / "autopilot.cron").write_text(cron_text, encoding="utf-8", newline="\n")
    document = {
        "task_name": task_name,
        "time": f"{hour:02d}:{minute:02d}",
        "task_xml": str(task_xml),
        "cron": str(out / "autopilot.cron"),
        "command": ["powershell.exe", *command],
        "log": str(log),
        "config": str(config_path),
    }
    write_json(out / "schedule.json", document)
    return document


def _stages():
    """The registered stages, once the stage registry exists (none before it)."""
    try:
        module = importlib.import_module("capstone.pipeline.autopilot.stages")
    except ModuleNotFoundError:
        return []
    return module.discover()


def main(argv=None):
    from capstone.pipeline.autopilot import config as config_module
    from capstone.pipeline.autopilot import ops

    parser = argparse.ArgumentParser(prog="autopilot-schedule", description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkout", default=str(CHECKOUT))
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        config = config_module.load(args.config, _stages(), extra=ops.discover()[0])
        document = render(config, python=args.python, checkout=args.checkout, out=args.out)
    except (ValueError, OSError, KeyError) as error:
        print("autopilot schedule: " + "; ".join(str(error).splitlines()), file=sys.stderr)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
