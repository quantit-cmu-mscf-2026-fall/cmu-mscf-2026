"""`python -m capstone.pipeline.autopilot tick|status|clear|preflight --config <toml>` and
`python -m capstone.pipeline.autopilot init --dir <folder>`.

Exit 0: done or progress. Exit 2: blocked or stopped, with one line on stderr.

`tick` runs every ready stage once, with the ops hooks (alerts, warnings, status) and the
spend guard (`spend.guard`, whose ceilings come from the config's `[spend]` section; the
defaults allow no model calls). It never pushes a branch or opens a pull request: the
autopilot is research-only when unattended. `status` lists every stage without running
or locking anything; `clear` records a person's decision on a STOP. `init` writes the
starter set without overwriting anything (`scaffold.py`); `preflight` is the read-only
checklist before a first run (`preflight.py`; exit 0 only when every required check is ok).
"""

from __future__ import annotations

import argparse
import json
import sys

from capstone.pipeline.autopilot import config as config_module
from capstone.pipeline.autopilot import ops, preflight, runner, scaffold, spend, stages


def tick(config, found, hooks, services=None):
    """One tick with the caller's spend guard."""
    return runner.tick(config, found, hooks=hooks, services=services, spend_guard=spend.guard)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="autopilot", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parsers = {name: commands.add_parser(name) for name in ("tick", "status", "clear")}
    for command in parsers.values():
        command.add_argument("--config", required=True)
    for flag in ("--stage", "--reviewer", "--reason"):
        parsers["clear"].add_argument(flag, required=True)
    preflight.add_arguments(commands.add_parser("preflight", help="read-only readiness checks"))
    scaffold.add_arguments(commands.add_parser("init", help="write a starter set"))
    args = parser.parse_args(argv)
    found = stages.discover()
    sections, hooks = ops.discover()
    if args.command == "preflight":  # read-only, its own loading and exit code
        return preflight.main(args, found, sections)
    if args.command == "init":
        return scaffold.main(args)
    try:
        config = config_module.load(args.config, found, extra=sections)
        if args.command == "tick":
            report = tick(config, found, hooks)
            print(report.line(), file=sys.stderr if report.exit_code else sys.stdout)
            return report.exit_code
        if args.command == "clear":
            runner.clear(config, found, args.stage, reviewer=args.reviewer, reason=args.reason)
            print(f"autopilot: cleared {args.stage}")
            return 0
        table = runner.status(config, found)
    except (ValueError, RuntimeError) as error:  # ConfigError, JournalError, lock, bad clear
        print("autopilot: " + "; ".join(str(error).splitlines()), file=sys.stderr)
        return 2
    print(json.dumps(table, indent=2))
    return 2 if any(row["state"] in ("blocked", "stopped") for row in table["stages"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
