"""`python -m capstone.pipeline.autopilot init --dir <folder>` and
`python -m capstone.pipeline.autopilot preflight --config <toml>`.

`init` writes the starter set without overwriting anything (`scaffold.py`); `preflight` is
the read-only checklist before a first run (`preflight.py`; exit 0 only when every required
check is ok, else 2).
"""

from __future__ import annotations

import argparse

from capstone.pipeline.autopilot import preflight, scaffold


def main(argv=None):
    parser = argparse.ArgumentParser(prog="autopilot", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight.add_arguments(commands.add_parser("preflight", help="read-only readiness checks"))
    scaffold.add_arguments(commands.add_parser("init", help="write a starter set"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        return preflight.main(args)
    return scaffold.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
