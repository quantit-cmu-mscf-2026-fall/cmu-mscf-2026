"""Run any experiment command and show it on the pipeline view's production line.

    python pipeline_run.py --name momentum-15-1 -- python experiments/momentum.py --lookback 315

The wrapper records start, heartbeat and exit for the command; the view moves the run to a
validation stage when a ledger entry with the same name appears. A script that also uses
`mscf_progress.track(...)` reports its own stations into the same run. The command's output
passes through unchanged and its exit code is returned.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mscf_progress import Run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--name", required=True, help="experiment name, as passed to runlog.log_run"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command to run")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")

    run = Run(args.name).start()
    run.stage("hypothesis", note=Path(command[-1]).name if command else "")
    env = dict(os.environ, MSCF_RUN_ID=run.run_id, MSCF_PROGRESS_DIR=str(run.dir))
    here = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH"), here) if p)
    try:
        code = subprocess.call(command, env=env)
    except KeyboardInterrupt:
        code = 130
    except OSError as exc:
        print(f"pipeline_run: {exc}", file=sys.stderr)
        code = 127
    run.finish("done" if code == 0 else "failed", code)
    return code


if __name__ == "__main__":
    sys.exit(main())
