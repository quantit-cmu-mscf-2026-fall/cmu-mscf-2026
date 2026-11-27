"""The first real search: `python -m capstone.pipeline.real_run --workdir <dir>`.

Runs the search the config names (default config/e2e_crsp.toml) with the real model:
papers, deepen and evaluation on the search period, every trial on the ledger, then the
gate config registered (`search.run_search`). It stops there: it has no way to open the
holdout, and it does not score discovery (gate stages 1-3), which is a separate step.

Before anything is loaded or sent it refuses:

- a placebo config unless `CAPSTONE_LEDGER_DIR` names the placebo ledger (checked
  first, so a placebo search never reaches the main trial count; `run_search` then
  refuses the placebo itself);
- no `ANTHROPIC_API_KEY` in the environment (the key is never read from a file or
  printed);
- no `CAPSTONE_LEDGER_DIR`: the ledger the trials go to is a deliberate choice;
- a config whose deepen doesn't stop on the budget (`[deepen] on_budget = "stop"`),
  so a used-up budget still ends in a registered search;
- a workdir that already holds a search (`run_summary.run_real_search`).

It writes `usage.jsonl` (written even if the run fails) and `run_summary.json` next to
`run_search`'s manifest (`capstone.pipeline.run_summary`). `--summarize-only` rebuilds
that summary from a finished search's files: no key, no model, no data.

The client is `anthropic.Anthropic(max_retries=0)` (the call budget refuses hidden SDK
retries) wrapped in `PromptCaching`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from capstone.pipeline import placebo
from capstone.pipeline.config import REPO, load_config
from capstone.pipeline.manifest import MANIFEST_FILE
from capstone.pipeline.run_summary import RUN_SUMMARY_FILE, run_real_search, write_summary
from capstone.pipeline.search import refuse_existing
from capstone.pipeline.usage import PromptCaching

__all__ = ["main"]


def _client():
    import anthropic

    # reads ANTHROPIC_API_KEY from the environment; no SDK retries; system text cacheable
    return PromptCaching(anthropic.Anthropic(max_retries=0))


def _refuse(message: str) -> int:
    print(f"refused: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None, *, client_factory=_client) -> int:
    parser = argparse.ArgumentParser(prog="python -m capstone.pipeline.real_run")
    parser.add_argument("--config", default=str(REPO / "config" / "e2e_crsp.toml"))
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="rebuild run_summary.json from a finished search's files; no model, no data",
    )
    args = parser.parse_args(argv)
    workdir = Path(args.workdir)
    if args.summarize_only:
        if not (workdir / MANIFEST_FILE).exists():
            return _refuse(f"{workdir} holds no search ({MANIFEST_FILE})")
        write_summary(workdir)
        print(f"summary      {workdir / RUN_SUMMARY_FILE}")
        return 0
    config = load_config(args.config)
    if config.placebo is not None:
        try:
            placebo.check_ledger()
        except placebo.PlaceboRefused as exc:
            return _refuse(str(exc))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _refuse("set ANTHROPIC_API_KEY in this shell first")
    if not os.environ.get("CAPSTONE_LEDGER_DIR"):
        return _refuse("set CAPSTONE_LEDGER_DIR to the ledger the trials go to")
    if config.deepen_on_budget != "stop":
        return _refuse(f'{args.config}: set [deepen] on_budget = "stop"')
    try:
        refuse_existing(workdir)
        summary = run_real_search(config, client=client_factory(), workdir=workdir)
    except (FileExistsError, placebo.PlaceboRefused) as exc:
        return _refuse(str(exc))
    usage = summary["usage"]
    print(f"model calls  {summary['model_calls']} of {config.max_model_calls}")
    print(f"est. cost    ${usage['usd']:.2f} (list prices; the console has the bill)")
    if usage["unpriced"]:
        print(f"unpriced     {', '.join(usage['unpriced'])}")
    print(f"trials       {summary['trial_count']} ({summary['run_tag']})")
    print(f"deepen       {summary['deepen']['rounds']} rounds {summary['deepen']['stopped']}")
    print(f"variants     {summary['n_variants']} in {len(summary['families'])} families")
    print(f"registered   {summary['gate_config_hash'][:16]}; holdout not opened")
    print(f"stage 3      {summary['stage3']}")
    print(f"summary      {workdir / RUN_SUMMARY_FILE}")
    print("discovery (gate stages 1-3) is a separate step; this run did not score it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
