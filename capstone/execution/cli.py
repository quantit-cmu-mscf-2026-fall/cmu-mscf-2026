"""Command line for the runner: a dry run only, and no connection unless configured.

    python -m capstone.execution --tag deploy-2026.10.07 --as-of 2026-10-06 \\
        --assumed-half-spread-bps 5

**This command builds no broker connection.** It holds no vendor client, reads
no host, port or account from the environment, and has no connect or live
subcommand. The broker and the bar source are supplied by the caller as
factories (`main(broker_factory=..., data_factory=...)`); with neither, `main`
validates its arguments and the deploy tag and then refuses, naming what is
missing, before anything is opened. A vendor-backed pair is its own,
separately reviewed change.

`--paper` is refused before anything is read or built: the only send path is a
frozen, journaled plan (`runner.SEND_PATH`).

Bar sources carry no spreads, so the half spread is an assumption the command
must state (`--assumed-half-spread-bps`, no default).

Before any factory is called, the command refuses an `--as-of` after the
discovery period (there is no flag that admits a later book), then reads the
tag and refuses a template universe and any strategy needing a field a bar
source does not have (shares, cap, mkt_return, fundamentals) or SIC codes
(sector neutrality).

Money and the kill switch come from the environment, never from files or flags
that end up in shell history:

- `CAPSTONE_PAPER_CAPITAL`: dollars the book is sized to (or `--capital`)
- `CAPSTONE_KILL_SWITCH`: kill switch file (default ~/.capstone/KILL_SWITCH)
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from capstone.execution.broker import Broker
from capstone.execution.data import MarketDataSource, check_servable
from capstone.execution.deploy import deployed_entries, deployed_universe, resolve_tag
from capstone.execution.runner import SEND_PATH_REMOVED, RiskLimits, run_once
from capstone.execution.universe import Universe
from capstone.factors.evaluation import DISCOVERY_END
from capstone.strategy.contract import CURRENT_ROUTE_MISSING

NO_CONNECTION = (
    "no broker connection is configured: this command never builds one by default. "
    "Pass a broker_factory and a data_factory to main() from code that owns that "
    "connection; the checks above ran, and nothing was opened."
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m capstone.execution", description=__doc__)
    parser.add_argument("--tag", required=True, help="deploy tag, deploy-YYYY.MM.DD")
    parser.add_argument("--as-of", required=True, type=pd.Timestamp, help="YYYY-MM-DD")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="removed: refused, naming the single send path",
    )
    parser.add_argument(
        "--assumed-half-spread-bps",
        type=float,
        default=None,
        help="required: the half spread assumed for every stock (bars carry none)",
    )
    parser.add_argument("--history", default="2 Y", help="history the data source is asked for")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-gross", type=float, default=RiskLimits.max_gross)
    parser.add_argument("--max-weight", type=float, default=RiskLimits.max_weight)
    parser.add_argument("--max-order-notional", type=float, default=RiskLimits.max_order_notional)
    parser.add_argument("--min-notional", type=float, default=100.0)
    parser.add_argument(
        "--incubation-cap",
        type=float,
        default=RiskLimits.incubation_cap,
        help="fraction of capital for all incubation-only strategies together",
    )
    return parser.parse_args(argv)


def main(
    argv=None,
    *,
    broker_factory: Callable[..., Broker] | None = None,
    data_factory: Callable[[Broker, Universe, argparse.Namespace], MarketDataSource] | None = None,
) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    env = os.environ
    if args.paper:
        raise SystemExit(SEND_PATH_REMOVED)
    if args.assumed_half_spread_bps is None:
        raise SystemExit(
            "set --assumed-half-spread-bps: bars carry no spreads, so the cost is an "
            "assumption the run must state"
        )
    capital = args.capital
    if capital is None:
        if "CAPSTONE_PAPER_CAPITAL" not in env:
            raise SystemExit("set --capital or CAPSTONE_PAPER_CAPITAL")
        capital = float(env["CAPSTONE_PAPER_CAPITAL"])
    kill = Path(env.get("CAPSTONE_KILL_SWITCH", Path.home() / ".capstone" / "KILL_SWITCH"))
    limits = RiskLimits(
        args.max_gross, args.max_weight, args.max_order_notional, kill, args.incubation_cap
    )
    if args.as_of > pd.Timestamp(DISCOVERY_END):
        raise SystemExit(CURRENT_ROUTE_MISSING)
    # Everything that can be refused from the tag alone is refused before a factory runs.
    commit = resolve_tag(args.repo, args.tag)
    universe = deployed_universe(args.repo, commit)
    for _, spec in deployed_entries(args.repo, commit):
        check_servable(spec.fields, spec.trading.neutral)
    if broker_factory is None or data_factory is None:
        raise SystemExit(NO_CONNECTION)
    broker = broker_factory(env, readonly=True)
    try:
        result = run_once(
            args.repo,
            args.tag,
            args.as_of,
            broker,
            data_factory(broker, universe, args),
            capital=capital,
            limits=limits,
            min_notional=args.min_notional,
            out_dir=args.out,
        )
    finally:
        disconnect = getattr(broker, "disconnect", None)
        if disconnect is not None:
            disconnect()
    print(f"dry-run: {len(result.orders)} orders, digest {result.digest}")
    print(f"plan {result.plan_path}")
    return 0
