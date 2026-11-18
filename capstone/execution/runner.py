"""The runner: one dry run of the deployed strategies; it never sends.

It has no send path. `mode="paper"` is refused before anything is read,
naming the single send path, `SEND_PATH` (a frozen, journaled, reconciled
instruction plan). It also never builds a broker: the caller passes one in
(a `FakeBroker` in tests), so a default configuration cannot open a connection.

`run_once(repo_root, tag, as_of, broker, data, capital=...)`:

1. **Stops** if the kill switch file exists, before the broker is asked
   anything.
2. **Reads the deploy at a tag**: `tag` must be an existing `deploy-YYYY.MM.DD`
   tag (never a branch tip). The registry, the enabled entries' spec files
   and the universe file are read from the tag's commit with `git show`, not
   from the working tree. Disabled entries are skipped; an enabled entry with
   no spec file, a spec that is not its id, or a gate report differing from
   the registry's is refused, as is a registry entry with no gate report.
   A tag with nothing enabled is refused (positions are left as they are),
   and so is the template universe file.
3. **Builds the book**: each spec's `target_weights` on the data source's
   panel through `as_of` (a panel with later dates is refused). An `as_of`
   after `DISCOVERY_END` is refused at entry, dry run included, before the
   broker or data source is asked anything: there is no flag that admits a
   later panel. The books are averaged across strategies (each gets an equal
   share of `capital`). An enabled entry with the role "incubation-only" (a
   family the gate labelled incubate) is **scaled, not refused**: all
   incubation-only entries together get at most `RiskLimits.incubation_cap` of
   capital, each min(1 / n, cap / n_incubation_only); the rest of their equal
   share stays in cash (it is not handed to the other strategies). The plan
   then shows each strategy's `role` and `capital_share`, and the cap in its
   limits; with no incubation-only entry enabled the plan carries neither.
4. **Checks risk limits** (`RiskLimits`): the book's gross and largest
   single-name weight, then each order's notional. Any breach stops the run.
5. **Orders**: current positions (shares, by symbol) become weights at the
   broker's prices; `strategy.orders` gives the weight trades and
   `to_share_orders` the whole-share orders. Every name, target or held, must
   be in the tag's universe file; the broker sees the venue's spelling.
6. **Logs**: one ledger entry (name "paper-run", tagged "dry-run") carrying
   the plan's digest, then the plan JSON and a log file in `out_dir`. It
   never calls `broker.place`.

The digest is the SHA-256 of the plan's canonical JSON, which holds no clock
time, mode or account: the same inputs give the same digest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import pandas as pd

from capstone.execution.broker import (
    Broker,
    ShareOrder,
    mask_account,
    to_share_orders,
    valid_price,
)
from capstone.execution.data import MarketDataSource
from capstone.execution.deploy import Refused, deployed_entries, deployed_universe, resolve_tag
from capstone.factors.evaluation import DISCOVERY_END, HoldoutError
from capstone.runlog import log_run
from capstone.strategy.contract import CURRENT_ROUTE_MISSING
from capstone.strategy.orders import orders as weight_orders
from capstone.strategy.portfolio import combined_book
from capstone.strategy.registry import INCUBATION_ONLY

LEDGER_NAME = "paper-run"
Mode = Literal["dry-run"]
MODES = get_args(Mode)
SEND_PATH = (
    "python -m capstone.execution.instructions <plan> --context <file> --journal <file> --paper"
)
SEND_PATH_REMOVED = (
    "the runner has no paper send path; it only dry-runs. Freeze a plan and send it with "
    "the single send path: " + SEND_PATH
)
EPS = 1e-9

log = logging.getLogger(__name__)


class RiskLimitError(Refused):
    """A risk limit (or the kill switch) stopped the run."""


@dataclass(frozen=True)
class RiskLimits:
    max_gross: float = 1.0  # sum of |weight| of the combined book
    max_weight: float = 0.10  # largest |weight| of one name
    max_order_notional: float = 50_000.0  # dollars, per order
    kill_switch: Path | None = None  # if this file exists, nothing runs
    incubation_cap: float = 0.10  # fraction of capital for all incubation-only entries

    def __post_init__(self) -> None:
        cap = self.incubation_cap
        if not (isinstance(cap, int | float) and 0.0 < cap <= 1.0):
            raise ValueError(f"incubation_cap must be a fraction in (0, 1], got {cap!r}")


@dataclass(frozen=True, eq=False)
class RunResult:
    digest: str
    plan: dict
    plan_path: Path
    log_path: Path
    orders: tuple[ShareOrder, ...]


def capital_shares(roles, cap: float) -> list[float]:
    """Each strategy's share of capital: 1 / n, and for incubation-only entries
    min(1 / n, cap / their count), so together they never exceed `cap`."""
    n, n_inc = len(roles), sum(r == INCUBATION_ONLY for r in roles)
    capped = min(1 / n, cap / n_inc) if n_inc else 0.0
    return [capped if r == INCUBATION_ONLY else 1 / n for r in roles]


def check_book(target: pd.Series, limits: RiskLimits) -> None:
    gross = float(target.abs().sum())
    if gross > limits.max_gross + EPS:
        raise RiskLimitError(f"book gross {gross:.6f} exceeds max_gross {limits.max_gross}")
    largest = float(target.abs().max()) if len(target) else 0.0
    if largest > limits.max_weight + EPS:
        raise RiskLimitError(
            f"single-name weight {largest:.6f} exceeds max_weight {limits.max_weight}"
        )


def _weights(series: pd.Series) -> dict[str, float]:
    return {str(int(p)): float(w) for p, w in series.items()}


def run_once(
    repo_root,
    tag: str,
    as_of,
    broker: Broker,
    data: MarketDataSource,
    *,
    capital: float,
    mode: Mode = "dry-run",
    limits: RiskLimits = RiskLimits(),  # noqa: B008 (frozen)
    min_trade: float = 0.0,
    min_notional: float = 100.0,
    out_dir=None,
) -> RunResult:
    """One run of the deploy at `tag` for the book after `as_of`'s close. See
    the module docstring."""
    if mode == "paper":
        raise Refused(SEND_PATH_REMOVED)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if pd.Timestamp(as_of) > pd.Timestamp(DISCOVERY_END):
        raise HoldoutError(CURRENT_ROUTE_MISSING)
    if not (math.isfinite(capital) and capital > 0):
        raise ValueError(f"capital must be a positive number, got {capital!r}")
    if limits.kill_switch is not None and Path(limits.kill_switch).exists():
        raise RiskLimitError(f"kill switch {limits.kill_switch} is set; nothing runs")
    repo = Path(repo_root)
    as_of = pd.Timestamp(as_of)
    if as_of != as_of.normalize():
        raise ValueError(f"as_of {as_of} must be a date")
    commit = resolve_tag(repo, tag)
    entries = deployed_entries(repo, commit)
    specs = tuple(spec for _, spec in entries)
    roles = [entry.role for entry, _ in entries]
    capped = INCUBATION_ONLY in roles  # otherwise the plan carries no role or share
    shares = capital_shares(roles, limits.incubation_cap) if capped else None
    universe = deployed_universe(repo, commit)

    target, info = combined_book(specs, data, as_of, shares)
    check_book(target, limits)
    symbols = {int(p): universe.venue_symbol(int(p)) for p in target.index}
    held = {s: int(n) for s, n in sorted(broker.positions().items()) if n != 0}
    for s in held:
        symbols[universe.permno(s)] = s
    try:
        quotes = broker.prices(sorted(set(symbols.values())))
        quotes = {s: valid_price(quotes, s) for s in sorted(set(symbols.values()))}
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    current = pd.Series(
        {universe.permno(s): n * quotes[s] / capital for s, n in held.items()}, dtype="float64"
    )
    trades = weight_orders(target, current, min_trade=min_trade).trades
    planned = to_share_orders(trades, symbols, quotes, capital, min_notional=min_notional)
    for o in planned:
        if o.notional > limits.max_order_notional + EPS:
            raise RiskLimitError(
                f"order {o.side} {abs(o.shares)} {o.symbol}: notional {o.notional:,.2f} "
                f"exceeds max_order_notional {limits.max_order_notional:,.2f}"
            )

    plan = {
        "tag": tag,
        "commit": commit,
        "as_of": str(as_of.date()),
        "capital": float(capital),
        "universe_version": universe.version,
        "data_versions": info["data_versions"],
        "limits": {
            "max_gross": limits.max_gross,
            "max_weight": limits.max_weight,
            "max_order_notional": limits.max_order_notional,
            "min_trade": min_trade,
            "min_notional": min_notional,
        },
        "strategies": [
            {
                "id": s.id,
                "expression": s.expression,
                "gate_report": s.gate_report,
                "data_version": s.data_version,
                "book": _weights(info["books"][s.id]),
            }
            for s in specs
        ],
        "target": _weights(target),
        "positions": held,
        "prices": quotes,
        "orders": [dataclasses.asdict(o) | {"side": o.side} for o in planned],
    }
    if capped:
        plan["limits"]["incubation_cap"] = limits.incubation_cap
        for row, role, share in zip(plan["strategies"], roles, shares, strict=True):
            row.update(role=role, capital_share=share)
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    account = mask_account(broker.account_id)

    log_run(
        LEDGER_NAME,
        params={
            "tag": tag,
            "commit": commit,
            "as_of": plan["as_of"],
            "mode": mode,
            "account": account,
            "capital": float(capital),
            "strategy_ids": [s.id for s in specs],
            "spec_data_versions": sorted({s.data_version for s in specs}),
            "data_version": ",".join(info["data_versions"]),
            "universe_version": universe.version,
            "plan_digest": digest,
        },
        metrics={
            "n_orders": len(planned),
            "gross": float(target.abs().sum()),
            "order_notional": float(sum(o.notional for o in planned)),
        },
        tags=[mode, "execution"],
        notes="paper runner plan; an execution record, not a research trial",
    )

    out = Path(out_dir) if out_dir is not None else repo / "experiments" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{plan['as_of']}_{mode}"
    plan_path, log_path = out / f"{stem}.plan.json", out / f"{stem}.log"
    document = {"digest": digest, "mode": mode, "account": account, "plan": plan}
    plan_path.write_bytes((json.dumps(document, sort_keys=True, indent=2) + "\n").encode())
    lines = [
        f"{mode} run of {tag} ({commit[:12]}) for {plan['as_of']}, account {account}",
        f"strategies {', '.join(s.id for s in specs)}; universe {universe.version}",
        f"plan digest {digest}",
        *(f"order {o.side} {abs(o.shares)} {o.symbol} @ {o.price:.4f}" for o in planned),
    ]
    for line in lines:
        log.info(line)
    log_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return RunResult(digest, plan, plan_path, log_path, planned)
