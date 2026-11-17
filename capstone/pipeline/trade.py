"""Steps 5-6 of the end-to-end driver: accepted families to strategies, trades and monitors.

Per accepted family (`accept`): its best variant becomes a `StrategySpec`
anchored at the search panel's first date and naming the gate report, a
registry entry under `<workdir>/deploy/` (enabled false), and the book and
orders (from an empty book) at the last search-period date; then an
incubation monitor at the gate's expected mean on a SYNTHETIC live series
(`incubate.live_continuation`, block-bootstrapped search returns; tagged "synthetic" in
the ledger), logged to the ledger. Realized, journal-attributed monitoring is
`lifecycle monitor` (`pipeline.monitoring`).
An incubate family goes the same way, after the accepted ones, with its
registry entry's role "incubation-only" (still disabled) and its monitor at its
own expected mean and scale: incubation governs it from there. It also runs the
confirming e-process (`incubation.confirm`, `confirms`) to `confirm_horizon`,
and its ledger entry records promote / retire / continue and why; an accepted
family does so only with `[incubation] confirm_accepted = true`.

The gate does not yet label any family incubate: `GateResult` has no
`incubate` field here (or, once it has one, it is empty unless the gate was
asked to promote to incubate), so the incubate path is inert by default and
`accept` iterates only the accepted families. Registering writes only under
`workdir`; nothing is deployed or enabled.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capstone.factors.evaluation import Panel, spec_from_summary
from capstone.incubate import Incubation, incubate, live_continuation
from capstone.pipeline.config import E2EConfig, IncubationSettings
from capstone.runlog import log_run
from capstone.scorecard import GateResult
from capstone.strategy import registry
from capstone.strategy.contract import StrategySpec, target_weights
from capstone.strategy.orders import Orders, orders
from capstone.strategy.registry import INCUBATION_ONLY

INCUBATION_NAME = "e2e-incubation"
EMPTY_REGISTRY = "# Deploy registry (capstone.strategy.registry), written by the e2e driver.\n"


@dataclass(frozen=True, eq=False)
class StrategyOutcome:
    """An accepted (or incubate) family as a registered strategy and its book on `as_of`."""

    family: str
    spec: StrategySpec
    entry: registry.RegistryEntry
    as_of: pd.Timestamp
    weights: pd.Series
    orders: Orders
    digest: str  # `book_digest(weights, orders)`
    role: registry.Role = "deploy"  # "incubation-only" for an incubate family


# Weights and trades enter the digest to this many decimals: platforms' NumPy
# builds differ in the last bits (~1e-17 here), far below 1e-12.
DIGEST_DECIMALS = 12


def _fixed(x: float) -> str:
    """`x` to DIGEST_DECIMALS decimals, a negative zero written as zero."""
    return f"{round(float(x), DIGEST_DECIMALS) + 0.0:.{DIGEST_DECIMALS}f}"


def book_digest(weights: pd.Series, book: Orders) -> str:
    """SHA-256 of the book and the trades, each to DIGEST_DECIMALS decimals."""
    lines = [f"w {p} {_fixed(w)}" for p, w in weights.items()]
    lines += [f"t {p} {_fixed(t)}" for p, t in book.trades.items()]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def confirms(settings: IncubationSettings, role: str) -> bool:
    """Whether a family's incubation runs the confirming e-process: always for an
    incubation-only one, for an accepted one only with `confirm_accepted`."""
    return role == INCUBATION_ONLY or settings.confirm_accepted


def prepare_registry(workdir: Path) -> None:
    registry_file = workdir / registry.REGISTRY_FILE
    if not registry_file.exists():
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text(EMPTY_REGISTRY + "strategies: []\n", encoding="utf-8")


def _strategy(config, workdir, family, row, panel, report, data_version, role) -> StrategyOutcome:
    dates = panel.universe.index
    spec = StrategySpec(
        expression=row["expression"],
        trading=spec_from_summary(row),
        data_version=data_version,
        anchor=dates[0].strftime("%Y-%m-%d"),
        gross=config.gross,
        hypothesis_id=str(family),
        gate_report=report.relative_to(workdir).as_posix(),
    )
    entry = registry.register(spec, workdir, role=role)
    weights = target_weights(spec, panel, dates[-1])
    empty = pd.Series(dtype="float64", index=pd.Index([], dtype=weights.index.dtype))
    book = orders(weights, empty, min_trade=config.min_trade)
    return StrategyOutcome(
        family, spec, entry, dates[-1], weights, book, book_digest(weights, book), role
    )


def accept(
    config: E2EConfig,
    workdir: Path,
    result: GateResult,
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    panel: Panel,
    report: Path,
    data_version: str,
) -> tuple[tuple[StrategyOutcome, ...], tuple[Incubation, ...]]:
    """Steps 5-6 per accepted, then incubate, family: registry entry, book,
    orders, monitor (logged)."""
    strategies, incubations = [], []
    # `incubate` is absent from a gate result that has no such outcome, and empty on one that
    # was not asked for it: either way no family takes the incubation-only role.
    incubating = getattr(result, "incubate", ())
    roles = {f: "deploy" for f in result.accepted} | {f: INCUBATION_ONLY for f in incubating}
    for family, role in roles.items():
        card = result.scorecard.loc[family]
        best = summary.loc[card["best"]]
        strategies.append(
            _strategy(config, workdir, family, best, panel, report, data_version, role)
        )
        search_returns = matrix[card["best"]]
        settings = config.incubation
        confirming = confirms(settings, role)
        live = live_continuation(
            search_returns,
            mean=card["expected_mean"] if settings.live_edge == "expected" else 0.0,
            periods=settings.confirm_horizon if confirming else settings.horizon,
            seed=config.seed,
            block=settings.block,
        )
        inc = incubate(
            settings,
            family=family,
            expected_mean=card["expected_mean"],
            scale=card["scale"],
            search_returns=search_returns,
            live=live,
            confirming=confirming,
        )
        params = {
            "run_id": config.run_id,
            "family": str(family),
            "variant_id": str(card["best"]),
            "expected_mean": inc.config.expected_mean,
            "scale": inc.config.scale,
            "lr_factor": inc.lr_factor,
            "live_edge": settings.live_edge,
            "data_version": data_version,
            **({"role": role} if role != "deploy" else {}),
        }
        metrics = {"decision": inc.decision, "e_value": inc.state.e_value}
        if inc.confirm is not None:
            params |= {
                "confirm_alpha": settings.confirm_alpha,
                "confirm_horizon": inc.config.horizon,
            }
            metrics |= {"confirm_e_value": inc.confirm.e_value, "reason": inc.confirm.reason}
        log_run(
            INCUBATION_NAME,
            params=params,
            metrics=metrics,
            seed=config.seed,
            tags=["incubation", "synthetic"],
        )
        incubations.append(inc)
    return tuple(strategies), tuple(incubations)
