"""Evaluate factor trees on the team's CRSP panel (QUANTIT-81).

This is where trials start. Each stored factor tree becomes one net daily
return series over the discovery period, and the series side by side are the
dates x candidates matrix that validation takes (`docs/validation.md` on the
validation stack), one column per `factor_id`.

- **Panel** (`load_panel`, `build_panel`): the shared S&P 500 daily file
  (`capstone.shared_data`), mapped onto the factor grammar's fields, dates x
  permno. Prices are divided by CRSP's cumulative price factor and shares and
  volume multiplied by its share factor, so a split moves nothing. Returns
  come from every row through `sd.daily_returns` (delistings included);
  holdings only from `in_universe`. `mkt_return` is the universe's
  value-weighted return, weighted by the previous day's cap.
- **Holdout guard**: the discovery period ends `DISCOVERY_END`; nothing after
  it is loaded, and a panel holding a later date is refused.
- **Factor to returns** (`factor_returns`): the factor's value on universe
  stocks, as dollar-neutral weights with gross exposure 1 (`backtest.to_weights`);
  a weight set at day t's close earns day t+1's return, as in
  `backtest.run_backtest`. The book is rebalanced every `hold` trading days,
  an option, not a fixed choice. Each trade pays half the stock's quoted
  spread that day (`sd.quoted_spread`): costs per stock and per day.
- **Variants** (`TradingSpec`, `grid`): a factor is traded by a spec, its
  holding period and sector neutrality; a (factor, spec) is a variant,
  recorded in the store's `variants` table with lineage to its factor. Each
  variant evaluated is a trial with its own matrix column, so a grid of
  specs is counted, not hidden. Costs are half the quoted spread; the full
  spread is reported alongside as a robustness view, not a separate trial.
- **Ledger** (`evaluate_factors`): every evaluated variant is logged with
  `log_run` as soon as its series exists, before anything uses it, with
  `sd.data_version()` in its params.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone import shared_data as sd
from capstone.backtest import to_weights
from capstone.factors import store
from capstone.factors.interpret import evaluate
from capstone.factors.tree import Node, unparse
from capstone.runlog import log_run

# Discovery 1990-2020, holdout 2021-2025: decided 2026-10-05 (PR #43). The
# holdout is looked at once, at the final decision, and never here.
DISCOVERY_START = "1990-01-01"
DISCOVERY_END = "2020-12-31"
DAILY_FILE = "sp500_daily_1990_2025"
COLUMNS = [
    "date",
    "permno",
    "in_universe",
    "dlyret",
    "dlyretmissflg",
    "dlydelflg",
    "delactiontype",
    "dlyopen",
    "dlyhigh",
    "dlylow",
    "dlyprc",
    "dlyvol",
    "shrout",
    "dlycap",
    "dlycumfacpr",
    "dlycumfacshr",
    "dlybid",
    "dlyask",
    "siccd",
]
LEDGER_NAME = "evaluate-factor"
PERIODS_PER_YEAR = 252


class HoldoutError(ValueError):
    """A date after the discovery period was asked for or found."""


@dataclass
class Panel:
    """The grammar's fields, dates x permno, with the universe and trading costs."""

    fields: dict[str, pd.DataFrame | pd.Series]
    universe: pd.DataFrame  # bool: may be held that day
    half_spread: pd.DataFrame  # cost of trading one unit of weight, as a fraction
    sic: pd.DataFrame | None = None  # SIC code, for sector neutrality
    data_version: str = ""


def _check_dates(dates: pd.Index) -> None:
    if len(dates) and pd.Timestamp(dates.max()) > pd.Timestamp(DISCOVERY_END):
        raise HoldoutError(f"dates after {DISCOVERY_END} are the holdout; they are not loaded here")


def build_panel(daily: pd.DataFrame, data_version: str = "") -> Panel:
    """The panel from daily rows (the `COLUMNS` of the shared daily file)."""
    _check_dates(pd.Index(daily["date"]))
    rows = daily.assign(
        returns=sd.daily_returns(daily),
        spread=sd.quoted_spread(daily),
        universe=daily["in_universe"].astype("float64").fillna(0).astype(bool),
    )
    price_factor = rows["dlycumfacpr"].where(rows["dlycumfacpr"] > 0)
    share_factor = rows["dlycumfacshr"].where(rows["dlycumfacshr"] > 0)
    rows["open"] = rows["dlyopen"] / price_factor
    rows["high"] = rows["dlyhigh"] / price_factor
    rows["low"] = rows["dlylow"] / price_factor
    rows["close"] = rows["dlyprc"].abs() / price_factor
    rows["volume"] = rows["dlyvol"] * share_factor
    rows["shares"] = rows["shrout"] * share_factor
    rows["cap"] = rows["dlycap"]

    def wide(column: str) -> pd.DataFrame:
        return rows.pivot(index="date", columns="permno", values=column).sort_index()

    fields = {
        name: wide(name)
        for name in ("open", "high", "low", "close", "volume", "returns", "shares", "cap")
    }
    universe = wide("universe").fillna(False).astype(bool)
    fields["mkt_return"] = _market_return(fields["returns"], fields["cap"], universe)
    half_spread = wide("spread") / 2
    sic = wide("siccd") if "siccd" in rows else None
    return Panel(fields, universe, half_spread, sic, data_version)


def _market_return(returns: pd.DataFrame, cap: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    """The universe's return each day, weighted by the previous day's cap."""
    weight = cap.where(universe).shift(1)
    held = weight.notna() & returns.notna()
    total = weight.where(held).sum(axis=1)
    out = (weight * returns).where(held).sum(axis=1) / total.where(total > 0)
    return out.rename("mkt_return")


def load_panel(start: str = DISCOVERY_START, end: str = DISCOVERY_END) -> Panel:
    """The discovery-period panel from the shared data. An `end` after the
    discovery period raises: the holdout is never loaded here."""
    if pd.Timestamp(end) > pd.Timestamp(DISCOVERY_END):
        raise HoldoutError(f"end {end} is after the discovery period, which ends {DISCOVERY_END}")
    daily = sd.load(DAILY_FILE, columns=COLUMNS, start=start, end=end)
    return build_panel(daily, sd.data_version())


@dataclass(frozen=True)
class FactorReturns:
    net: pd.Series  # daily return after costs; NaN on the first day (nothing held)
    gross: pd.Series
    cost: pd.Series
    turnover: pd.Series  # sum of |weight change| that day

    @property
    def net_full_spread(self) -> pd.Series:
        """Net of the full quoted spread instead of half: a pessimistic cost view."""
        return (self.gross - 2 * self.cost).rename("net_full_spread")

    def sharpe(self, which: str = "net") -> float:
        r = getattr(self, which).dropna()
        sd_ = r.std(ddof=1)
        return float(r.mean() / sd_ * np.sqrt(PERIODS_PER_YEAR)) if sd_ > 0 else float("nan")


# ---------------------------------------------------------------------------
# Trading specs: how a factor is traded. A (factor, spec) is a variant.

NEUTRALITY = ("none", "sector")
# A 2-digit SIC group with fewer universe stocks than this on a day falls back
# to its SIC division for that day, so no group is too thin to demean within.
MIN_SECTOR_SIZE = 5
_DIVISIONS = (  # SIC divisions by 2-digit major group
    (1, 9, "agriculture"),
    (10, 14, "mining"),
    (15, 17, "construction"),
    (20, 39, "manufacturing"),
    (40, 49, "transport-utilities"),
    (50, 51, "wholesale"),
    (52, 59, "retail"),
    (60, 67, "finance"),
    (70, 89, "services"),
    (91, 99, "public"),
)


@dataclass(frozen=True)
class TradingSpec:
    """How a factor is traded: rebalance every `hold` trading days; `neutral`
    "sector" demeans the signal within sector each day. Weights are always
    proportional to the (demeaned) signal, dollar-neutral, gross exposure 1."""

    hold: int = 1
    neutral: str = "none"

    def __post_init__(self) -> None:
        if self.hold < 1:
            raise ValueError("hold must be at least 1 day")
        if self.neutral not in NEUTRALITY:
            raise ValueError(f"neutral must be one of {NEUTRALITY}, got {self.neutral!r}")

    def as_dict(self) -> dict:
        return {"hold": self.hold, "neutral": self.neutral, "weighting": "signal"}


def grid(holds, neutral) -> list[TradingSpec]:
    """Every combination of the given holding periods and neutralities."""
    return [TradingSpec(h, n) for n in neutral for h in holds]


def _division(sic2: float) -> str:
    for low, high, name in _DIVISIONS:
        if low <= sic2 <= high:
            return name
    return "unclassified"


def sector_groups(panel: Panel) -> pd.Series:
    """Each universe (date, permno)'s sector: its 2-digit SIC group, or its SIC
    division on a day its group has fewer than `MIN_SECTOR_SIZE` universe stocks."""
    if panel.sic is None:
        raise ValueError("this panel has no SIC codes")
    sic = panel.sic.where(panel.universe).stack()
    sic2 = (sic // 100).astype(int)
    dates = sic2.index.get_level_values(0)
    size = sic2.groupby([dates, sic2.values]).transform("size")
    division = sic2.map(_division)
    return ("sic" + sic2.astype(str)).where(size >= MIN_SECTOR_SIZE, division).rename("sector")


def factor_signal(tree: Node, panel: Panel) -> pd.DataFrame:
    """The factor's value on universe stocks (NaN elsewhere)."""
    return evaluate(tree, panel.fields).where(panel.universe)


def signal_weights(signal: pd.DataFrame, panel: Panel, neutral: str = "none") -> pd.DataFrame:
    """Dollar-neutral weights, gross 1, proportional to the signal demeaned across
    all universe stocks or, with neutral="sector", within each sector.

    A day with no signal on any universe stock (a lookback still warming up,
    say) is a row of NaN, not of zeros, so `returns_from_weights` can tell
    "no signal" from "a signal that nets to zero weight".
    """
    has_signal = signal.notna().any(axis=1)
    if neutral == "sector":
        long = signal.stack()
        sectors = sector_groups(panel).reindex(long.index)
        dates = long.index.get_level_values(0)
        demeaned = long - long.groupby([dates, sectors.values]).transform("mean")
        signal = demeaned.unstack().reindex_like(signal)
        weights = to_weights(signal, demean=False, gross=1.0).fillna(0.0)
    else:
        weights = to_weights(signal, demean=True, gross=1.0).fillna(0.0)
    return weights.where(has_signal, axis=0)


def factor_weights(tree: Node, panel: Panel, neutral: str = "none") -> pd.DataFrame:
    """The factor as dollar-neutral weights (gross 1) on universe stocks, each day."""
    return signal_weights(factor_signal(tree, panel), panel, neutral)


def returns_from_weights(weights: pd.DataFrame, panel: Panel, hold: int = 1) -> FactorReturns:
    """Returns of trading `weights`, rebalancing every `hold` trading days.

    On a rebalance day (every `hold`-th day from the first) the book is set to
    that day's weights and held until the next one; `hold=1` rebalances
    daily. A weight set at day t's close earns day t+1's return (positions
    are the weights shifted one day, as in `backtest.run_backtest`). Trading
    the change in weight pays half that day's quoted spread. Between
    rebalances the weights are held fixed, so the small trades that would
    keep them fixed as prices drift are not charged.

    A row of NaN weights means no signal that day. A day whose book came from
    such a row, with nothing traded, holds no position: its returns, cost and
    turnover are NaN, not 0.0, so a lookback's warm-up doesn't count as flat
    days (the rule `backtest` adopts in #55). A real signal that nets to zero
    weight still earns 0.0.
    """
    if hold < 1:
        raise ValueError("hold must be at least 1 day")
    has_signal = weights.notna().any(axis=1).astype(float)
    weights = weights.fillna(0.0)
    if hold > 1:
        rebalance = pd.Series(np.arange(len(weights)) % hold == 0, index=weights.index)
        weights = weights.where(rebalance, axis=0).ffill()
        has_signal = has_signal.where(rebalance).ffill()
    returns = panel.fields["returns"]
    positions = weights.shift(1)
    gross = (positions * returns).sum(axis=1, min_count=1)
    trades = weights.diff().abs()
    trades.iloc[0] = weights.iloc[0].abs()  # building the first book is a trade too
    spread = panel.half_spread.reindex_like(trades)
    # A stock with no spread that day (out of the data): that day's median spread.
    spread = spread.where(spread.notna(), spread.median(axis=1), axis=0)
    cost = (trades * spread).sum(axis=1).shift(1)  # charged on the day the position is first held
    turnover = trades.sum(axis=1).shift(1)
    gross.iloc[:1] = np.nan
    # The book held on day t was set at t-1: idle if that came from no signal.
    idle = (has_signal.shift(1) == 0) & (turnover == 0)
    gross[idle] = np.nan
    cost[idle] = np.nan
    turnover[idle] = np.nan
    net = gross - cost
    return FactorReturns(net.rename("net"), gross.rename("gross"), cost.rename("cost"), turnover)


def factor_returns(tree: Node, panel: Panel, spec: TradingSpec | int = 1) -> FactorReturns:
    """The returns of trading the factor by `spec` (or a holding period in days)."""
    spec = spec if isinstance(spec, TradingSpec) else TradingSpec(hold=spec)
    return returns_from_weights(factor_weights(tree, panel, spec.neutral), panel, spec.hold)


def evaluate_factors(
    trees: Mapping[str, Node],
    panel: Panel,
    *,
    specs: Sequence[TradingSpec] = (TradingSpec(),),
    con: sqlite3.Connection | None = None,
    extra_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every variant's net returns, side by side: each factor traded by each spec.

    A variant (factor, spec) is a candidate and a trial: logged to the ledger
    the moment its returns exist, before they are used, as a matrix column
    named by its variant id (`store.variant_id`). With `con`, the variants
    are recorded in the store's `variants` table. A factor that cannot be
    computed is logged once per spec (each was a trial) and gets no column.
    Costs are half the quoted spread; the full-spread Sharpe is reported
    alongside as a robustness view, not a separate candidate. Returns
    (matrix: dates x variant net returns, summary: one row per variant).
    """
    columns, summary = {}, []
    for fid, tree in trees.items():
        try:
            signal, error = factor_signal(tree, panel), ""
        except (ValueError, ArithmeticError) as exc:
            signal, error = None, str(exc)
        weights_by_neutral: dict[str, pd.DataFrame] = {}
        for spec in specs:
            vid = (
                store.add_variant(con, fid, spec.as_dict())
                if con is not None
                else store.variant_id(fid, spec.as_dict())
            )
            params = {
                "factor_id": fid,
                "variant_id": vid,
                "spec": spec.as_dict(),
                "expression": unparse(tree),
                "data_version": panel.data_version,
                "period": [DISCOVERY_START, DISCOVERY_END],
                "costs": "half quoted spread per unit traded",
                **(extra_params or {}),
            }
            if signal is None:
                log_run(LEDGER_NAME, params=params, metrics={"error": error}, tags=["evaluation"])
                continue
            if spec.neutral not in weights_by_neutral:
                weights_by_neutral[spec.neutral] = signal_weights(signal, panel, spec.neutral)
            result = returns_from_weights(weights_by_neutral[spec.neutral], panel, spec.hold)
            full = result.net_full_spread.dropna()
            metrics = {
                "sharpe_net": result.sharpe("net"),
                "sharpe_gross": result.sharpe("gross"),
                "sharpe_net_full_spread": float(
                    full.mean() / full.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
                )
                if full.std(ddof=1) > 0
                else float("nan"),
                "mean_turnover": float(result.turnover.mean()),
                "mean_cost": float(result.cost.mean()),
                "days": int(result.net.notna().sum()),
            }
            log_run(LEDGER_NAME, params=params, metrics=metrics, tags=["evaluation"])
            columns[vid] = result.net
            summary.append(
                {
                    "variant_id": vid,
                    "factor_id": fid,
                    "hold": spec.hold,
                    "neutral": spec.neutral,
                    "expression": params["expression"],
                    **metrics,
                }
            )
    matrix = pd.DataFrame(columns)
    matrix.columns.name = "variant_id"
    table = pd.DataFrame(summary).set_index("variant_id") if summary else pd.DataFrame()
    return matrix.iloc[1:], table


# ---------------------------------------------------------------------------
# Trial counting, with corrections

CORRECTION_NAME = "ledger-correction"


def excluded_entries(entries: list[dict]) -> set[tuple[str, str]]:
    """(ts_utc, factor_id) of evaluation entries a ledger correction excludes.

    The ledger is append-only: a mistaken or duplicate run is never deleted.
    Instead a correction entry (name `CORRECTION_NAME`) names the entries to
    leave out of the trial count, and says why; the record stays complete.
    """
    out = set()
    for e in entries:
        if e.get("name") == CORRECTION_NAME and e["params"].get("ledger_name") == LEDGER_NAME:
            out.update((x["ts_utc"], x["factor_id"]) for x in e["params"]["excludes"])
    return out


def trial_count(entries: list[dict] | None = None) -> int:
    """Evaluation trials in the ledger: `LEDGER_NAME` entries no correction excludes."""
    if entries is None:
        from capstone.runlog import _read_entries

        entries = _read_entries()
    skip = excluded_entries(entries)
    return sum(
        1
        for e in entries
        if e.get("name") == LEDGER_NAME
        and (e.get("ts_utc"), e["params"].get("factor_id")) not in skip
    )
