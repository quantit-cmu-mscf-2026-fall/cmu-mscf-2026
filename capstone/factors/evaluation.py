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
  stocks, traded by the project's one backtest (`backtest.backtest_frame`)
  as dollar-neutral weights with gross exposure 1; a weight set at day t's
  close earns day t+1's return, and a lookback's warm-up is no position
  (NaN), not flat days. The book is rebalanced every `hold` trading days,
  an option, not a fixed choice. Each trade pays half the stock's quoted
  spread that day (`sd.quoted_spread`): costs per stock and per day. This
  module keeps no positions, trades or cost loop of its own.
- **Variants** (`TradingSpec`, `grid`): a factor is traded by a spec, its
  holding period and sector neutrality; a (factor, spec) is a variant,
  recorded in the store's `variants` table with lineage to its factor. Each
  variant evaluated is a trial with its own matrix column, so a grid of
  specs is counted, not hidden. A spec may add a no-trade band (`band`),
  traded by the same backtest. Costs are half the quoted spread; the full
  spread is reported alongside as a robustness view, not a separate trial.
- **Ledger** (`evaluate_factors`): every evaluated variant is logged with
  `log_run` as soon as its series exists, before anything uses it, with
  `sd.data_version()` and the panel's own first and last dates in its params;
  a caller's `tags` ride on each entry's "evaluation" tag.
- **Screen statistic** (`screen_pvalue`): each variant's one-sided HAC
  p-value against a Sharpe ratio of zero, on its net returns. It is the
  per-variant input to the validation funnel's stage 1 and the deepen move's
  reward. The stage 1 gate itself (each hypothesis family's search-adjusted
  p-value, then BH across families) belongs to validation, so the summary
  only adds each variant's `hypothesis_id`, the family it belongs to.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np
import pandas as pd

from capstone import shared_data as sd
from capstone.backtest import backtest_frame, sharpe_ratio, to_weights
from capstone.evaluate import sharpe_test
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
    calendar: pd.DatetimeIndex | None = None
    availability: dict[str, pd.Series] | None = None
    # Optional inputs the research constructions read when a caller attaches them.
    company: pd.DataFrame | None = None  # permco by date, for company-level signals
    fundamentals: pd.DataFrame | None = None  # accounting vintages with their publication dates
    # Raw closes a share planner sizes with; bound by a replay for formation eligibility.
    sizing_prices: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    # How a panel past the discovery period was authorized; None for every discovery panel.
    provenance: object | None = field(default=None, repr=False, compare=False)


def panel_period(panel: Panel) -> list[str]:
    """The panel's first and last dates, as logged in a trial's `period` param.

    The panel's own dates, not `DISCOVERY_START`/`DISCOVERY_END`: a synthetic
    or shortened panel is not 1990-2020, and the ledger records what was used.
    """
    dates = pd.DatetimeIndex(panel.universe.index)
    return [str(dates.min().date()), str(dates.max().date())]


def _check_dates(dates: pd.Index) -> None:
    if len(dates) and pd.Timestamp(dates.max()) > pd.Timestamp(DISCOVERY_END):
        raise HoldoutError(f"dates after {DISCOVERY_END} are the holdout; they are not loaded here")


def require_discovery(panel: Panel) -> None:
    """Raise `HoldoutError` if `panel` holds a date after `DISCOVERY_END`."""
    _check_dates(pd.Index(panel.universe.index))


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
    """That day's universe's return, each stock weighted by its previous day's cap.

    Membership is the day's own (`tree.py`'s definition of `mkt_return`): a
    stock that joins on day t counts in `mkt_return[t]`, one that leaves on t
    does not. Only the weight is lagged.
    """
    weight = cap.shift(1).where(universe)
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

    def sharpe(
        self, which: Literal["net", "gross", "cost", "turnover", "net_full_spread"] = "net"
    ) -> float:
        """Annualised Sharpe of one of the series: `backtest.sharpe_ratio`, the
        project's one definition (the one `backtest.summarize` reports)."""
        return sharpe_ratio(getattr(self, which), PERIODS_PER_YEAR)


# ---------------------------------------------------------------------------
# Trading specs: how a factor is traded. A (factor, spec) is a variant.

Neutral = Literal["none", "sector"]
NEUTRALITY: tuple[Neutral, ...] = get_args(Neutral)
# A 2-digit SIC group with fewer universe stocks than this on a day falls back
# to its SIC division for that day. A division's fallback group can still be thin.
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
    proportional to the (demeaned) signal, dollar-neutral, gross exposure 1.

    `band` is a no-trade band, in units of weight of the gross-1 book: on a
    rebalance a name trades to its target only when |target - held| > band
    (`backtest.band_book`). 0, the default, is no band; a spec without one
    has the dict and variant id it had before bands existed."""

    hold: int = 1
    neutral: Neutral = "none"
    band: float = 0.0

    def __post_init__(self) -> None:
        if self.hold < 1:
            raise ValueError("hold must be at least 1 day")
        if self.neutral not in NEUTRALITY:
            raise ValueError(f"neutral must be one of {NEUTRALITY}, got {self.neutral!r}")
        band = self.band
        if isinstance(band, bool) or not isinstance(band, int | float):
            raise ValueError(f"band must be a number, got {band!r}")
        if not (math.isfinite(band) and 0.0 <= band < 1.0):
            raise ValueError(f"band must be in [0, 1), got {band!r}")
        object.__setattr__(self, "band", float(band))

    def as_dict(self) -> dict:
        spec = {"hold": self.hold, "neutral": self.neutral, "weighting": "signal"}
        return {**spec, "band": self.band} if self.band > 0 else spec


def grid(holds, neutral, bands=(0.0,)) -> list[TradingSpec]:
    """Every combination of the given holding periods, neutralities and bands
    (bands innermost; the default is no band)."""
    return [TradingSpec(h, n, b) for n in neutral for h in holds for b in bands]


def spec_from_summary(row) -> TradingSpec:
    """The spec of one `evaluate_factors` summary row (no "band", or NaN: none)."""
    band = row.get("band", 0.0)
    band = 0.0 if band is None or pd.isna(band) else float(band)
    return TradingSpec(int(row["hold"]), row["neutral"], band)


def _division(sic2: float) -> str:
    for low, high, name in _DIVISIONS:
        if low <= sic2 <= high:
            return name
    return "unclassified"


def sector_groups(panel: Panel) -> pd.Series:
    """Each universe (date, permno)'s sector: its 2-digit SIC group, or its SIC
    division on a day its group has fewer than `MIN_SECTOR_SIZE` universe stocks.
    A universe stock with no SIC code that day has no sector (no row here)."""
    if panel.sic is None:
        raise ValueError("this panel has no SIC codes")
    # dropna: pandas 3's stack() keeps NaN cells, pandas 2's dropped them.
    sic = panel.sic.where(panel.universe).stack().dropna()
    sic2 = (sic // 100).astype(int)
    dates = sic2.index.get_level_values(0)
    size = sic2.groupby([dates, sic2.values]).transform("size")
    division = sic2.map(_division)
    return ("sic" + sic2.astype(str)).where(size >= MIN_SECTOR_SIZE, division).rename("sector")


def factor_signal(tree: Node, panel: Panel) -> pd.DataFrame:
    """The factor's value on the stocks that may be held that day, NaN elsewhere."""
    return evaluate(tree, panel.fields).where(panel.universe)


def trading_signal(signal: pd.DataFrame, panel: Panel, neutral: Neutral = "none") -> pd.DataFrame:
    """The signal the backtest trades: `signal` itself, or with neutral="sector"
    the signal less its sector's mean each day (traded without a further
    demean). A stock with a signal but no sector gets 0, not NaN, so a day
    with any signal is never mistaken for a day with none.
    """
    if neutral == "none":
        return signal
    long = signal.stack().dropna()  # dropna: pandas 3's stack() keeps NaN cells
    sectors = sector_groups(panel).reindex(long.index)
    dates = long.index.get_level_values(0)
    demeaned = long - long.groupby([dates, sectors.values]).transform("mean")
    demeaned = demeaned.unstack().reindex_like(signal)
    return demeaned.where(signal.isna(), demeaned.fillna(0.0))


def signal_weights(signal: pd.DataFrame, panel: Panel, neutral: Neutral = "none") -> pd.DataFrame:
    """Dollar-neutral weights, gross 1, proportional to the signal demeaned across
    all universe stocks or, with neutral="sector", within each sector: the
    book `signal_returns` trades on each rebalance day.

    A day with no signal on any universe stock (a lookback still warming up,
    say) is a row of NaN, not of zeros: "no signal" is not "a signal that
    nets to zero weight".
    """
    traded = trading_signal(signal, panel, neutral)
    weights = to_weights(traded, demean=neutral == "none", gross=1.0).fillna(0.0)
    return weights.where(signal.notna().any(axis=1), axis=0)


def factor_weights(tree: Node, panel: Panel, neutral: Neutral = "none") -> pd.DataFrame:
    """The factor as dollar-neutral weights (gross 1) on universe stocks, each day."""
    return signal_weights(factor_signal(tree, panel), panel, neutral)


def signal_returns(signal: pd.DataFrame, panel: Panel, spec: TradingSpec) -> FactorReturns:
    """Returns of trading `signal` by `spec`, from `backtest.backtest_frame`.

    The book is `signal_weights`, set on every `spec.hold`-th day counted from
    the panel's first (`hold=1` rebalances daily) and held in between; a
    weight set at day t's close earns day t+1's return. Trading one unit of a
    stock's weight pays half its quoted spread that day; a stock with no
    spread pays that day's median, and a trade on a day where no stock has a
    spread raises `ValueError` rather than trading for free. A day whose book
    came from no signal, with nothing traded, holds no position: its returns,
    cost and turnover are NaN, not 0.0, so a lookback's warm-up doesn't count
    as flat days (#55). A real signal that nets to zero weight still earns 0.0.
    """
    return _backtest(trading_signal(signal, panel, spec.neutral), panel, spec)


def _backtest(traded: pd.DataFrame, panel: Panel, spec: TradingSpec) -> FactorReturns:
    """`signal_returns` from `trading_signal`'s output for `spec.neutral`."""
    frame = backtest_frame(
        traded,
        panel.fields["returns"],
        cost_bps=panel.half_spread * 1e4,  # the backtest takes basis points
        demean=spec.neutral == "none",
        gross=1.0,
        hold=spec.hold,
        band=spec.band,
    )
    return FactorReturns(frame["net"], frame["gross"], frame["cost"], frame["turnover"])


def factor_returns(tree: Node, panel: Panel, spec: TradingSpec | int = 1) -> FactorReturns:
    """The returns of trading the factor by `spec` (or a holding period in days)."""
    spec = spec if isinstance(spec, TradingSpec) else TradingSpec(hold=spec)
    return signal_returns(factor_signal(tree, panel), panel, spec)


def screen_pvalue(net: pd.Series) -> float:
    """One-sided p-value of H0: Sharpe ratio <= 0, with a HAC variance (`sharpe_test`).

    Computed as 1 - PSR, which equals `sharpe_test(...).pvalue_greater` on the
    validation stack (#31) to rounding; switch to that field once #31 is on
    `main`. NaN when the series has no variance.
    """
    clean = net.dropna()
    if len(clean) < 2 or not clean.std(ddof=1) > 0:
        return float("nan")
    p = 1.0 - sharpe_test(clean, periods_per_year=PERIODS_PER_YEAR).psr
    return float(p) if np.isfinite(p) else float("nan")


def hypothesis_of(con: sqlite3.Connection, fid: str) -> str | None:
    """The hypothesis a factor belongs to: the one whose proposal first stored it.

    A factor proposed again under another hypothesis is recorded as a
    duplicate there and stays in its first family.
    """
    rows = store.lineage(con, fid)
    stored = [r for r in rows if r["status"] == "stored"] or rows
    return stored[0]["hypothesis_id"] if stored else None


def evaluate_factors(
    trees: Mapping[str, Node],
    panel: Panel,
    *,
    specs: Sequence[TradingSpec] = (TradingSpec(),),
    con: sqlite3.Connection | None = None,
    families: Mapping[str, str | None] | None = None,
    extra_params: dict | None = None,
    tags: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every variant's net returns, side by side: each factor traded by each spec.

    A variant (factor, spec) is a candidate and a trial: logged to the ledger
    the moment its returns exist, before they are used, as a matrix column
    named by its variant id (`store.variant_id`). With `con`, the variants
    are recorded in the store's `variants` table. A factor that cannot be
    computed is logged once per spec (each was a trial) and gets no column.
    Costs are half the quoted spread; the full-spread Sharpe is reported
    alongside as a robustness view, not a separate candidate. Each variant's
    screen p-value (`screen_pvalue`) is logged with its metrics, and the
    summary names its hypothesis family from `families` (factor id ->
    hypothesis id, as `hypothesis_of` gives). `tags` are added to each
    entry's "evaluation" tag (a run's tag, to scope its trial count). Returns
    (matrix: dates x variant net returns, summary: one row per variant).
    A panel holding a date after `DISCOVERY_END` raises `HoldoutError`.
    """
    require_discovery(panel)
    columns, summary = {}, []
    for fid, tree in trees.items():
        family = (families or {}).get(fid)
        try:
            signal, error = factor_signal(tree, panel), ""
        except (ValueError, ArithmeticError) as exc:
            signal, error = None, str(exc)
        traded: dict[str, pd.DataFrame] = {}  # the sector demean is slow: once per factor
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
                "period": panel_period(panel),
                "costs": "half quoted spread per unit traded",
                **(extra_params or {}),
            }
            if signal is None:
                log_run(
                    LEDGER_NAME, params=params, metrics={"error": error}, tags=["evaluation", *tags]
                )
                continue
            if spec.neutral not in traded:
                traded[spec.neutral] = trading_signal(signal, panel, spec.neutral)
            try:
                result = _backtest(traded[spec.neutral], panel, spec)
            except ValueError as exc:
                # The backtest refused (no cost rate on a trading day): the trial
                # was run, so it is counted, and the refusal is not swallowed.
                log_run(
                    LEDGER_NAME,
                    params=params,
                    metrics={"error": str(exc)},
                    tags=["evaluation", *tags],
                )
                raise
            metrics = {
                "sharpe_net": result.sharpe("net"),
                "sharpe_gross": result.sharpe("gross"),
                "sharpe_net_full_spread": result.sharpe("net_full_spread"),
                "mean_turnover": float(result.turnover.mean()),
                "mean_cost": float(result.cost.mean()),
                "days": int(result.net.notna().sum()),
                "pvalue_screen": screen_pvalue(result.net),
            }
            log_run(LEDGER_NAME, params=params, metrics=metrics, tags=["evaluation", *tags])
            columns[vid] = result.net
            summary.append(
                {
                    "variant_id": vid,
                    "factor_id": fid,
                    "hypothesis_id": family,
                    "hold": spec.hold,
                    "neutral": spec.neutral,
                    **({"band": spec.band} if spec.band > 0 else {}),
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
        from capstone.runlog import read_entries

        entries = read_entries()
    skip = excluded_entries(entries)
    return sum(
        1
        for e in entries
        if e.get("name") == LEDGER_NAME
        and (e.get("ts_utc"), e["params"].get("factor_id")) not in skip
    )
