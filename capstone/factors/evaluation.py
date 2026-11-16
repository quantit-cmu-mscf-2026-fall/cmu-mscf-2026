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
- **Trials**: every (factor, holding period) evaluated is a trial with its
  own matrix column, so trying several holding periods is counted, not hidden.
- **Ledger** (`evaluate_factors`): every evaluated (factor, holding period) is logged with
  `log_run` as soon as its series exists, before anything uses it, with
  `sd.data_version()` in its params.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from capstone import shared_data as sd
from capstone.backtest import backtest_frame, to_weights
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
    return Panel(fields, universe, half_spread, data_version)


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

    def sharpe(self, which: str = "net") -> float:
        r = getattr(self, which).dropna()
        sd_ = r.std(ddof=1)
        return float(r.mean() / sd_ * np.sqrt(PERIODS_PER_YEAR)) if sd_ > 0 else float("nan")


def factor_signal(tree: Node, panel: Panel) -> pd.DataFrame:
    """The factor's value on the stocks that may be held that day, NaN elsewhere."""
    return evaluate(tree, panel.fields).where(panel.universe)


def factor_weights(tree: Node, panel: Panel) -> pd.DataFrame:
    """The factor as dollar-neutral weights (gross 1) on universe stocks, each day:
    the book `signal_returns` trades on a rebalance day.

    A day with no signal on any universe stock (a lookback still warming up,
    say) is a row of NaN, not of zeros: "no signal" is not "a signal that
    nets to zero weight".
    """
    signal = factor_signal(tree, panel)
    weights = to_weights(signal, demean=True, gross=1.0).fillna(0.0)
    return weights.where(signal.notna().any(axis=1), axis=0)


def signal_returns(signal: pd.DataFrame, panel: Panel, hold: int = 1) -> FactorReturns:
    """Returns of trading `signal`, rebalancing every `hold` trading days, from
    `backtest.backtest_frame`.

    On a rebalance day (every `hold`-th day counted from the panel's first) the
    book is set to that day's weights and held until the next one; `hold=1`
    rebalances daily. A weight set at day t's close earns day t+1's return.
    Trading one unit of a stock's weight pays half its quoted spread that day;
    a stock with no spread pays that day's median, and a trade on a day where
    no stock has a spread raises `ValueError` rather than trading for free.
    A day whose book came from no signal, with nothing traded, holds no
    position: its returns, cost and turnover are NaN, not 0.0, so a lookback's
    warm-up doesn't count as flat days (#55). A real signal that nets to zero
    weight still earns 0.0.
    """
    frame = backtest_frame(
        signal,
        panel.fields["returns"],
        cost_bps=panel.half_spread * 1e4,  # the backtest takes basis points
        demean=True,
        gross=1.0,
        hold=hold,
    )
    return FactorReturns(frame["net"], frame["gross"], frame["cost"], frame["turnover"])


def factor_returns(tree: Node, panel: Panel, hold: int = 1) -> FactorReturns:
    """The factor's portfolio returns, rebalanced every `hold` trading days."""
    return signal_returns(factor_signal(tree, panel), panel, hold)


def column_name(fid: str, hold: int) -> str:
    """A matrix column: the factor, and the holding period it was traded with."""
    return f"{fid}@{hold}"


def evaluate_factors(
    trees: Mapping[str, Node],
    panel: Panel,
    *,
    holds: tuple[int, ...] = (1,),
    extra_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every factor's net returns under every holding period, side by side.

    Each (factor, holding period) is a trial: logged to the ledger the moment
    its returns exist, before they are used, with its own matrix column
    (`column_name`). A factor that cannot be computed is logged once per
    holding period (each was a trial) and gets no column. Returns (matrix:
    dates x column net returns, summary: one row per column).
    """
    columns, summary = {}, []
    for fid, tree in trees.items():
        try:
            signal, error = factor_signal(tree, panel), ""
        except (ValueError, ArithmeticError) as exc:
            signal, error = None, str(exc)
        for hold in holds:
            params = {
                "factor_id": fid,
                "expression": unparse(tree),
                "hold": hold,
                "data_version": panel.data_version,
                "period": [DISCOVERY_START, DISCOVERY_END],
                "portfolio": "dollar-neutral, gross 1, signal at t earns t+1",
                "costs": "half quoted spread per unit traded",
                **(extra_params or {}),
            }
            if signal is None:
                log_run(LEDGER_NAME, params=params, metrics={"error": error}, tags=["evaluation"])
                continue
            try:
                result = signal_returns(signal, panel, hold)
            except ValueError as exc:
                # The backtest refused (no cost rate on a trading day): the trial
                # was run, so it is counted, and the refusal is not swallowed.
                log_run(
                    LEDGER_NAME, params=params, metrics={"error": str(exc)}, tags=["evaluation"]
                )
                raise
            metrics = {
                "sharpe_net": result.sharpe("net"),
                "sharpe_gross": result.sharpe("gross"),
                "mean_turnover": float(result.turnover.mean()),
                "mean_cost": float(result.cost.mean()),
                "days": int(result.net.notna().sum()),
            }
            log_run(LEDGER_NAME, params=params, metrics=metrics, tags=["evaluation"])
            name = column_name(fid, hold)
            columns[name] = result.net
            summary.append(
                {
                    "column": name,
                    "factor_id": fid,
                    "hold": hold,
                    "expression": params["expression"],
                    **metrics,
                }
            )
    matrix = pd.DataFrame(columns)
    matrix.columns.name = "candidate"
    table = pd.DataFrame(summary).set_index("column") if summary else pd.DataFrame()
    return matrix.iloc[1:], table
