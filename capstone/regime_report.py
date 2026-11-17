"""Each family's discovery result split by market regime.

Report-only. `regime_split` takes a finished discovery result (duck-typed: `survivors`,
a family -> variant mapping, and `scorecard["best"]`, a family -> best-variant mapping),
the search's dates x variants return matrix and the market return, and for each
family's best variant (survivors flagged) reports the annualised return, Sharpe ratio and
number of days in each regime state (`capstone.regimes`):

- `high_vol` / `low_vol`: rv126 above / not above its expanding median;
- `bear` / `non_bear`: the past 504 days' cumulative market return < 0 / >= 0;
- `panic_top` / `panic_rest`: the panic index above / not above its own expanding
  80th percentile through t-1 (a point-in-time top quintile).

It reads only dates up to the search end (never past `DISCOVERY_END`), changes no
gate decision (it runs after the gate and only reads its result) and never writes
to the ledger. A date whose regime is still in warm-up (NaN) is in no state.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from capstone import regimes
from capstone.factors.evaluation import DISCOVERY_END

__all__ = ["PANIC_QUANTILE", "STATES", "regime_markdown", "regime_split", "regime_states"]

TRADING_DAYS = 252
PANIC_QUANTILE = 0.8  # the top quintile
STATES = ("high_vol", "low_vol", "bear", "non_bear", "panic_top", "panic_rest")


def _end(end: str | None) -> pd.Timestamp:
    cap = pd.Timestamp(DISCOVERY_END)
    return cap if end is None else min(pd.Timestamp(end), cap)


def regime_states(mkt_return: pd.Series, *, end: str | None = None) -> pd.DataFrame:
    """Dates x `STATES` booleans (NaN in warm-up), on dates <= the search end only."""
    market = mkt_return.loc[: _end(end)].astype(float)
    frame = regimes.regime_series(market)
    vol, down, pan = frame["high_vol"], frame["bear"], frame["panic"]
    cutoff = pan.expanding(min_periods=1).quantile(PANIC_QUANTILE).shift(1)
    top = (pan > cutoff).astype(float).where(pan.notna() & cutoff.notna())
    pairs = {"high_vol": vol, "bear": down, "panic_top": top}
    out = {}
    for name, series in pairs.items():
        out[name] = series
        out[STATES[STATES.index(name) + 1]] = (1.0 - series).where(series.notna())
    return pd.DataFrame(out, index=market.index)[list(STATES)]


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    days = int(len(r))
    if days == 0:
        return {"days": 0, "ann_return": None, "sharpe": None}
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if days > 1 else float("nan")
    sharpe = mean / std * math.sqrt(TRADING_DAYS) if std > 0 and math.isfinite(std) else None
    return {"days": days, "ann_return": mean * TRADING_DAYS, "sharpe": sharpe}


def regime_split(
    found, returns: pd.DataFrame, mkt_return: pd.Series, *, end: str | None = None
) -> dict:
    """Each family's best variant (`found.scorecard['best']`), split by regime state."""
    states = regime_states(mkt_return, end=end)
    window = returns.loc[: _end(end)]
    states = states.reindex(window.index)
    survivors = {str(k): str(v) for k, v in found.survivors.items()}
    rows = []
    for family, best in found.scorecard["best"].items():
        variant = str(best)
        series = window[best] if best in window else pd.Series(dtype=float)
        split = {}
        for state in STATES:
            mask = states[state].eq(1.0)
            split[state] = _stats(series[mask.reindex(series.index, fill_value=False)])
        rows.append(
            {
                "family": str(family),
                "variant": variant,
                "survivor": survivors.get(str(family)) == variant,
                "all": _stats(series),
                "regimes": split,
            }
        )
    last = window.index.max() if len(window.index) else None
    return {
        "report_only": True,
        "end": None if last is None else pd.Timestamp(last).date().isoformat(),
        "states": list(STATES),
        "state_days": {s: int(states[s].eq(1.0).sum()) for s in STATES},
        "rows": rows,
    }


def _cell(stats: dict) -> str:
    sharpe = stats["sharpe"]
    shown = "-" if sharpe is None or not np.isfinite(sharpe) else f"{sharpe:.2f}"
    return f"{shown} ({stats['days']})"


def regime_markdown(split: dict) -> str:
    """The regime-split section of RESULT.md: Sharpe (days) per state."""
    states = split["states"]
    lines = [
        "",
        "## Regime split (report-only)",
        "",
        "Each family's best variant on the search period, split by point-in-time market",
        "regime (`capstone.regimes`). Sharpe ratio (annualised), days in brackets. This",
        "table changes no gate decision and logs no trial.",
        "",
        "| family | variant | survivor | all | " + " | ".join(states) + " |",
        "|---|---|---|---|" + "---|" * len(states),
    ]
    for row in split["rows"]:
        cells = [row["family"], row["variant"], "yes" if row["survivor"] else "no"]
        cells.append(_cell(row["all"]))
        cells.extend(_cell(row["regimes"][s]) for s in states)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
