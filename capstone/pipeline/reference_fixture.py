"""Synthetic reference inputs, including calendar and market-wide FF3 fields."""

import numpy as np
import pandas as pd

from capstone.factors.evaluation import Panel


def reference_panel():
    dates = pd.bdate_range("1995-01-02", "2006-02-02")
    calendar = pd.bdate_range("1995-01-02", "2006-03-01")
    names = pd.Index(range(1, 121), name="permno")
    rng = np.random.default_rng(404)
    trend = np.linspace(-0.0003, 0.0003, len(names))
    raw = rng.normal(0, 0.008, (len(dates), len(names))) + trend
    returns = pd.DataFrame(raw, index=dates, columns=names)
    close = 100 * (1 + returns).cumprod()
    volume = pd.DataFrame(
        np.broadcast_to(np.linspace(300_000, 100_000, len(names)), raw.shape),
        index=dates,
        columns=names,
    )
    fields = {
        "returns": returns,
        "close": close,
        "open": close / (1 + returns),
        "shares": close * 0 + 1_000_000,
        "cap": close * 1_000_000,
        "volume": volume,
        "mkt_return": returns.mul(0).add(returns.mean(axis=1), axis=0),
    }
    for name in ("mkt_excess", "smb", "hml"):
        series = rng.normal(0, 0.005, len(dates))
        fields[name] = pd.DataFrame(
            np.repeat(series[:, None], len(names), axis=1), index=dates, columns=names
        )
    fields["rf"] = close * 0 + 0.0001
    return Panel(
        fields,
        close.notna(),
        close * 0 + 0.0005,
        data_version="synthetic-reference-v2",
        calendar=calendar,
        availability={name: pd.Series(dates, index=dates) for name in fields},
    )
