"""Synthetic markets and a stand-in holdout provider for the gate's tests.

Every market is synthetic and ends years before any real holdout; the gate
tests point `CAPSTONE_LEDGER_DIR` at a temporary directory (their `ledger`
fixture). No real data is read.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

import pandas as pd

from capstone import calibration, gate, runlog
from capstone.funnel import Thresholds
from capstone.gate import GateConfig, TrialScope, register, run_gate
from capstone.scorecard import GateResult

N_FAMILIES = 10
K = 4
N_BOOT = 199
N_SEARCH = 6 * 252
N_HOLDOUT = 3 * 252
TRIAL_NAME = "evaluate-factor"
AUTHORIZER = "test-gate-owner"
STAGE_1_TO_3 = [
    "best",
    "k",
    "s1_stat",
    "s1_pass",
    "s2_stat",
    "s2_pass",
    "s3_pass",
    "p1",
    "by",
    "holm",
    "sharpe",
    "psr",
    "min_trl_years",
    "is_mean",
]


@dataclasses.dataclass(frozen=True)
class Market:
    returns: pd.DataFrame
    family: pd.Series
    truth: pd.Series
    search_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp


def market(seed: int, share_real: float = 0.0, sharpe: float = 1.5, **defects) -> Market:
    """6 years of search and 3 of holdout, on business days from 1990."""
    fs = calibration.make_family_set(
        N_FAMILIES,
        K,
        share_real,
        sharpe,
        seed=seed,
        n_search=N_SEARCH,
        n_holdout=N_HOLDOUT,
        **defects,
    )
    dates = pd.bdate_range("1990-01-01", periods=len(fs.returns))
    return Market(
        returns=fs.returns.set_axis(dates),
        family=fs.family,
        truth=fs.truth,
        search_end=dates[fs.n_search - 1],
        holdout_start=dates[fs.n_search],
        holdout_end=dates[-1],
    )


def config_for(m: Market, **overrides) -> GateConfig:
    kwargs = dict(
        thresholds=Thresholds(stage1="bootstrap"),
        n_boot=N_BOOT,
        seed=0,
        search_end=m.search_end,
        holdout_start=m.holdout_start,
        holdout_end=m.holdout_end,
        trial_scope=TrialScope(name=TRIAL_NAME),
        data_version="synthetic",
        generator="test-gen",
    )
    kwargs.update(overrides)
    return GateConfig(**kwargs)


def log_trials(n: int, name: str = TRIAL_NAME) -> None:
    """Append `n` trial entries to the (temporary) ledger in log_run's shape."""
    path = runlog.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for i in range(n):
            entry = {
                "ts_utc": datetime.now(UTC).isoformat(),
                "name": name,
                "seed": i,
                "params": {"variant": i},
                "tags": [],
            }
            fh.write(json.dumps(entry) + "\n")


def search_of(m: Market) -> pd.DataFrame:
    """The gate's input: the search period only."""
    return m.returns.loc[m.returns.index <= m.search_end]


class Holdout:
    """A holdout provider over `returns` that records each call: the ids, the
    period, and how many look entries the ledger held at that moment."""

    def __init__(self, returns: pd.DataFrame, config_hash: str = ""):
        self.returns = returns
        self.config_hash = config_hash
        self.calls: list[tuple] = []

    def __call__(self, ids, start, end) -> pd.DataFrame:
        n_looks = len(looks(self.config_hash)) if self.config_hash else None
        self.calls.append((tuple(ids), pd.Timestamp(start), pd.Timestamp(end), n_looks))
        idx = self.returns.index
        return self.returns.loc[(idx >= start) & (idx <= end), list(ids)]


def run(config: GateConfig, m: Market, *, holdout=None, authorizer=AUTHORIZER) -> GateResult:
    """Run a registered config on `m`'s search period, its holdout on demand."""
    holdout = Holdout(m.returns, config.config_hash) if holdout is None else holdout
    return run_gate(config, search_of(m), m.family, holdout=holdout, authorizer=authorizer)


def gate_once(m: Market, *, n_trials: int | None = None, holdout=None, **overrides) -> GateResult:
    """Register a config on the current ledger and run the gate once."""
    log_trials(m.returns.shape[1] if n_trials is None else n_trials)
    config = config_for(m, **overrides)
    register(config)
    return run(config, m, holdout=holdout)


def looks(config_hash: str) -> list[dict]:
    return [
        e
        for e in runlog.read_entries()
        if e["name"] == gate.LOOK_NAME and e["params"]["config_hash"] == config_hash
    ]
