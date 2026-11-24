"""The report-only regime split of a discovery result (`capstone.regime_report`).

- one row per family's best variant, survivors flagged, with days, annualised
  return and Sharpe per regime state, matching a direct computation;
- it changes no gate decision and writes nothing to the ledger;
- it never reads a date after the search end or `DISCOVERY_END`.

`regime_split` only reads `found.survivors` and `found.scorecard["best"]`. Until the
gate's discovery run is public, `StandIn` below has exactly those two attributes (plus
`counts`, to check they are left alone); `test_split_of_a_real_discovery_result` is
the integration test to switch on when the gate lands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from capstone import regime_report, regimes, runlog
from capstone.factors.evaluation import DISCOVERY_END
from capstone.regime_report import STATES, regime_split, regime_states

FAMILIES = {"mom": ["mom_a", "mom_b"], "value": ["val_a", "val_b"], "qual": ["qual_a"]}
BEST = {"mom": "mom_b", "value": "val_a", "qual": "qual_a"}
SURVIVORS = {"mom": "mom_b", "qual": "qual_a"}


@dataclass
class StandIn:
    """The two attributes `regime_split` reads from a gate discovery result."""

    scorecard: pd.DataFrame
    survivors: dict
    counts: dict


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


def ledger_bytes() -> bytes:
    path = runlog.ledger_path()
    return path.read_bytes() if path.exists() else b""


def market_return(index: pd.DatetimeIndex, seed: int = 3) -> pd.Series:
    """A market with a calm and a stormy, falling stretch, so every state occurs."""
    rng = np.random.default_rng(seed)
    n = len(index)
    vol = np.where((np.arange(n) // 700) % 2 == 1, 0.025, 0.007)
    drift = np.where((np.arange(n) // 700) % 2 == 1, -0.0015, 0.0006)
    return pd.Series(drift + vol * rng.standard_normal(n), index=index, name="mkt_return")


def search_returns(n: int = 2400, seed: int = 11) -> pd.DataFrame:
    idx = pd.bdate_range("2005-01-03", periods=n)
    rng = np.random.default_rng(seed)
    cols = [v for variants in FAMILIES.values() for v in variants]
    return pd.DataFrame(rng.normal(0.0004, 0.01, (n, len(cols))), index=idx, columns=cols)


def stand_in() -> StandIn:
    card = pd.DataFrame({"best": BEST}, index=pd.Index(list(BEST), name="family"))
    return StandIn(card, dict(SURVIVORS), {"families": len(card)})


def test_rows_match_a_direct_computation(ledger):
    found, returns = stand_in(), search_returns()
    mkt = market_return(returns.index)
    split = regime_split(found, returns, mkt)
    assert split["report_only"] is True and split["states"] == list(STATES)
    assert [r["family"] for r in split["rows"]] == [str(f) for f in found.scorecard.index]
    assert {r["family"] for r in split["rows"] if r["survivor"]} == set(found.survivors)
    # A family whose best variant is not the surviving one is not flagged.
    assert not [r for r in split["rows"] if r["family"] == "value"][0]["survivor"]

    states = regime_states(mkt)
    for state in STATES:
        assert split["state_days"][state] > 0, state
    row = split["rows"][0]
    series = returns[row["variant"]]
    in_state = series[states["high_vol"].reindex(series.index).eq(1.0)].dropna()
    got = row["regimes"]["high_vol"]
    assert got["days"] == len(in_state)
    assert got["ann_return"] == pytest.approx(in_state.mean() * 252)
    assert got["sharpe"] == pytest.approx(in_state.mean() / in_state.std() * math.sqrt(252))
    for a, b in (("high_vol", "low_vol"), ("bear", "non_bear"), ("panic_top", "panic_rest")):
        both = row["regimes"][a]["days"] + row["regimes"][b]["days"]
        assert both <= row["all"]["days"]


def test_states_are_the_regime_series(ledger):
    idx = pd.bdate_range("2000-01-03", periods=1500)
    mkt = market_return(idx)
    states = regime_states(mkt)
    pd.testing.assert_series_equal(
        states["high_vol"], regimes.regime_series(mkt)["high_vol"], check_names=False
    )
    pd.testing.assert_series_equal(
        states["bear"], regimes.regime_series(mkt)["bear"], check_names=False
    )
    warm = states["bear"].isna()
    assert warm.iloc[: regimes.BEAR_WINDOW].all()  # warm-up stays NaN, in no state
    assert states.loc[warm, ["bear", "non_bear"]].isna().all().all()
    known = states["high_vol"].notna()
    assert (states.loc[known, "high_vol"] + states.loc[known, "low_vol"] == 1).all()


def test_report_only_no_gate_change_no_ledger_write(ledger):
    found, returns = stand_in(), search_returns()
    card = found.scorecard.copy()
    counts, survivors = dict(found.counts), dict(found.survivors)
    before = ledger_bytes()

    def refuse(*args, **kwargs):
        raise AssertionError("the regime split wrote to the ledger")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runlog, "log_run", refuse)
        regime_split(found, returns, market_return(returns.index))
    pd.testing.assert_frame_equal(found.scorecard, card)
    assert found.counts == counts and found.survivors == survivors
    assert ledger_bytes() == before
    assert not ledger.exists()


def test_nothing_after_the_search_end_is_read(ledger):
    found, returns = stand_in(), search_returns()
    mkt = market_return(returns.index)
    end = returns.index[len(returns) // 2]
    cut = regime_split(found, returns.loc[:end], mkt.loc[:end])
    # Wild values after the end, in both the returns and the market, change nothing.
    noisy_returns = returns.copy()
    noisy_returns.loc[noisy_returns.index > end] = 1e6
    noisy_mkt = mkt.copy()
    noisy_mkt.loc[noisy_mkt.index > end] = -0.9
    full = regime_split(found, noisy_returns, noisy_mkt, end=str(end.date()))
    assert full == cut


def test_discovery_end_caps_the_window(ledger):
    idx = pd.bdate_range("2019-01-01", "2022-06-30")
    mkt = market_return(idx)
    states = regime_states(mkt, end="2022-06-30")
    assert states.index.max() <= pd.Timestamp(DISCOVERY_END)


def test_markdown_section(ledger):
    found, returns = stand_in(), search_returns()
    text = regime_report.regime_markdown(regime_split(found, returns, market_return(returns.index)))
    assert "## Regime split (report-only)" in text
    assert text.count("\n| ") == len(found.scorecard) + 1


@pytest.mark.skip(reason="switch on when the gate's `run_discovery` is public (S-17)")
def test_split_of_a_real_discovery_result(ledger):
    """Same checks as `test_rows_match_a_direct_computation`, on `run_discovery`'s result."""
    raise NotImplementedError("build `found` with the gate's discovery run, drop `StandIn`")
