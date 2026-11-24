"""The null-calibration test for regime variants (the key safety test).

On signal-free return matrices (`synth.make_return_matrix`, with correlated `rho`
and autocorrelated `ar1` nulls) each family gets regime-split variants of its first
variant: gated off in a bear state, kept only in high vol, and volatility-scaled.
The regimes come from an independent synthetic market and are block-shuffled, so
they carry no information. Adding them must keep the funnel's false-discovery rate
at stage 1 (BH at `q1`) and after stage 2 at or below the nominal `q1`: on a null
every discovery is false, so the FDR is the share of worlds with any discovery.
A modest number of worlds keeps it CI-sized; the bound allows two standard errors.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from capstone import funnel, regimes
from capstone.factors.interpret import VOLSCALE_TARGET
from capstone.funnel import Thresholds
from capstone.synth import make_return_matrix

# 100 worlds take about a minute per case. The default is a CI-sized 30; set
# REGIME_NULL_WORLDS=100 for the full run.
N_WORLDS = int(os.environ.get("REGIME_NULL_WORLDS", "30"))
N_OBS = 2520 + regimes.BEAR_WINDOW + 1
K = 4  # variants per family, before the regime variants
N_FAMILIES = 15
BLOCK = 63  # a quarter: shuffled in blocks so regimes keep their persistence
N_BOOT = 499  # stage 1 is the registered bootstrap; fewer draws than the gate, CI-sized


def _shuffled_regimes(n: int, index: pd.DatetimeIndex, rng) -> pd.DataFrame:
    """bear, high_vol and the volscale factor from an independent market, block-shuffled."""
    vol = np.repeat(rng.choice([0.007, 0.02], size=n // BLOCK + 1), BLOCK)[:n]
    # The synthetic market lives on search-period dates (regime_series refuses holdout
    # dates); its regimes are then laid on the return matrix's own dates.
    market = pd.Series(
        vol * rng.standard_normal(n) - 0.0002, index=pd.bdate_range("1990-01-01", periods=n)
    )
    series = regimes.regime_series(market).set_axis(index)
    frame = pd.DataFrame(
        {
            "bear": series["bear"],
            "high_vol": series["high_vol"],
            "scale": VOLSCALE_TARGET / series["rv126"],
        }
    ).dropna()
    blocks = [frame.iloc[i : i + BLOCK] for i in range(0, len(frame), BLOCK)]
    order = rng.permutation(len(blocks))
    shuffled = pd.concat([blocks[i] for i in order])
    return shuffled.set_axis(frame.index)


def _world(seed: int, *, rho: float, ar1: float):
    """A null world: K variants per family plus three regime variants of the first."""
    m = make_return_matrix(n_obs=N_OBS, n_candidates=N_FAMILIES * K, rho=rho, ar1=ar1, seed=seed)
    returns = m.returns
    family = pd.Series([f"h{i // K}" for i in range(returns.shape[1])], index=returns.columns)
    rng = np.random.default_rng(10_000 + seed)
    states = _shuffled_regimes(len(returns), returns.index, rng)
    returns = returns.loc[states.index]
    extra, labels = {}, {}
    for f in sorted(family.unique()):
        base = returns[family.index[family == f][0]]
        for name, series in (
            ("bear_gate", base * (1.0 - states["bear"])),
            ("high_vol", base.where(states["high_vol"] == 1.0, 0.0)),
            ("volscale", base * states["scale"] / states["scale"].mean()),
        ):
            extra[f"{base.name}:{name}"] = series
            labels[f"{base.name}:{name}"] = f
    returns = pd.concat([returns, pd.DataFrame(extra)], axis=1)
    return returns, pd.concat([family, pd.Series(labels)])


def _discoveries(returns: pd.DataFrame, family: pd.Series, th: Thresholds) -> tuple[int, int]:
    stats = funnel.family_statistics(
        returns, family, search_end=returns.index[-1], n_boot=N_BOOT, q=th.q1
    )["reuse"]
    stages = funnel.screen(stats, th)
    return int(stages["stage1"].sum()), int(stages["stage2"].sum())


@pytest.mark.parametrize("rho, ar1", [(0.0, 0.0), (0.3, 0.1)])
def test_regime_variants_keep_the_null_fdr_at_nominal(rho, ar1):
    th = Thresholds()
    any1 = any2 = without = 0
    for seed in range(N_WORLDS):
        returns, family = _world(seed, rho=rho, ar1=ar1)
        assert returns.shape[1] == N_FAMILIES * (K + 3)
        assert not returns.isna().any().any()
        s1, s2 = _discoveries(returns, family, th)
        any1 += s1 > 0
        any2 += s2 > 0
        plain = [c for c in returns.columns if ":" not in c]  # the same world, no regimes
        without += _discoveries(returns[plain], family[plain], th)[0] > 0
    q = th.q1
    se = math.sqrt(q * (1 - q) / N_WORLDS)
    assert any1 / N_WORLDS <= q + 2 * se, (any1, without, N_WORLDS, q)
    assert any2 <= any1 and any2 / N_WORLDS <= q + 2 * se
    # Paired: the regime variants do not raise the rate over the same worlds without them.
    assert (any1 - without) / N_WORLDS <= 2 * se, (any1, without, N_WORLDS)


def test_the_regime_variants_are_built_as_described():
    returns, family = _world(0, rho=0.0, ar1=0.0)
    base = returns.iloc[:, 0]
    gated = returns[f"{base.name}:bear_gate"]
    assert ((gated == 0) | np.isclose(gated, base)).all()
    assert (gated == 0).any() and (gated != 0).any()
    assert family[f"{base.name}:volscale"] == family[base.name]
