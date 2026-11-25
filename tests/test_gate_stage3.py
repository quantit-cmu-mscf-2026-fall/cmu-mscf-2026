"""Tests for stage 3 of `gate.run_gate`: the spanning alpha test, opt-in by registration.

A config that registers a factor frame (`factors_digest`) and `thresholds.alpha3`
has each stage-2 survivor's chosen variant tested with `spanning.factor_alpha`
(excess returns as given, one-sided alpha p, BH across the families tested at
`alpha3`); a config that registers neither behaves as it did before stage 3
existed. Every market and factor frame is synthetic and no real data is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gate_helpers import (
    AUTHORIZER,
    N_SEARCH,
    Holdout,
    Market,
    config_for,
    log_trials,
    looks,
    market,
    search_of,
)
from scipy import stats

from capstone import gate
from capstone.data import FF5_MOMENTUM_COLUMNS, FF5_MOMENTUM_REGRESSORS
from capstone.evaluate import average_correlation
from capstone.funnel import Thresholds, thresholds_dict
from capstone.gate import factors_frame_digest, register, run_gate
from capstone.search_fdr import search_adjusted_pvalue

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
CRSP_V1_HASH = "5f590f13f63e94f33fbdf960089c1fc01881a6fedf2437d8afbc3a0c396af7da"
ALPHA3 = 0.05


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path / "ledger"


def factor_frame(m: Market, seed: int = 7) -> pd.DataFrame:
    """Daily FF5 + Mom on the search dates; the market factor has a clear drift."""
    dates = search_of(m).index
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.005, (len(dates), len(FF5_MOMENTUM_COLUMNS)))
    frame = pd.DataFrame(data, index=dates, columns=FF5_MOMENTUM_COLUMNS)
    frame["Mkt-RF"] = rng.normal(0.003, 0.01, len(dates))
    frame["RF"] = 0.0001
    return frame


def planted(seed: int = 3) -> tuple[Market, str, str]:
    """A null market whose first family is a pure market exposure and whose
    second has alpha orthogonal to the factors; returns (market, exposed, alpha)."""
    m = market(seed)
    factors = factor_frame(m)
    rng = np.random.default_rng(seed + 100)
    returns = m.returns.copy()
    exposed, alpha = list(dict.fromkeys(m.family))[:2]
    mkt = factors["Mkt-RF"].reindex(returns.index).fillna(0.0)
    for column in m.family.index[m.family == exposed]:
        returns[column] = mkt + rng.normal(0.0, 0.002, len(returns))
    for column in m.family.index[m.family == alpha]:
        returns[column] = 0.004 + rng.normal(0.0, 0.01, len(returns))
    return (
        Market(returns, m.family, m.truth, m.search_end, m.holdout_start, m.holdout_end),
        (exposed),
        alpha,
    )


def stage3_config(m: Market, factors: pd.DataFrame, **overrides):
    overrides.setdefault("thresholds", Thresholds(stage1="bootstrap", alpha3=ALPHA3))
    overrides.setdefault("factors_digest", factors_frame_digest(factors))
    return config_for(m, **overrides)


def run3(config, m: Market, factors, holdout=None, returns=None):
    holdout = Holdout(m.returns, config.config_hash) if holdout is None else holdout
    return run_gate(
        config,
        search_of(m) if returns is None else returns,
        m.family,
        holdout=holdout,
        authorizer=AUTHORIZER,
        factors=factors,
    )


def registered(m: Market, factors, **overrides):
    log_trials(m.returns.shape[1])
    config = stage3_config(m, factors, **overrides)
    register(config)
    return config


class TestMeasured:
    def test_a_pure_factor_exposure_fails_and_orthogonal_alpha_passes(self, ledger):
        m, exposed, alpha = planted()
        factors = factor_frame(m)
        config = registered(m, factors)
        holdout = Holdout(m.returns, config.config_hash)
        card = run3(config, m, factors, holdout).scorecard

        assert card.loc[[exposed, alpha], "s2_pass"].all()  # both look good before stage 3
        assert card.loc[[exposed, alpha], "s3_measured"].all()
        assert not card.loc[exposed, "s3_pass"]
        assert card.loc[exposed, "s3_stat"] > ALPHA3
        assert card.loc[alpha, "s3_pass"]
        assert card.loc[alpha, "s3_stat"] <= ALPHA3
        assert (card["s3_threshold"].dropna() == ALPHA3).all()
        assert card.loc[alpha, "s3_t"] > 3
        assert card.loc[alpha, "s3_alpha"] == pytest.approx(0.004 * gate.TRADING_DAYS, rel=0.3)
        # Stage 4 gets the stage-3 survivors' chosen variants only.
        assert holdout.calls[0][0] == (card.loc[alpha, "best"],)

    def test_untested_families_are_not_marked_measured(self, ledger):
        m, exposed, alpha = planted()
        factors = factor_frame(m)
        card = run3(registered(m, factors), m, factors).scorecard
        untested = ~card["s2_pass"]
        assert untested.any()
        assert not card.loc[untested, "s3_measured"].any()
        assert card.loc[untested, "s3_stat"].isna().all()
        assert not card.loc[untested, "s3_pass"].any()

    def test_loadings_and_header_record_stage_3(self, ledger):
        m, exposed, alpha = planted()
        factors = factor_frame(m)
        config = registered(m, factors)
        result = run3(config, m, factors)
        assert result.header.factors_digest == config.factors_digest
        assert result.header.thresholds.alpha3 == ALPHA3
        assert list(result.loadings.columns) == FF5_MOMENTUM_REGRESSORS
        assert result.loadings.loc[exposed, "Mkt-RF"] == pytest.approx(1.0, abs=0.05)

    def test_stage_3_p_is_adjusted_for_the_family_variant_count(self, ledger):
        # Stage 3 tests the chosen variant's alpha, and the choice by Sharpe also
        # selects on alpha: the raw p is adjusted for the family's k (with the
        # variants' residual correlation), then BH runs across families.
        m, exposed, alpha = planted()
        factors = factor_frame(m)
        search = search_of(m)
        best = pd.Series({exposed: search.columns[(m.family == exposed).to_numpy()][0]})
        one = pd.Series([True], index=[exposed])
        cols, _ = gate._stage3(search, m.family, best, one, factors, ALPHA3)
        raw = float(stats.norm.sf(cols.loc[exposed, "s3_t"]))
        variants = m.family.index[m.family == exposed]
        rho = max(average_correlation(gate._factor_residuals(search[variants], factors)), 0.0)
        expected = search_adjusted_pvalue(pd.Series([raw]), len(variants), rho=rho).iloc[0]
        assert cols.loc[exposed, "s3_stat"] == pytest.approx(expected)
        assert cols.loc[exposed, "s3_stat"] > raw

        # More variants in the family, the same chosen one: the adjusted p grows.
        extra = {
            f"x{i}": np.random.default_rng(i).normal(0.0, 0.01, len(search)) for i in range(20)
        }
        wider = search.assign(**extra)
        family = pd.concat([m.family, pd.Series(exposed, index=list(extra))])
        wide, _ = gate._stage3(wider, family, best, one, factors, ALPHA3)
        assert wide.loc[exposed, "s3_t"] == cols.loc[exposed, "s3_t"]
        assert wide.loc[exposed, "s3_stat"] > cols.loc[exposed, "s3_stat"]

    def test_selection_inflates_the_raw_p_and_the_adjustment_restores_the_size(self):
        # Null families (zero alpha) of variants sharing one factor exposure and
        # correlated noise; the best-Sharpe variant is tested. Its raw p passes
        # alpha3 more often than alpha3; the adjusted p holds the size.
        level = 0.10
        raw, adjusted = [], []
        for seed in range(60):
            returns, family, frame = null_families(seed)
            sharpe = returns.mean() / returns.std()
            best = sharpe.groupby(family).idxmax()
            for fam in best.index:
                only = pd.Series(best.index == fam, index=best.index)
                cols, _ = gate._stage3(returns, family, best, only, frame, level)
                raw.append(stats.norm.sf(cols.loc[fam, "s3_t"]) <= level)
                adjusted.append(cols.loc[fam, "s3_stat"] <= level)
        tol = 3.0 * np.sqrt(level * (1 - level) / len(adjusted))
        assert np.mean(raw) > level + tol, np.mean(raw)
        assert np.mean(adjusted) <= level + tol, np.mean(adjusted)


def null_families(seed: int, n_families=8, k=5, rho=0.5, years=10):
    """Families of k zero-alpha variants with one shared factor exposure each."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("1990-01-01", periods=years * 252)
    frame = pd.DataFrame(
        rng.normal(0.0, 0.005, (len(dates), len(FF5_MOMENTUM_COLUMNS))),
        index=dates,
        columns=FF5_MOMENTUM_COLUMNS,
    )
    cols, labels = {}, {}
    for h in range(n_families):
        exposure = frame[FF5_MOMENTUM_REGRESSORS].to_numpy() @ rng.uniform(-0.6, 0.6, 6)
        shared = rng.standard_normal(len(dates))
        for j in range(k):
            noise = 0.01 * (
                np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal(len(dates))
            )
            cols[f"h{h}_v{j}"] = exposure + noise
            labels[f"h{h}_v{j}"] = f"h{h}"
    return pd.DataFrame(cols, index=dates), pd.Series(labels), frame


class TestRefusals:
    def test_a_digest_mismatch_is_refused_before_any_look(self, ledger):
        m, _, _ = planted()
        factors = factor_frame(m)
        config = registered(m, factors)
        holdout = Holdout(m.returns, config.config_hash)
        other = factors.assign(SMB=factors["SMB"] + 1e-6)
        with pytest.raises(ValueError, match="digest"):
            run3(config, m, other, holdout)
        assert looks(config.config_hash) == [] and holdout.calls == []

    def test_factors_missing_search_dates_are_refused_before_any_look(self, ledger):
        m, _, _ = planted()
        full = factor_frame(m)
        short = full.iloc[:-5]
        config = registered(m, short)  # a consistent digest, but 5 dates short
        with pytest.raises(ValueError, match="lack 5 of the"):
            run3(config, m, short)
        assert looks(config.config_hash) == []

    def test_factors_past_search_end_are_refused(self, ledger):
        m, _, _ = planted()
        dates = pd.bdate_range(m.returns.index[0], m.holdout_end)
        full = factor_frame(m).reindex(dates).fillna(0.0)
        config = registered(m, full)
        with pytest.raises(ValueError, match="search period only"):
            run3(config, m, full)
        assert looks(config.config_hash) == []

    def test_a_registered_frame_must_be_passed(self, ledger):
        m, _, _ = planted()
        config = registered(m, factor_frame(m))
        with pytest.raises(ValueError, match="pass it as factors"):
            run3(config, m, None)
        assert looks(config.config_hash) == []

    @pytest.mark.parametrize("which", ["digest_only", "alpha3_only"])
    def test_a_half_registered_stage_3_is_refused(self, ledger, which):
        m, _, _ = planted()
        factors = factor_frame(m)
        if which == "digest_only":
            overrides = {"thresholds": Thresholds(stage1="bootstrap")}
        else:
            overrides = {"factors_digest": None}
        config = registered(m, factors, **overrides)
        with pytest.raises(ValueError, match="needs both"):
            run3(config, m, factors)
        assert looks(config.config_hash) == []

    def test_a_frame_for_an_unregistered_stage_3_is_refused(self, ledger):
        m, _, _ = planted()
        log_trials(m.returns.shape[1])
        config = config_for(m)
        register(config)
        with pytest.raises(ValueError, match="registers none"):
            run3(config, m, factor_frame(m))
        assert looks(config.config_hash) == []


class TestUnregistered:
    def test_no_factor_frame_passes_stage_2_through_as_before(self, ledger):
        m, _, _ = planted()
        log_trials(m.returns.shape[1])
        config = config_for(m)
        register(config)
        card = run3(config, m, None).scorecard
        assert not card["s3_measured"].any()
        assert card["s3_stat"].isna().all() and card["s3_threshold"].isna().all()
        assert (card["s3_pass"] == card["s2_pass"]).all()

    def test_the_registered_crsp_v1_hash_is_unchanged(self):
        config = gate.load_registered(SCRIPTS / "gate_crsp_v1.json")
        assert config.config_hash == CRSP_V1_HASH
        assert config.factors_digest is None and config.thresholds.alpha3 is None
        assert "alpha3" not in thresholds_dict(config.thresholds)
        assert "alpha3" not in json.dumps(config.to_dict())

    def test_registering_alpha3_changes_the_hash(self):
        m = market(1)
        assert (
            config_for(m, thresholds=Thresholds(stage1="bootstrap", alpha3=0.05)).config_hash
            != config_for(m).config_hash
        )

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, True, "0.05"])
    def test_alpha3_is_a_level_in_the_open_unit_interval(self, bad):
        with pytest.raises((ValueError, TypeError), match="alpha3"):
            Thresholds(alpha3=bad)


def test_the_digest_depends_on_columns_dates_and_values():
    m = market(1)
    f = factor_frame(m)
    assert factors_frame_digest(f) == factors_frame_digest(f.copy())
    assert factors_frame_digest(f) != factors_frame_digest(f.rename(columns={"Mom": "UMD"}))
    assert factors_frame_digest(f) != factors_frame_digest(f.iloc[1:])
    assert factors_frame_digest(f) != factors_frame_digest(f * 1.0001)
    assert len(f) == N_SEARCH
