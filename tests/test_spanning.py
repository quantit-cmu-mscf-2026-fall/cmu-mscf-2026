from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from capstone import spanning
from capstone.data import FF5_MOMENTUM_COLUMNS, FF5_MOMENTUM_REGRESSORS
from capstone.evaluate import (
    bh_adjusted,
    by_adjusted,
    evidence_profile,
    holm_adjusted,
    newey_west_lags,
    sharpe_test,
)
from capstone.spanning import BETA_COLUMNS, MAX_MEDIAN_GAP_DAYS, OUTPUT_COLUMNS, factor_alpha
from capstone.synth import make_return_matrix

N_OBS = 2520
START = "2005-01-03"
LEVEL = 0.05
Z_LEVEL = float(stats.norm.isf(LEVEL))
BAND_Z = 3.5
FACTOR_VOL = 0.008
FACTOR_MEANS = np.array([4e-4, 1e-4, 1e-4, 1.5e-4, 1e-4, 4e-4])
AR1 = 0.3
RHO = 0.3


def _factors(n_obs: int = N_OBS, *, seed: int, start: str = START) -> pd.DataFrame:
    rng = np.random.default_rng([seed, 101])
    dates = pd.bdate_range(start, periods=n_obs)
    regressors = FACTOR_MEANS + FACTOR_VOL * rng.standard_normal((n_obs, len(FACTOR_MEANS)))
    frame = pd.DataFrame(regressors, index=dates, columns=FF5_MOMENTUM_REGRESSORS)
    frame["RF"] = 2e-4 + 5e-5 * rng.standard_normal(n_obs)
    return frame[FF5_MOMENTUM_COLUMNS]


def _betas(n_candidates: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 202])
    return rng.uniform(0.2, 1.2, size=(len(FF5_MOMENTUM_REGRESSORS), n_candidates))


def _exposed(factors: pd.DataFrame, noise: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    exposure = factors[FF5_MOMENTUM_REGRESSORS].to_numpy() @ _betas(noise.shape[1], seed=seed)
    return noise.set_axis(factors.index) + exposure


def _reference_fit(y: pd.Series, factors: pd.DataFrame, **fit_kwargs):
    design = sm.add_constant(factors.loc[y.index, FF5_MOMENTUM_REGRESSORS].to_numpy())
    return sm.OLS(y.to_numpy(), design).fit(**fit_kwargs)


def _bartlett_target_lrv_ar1(phi: float, lags: int) -> float:
    return 1.0 + 2.0 * sum((1 - j / (lags + 1)) * phi**j for j in range(1, lags + 1))


def _ar1_lrv(phi: float) -> float:
    return (1.0 + phi) / (1.0 - phi)


def _rate_band(expected: float, n_sims: int, n_per_sim: int, pair_corr: float = 0.0) -> float:
    p = expected
    both = 0.0
    if pair_corr:
        z = float(stats.norm.isf(p))
        cov = [[1.0, pair_corr], [pair_corr, 1.0]]
        both = float(stats.multivariate_normal(mean=[0, 0], cov=cov).cdf([-z, -z]))
    c = both - p**2 if pair_corr else 0.0
    variance = (p * (1 - p) + (n_per_sim - 1) * c) / (n_per_sim * n_sims)
    return BAND_Z * float(np.sqrt(variance))


NULL_SCENARIOS = {
    "iid": dict(n_sims=20, n_candidates=100, noise={}),
    "ar1": dict(n_sims=20, n_candidates=100, noise={"ar1": AR1}),
    "rho": dict(n_sims=200, n_candidates=10, noise={"rho": RHO}),
}


@functools.cache
def _null_study(scenario: str) -> SimpleNamespace:
    spec = NULL_SCENARIOS[scenario]
    hac_reject, naive_reject, se_ratio, raw_sharpe_reject = [], [], [], []
    for seed in range(spec["n_sims"]):
        factors = _factors(seed=seed)
        noise = make_return_matrix(
            n_obs=N_OBS, n_candidates=spec["n_candidates"], seed=seed, **spec["noise"]
        ).returns
        returns = _exposed(factors, noise, seed=seed)
        result = factor_alpha(returns, factors, returns_are_excess=True)
        hac_reject.append(result["pvalue_greater"].to_numpy() < LEVEL)
        for name in returns.columns:
            naive = _reference_fit(returns[name], factors)
            naive_reject.append(naive.tvalues[0] > Z_LEVEL)
            se_ratio.append(result.at[name, "se"] / naive.bse[0])
            if seed == 0:
                raw_sharpe_reject.append(sharpe_test(returns[name]).pvalue_greater < LEVEL)
    return SimpleNamespace(
        hac=float(np.concatenate(hac_reject).mean()),
        naive=float(np.mean(naive_reject)),
        se_ratio=float(np.mean(se_ratio)),
        raw_sharpe=float(np.mean(raw_sharpe_reject)),
        n_sims=spec["n_sims"],
        n_per_sim=spec["n_candidates"],
    )


class TestNullCalibration:
    """Factor exposure, no alpha: the test rejects at its nominal level.

    IID and rho null: expected rate 0.05. AR(1) null: the Bartlett estimator
    at L = newey_west_lags(T) targets less than the true long-run variance, so
    the expected rate is P(Z > z_0.05 sqrt(LRV_Bartlett / LRV_true)), about
    0.057 at phi = 0.3, T = 2520 (Newey & West 1987, Bartlett weights 1 - j/(L+1)).
    """

    def test_iid_null_rejects_at_the_nominal_level(self):
        study = _null_study("iid")
        band = _rate_band(LEVEL, study.n_sims, study.n_per_sim)
        assert abs(study.hac - LEVEL) <= band

    def test_ar1_null_rejects_near_the_level_the_lag_rule_implies(self):
        study = _null_study("ar1")
        lags = newey_west_lags(N_OBS)
        expected = float(
            stats.norm.sf(Z_LEVEL * np.sqrt(_bartlett_target_lrv_ar1(AR1, lags) / _ar1_lrv(AR1)))
        )
        band = _rate_band(expected, study.n_sims, study.n_per_sim)
        assert abs(study.hac - expected) <= band

    def test_correlated_null_rejects_at_the_nominal_level(self):
        study = _null_study("rho")
        band = _rate_band(LEVEL, study.n_sims, study.n_per_sim, pair_corr=RHO)
        assert abs(study.hac - LEVEL) <= band

    def test_factor_premia_alone_pass_a_raw_sharpe_test_but_not_the_alpha_test(self):
        study = _null_study("iid")
        assert study.raw_sharpe > 0.5
        assert study.hac < 0.1


class TestHacInference:
    def test_naive_ols_over_rejects_ar1_nulls_and_hac_does_not(self):
        study = _null_study("ar1")
        naive_expected = float(stats.norm.sf(Z_LEVEL / np.sqrt(_ar1_lrv(AR1))))
        naive_band = _rate_band(naive_expected, study.n_sims, study.n_per_sim)
        assert abs(study.naive - naive_expected) <= naive_band
        assert study.naive - study.hac > naive_band

    def test_hac_widens_the_se_by_the_bartlett_long_run_factor_under_ar1(self):
        lags = newey_west_lags(N_OBS)
        assert _null_study("ar1").se_ratio == pytest.approx(
            np.sqrt(_bartlett_target_lrv_ar1(AR1, lags)), abs=0.03
        )
        assert _null_study("iid").se_ratio == pytest.approx(1.0, abs=0.03)

    def test_default_lags_follow_newey_west_lags_per_candidate(self):
        factors = _factors(seed=3)
        noise = make_return_matrix(n_obs=N_OBS, n_candidates=3, seed=3, ar1=AR1).returns
        returns = _exposed(factors, noise, seed=3)
        returns.iloc[:1520, 1] = np.nan
        returns.iloc[:2220, 2] = np.nan
        result = factor_alpha(returns, factors, returns_are_excess=True)
        for name in returns.columns:
            y = returns[name].dropna()
            lags = newey_west_lags(len(y))
            reference = _reference_fit(
                y, factors, cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=False
            )
            row = result.loc[name]
            assert row["lags"] == lags
            assert row["alpha"] == pytest.approx(reference.params[0], rel=1e-10)
            assert row["se"] == pytest.approx(reference.bse[0], rel=1e-10)
            assert row["pvalue_greater"] == pytest.approx(
                stats.norm.sf(reference.params[0] / reference.bse[0]), rel=1e-10
            )
        assert result["lags"].tolist() == [8, 6, 5]

    def test_explicit_lags_override_the_default_and_zero_is_white(self):
        factors = _factors(600, seed=4)
        noise = make_return_matrix(n_obs=600, n_candidates=1, seed=4).returns
        returns = _exposed(factors, noise, seed=4)
        white = _reference_fit(returns.iloc[:, 0], factors, cov_type="HC0")
        result = factor_alpha(returns, factors, returns_are_excess=True, lags=0)
        assert result["lags"].iloc[0] == 0
        assert result["se"].iloc[0] == pytest.approx(white.bse[0], rel=1e-10)
        three = factor_alpha(returns, factors, returns_are_excess=True, lags=3)
        assert three["lags"].iloc[0] == 3

    @pytest.mark.parametrize("lags", [-1, 1.5, True, "3"])
    def test_invalid_lags_are_refused(self, lags):
        factors = _factors(100, seed=5)
        returns = pd.DataFrame({"c": np.zeros(100)}, index=factors.index)
        with pytest.raises(ValueError, match="lags"):
            factor_alpha(returns, factors, returns_are_excess=True, lags=lags)

    def test_lags_not_below_n_obs_give_nan_for_that_candidate_only(self):
        factors = _factors(300, seed=6)
        noise = make_return_matrix(n_obs=300, n_candidates=2, seed=6).returns
        returns = _exposed(factors, noise, seed=6)
        returns.iloc[10:, 0] = np.nan
        result = factor_alpha(returns, factors, returns_are_excess=True, lags=10)
        assert result.iloc[0][["alpha", "se", "pvalue_greater"]].isna().all()
        assert np.isfinite(result.iloc[1]["pvalue_greater"])
        assert np.isfinite(
            factor_alpha(returns, factors, returns_are_excess=True, lags=9).iloc[0]["se"]
        )


def _power(n_obs: int, ar1: float, lags: int) -> float:
    delta = np.sqrt(n_obs / 252)
    bartlett = _bartlett_target_lrv_ar1(ar1, lags)
    shift = (delta / np.sqrt(bartlett) - Z_LEVEL) / np.sqrt(_ar1_lrv(ar1) / bartlett)
    return float(stats.norm.cdf(shift))


class TestPower:
    """Planted alpha of annual Sharpe 1.0 against the residual risk, on top of factor exposure.

    Over ten years that is a mean t of sqrt(10) = 3.16, so the expected power at 5%
    is 0.93 for IID residuals and about 0.77 at AR(1) 0.3 (HAC t-statistic,
    normal approximation).
    """

    @pytest.mark.parametrize("ar1", [0.0, AR1], ids=["iid", "ar1"])
    def test_planted_alpha_is_found(self, ar1):
        found = []
        for seed in range(2):
            factors = _factors(seed=100 + seed)
            noise = make_return_matrix(
                n_obs=N_OBS, n_candidates=200, n_real=200, sharpe_real=1.0, ar1=ar1, seed=100 + seed
            ).returns
            result = factor_alpha(
                _exposed(factors, noise, seed=100 + seed), factors, returns_are_excess=True
            )
            found.append(result["pvalue_greater"].to_numpy() < LEVEL)
        expected = _power(N_OBS, ar1, newey_west_lags(N_OBS))
        assert np.concatenate(found).mean() >= expected - _rate_band(expected, 2, 200)

    def test_planted_alpha_is_estimated_without_bias(self):
        factors = _factors(seed=110)
        vol = 0.01
        alpha = 1.0 / np.sqrt(252) * vol
        noise = make_return_matrix(
            n_obs=N_OBS, n_candidates=200, n_real=200, sharpe_real=1.0, vol=vol, seed=110
        ).returns
        result = factor_alpha(_exposed(factors, noise, seed=110), factors, returns_are_excess=True)
        standard_error = vol / np.sqrt(N_OBS) / np.sqrt(200)
        assert result["alpha"].mean() == pytest.approx(alpha, abs=BAND_Z * standard_error)


def _small_case(n_obs: int = 400, n_candidates: int = 3, *, seed: int = 7):
    factors = _factors(n_obs, seed=seed)
    noise = make_return_matrix(n_obs=n_obs, n_candidates=n_candidates, seed=seed).returns
    return _exposed(factors, noise, seed=seed), factors


class TestOutputContract:
    def test_rows_are_the_candidates_in_order_with_the_documented_columns(self):
        returns, factors = _small_case()
        returns = returns[["cand_0002", "cand_0000", "cand_0001"]]
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert list(result.columns) == OUTPUT_COLUMNS
        assert BETA_COLUMNS == [f"beta_{name}" for name in FF5_MOMENTUM_REGRESSORS]
        assert result.index.equals(returns.columns)
        assert result.index.name == "candidate"
        assert result["n_obs"].dtype == np.int64
        assert result["n_dropped"].dtype == np.int64
        assert str(result["lags"].dtype) == "Int64"

    def test_betas_recover_the_planted_exposures(self):
        factors = _factors(seed=8)
        noise = make_return_matrix(n_obs=N_OBS, n_candidates=4, seed=8).returns
        result = factor_alpha(_exposed(factors, noise, seed=8), factors, returns_are_excess=True)
        np.testing.assert_allclose(result[BETA_COLUMNS].to_numpy().T, _betas(4, seed=8), atol=0.1)

    def test_pvalue_greater_feeds_the_multiple_testing_helpers(self):
        returns, factors = _small_case(n_candidates=6)
        returns.iloc[5:, 0] = np.nan
        pvalues = factor_alpha(returns, factors, returns_are_excess=True)["pvalue_greater"]
        assert isinstance(pvalues, pd.Series)
        assert pvalues.dtype == float
        assert np.isnan(pvalues.iloc[0])
        for adjusted in (holm_adjusted(pvalues), by_adjusted(pvalues), bh_adjusted(pvalues)):
            assert adjusted.index.equals(pvalues.index)
            assert np.isnan(adjusted.iloc[0])
            valid = adjusted.iloc[1:]
            assert ((valid >= pvalues.iloc[1:] - 1e-12) & (valid <= 1)).all()
        profile = evidence_profile(pvalues)
        assert set(profile.index) == set(pvalues.index)

    def test_an_empty_candidate_set_returns_an_empty_frame(self):
        _, factors = _small_case()
        empty = pd.DataFrame(index=factors.index, columns=pd.Index([], name="candidate"))
        result = factor_alpha(empty.astype(float), factors, returns_are_excess=True)
        assert result.empty
        assert list(result.columns) == OUTPUT_COLUMNS


class TestRiskFreeRate:
    def test_rf_is_subtracted_once_when_returns_are_not_excess(self):
        returns, factors = _small_case()
        raw = returns.add(factors["RF"], axis=0)
        from_raw = factor_alpha(raw, factors, returns_are_excess=False)
        from_excess = factor_alpha(returns, factors, returns_are_excess=True)
        pd.testing.assert_frame_equal(from_raw, from_excess, rtol=1e-9, atol=1e-12)

    def test_returns_are_left_alone_when_they_are_excess(self):
        returns, factors = _small_case()
        result = factor_alpha(returns, factors, returns_are_excess=True)
        reference = _reference_fit(returns.iloc[:, 0], factors)
        assert result["alpha"].iloc[0] == pytest.approx(reference.params[0], rel=1e-10)
        shifted = factor_alpha(returns.add(factors["RF"], axis=0), factors, returns_are_excess=True)
        assert (shifted["alpha"] - result["alpha"]).to_numpy() == pytest.approx(
            np.full(3, factors["RF"].mean()), rel=0.2
        )

    def test_rf_is_never_a_regressor(self):
        returns, factors = _small_case()
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert "beta_RF" not in result.columns
        scrambled = factors.assign(RF=np.random.default_rng(9).normal(0, 0.05, len(factors)))
        pd.testing.assert_frame_equal(
            factor_alpha(returns, scrambled, returns_are_excess=True), result
        )
        no_rf = factor_alpha(returns, factors.drop(columns="RF"), returns_are_excess=True)
        pd.testing.assert_frame_equal(no_rf, result)

    def test_rf_is_required_only_when_it_must_be_subtracted(self):
        returns, factors = _small_case()
        with pytest.raises(ValueError, match="RF"):
            factor_alpha(returns, factors.drop(columns="RF"), returns_are_excess=False)

    @pytest.mark.parametrize("flag", [None, 1, "yes"])
    def test_the_excess_flag_must_be_a_bool(self, flag):
        returns, factors = _small_case()
        with pytest.raises(TypeError, match="returns_are_excess"):
            factor_alpha(returns, factors, returns_are_excess=flag)


class TestAlignment:
    def _case(self):
        returns, factors = _small_case(600, 3, seed=11)
        dates = factors.index
        factors = factors.drop(dates[[20, 21, 22, 300, 301]])
        factors.loc[dates[[50, 51, 52]], "HML"] = np.nan
        factors.loc[dates[[80, 81]], "RF"] = np.nan
        returns.iloc[[20, 50, 100, 101, 102, 103], 0] = np.nan
        returns.iloc[:, 2] = np.nan
        return returns, factors

    @pytest.mark.parametrize("excess", [True, False])
    def test_counts_follow_each_candidates_own_dates(self, excess):
        returns, factors = self._case()
        needed = FF5_MOMENTUM_REGRESSORS + ([] if excess else ["RF"])
        complete = set(factors[needed].dropna().index)
        result = factor_alpha(returns, factors, returns_are_excess=excess)
        for name in returns.columns:
            available = set(returns[name].dropna().index)
            used = available & complete
            assert result.at[name, "n_obs"] == len(used)
            assert result.at[name, "n_dropped"] == len(available - used)
        assert result.loc["cand_0000", "n_dropped"] == (6 if excess else 8)
        assert result.loc["cand_0001", "n_dropped"] == (8 if excess else 10)

    def test_estimates_use_exactly_the_aligned_dates(self):
        returns, factors = self._case()
        result = factor_alpha(returns, factors, returns_are_excess=True)
        y = returns["cand_0000"].dropna()
        y = y[y.index.isin(factors[FF5_MOMENTUM_REGRESSORS].dropna().index)]
        reference = _reference_fit(
            y,
            factors,
            cov_type="HAC",
            cov_kwds={"maxlags": newey_west_lags(len(y))},
            use_t=False,
        )
        assert result.at["cand_0000", "alpha"] == pytest.approx(reference.params[0], rel=1e-10)
        assert result.at["cand_0000", "se"] == pytest.approx(reference.bse[0], rel=1e-10)

    def test_an_all_missing_candidate_is_nan_with_zero_counts(self):
        returns, factors = self._case()
        row = factor_alpha(returns, factors, returns_are_excess=True).loc["cand_0002"]
        assert row["n_obs"] == 0 and row["n_dropped"] == 0
        assert pd.isna(row["lags"])
        assert row[["alpha", "se", "tstat", "pvalue_greater", *BETA_COLUMNS]].isna().all()

    def test_factor_dates_beyond_the_candidates_are_ignored(self):
        returns, factors = _small_case()
        longer = _factors(len(factors) + 50, seed=7)
        longer.iloc[: len(factors)] = factors.to_numpy()
        pd.testing.assert_frame_equal(
            factor_alpha(returns, longer, returns_are_excess=True),
            factor_alpha(returns, factors, returns_are_excess=True),
        )


class TestSearchPeriod:
    def _frames(self, end: str):
        dates = pd.bdate_range(end=end, periods=300)
        factors = _factors(300, seed=12).set_axis(dates)
        returns = pd.DataFrame({"c": np.random.default_rng(12).normal(0, 0.01, 300)}, index=dates)
        return returns, factors

    def test_the_last_search_day_is_accepted(self):
        returns, factors = self._frames("2020-12-31")
        factor_alpha(returns, factors, returns_are_excess=True)

    @pytest.mark.parametrize("end", ["2021-01-01", "2021-01-04", "2025-12-31"])
    def test_holdout_dates_are_refused(self, end):
        returns, factors = self._frames(end)
        with pytest.raises(ValueError, match="holdout"):
            factor_alpha(returns, factors, returns_are_excess=True)

    def test_a_holdout_row_is_refused_even_when_it_is_empty(self):
        returns, factors = self._frames("2021-01-04")
        returns.loc[returns.index >= "2021-01-01"] = np.nan
        with pytest.raises(ValueError, match="holdout"):
            factor_alpha(returns, factors, returns_are_excess=True)

    def test_factors_reaching_into_the_holdout_are_fine(self):
        returns, factors = self._frames("2020-12-31")
        later = pd.bdate_range("2021-01-04", periods=20)
        extended = pd.concat([factors, _factors(20, seed=13).set_axis(later)])
        factor_alpha(returns, extended, returns_are_excess=True)


def _on_index(index: pd.DatetimeIndex, *, seed: int = 14):
    factors = _factors(len(index), seed=seed).set_axis(index)
    noise = make_return_matrix(n_obs=len(index), n_candidates=2, seed=seed).returns
    return _exposed(factors, noise, seed=seed), factors


class TestDailyFrequency:
    def test_holidays_and_missing_sessions_are_accepted(self):
        dates = pd.bdate_range("2010-01-04", "2014-12-31")
        holidays = dates[(dates.month == 1) & (dates.day == 1)].union(
            dates[(dates.month == 12) & (dates.day == 25)]
        )
        skipped = pd.DatetimeIndex(np.random.default_rng(15).choice(dates, 60, replace=False))
        index = dates.difference(holidays.union(skipped))
        returns, factors = _on_index(index)
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert result["pvalue_greater"].notna().all()

    def test_a_median_gap_at_the_limit_is_accepted(self):
        returns, factors = _on_index(pd.date_range("2000-01-03", periods=400, freq="4D"))
        assert MAX_MEDIAN_GAP_DAYS == 4
        factor_alpha(returns, factors, returns_are_excess=True)

    @pytest.mark.parametrize("freq", ["5D", "W-FRI", "BME"])
    def test_sparser_returns_are_refused(self, freq):
        returns, factors = _on_index(pd.date_range("2000-01-03", periods=200, freq=freq))
        daily = _factors(seed=16, start="1999-01-04")
        with pytest.raises(ValueError, match="returns.*not daily"):
            factor_alpha(returns, daily, returns_are_excess=True)

    def test_monthly_factors_are_refused(self):
        returns, _ = _small_case()
        _, monthly = _on_index(pd.date_range("1990-01-31", periods=200, freq="BME"))
        with pytest.raises(ValueError, match="factors.*not daily"):
            factor_alpha(returns, monthly, returns_are_excess=True)

    def test_intraday_stamps_are_refused(self):
        returns, factors = _on_index(pd.date_range("2010-01-04", periods=300, freq="12h"))
        with pytest.raises(ValueError, match="less than one day"):
            factor_alpha(returns, factors, returns_are_excess=True)


def _with_value(frame: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    changed = frame.copy()
    changed.iloc[5, changed.columns.get_loc(column)] = value
    return changed


def _breaks(returns: pd.DataFrame, factors: pd.DataFrame):
    return {
        "non-datetime index": (returns.reset_index(drop=True), factors, "DatetimeIndex"),
        "duplicate dates": (
            pd.concat([returns.iloc[:1], returns]),
            factors,
            "duplicate dates",
        ),
        "decreasing dates": (returns.iloc[::-1], factors, "increasing"),
        "single date": (returns.iloc[:1], factors, "two dates"),
        "duplicate candidates": (
            returns.set_axis(["a", "a", "b"], axis=1),
            factors,
            "duplicate candidate",
        ),
        "missing factor": (returns, factors.drop(columns="Mom"), r"missing columns \['Mom'\]"),
        "duplicate factor": (
            returns,
            pd.concat([factors, factors[["SMB"]]], axis=1),
            "duplicate column",
        ),
        "text returns": (returns.astype(str), factors, "non-numeric"),
        "bool returns": (returns > 0, factors, "non-numeric"),
        "infinite return": (_with_value(returns, "cand_0000", np.inf), factors, "infinite"),
        "infinite factor": (returns, _with_value(factors, "HML", -np.inf), "infinite"),
        "series not frame": (returns.iloc[:, 0], factors, "DataFrame"),
        "tz-aware returns": (returns.tz_localize("UTC"), factors, "timezone"),
        "tz-aware factors": (returns, factors.tz_localize("UTC"), "timezone"),
        "both tz-aware": (returns.tz_localize("UTC"), factors.tz_localize("UTC"), "timezone"),
        "no shared dates": (
            returns.set_axis(returns.index + pd.Timedelta(hours=16)),
            factors,
            "share no dates",
        ),
    }


class TestStructuralInput:
    @pytest.mark.parametrize(
        "case",
        [
            "non-datetime index",
            "duplicate dates",
            "decreasing dates",
            "single date",
            "duplicate candidates",
            "missing factor",
            "duplicate factor",
            "text returns",
            "bool returns",
            "infinite return",
            "infinite factor",
            "series not frame",
            "tz-aware returns",
            "tz-aware factors",
            "both tz-aware",
            "no shared dates",
        ],
    )
    def test_is_refused(self, case):
        returns, factors = _small_case()
        bad_returns, bad_factors, message = _breaks(returns, factors)[case]
        with pytest.raises(ValueError, match=message):
            factor_alpha(bad_returns, bad_factors, returns_are_excess=True)

    def test_extra_factor_columns_are_ignored(self):
        returns, factors = _small_case()
        extra = factors.assign(Other=np.random.default_rng(17).normal(size=len(factors)))
        pd.testing.assert_frame_equal(
            factor_alpha(returns, extra, returns_are_excess=True),
            factor_alpha(returns, factors, returns_are_excess=True),
        )


STATISTICS = ["alpha", "se", "tstat", "pvalue_greater", *BETA_COLUMNS]


class TestUnestimableCandidates:
    def test_seven_observations_are_too_few_and_eight_are_enough(self):
        returns, factors = _small_case(400, 3, seed=18)
        returns.iloc[7:, 0] = np.nan
        returns.iloc[8:, 1] = np.nan
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert result.iloc[0][STATISTICS].isna().all()
        assert result.iloc[0]["n_obs"] == 7
        assert result.iloc[0]["lags"] == newey_west_lags(7)
        assert result.iloc[1][STATISTICS].notna().all()
        assert result.iloc[2][STATISTICS].notna().all()

    def test_a_factor_constant_over_one_candidates_sample_is_rank_deficient(self):
        returns, factors = _small_case(400, 2, seed=19)
        factors.iloc[100:200, factors.columns.get_loc("CMA")] = 0.0
        returns.iloc[np.r_[0:100, 200:400], 0] = np.nan
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert result.iloc[0]["n_obs"] == 100
        assert result.iloc[0][STATISTICS].isna().all()
        assert result.iloc[1][STATISTICS].notna().all()

    def test_collinear_factors_give_nan_for_every_candidate_without_raising(self):
        returns, factors = _small_case(seed=20)
        collinear = factors.assign(HML=2.0 * factors["SMB"])
        result = factor_alpha(returns, collinear, returns_are_excess=True)
        assert result[STATISTICS].isna().all().all()
        assert (result["n_obs"] == len(factors)).all()

    @pytest.mark.parametrize("shift", [0.0, 3e-4], ids=["copy", "copy-plus-constant"])
    def test_an_exact_factor_mixture_is_degenerate(self, shift):
        returns, factors = _small_case(seed=21)
        exact = factors[FF5_MOMENTUM_REGRESSORS].to_numpy() @ _betas(1, seed=21)[:, 0] + shift
        returns["exact"] = exact
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert result.loc["exact", STATISTICS].isna().all()
        assert result.drop(index="exact")[STATISTICS].notna().all().all()

    def test_a_constant_candidate_is_degenerate(self):
        returns, factors = _small_case(seed=22)
        returns["flat"] = 1e-3
        returns["zero"] = 0.0
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert result.loc[["flat", "zero"], STATISTICS].isna().all().all()

    @pytest.mark.parametrize("bse", [np.nan, np.inf, 0.0, -1.0])
    def test_a_non_finite_or_non_positive_se_gives_nan(self, monkeypatch, bse):
        real_ols = sm.OLS

        class FakeOLS:
            def __init__(self, y, design):
                self._model = real_ols(y, design)

            def fit(self, **kwargs):
                fit = self._model.fit(**kwargs)
                return SimpleNamespace(params=fit.params, ssr=fit.ssr, bse=np.r_[bse, fit.bse[1:]])

        monkeypatch.setattr(
            spanning, "sm", SimpleNamespace(OLS=FakeOLS, add_constant=sm.add_constant)
        )
        returns, factors = _small_case(seed=23)
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert result[STATISTICS].isna().all().all()
        assert (result["n_obs"] == len(factors)).all()


class TestDegenerateCutoff:
    @pytest.mark.parametrize("noise_scale", [1e-6, 1e-8])
    def test_small_but_real_noise_is_estimated(self, noise_scale):
        factors = _factors(1000, seed=24)
        noise = make_return_matrix(n_obs=1000, n_candidates=2, vol=noise_scale, seed=24).returns
        returns = _exposed(factors, noise, seed=24)
        result = factor_alpha(returns, factors, returns_are_excess=True)
        assert result[STATISTICS].notna().all().all()
        reference = _reference_fit(
            returns.iloc[:, 0],
            factors,
            cov_type="HAC",
            cov_kwds={"maxlags": newey_west_lags(1000)},
            use_t=False,
        )
        assert result.iloc[0]["tstat"] == pytest.approx(reference.tvalues[0], rel=1e-6)

    def test_rescaling_the_candidate_does_not_change_the_verdict(self):
        factors = _factors(1000, seed=25)
        noise = make_return_matrix(n_obs=1000, n_candidates=2, vol=1e-6, seed=25).returns
        returns = _exposed(factors, noise, seed=25)
        base = factor_alpha(returns, factors, returns_are_excess=True)
        tiny = factor_alpha(returns * 1e-4, factors, returns_are_excess=True)
        np.testing.assert_allclose(tiny["tstat"], base["tstat"], rtol=1e-6)
        np.testing.assert_allclose(tiny["pvalue_greater"], base["pvalue_greater"], rtol=1e-6)
