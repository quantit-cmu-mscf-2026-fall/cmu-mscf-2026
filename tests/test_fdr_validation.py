import numpy as np
import pandas as pd
import pytest

from capstone.fdr_validation import (
    fdr_stepup,
    strategy_pvalues,
    summarize,
    validate_strategies,
)


def _noise(rng, n_obs, n_strategies, sigma=0.01):
    """Signal-free returns: every column is a true null."""
    return pd.DataFrame(
        rng.normal(0.0, sigma, (n_obs, n_strategies)),
        columns=[f"noise_{i}" for i in range(n_strategies)],
    )


def test_bh_and_by_select_expected_strategies():
    pvals = np.array([0.003, 0.018, 0.027, 0.041, 0.120])

    bh_reject, bh_adjusted, bh_cutoff = fdr_stepup(pvals, q=0.05, method="bh")
    by_reject, by_adjusted, by_cutoff = fdr_stepup(pvals, q=0.05, method="by")

    assert bh_reject.tolist() == [True, True, True, False, False]
    assert by_reject.tolist() == [True, False, False, False, False]
    assert bh_cutoff == 0.027
    assert by_cutoff == 0.003
    assert (bh_adjusted <= 0.05).tolist() == bh_reject.tolist()
    assert (by_adjusted <= 0.05).tolist() == by_reject.tolist()


def test_bh_uses_largest_passing_rank():
    # Rank 2 fails its threshold, but rank 3 passes.
    pvals = [0.009, 0.025, 0.028, 0.20, 0.30]

    reject, _, cutoff = fdr_stepup(pvals, q=0.05, method="bh")

    assert reject.tolist() == [True, True, True, False, False]
    assert cutoff == 0.028


# --------------------------------------------------------------------------- #
# A constant-return strategy has no risk, so it has no test
# --------------------------------------------------------------------------- #
def test_constant_strategy_is_not_discovered_with_hac():
    # A cash sleeve: positive mean, zero variance. Flooring the long-run
    # variance used to turn this into a t-stat of ~3e6 and p_raw of exactly 0,
    # which every correction then "discovered".
    rng = np.random.default_rng(0)
    returns = _noise(rng, 1000, 20)
    returns["cash"] = 0.0001

    res = validate_strategies(returns, q=0.05)

    assert np.isnan(res.loc["cash", "p_raw"])
    assert not res.loc["cash", "reject_bh"]
    assert not res.loc["cash", "reject_by"]
    assert np.isnan(res.loc["cash", "p_bh"])
    assert np.isnan(res.loc["cash", "p_by"])


def test_constant_strategy_is_not_discovered_without_hac():
    # The hac=False path had the same hole for a subtler reason: the sample std
    # of a constant series is floating-point noise (~1e-20), not exactly 0, so
    # an `sd > 0` guard let it through.
    rng = np.random.default_rng(1)
    returns = _noise(rng, 500, 5)
    returns["cash"] = 0.0001

    assert returns["cash"].std(ddof=1) > 0  # the old guard would have passed
    pvals = strategy_pvalues(returns, hac=False)

    assert np.isnan(pvals["cash"])
    assert pvals.drop("cash").notna().all()


def test_near_constant_strategy_is_not_discovered():
    # Variation at the float-epsilon scale is rounding, not risk.
    rng = np.random.default_rng(2)
    returns = _noise(rng, 300, 3)
    returns["carry"] = 0.0002 + rng.normal(0.0, 1e-19, 300)

    assert np.isnan(strategy_pvalues(returns)["carry"])
    assert np.isnan(strategy_pvalues(returns, hac=False)["carry"])


def test_genuine_low_volatility_strategy_is_still_tested():
    # The degeneracy guard must not swallow a real strategy that simply has a
    # very high Sharpe: tiny vol is not zero vol.
    rng = np.random.default_rng(3)
    returns = _noise(rng, 500, 3)
    returns["smooth"] = rng.normal(0.0005, 1e-5, 500)

    p = strategy_pvalues(returns)["smooth"]

    assert not np.isnan(p)
    assert p < 1e-6


# --------------------------------------------------------------------------- #
# A failed backtest is still a trial: it counts toward m
# --------------------------------------------------------------------------- #
def test_fdr_stepup_counts_nan_toward_family_size():
    # One usable p-value of 0.01 among ten trials. BH's rank-1 threshold is
    # q * 1/10 = 0.005, so it must NOT be rejected. Dropping the NaNs would
    # make m = 1, a threshold of 0.05, and a false discovery.
    pvals = [0.01] + [np.nan] * 9

    reject, p_adj, cutoff = fdr_stepup(pvals, q=0.05, method="bh")

    assert reject.tolist() == [False] * 10
    assert np.isnan(cutoff)
    assert p_adj[0] == pytest.approx(0.1)  # 0.01 * m / rank = 0.01 * 10 / 1
    assert np.isnan(p_adj[1:]).all()


def test_fdr_stepup_nan_is_never_rejected_but_shrinks_power():
    # Thresholds are rank / (m * c_m) * q, so m decides how much evidence it
    # takes. At m = 3 the rank-2 threshold is 0.0333 and 0.012 clears it; at
    # m = 10 it is 0.010 and the same p-value does not.
    usable = [0.004, 0.012, 0.2]
    padded = usable + [np.nan] * 7

    reject_small, _, _ = fdr_stepup(usable, q=0.05, method="bh")
    reject_large, _, _ = fdr_stepup(padded, q=0.05, method="bh")

    assert reject_small.tolist() == [True, True, False]
    assert reject_large.tolist() == [True, False, False] + [False] * 7
    assert not reject_large[3:].any()  # the NaN trials themselves never reject


def test_fdr_stepup_all_nan_rejects_nothing():
    reject, p_adj, cutoff = fdr_stepup([np.nan, np.nan, np.nan], q=0.05, method="by")

    assert not reject.any()
    assert np.isnan(p_adj).all()
    assert np.isnan(cutoff)


def test_validate_strategies_keeps_untestable_strategies_in_m():
    # Twelve trials: ten testable, one that returned nothing, one that stopped
    # after five days. m must be 12, not 10.
    rng = np.random.default_rng(4)
    returns = _noise(rng, 1000, 10)
    returns["crashed"] = np.nan
    returns["too_short"] = [0.01] * 5 + [np.nan] * 995

    res = validate_strategies(returns, q=0.05)

    assert res.attrs["m"] == 12
    assert res.attrs["n_usable"] == 10
    assert len(res) == 12
    for name in ("crashed", "too_short"):
        assert np.isnan(res.loc[name, "p_raw"])
        assert not res.loc[name, "reject_bh"]
        assert not res.loc[name, "reject_by"]
    assert "Untestable (no p-value, still counted in m): 2" in summarize(res)


def test_validate_strategies_raises_when_nothing_is_testable():
    returns = pd.DataFrame({"a": [np.nan] * 20, "b": [0.01] * 3 + [np.nan] * 17})

    with pytest.raises(ValueError, match="No valid p-values"):
        validate_strategies(returns)


# --------------------------------------------------------------------------- #
# Signal-free data: the procedure has to be able to say "nothing here"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hac", [True, False])
def test_null_calibration_on_signal_free_returns(hac):
    # CLAUDE.md: run validation methods on signal-free data before trusting
    # them. 200 independent families of 50 true nulls, T = 500, IID returns.
    # Nominal family-wise rejection rate at q = 0.05 is 0.05.
    rng = np.random.default_rng(12345)
    reps = 200
    bh_hits = 0
    by_hits = 0
    for _ in range(reps):
        res = validate_strategies(_noise(rng, 500, 50), q=0.05, hac=hac)
        bh_hits += bool(res["reject_bh"].any())
        by_hits += bool(res["reject_by"].any())

    bh_rate = bh_hits / reps
    by_rate = by_hits / reps

    # The point of the test: it mostly finds nothing, and it is in the right
    # neighbourhood of the nominal level rather than wildly liberal.
    assert bh_rate <= 0.12, f"BH family-wise rejection rate {bh_rate} on pure noise"
    assert by_rate <= bh_rate, "BY must be no more liberal than BH"
    assert bh_hits < reps // 2, "a procedure that cannot return 'nothing here'"


def test_single_null_family_mostly_finds_nothing():
    rng = np.random.default_rng(7)
    res = validate_strategies(_noise(rng, 1000, 100), q=0.05)

    assert res.attrs["m"] == 100
    assert res.attrs["n_reject_bh"] == 0
    assert res.attrs["n_reject_by"] == 0
    assert np.isnan(res.attrs["threshold_bh"])


# --------------------------------------------------------------------------- #
# strategy_pvalues and validate_strategies, the parts tests never reached
# --------------------------------------------------------------------------- #
def test_strategy_pvalues_flags_short_series():
    rng = np.random.default_rng(8)
    returns = _noise(rng, 50, 2)
    returns["nine_obs"] = [0.01] * 9 + [np.nan] * 41
    returns["ten_obs"] = [0.01, -0.02] * 5 + [np.nan] * 40

    pvals = strategy_pvalues(returns)

    assert np.isnan(pvals["nine_obs"])  # T < 10
    assert not np.isnan(pvals["ten_obs"])


def test_strategy_pvalues_detects_a_real_edge():
    rng = np.random.default_rng(9)
    returns = _noise(rng, 1000, 3)
    returns["alpha"] = rng.normal(0.001, 0.01, 1000)

    pvals = strategy_pvalues(returns)

    assert pvals["alpha"] < 0.01
    assert (pvals.drop("alpha") > 0.01).all()


def test_strategy_pvalues_one_sided_and_two_sided_agree():
    rng = np.random.default_rng(10)
    returns = _noise(rng, 400, 4)
    returns["alpha"] = rng.normal(0.001, 0.01, 400)

    one = strategy_pvalues(returns, one_sided=True)
    two = strategy_pvalues(returns, one_sided=False)

    # For a positive t-stat the two-sided p-value is exactly twice the
    # one-sided one.
    assert two["alpha"] == pytest.approx(2.0 * one["alpha"])


def test_validate_strategies_accepts_precomputed_pvalues():
    pvals = pd.Series({"a": 0.001, "b": 0.02, "c": 0.4, "d": 0.9})

    res = validate_strategies(pvalues=pvals, q=0.05)

    assert list(res.index) == ["a", "b", "c", "d"]  # sorted by raw p-value
    assert "sharpe_ann" not in res.columns  # no returns, no performance columns
    assert res.attrs["m"] == 4
    assert res.attrs["n_usable"] == 4
    assert res.loc["a", "reject_bh"]


def test_validate_strategies_requires_exactly_one_input():
    rng = np.random.default_rng(11)
    returns = _noise(rng, 100, 2)

    with pytest.raises(ValueError, match="exactly one"):
        validate_strategies()
    with pytest.raises(ValueError, match="exactly one"):
        validate_strategies(returns=returns, pvalues=pd.Series({"a": 0.01}))


def test_validate_strategies_rejects_out_of_range_pvalues():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_strategies(pvalues=pd.Series({"a": 1.4, "b": 0.2}))
