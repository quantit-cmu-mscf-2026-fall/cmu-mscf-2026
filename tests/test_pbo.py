"""Tests for CSCV / probability of backtest overfitting.

These tests pin the statistic against panels whose overfitting structure is
known by construction, because PBO's own calibration is the thing most easily
got wrong. Two baselines matter and they are not the same number:

* **Independent no-skill candidates** -- what `synth.make_panel(n_real=0)`
  generates -- score PBO near **0.5**, not near 1. Each candidate is its own
  draw, so its IS and OOS halves are independent and the winner's OOS rank is
  uniform. Half the splits land below the median by symmetry. Measured over
  seeds 0-5 of the configuration used here: 0.34 to 0.69, mean 0.55.
* **Coupled candidates** -- variants of one search over one sample, so their
  full-sample performances cluster while their IS/OOS splits do not -- score
  near **1.0**. That is the overfit-parameter-sweep regime CSCV exists to
  catch, and `TestCoupledSweepPanel` is where the "a noise winner usually
  flops out of sample" claim is actually true.

So a high PBO is evidence about the *search*, not about the *data being
noise*. Asserting PBO > 0.7 on an independent null panel would not be a
stricter test; it would be a wrong one, and the only way to make it pass would
be to break the estimator.

`TestSabotage` is the guard that keeps the null calibration honest: if IS and
OOS are fed the same rows, `_assert_independent_null_calibration` must raise.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from capstone.backtest import run_backtest
from capstone.pbo import cscv
from capstone.synth import candidate_frames, make_panel

# Kept small enough that the whole file runs in seconds, but wide enough that
# the winner is picked from a real crowd: 100 candidates over ~2 years.
N_DATES = 520
N_ASSETS = 20
N_CANDIDATES = 100
N_BLOCKS = 16

_MATRIX_CACHE: dict[tuple, pd.DataFrame] = {}


def strategy_matrix(*, n_real: int = 0, effect_size: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """T x N per-period strategy returns, one column per candidate.

    `backtest.sweep` returns per-candidate summaries, not the per-period matrix
    CSCV needs, so each candidate goes through `run_backtest` and the resulting
    series are concatenated column-wise. Cached because several tests reuse the
    same panel and the backtests dominate this file's runtime.
    """
    key = (n_real, effect_size, seed)
    if key not in _MATRIX_CACHE:
        panel = make_panel(
            n_dates=N_DATES,
            n_assets=N_ASSETS,
            n_candidates=N_CANDIDATES,
            n_real=n_real,
            effect_size=effect_size,
            seed=seed,
        )
        _MATRIX_CACHE[key] = pd.DataFrame(
            {name: run_backtest(frame, panel.returns) for name, frame, _ in candidate_frames(panel)}
        )
    return _MATRIX_CACHE[key]


def coupled_sweep_matrix(seed: int = 0, n_strategies: int = N_CANDIDATES) -> pd.DataFrame:
    """A signal-free panel with the structure of an overfit parameter sweep.

    A sweep over one price path yields variants that all end the sample in much
    the same place -- they differ in *when* they made the money, not in how
    much. Removing each column's full-sample mean imposes exactly that
    constraint, which is what couples an exceptional IS half to a poor OOS half.
    No column has any edge: this is noise, arranged the way searching over one
    sample arranges it.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((N_DATES, n_strategies)) * 0.01
    return pd.DataFrame(raw - raw.mean(axis=0))


def _assert_independent_null_calibration(result: dict) -> None:
    """The claim `TestSabotage` breaks: independent nulls sit at the median.

    Kept as a helper rather than inlined so the sabotage test can assert that
    these exact assertions fail when IS and OOS share rows. Bounds are wide
    because a single panel's PBO is itself a noisy statistic -- the measured
    spread over seeds 0-5 is 0.34 to 0.69.
    """
    assert 0.20 < result["pbo"] < 0.85, (
        f"PBO {result['pbo']:.3f} is outside the range independent no-skill candidates "
        "produce; near 0 means IS selection is predicting OOS, which on a signal-free "
        "panel means the split is leaking"
    )
    rank_mean = float(result["oos_ranks"].mean())
    assert 0.25 < rank_mean < 0.75, (
        f"winner's mean relative OOS rank {rank_mean:.3f} is far from the 0.5 that "
        "an uninformative IS ranking must give"
    )


class TestIndependentNullPanel:
    """`make_panel(n_real=0)`: no candidate has an edge, and none are coupled."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_winner_lands_near_the_oos_median(self, seed):
        # Picking the IS winner out of 100 pure-noise candidates tells you
        # nothing about OOS, so it lands on either side of the median about
        # equally often. This is the calibration the sabotage test protects.
        result = cscv(strategy_matrix(seed=seed), n_blocks=N_BLOCKS)
        _assert_independent_null_calibration(result)

    def test_pbo_averages_near_one_half_across_seeds(self):
        # Per-panel PBO is noisy; the average over seeds is what should sit at
        # 0.5. Measured mean over these three seeds is 0.46.
        values = [cscv(strategy_matrix(seed=s), n_blocks=N_BLOCKS)["pbo"] for s in range(3)]
        mean_pbo = float(np.mean(values))
        assert 0.30 < mean_pbo < 0.75, f"mean PBO {mean_pbo:.3f} over seeds {values}"

    def test_degradation_slope_is_reported_and_finite(self):
        result = cscv(strategy_matrix(seed=0), n_blocks=N_BLOCKS)
        assert math.isfinite(result["degradation_slope"])
        assert math.isfinite(result["degradation_intercept"])


class TestCoupledSweepPanel:
    """The regime PBO exists to catch: a search over one sample, no real edge."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_pbo_is_high_when_candidates_are_coupled(self, seed):
        # Every variant ends the sample in the same place, so an exceptional IS
        # half is borrowed from its OOS half. The IS winner is the one that
        # borrowed most, and it reverts almost every time.
        result = cscv(coupled_sweep_matrix(seed=seed), n_blocks=N_BLOCKS)
        assert result["pbo"] > 0.7, (
            f"PBO {result['pbo']:.3f} on a coupled signal-free sweep; the IS winner "
            "should nearly always flop out of sample here"
        )

    def test_winner_ranks_near_the_bottom_out_of_sample(self):
        result = cscv(coupled_sweep_matrix(seed=0), n_blocks=N_BLOCKS)
        assert float(result["oos_ranks"].mean()) < 0.25

    def test_coupling_is_what_separates_it_from_the_independent_null(self):
        # Same generator, same noise, differing only in whether the full-sample
        # constraint is imposed. The gap is the whole point of the statistic.
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((N_DATES, N_CANDIDATES)) * 0.01
        uncoupled = cscv(pd.DataFrame(raw), n_blocks=N_BLOCKS)["pbo"]
        coupled = cscv(pd.DataFrame(raw - raw.mean(axis=0)), n_blocks=N_BLOCKS)["pbo"]
        assert coupled - uncoupled > 0.3, f"coupled {coupled:.3f} vs uncoupled {uncoupled:.3f}"


class TestPlantedSignalPanel:
    """With a strong real signal, IS selection genuinely predicts OOS."""

    def test_pbo_is_materially_lower_than_the_null(self):
        planted = cscv(strategy_matrix(n_real=10, effect_size=0.30, seed=0), n_blocks=N_BLOCKS)[
            "pbo"
        ]
        null = cscv(strategy_matrix(seed=0), n_blocks=N_BLOCKS)["pbo"]
        assert planted < 0.05, f"PBO {planted:.3f} with a strong planted signal"
        assert null - planted > 0.25, f"null {null:.3f} vs planted {planted:.3f}"

    def test_winner_is_a_planted_candidate(self):
        # PBO only means something if the IS winner really is one of the real
        # signals; otherwise the low PBO is measuring something else.
        panel = make_panel(
            n_dates=N_DATES,
            n_assets=N_ASSETS,
            n_candidates=N_CANDIDATES,
            n_real=10,
            effect_size=0.30,
            seed=0,
        )
        real = set(panel.truth.index[panel.truth])
        result = cscv(strategy_matrix(n_real=10, effect_size=0.30, seed=0), n_blocks=N_BLOCKS)
        share_real = np.mean([c in real for c in result["winner_columns"]])
        assert share_real > 0.9, f"only {share_real:.1%} of split winners were planted signals"

    def test_winner_ranks_near_the_top_out_of_sample(self):
        result = cscv(strategy_matrix(n_real=10, effect_size=0.30, seed=0), n_blocks=N_BLOCKS)
        assert float(result["oos_ranks"].mean()) > 0.75


class TestSabotage:
    """Break the block split; the null calibration must go red.

    A validation procedure that cannot fail is not a validation procedure. The
    check here is the one that catches the worst possible bug in this module --
    IS and OOS scoring the same rows -- which would make every panel look like
    it generalises perfectly.
    """

    @staticmethod
    def _shared_rows_matrix(honest: pd.DataFrame) -> pd.DataFrame:
        """One block, tiled: every split's IS and OOS hold identical data.

        Sabotaging the data this way rather than reaching into `cscv` reproduces
        the exact condition a broken split would create -- IS and OOS Sharpes
        equal, column for column -- without the test having to restate the
        implementation it is checking.
        """
        clean = honest.dropna(how="any")
        block = clean.iloc[: len(clean) // N_BLOCKS]
        return pd.concat([block] * N_BLOCKS, ignore_index=True)

    def test_null_calibration_fails_when_is_and_oos_share_rows(self):
        sabotaged = self._shared_rows_matrix(strategy_matrix(seed=0))
        with warnings.catch_warnings():
            # linregress warns on the degenerate fit; the ranks are the point.
            warnings.simplefilter("ignore", RuntimeWarning)
            result = cscv(sabotaged, n_blocks=N_BLOCKS)

        assert result["pbo"] == pytest.approx(0.0), (
            f"PBO {result['pbo']:.3f} with IS and OOS on identical rows; the IS winner "
            "is by construction also the OOS winner"
        )
        assert float(result["oos_ranks"].mean()) > 0.95

        with pytest.raises(AssertionError, match="outside the range"):
            _assert_independent_null_calibration(result)

    def test_the_honest_split_still_passes(self):
        # The restore half of the sabotage check: the assertions above are not
        # simply unsatisfiable.
        result = cscv(strategy_matrix(seed=0), n_blocks=N_BLOCKS)
        _assert_independent_null_calibration(result)

    def test_coupled_panel_also_goes_red_under_shared_rows(self):
        honest = cscv(coupled_sweep_matrix(seed=0), n_blocks=N_BLOCKS)["pbo"]
        sabotaged_matrix = self._shared_rows_matrix(coupled_sweep_matrix(seed=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sabotaged = cscv(sabotaged_matrix, n_blocks=N_BLOCKS)["pbo"]
        assert honest > 0.7
        assert sabotaged == pytest.approx(0.0)


class TestSplitMechanics:
    def test_every_balanced_split_is_evaluated(self):
        result = cscv(coupled_sweep_matrix(n_strategies=5), n_blocks=8)
        assert result["n_splits"] == math.comb(8, 4)
        assert result["n_strategies"] == 5
        assert len(result["oos_ranks"]) == result["n_splits"]
        assert len(result["winner_columns"]) == result["n_splits"]

    def test_blocks_are_equal_length_and_the_remainder_is_trimmed(self):
        # 103 rows over 10 blocks is 10 per block; the last 3 rows are dropped
        # so both sides of every split carry the same estimation error.
        matrix = coupled_sweep_matrix(n_strategies=4).iloc[:103]
        result = cscv(matrix, n_blocks=10)
        assert result["block_size"] == 10
        assert result["n_obs_used"] == 100

    def test_ranks_are_strictly_inside_the_unit_interval(self):
        result = cscv(coupled_sweep_matrix(n_strategies=20), n_blocks=8)
        assert result["oos_ranks"].between(0.0, 1.0, inclusive="neither").all()

    def test_ranks_do_not_depend_on_the_annualisation_factor(self):
        matrix = coupled_sweep_matrix(n_strategies=10)
        daily = cscv(matrix, n_blocks=8, periods_per_year=252)
        monthly = cscv(matrix, n_blocks=8, periods_per_year=12)
        assert daily["pbo"] == monthly["pbo"]
        assert np.allclose(daily["oos_ranks"], monthly["oos_ranks"])

    def test_a_dominant_strategy_scores_zero(self):
        # One column that beats the rest in every block must win IS and rank
        # first OOS on every split.
        rng = np.random.default_rng(0)
        matrix = pd.DataFrame(rng.standard_normal((200, 6)) * 0.01)
        matrix[0] = matrix[0].abs() + 0.05
        result = cscv(matrix, n_blocks=8)
        assert result["pbo"] == pytest.approx(0.0)
        assert set(result["winner_columns"]) == {0}


class TestGuardrails:
    @pytest.mark.parametrize("n_blocks", [3, 15, 1, 0, -2])
    def test_rejects_block_counts_that_cannot_be_halved(self, n_blocks):
        with pytest.raises(ValueError, match="even and at least 2"):
            cscv(coupled_sweep_matrix(n_strategies=4), n_blocks=n_blocks)

    def test_rejects_a_block_count_that_would_not_finish(self):
        with pytest.raises(ValueError, match="use fewer blocks"):
            cscv(coupled_sweep_matrix(n_strategies=4), n_blocks=26)

    def test_rejects_a_single_strategy(self):
        # A lone column ranks first against itself on every split, so PBO would
        # be 0 no matter what the data said.
        with pytest.raises(ValueError, match="at least 2 strategies"):
            cscv(coupled_sweep_matrix(n_strategies=1), n_blocks=8)

    def test_rejects_more_blocks_than_rows(self):
        with pytest.raises(ValueError, match="usable rows"):
            cscv(coupled_sweep_matrix(n_strategies=4).iloc[:6], n_blocks=8)

    def test_rejects_blocks_too_thin_to_carry_a_variance(self):
        # 3 rows over 2 blocks is one row per side, and a Sharpe needs a
        # variance. Note the binding constraint is observations per *side*, not
        # per block: 8 blocks of 1 row still leaves 4 observations a side.
        with pytest.raises(ValueError, match="a variance needs at least 2"):
            cscv(coupled_sweep_matrix(n_strategies=4).iloc[:3], n_blocks=2)

    def test_incomplete_rows_are_dropped_so_blocks_line_up(self):
        # run_backtest leaves a NaN first row; every column must be scored on
        # identical periods or the IS/OOS comparison is between unlike things.
        matrix = coupled_sweep_matrix(n_strategies=4).copy()
        matrix.iloc[0, 0] = np.nan
        matrix.iloc[5, 2] = np.nan
        result = cscv(matrix, n_blocks=8)
        assert result["n_obs_used"] <= len(matrix) - 2
