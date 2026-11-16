"""Tests for capstone.intake: what the validation funnel takes in.

Any generator's dates x candidate returns and one family label per candidate
become a `FunnelInput`, checked once. `stage_inputs` feeds it to the funnel's
per-family statistics with periods cut by date, so stages 1-3 never see a
holdout row.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from capstone import calibration, funnel
from capstone.evaluate import benjamini_hochberg
from capstone.intake import FunnelInput, from_returns, stage_inputs


def _tiny(n=30):
    dates = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=dates, columns=["a", "b", "c"])
    family = pd.Series({"a": "h1", "b": "h1", "c": "h2"})
    return returns, family


def _family_set(n_families, share_real, seed, *, sharpe_real=2.0, n_holdout=0):
    """Simulated families (k=5) on business days from 2015, 3 years of search."""
    fs = calibration.make_family_set(
        n_families, 5, share_real, sharpe_real, n_search=3 * 252, n_holdout=n_holdout, seed=seed
    )
    assert isinstance(fs.returns.index, pd.DatetimeIndex)
    assert fs.returns.index.max() <= pd.Timestamp("2020-12-31")  # synthetic, pre-holdout dates
    return fs


# ---------------------------------------------------------------------------
# from_returns: the same checks for any generator


def test_from_returns_keeps_the_generator_and_aligns_family_to_columns():
    returns, family = _tiny()
    fi = from_returns(returns, family.iloc[::-1], generator="alpha-gpt", source={"v": "x"})
    assert isinstance(fi, FunnelInput)
    assert [f.name for f in dataclasses.fields(FunnelInput)] == [
        "returns",
        "family",
        "generator",
        "source",
    ]
    assert fi.generator == "alpha-gpt"
    assert list(fi.family.index) == ["a", "b", "c"]
    assert fi.source["variant_ids"] == ["a", "b", "c"]
    assert fi.source["v"] == "x"
    assert (fi.source["start"], fi.source["end"]) == ("2010-01-04", "2010-02-12")


def test_from_returns_drops_all_missing_rows_only():
    returns, family = _tiny()
    returns.iloc[0] = np.nan  # untraded first row
    returns.iloc[5, 0] = np.nan  # one candidate missing one day: kept
    fi = from_returns(returns, family, generator="g")
    assert len(fi.returns) == len(returns) - 1
    assert fi.returns.index[0] == returns.index[1]
    assert np.isnan(fi.returns.loc[returns.index[5], "a"])
    assert fi.source["n_dropped_rows"] == 1


def test_a_missing_family_label_raises_naming_the_candidate():
    returns, family = _tiny()
    with pytest.raises(ValueError, match="'c'"):
        from_returns(returns, family.drop("c"), generator="g")
    family["b"] = np.nan
    with pytest.raises(ValueError, match="'b'"):
        from_returns(returns, family, generator="g")


def test_a_label_for_no_candidate_raises():
    returns, family = _tiny()
    family["z"] = "h3"
    with pytest.raises(ValueError, match="'z'"):
        from_returns(returns, family, generator="g")


def test_a_non_date_index_raises():
    returns, family = _tiny()
    with pytest.raises(TypeError, match="date"):
        from_returns(returns.reset_index(drop=True), family, generator="g")


def test_unsorted_or_repeated_dates_raise():
    returns, family = _tiny()
    with pytest.raises(ValueError, match="increasing"):
        from_returns(returns.iloc[::-1], family, generator="g")
    with pytest.raises(ValueError, match="increasing"):
        from_returns(pd.concat([returns.iloc[:3], returns.iloc[2:]]), family, generator="g")


def test_repeated_candidates_raise():
    returns, family = _tiny()
    with pytest.raises(ValueError, match="unique"):
        from_returns(returns.rename(columns={"b": "a"}), family.drop("b"), generator="g")
    with pytest.raises(ValueError, match="more than once"):
        from_returns(returns, pd.concat([family, family.iloc[:1]]), generator="g")


def test_infinite_returns_and_empty_candidates_raise():
    returns, family = _tiny()
    bad = returns.copy()
    bad.iloc[3, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        from_returns(bad, family, generator="g")
    bad = returns.copy()
    bad["c"] = np.nan
    with pytest.raises(ValueError, match="'c'"):
        from_returns(bad, family, generator="g")


def test_a_blank_generator_raises():
    returns, family = _tiny()
    with pytest.raises(ValueError, match="generator"):
        from_returns(returns, family, generator="")


def test_the_input_is_frozen_and_checked_when_built_directly():
    returns, family = _tiny()
    fi = from_returns(returns, family, generator="g")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fi.generator = "other"
    with pytest.raises(ValueError, match="'c'"):
        FunnelInput(returns, family.drop("c"), "g", {})


def test_the_intake_imports_no_generator_code():
    # Validation never calls strategy code: importing the intake must not pull
    # in a generator, the backtest or the data layer.
    code = (
        "import sys, capstone.intake; "
        "print('\\n'.join(m for m in sys.modules if m.startswith('capstone')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    loaded = set(out.stdout.split())
    assert "capstone.intake" in loaded
    banned = ("capstone.factors", "capstone.backtest", "capstone.shared_data", "capstone.data")
    assert not [m for m in loaded if m.startswith(banned)]


# ---------------------------------------------------------------------------
# stage_inputs: periods by date; stages 1-3 never see the holdout


def test_stage_inputs_never_shows_holdout_rows_to_stages_1_to_3(monkeypatch):
    fs = _family_set(6, 1 / 3, seed=1, n_holdout=252)
    search_end = fs.returns.index[fs.n_search - 1]
    holdout_start = fs.returns.index[fs.n_search]
    clean = from_returns(fs.returns, fs.family, generator="synthetic")
    poisoned_returns = fs.returns.copy()
    poisoned_returns.loc[holdout_start:] += 1e6
    poisoned = from_returns(poisoned_returns, fs.family, generator="synthetic")

    seen = []
    original = funnel.bootstrap_family_pvalues

    def spy(window, family, **kwargs):
        seen.append(window.index.max())
        return original(window, family, **kwargs)

    monkeypatch.setattr(funnel, "bootstrap_family_pvalues", spy)
    kwargs = dict(search_end=search_end, holdout_start=holdout_start, n_boot=199, seed=3, q=0.05)
    a = stage_inputs(clean, **kwargs)
    b = stage_inputs(poisoned, **kwargs)

    assert seen and all(d <= search_end for d in seen)
    cols = ["best", "p1_independent", "p1_rho", "p1_bootstrap", "p2"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert (b["ho_sharpe"] != a["ho_sharpe"]).all()  # stage 4 does read the holdout
    # The same statistics as on the search period alone.
    search_only = from_returns(fs.returns.loc[:search_end], fs.family, generator="synthetic")
    c = stage_inputs(search_only, search_end=search_end, n_boot=199, seed=3, q=0.05)
    pd.testing.assert_frame_equal(a[cols], c[cols])


def test_stage_inputs_refuses_post_search_dates_without_a_holdout():
    fs = _family_set(4, 0.0, seed=2, n_holdout=60)
    fi = from_returns(fs.returns, fs.family, generator="synthetic")
    with pytest.raises(ValueError, match="holdout_start"):
        stage_inputs(fi, search_end=fs.returns.index[fs.n_search - 1], n_boot=199, seed=0, q=0.05)


def test_too_few_bootstrap_draws_for_the_registered_q_raise():
    # BH at q=0.01 across 20 families needs p-values down to 0.0005; 499 draws
    # floor them at 0.002, so stage 1 could pass nothing. Refuse it up front.
    fs = calibration.make_family_set(20, 5, 0.15, 2.0, n_search=3 * 252, n_holdout=0, seed=4)
    fi = from_returns(fs.returns, fs.family, generator="synthetic")
    with pytest.raises(ValueError, match="can't resolve BH at 0.01"):
        stage_inputs(fi, search_end=fs.returns.index[-1], n_boot=499, seed=0, q=0.01)


def test_planted_families_get_small_stage1_pvalues():
    fs = _family_set(10, 0.3, seed=4)
    fi = from_returns(fs.returns, fs.family, generator="synthetic")
    stats = stage_inputs(fi, search_end=fs.returns.index[-1], n_boot=199, seed=0, q=0.10)
    planted = fs.truth[fs.truth].index
    assert len(planted) == 3
    assert (stats.loc[planted, "p1_bootstrap"] <= 0.01).all()
    passed = benjamini_hochberg(stats["p1_bootstrap"], 0.10)
    assert passed[planted].all()


@pytest.mark.parametrize("defect", [{}, {"rho": 0.3}, {"ar1": 0.2}])
def test_stage1_on_null_families_accepts_at_most_q1_in_expectation(defect):
    q1, accepted, any_accepted = 0.10, [], 0
    for seed in range(20):
        fs = calibration.make_family_set(
            10, 5, 0.0, 0.0, n_search=3 * 252, n_holdout=0, seed=100 + seed, **defect
        )
        fi = from_returns(fs.returns, fs.family, generator="synthetic")
        stats = stage_inputs(fi, search_end=fs.returns.index[-1], n_boot=199, seed=seed, q=q1)
        passed = benjamini_hochberg(stats["p1_bootstrap"], q1)
        accepted.append(passed.mean())
        any_accepted += bool(passed.any())
    assert np.mean(accepted) <= q1
    assert any_accepted <= 5  # BH under the complete null: P(any) <= q1, so ~2 of 20


# ---------------------------------------------------------------------------
# The funnel's date/period contract, applied at the boundary


def _periods(seed=2):
    fs = _family_set(4, 0.0, seed=seed, n_holdout=252)
    return fs, fs.returns.index[fs.n_search - 1], fs.returns.index[fs.n_search]


def test_timezone_aware_or_timed_dates_raise_at_intake():
    returns, family = _tiny()
    aware = returns.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone"):
        from_returns(aware, family, generator="g")
    timed = returns.copy()
    timed.index = timed.index + pd.Timedelta(hours=16)
    with pytest.raises(ValueError, match="normalised"):
        from_returns(timed, family, generator="g")
    with pytest.raises(ValueError, match="timezone"):
        FunnelInput(aware, family, "g", {})


@pytest.mark.parametrize("period", ["search period", "holdout"])
def test_a_candidate_short_of_returns_in_a_period_is_refused_by_name(period):
    fs, search_end, holdout_start = _periods()
    returns = fs.returns.copy()
    victim = returns.columns[1]
    if period == "holdout":
        returns.loc[holdout_start:, victim] = np.nan
    else:
        returns.loc[:search_end, victim] = np.nan
    # With the dates, intake itself refuses it ...
    with pytest.raises(ValueError, match=rf"{period}.*{victim}"):
        from_returns(
            returns, fs.family, generator="g", search_end=search_end, holdout_start=holdout_start
        )
    # ... and without them stage_inputs does, before the funnel runs.
    fi = from_returns(returns, fs.family, generator="g")
    with pytest.raises(ValueError, match=rf"{period}.*{victim}"):
        stage_inputs(
            fi, search_end=search_end, holdout_start=holdout_start, n_boot=0, seed=0, q=0.05
        )


def test_one_return_in_a_period_is_not_enough():
    fs, search_end, holdout_start = _periods()
    returns = fs.returns.copy()
    victim = returns.columns[2]
    returns.loc[holdout_start:, victim] = np.nan
    returns.loc[holdout_start, victim] = 0.001
    with pytest.raises(ValueError, match=victim):
        from_returns(
            returns, fs.family, generator="g", search_end=search_end, holdout_start=holdout_start
        )


def test_adequate_periods_pass_and_match_the_unchecked_intake():
    fs, search_end, holdout_start = _periods()
    checked = from_returns(
        fs.returns, fs.family, generator="g", search_end=search_end, holdout_start=holdout_start
    )
    plain = from_returns(fs.returns, fs.family, generator="g")
    pd.testing.assert_frame_equal(checked.returns, plain.returns)


def test_a_blank_family_label_raises_naming_the_candidate():
    returns, family = _tiny()
    family["b"] = " "
    with pytest.raises(ValueError, match="blank family.*'b'"):
        from_returns(returns, family, generator="g")
