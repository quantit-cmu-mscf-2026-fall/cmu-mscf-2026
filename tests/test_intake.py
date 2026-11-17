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
from test_factor_evaluation import fake_rows

from capstone import calibration, funnel
from capstone.evaluate import benjamini_hochberg
from capstone.factors import evaluation as ev
from capstone.factors import store
from capstone.factors.tree import parse
from capstone.intake import FunnelInput, from_evaluation, from_returns, stage_inputs


def _tiny(n=30):
    dates = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=dates, columns=["a", "b", "c"])
    family = pd.Series({"a": "h1", "b": "h1", "c": "h2"})
    return returns, family


def _evaluated(tmp_path, monkeypatch, expressions):
    """evaluate_factors on fake rows, families from a small store.

    `expressions` maps a hypothesis name to the factor expressions proposed
    under it. Returns (matrix, summary, {hypothesis name: hypothesis id}).
    """
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    con = store.connect(tmp_path / "factors.db")
    store.add_paper(con, "p", "Paper")
    hids, trees = {}, {}
    for name, exprs in expressions.items():
        hids[name] = store.add_hypothesis(con, store.Hypothesis("p", name, "k", "j", "s", "f"))
        for e in exprs:
            fid, _ = store.add_factor(con, parse(e), hids[name])
            trees[fid] = parse(e)
    families = {fid: ev.hypothesis_of(con, fid) for fid in trees}
    panel = ev.build_panel(fake_rows(n_stocks=12, n_days=80), data_version="v-test")
    specs = ev.grid(holds=[1, 5], neutral=["none"])
    matrix, summary = ev.evaluate_factors(trees, panel, specs=specs, con=con, families=families)
    return matrix, summary, hids


def _family_set(n_families, share_real, seed, *, sharpe_real=2.0, n_holdout=0):
    """Simulated families (k=5) on business days from 2015, 3 years of search."""
    fs = calibration.make_family_set(
        n_families, 5, share_real, sharpe_real, n_search=3 * 252, n_holdout=n_holdout, seed=seed
    )
    assert isinstance(fs.returns.index, pd.DatetimeIndex)
    assert fs.returns.index.max() <= pd.Timestamp("2020-12-31")  # synthetic, pre-holdout dates
    return fs


# ---------------------------------------------------------------------------
# from_evaluation


def test_from_evaluation_gives_one_family_per_hypothesis(tmp_path, monkeypatch):
    matrix, summary, hids = _evaluated(
        tmp_path,
        monkeypatch,
        {"volume": ["volume", "open"], "reversal": ["ts_mean(returns, 5)"]},
    )
    fi = from_evaluation(matrix, summary, data_version="v-test")

    assert isinstance(fi, FunnelInput)
    assert [f.name for f in dataclasses.fields(FunnelInput)] == [
        "returns",
        "family",
        "generator",
        "source",
    ]
    assert list(fi.returns.columns) == list(matrix.columns)
    assert list(fi.family.index) == list(matrix.columns)
    assert fi.family.to_dict() == summary["hypothesis_id"].to_dict()
    assert fi.family.value_counts().to_dict() == {hids["volume"]: 4, hids["reversal"]: 2}
    assert fi.generator == "factor-trees"
    assert fi.source["data_version"] == "v-test"
    assert fi.source["variant_ids"] == list(matrix.columns)
    assert isinstance(fi.returns.index, pd.DatetimeIndex)


def test_from_evaluation_drops_rows_no_variant_traded(tmp_path, monkeypatch):
    # Every factor has a lookback, so the first days hold nothing anywhere.
    matrix, summary, _ = _evaluated(
        tmp_path, monkeypatch, {"a": ["ts_mean(returns, 5)"], "b": ["ts_mean(volume, 10)"]}
    )
    empty = matrix.isna().all(axis=1)
    assert empty.any()  # the warm-up is in evaluation's matrix
    fi = from_evaluation(matrix, summary)
    assert not fi.returns.isna().all(axis=1).any()
    pd.testing.assert_frame_equal(fi.returns, matrix.loc[~empty], check_freq=False)
    assert fi.source["n_dropped_rows"] == int(empty.sum())


def test_a_variant_with_no_family_raises_naming_it(tmp_path, monkeypatch):
    matrix, summary, _ = _evaluated(tmp_path, monkeypatch, {"volume": ["volume"]})
    orphan = summary.index[1]
    summary = summary.copy()
    summary.loc[orphan, "hypothesis_id"] = None
    with pytest.raises(ValueError, match=orphan):
        from_evaluation(matrix, summary)


def test_a_variant_missing_from_the_summary_raises_naming_it(tmp_path, monkeypatch):
    matrix, summary, _ = _evaluated(tmp_path, monkeypatch, {"volume": ["volume"]})
    with pytest.raises(ValueError, match=summary.index[0]):
        from_evaluation(matrix, summary.iloc[1:])


def test_a_summary_without_a_hypothesis_column_raises(tmp_path, monkeypatch):
    matrix, summary, _ = _evaluated(tmp_path, monkeypatch, {"volume": ["volume"]})
    with pytest.raises(ValueError, match="hypothesis_id"):
        from_evaluation(matrix, summary.drop(columns="hypothesis_id"))


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


@pytest.mark.parametrize("n_obs", [0, 1])
def test_a_candidate_short_of_search_returns_is_refused_by_name(n_obs):
    fs, search_end, holdout_start = _periods()
    returns = fs.returns.copy()
    victim = returns.columns[1]
    returns.loc[:search_end, victim] = np.nan
    if n_obs:
        returns.loc[returns.index[0], victim] = 0.001
    # With the dates, intake itself refuses it ...
    with pytest.raises(ValueError, match=rf"search period.*{victim}"):
        from_returns(
            returns, fs.family, generator="g", search_end=search_end, holdout_start=holdout_start
        )
    # ... and without them stage_inputs does, before the funnel runs.
    fi = from_returns(returns, fs.family, generator="g")
    with pytest.raises(ValueError, match=rf"search period.*{victim}"):
        stage_inputs(
            fi, search_end=search_end, holdout_start=holdout_start, n_boot=0, seed=0, q=0.05
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


def _six_variants():
    dates = pd.bdate_range("2020-01-02", periods=400)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (len(dates), 6)), index=dates, columns=[f"v{i}" for i in range(6)]
    )
    returns["v1"] += 0.002
    returns["v5"] += 0.003
    family = pd.Series(["A"] * 3 + ["B"] * 3, index=returns.columns)
    kwargs = dict(search_end="2020-12-31", holdout_start="2021-01-01", n_boot=0, seed=0, q=0.05)
    return returns, family, kwargs


@pytest.mark.parametrize("check_at_intake", [False, True])
def test_an_unchosen_variant_without_holdout_matches_the_funnel(check_at_intake):
    returns, family, kwargs = _six_variants()
    clean = funnel.family_statistics(returns, family, **kwargs)["reuse"]
    assert list(clean["best"]) == ["v1", "v5"]
    returns.loc[kwargs["holdout_start"] :, "v0"] = np.nan
    direct = funnel.family_statistics(returns, family, **kwargs)["reuse"]
    pd.testing.assert_frame_equal(direct, clean)

    periods = {k: kwargs[k] for k in ("search_end", "holdout_start")} if check_at_intake else {}
    fi = from_returns(returns, family, generator="synthetic", **periods)
    got = stage_inputs(fi, **kwargs)
    pd.testing.assert_frame_equal(got, direct)
    assert got.attrs["n_variants"] == 6
    pd.testing.assert_frame_equal(fi.returns, returns)


@pytest.mark.parametrize("n_obs", [0, 1])
def test_chosen_variant_holdout_sufficiency_stays_with_the_funnel(n_obs):
    returns, family, kwargs = _six_variants()
    victim = funnel.family_statistics(returns, family, **kwargs)["reuse"]["best"].iloc[0]
    returns.loc[kwargs["holdout_start"] :, victim] = np.nan
    if n_obs:
        first_holdout = returns.index[returns.index >= kwargs["holdout_start"]][0]
        returns.loc[first_holdout, victim] = 0.001
    fi = from_returns(
        returns,
        family,
        generator="synthetic",
        search_end=kwargs["search_end"],
        holdout_start=kwargs["holdout_start"],
    )
    with pytest.raises(ValueError, match=rf"{victim}.*holdout"):
        funnel.family_statistics(returns, family, **kwargs)
    with pytest.raises(ValueError, match=rf"{victim}.*holdout"):
        stage_inputs(fi, **kwargs)
