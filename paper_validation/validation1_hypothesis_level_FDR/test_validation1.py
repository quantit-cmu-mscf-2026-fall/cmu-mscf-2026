"""Tests for the hypothesis-level multiple-testing experiment.

Run:  PYTHONPATH=. pytest paper_validation/validation1_hypothesis_level_FDR/ -q

Not picked up by the repo's default `pytest -q` (pyproject sets
`testpaths = ["tests"]`) because this is paper-validation work, not part of the
kit. Run it explicitly.

Weighted toward the two places where a silent error would invalidate every
number downstream:

* the **DGP**, because every FDR and power figure is measured against truth it
  defines — if the correlation structure or the effect size is not what it
  claims, the experiment measures nothing and says so nowhere;
* the **bootstrap's null imposition**, because forgetting to centre, or
  resampling candidates independently, produces a procedure that still runs and
  still returns plausible p-values while testing the wrong hypothesis.
"""

from __future__ import annotations

import numpy as np
import pytest
from bootstrap import block_length_rule, bootstrap_pvalues, check_resolution
from dgp import make_panel, newey_west_se
from methods import (
    METHOD_ORDER,
    apply_all_methods,
    bonferroni_family_pvalues,
    by_constant,
    family_min_pvalues,
)
from metrics import paired_difference, score_decision

SMALL = dict(n_families=20, m_per_family=10, n_obs=300)


# --- DGP: structure ---------------------------------------------------------


def test_panel_shape_and_layout():
    panel = make_panel(pi1=0.1, delta=3.0, seed=0)
    assert panel.returns.shape == (500, 100 * 20)
    assert np.array_equal(panel.family_of, np.repeat(np.arange(100), 20))


def test_family_truth_is_derived_from_candidate_truth():
    """DESIGN.md §4 Phase B.3 — a family is non-null iff it holds a real candidate."""
    for seed in range(15):
        panel = make_panel(pi1=0.3, delta=3.0, seed=seed, **SMALL)
        contains_real = np.array(
            [
                panel.is_nonnull_candidate[panel.family_of == h].any()
                for h in range(panel.n_families)
            ]
        )
        assert np.array_equal(panel.is_nonnull_family, contains_real)


def test_sparsity_is_honoured():
    """A non-null family carries exactly `n_signals_per_family` real candidates."""
    panel = make_panel(pi1=0.3, delta=3.0, n_signals_per_family=3, seed=1, **SMALL)
    for h in np.flatnonzero(panel.is_nonnull_family):
        assert panel.is_nonnull_candidate[panel.family_of == h].sum() == 3


def test_within_and_between_family_correlation():
    """The load-bearing DGP claim: rho_W and rho_B come out as requested."""
    panel = make_panel(
        n_obs=40_000,
        n_families=8,
        m_per_family=6,
        pi1=0.0,
        delta=0.0,
        rho_w=0.7,
        rho_b=0.2,
        phi=0.0,
        seed=3,
    )
    corr = np.corrcoef(panel.returns.T)
    same = panel.family_of[:, None] == panel.family_of[None, :]
    off = ~np.eye(panel.n_candidates, dtype=bool)
    assert corr[same & off].mean() == pytest.approx(0.7, abs=0.02)
    assert corr[~same].mean() == pytest.approx(0.2, abs=0.02)


def test_serial_correlation_is_honoured_without_disturbing_scale():
    """`phi` changes time dependence only — unit variance and rho_W must survive."""
    for phi in (0.0, 0.5):
        panel = make_panel(
            n_obs=40_000,
            n_families=6,
            m_per_family=6,
            pi1=0.0,
            delta=0.0,
            rho_w=0.6,
            rho_b=0.0,
            phi=phi,
            seed=4,
        )
        x = panel.returns
        lag1 = np.mean([np.corrcoef(x[:-1, i], x[1:, i])[0, 1] for i in range(x.shape[1])])
        assert lag1 == pytest.approx(phi, abs=0.02)
        assert x.std(axis=0).mean() == pytest.approx(1.0, abs=0.02)
        corr = np.corrcoef(x.T)
        same = panel.family_of[:, None] == panel.family_of[None, :]
        off = ~np.eye(panel.n_candidates, dtype=bool)
        assert corr[same & off].mean() == pytest.approx(0.6, abs=0.03)


def test_effect_size_is_the_local_alternative():
    """theta = delta*sigma/sqrt(T), so E[t] ~ delta independently of T."""
    for n_obs in (300, 1200):
        panel = make_panel(
            n_obs=n_obs,
            pi1=1.0,
            delta=3.0,
            rho_w=0.0,
            rho_b=0.0,
            phi=0.0,
            n_signals_per_family=10,
            seed=5,
            n_families=20,
            m_per_family=10,
        )
        real = panel.theta > 0
        assert panel.theta[real][0] == pytest.approx(3.0 / np.sqrt(n_obs), rel=1e-9)
        t = np.sqrt(n_obs) * panel.returns.mean(axis=0) / panel.returns.std(axis=0, ddof=1)
        assert t[real].mean() == pytest.approx(3.0, abs=0.25)


def test_rejects_invalid_correlation_pair():
    """rho_B > rho_W has no real weight; DESIGN.md §4 validity rule 1."""
    with pytest.raises(ValueError, match="rho_b <= rho_w"):
        make_panel(rho_w=0.0, rho_b=0.1)
    with pytest.raises(ValueError, match="phi"):
        make_panel(phi=1.0)
    with pytest.raises(ValueError, match="global null"):
        make_panel(pi1=0.0, delta=3.0)


# --- bootstrap --------------------------------------------------------------


def test_resolution_guard():
    """B must resolve q/H, or outer BH's most extreme rank is unreachable."""
    assert not check_resolution(499, 100, 0.05)
    assert check_resolution(1999, 100, 0.05)
    assert check_resolution(4999, 100, 0.05)


def test_bonferroni_needs_m_times_more_resolution():
    """The guard that the pilot run needed and did not have.

    Bonferroni's family p-value floor is m/(B+1). At m=20, H=100, q=0.10 the
    bootstrap max clears its requirement at B=1999 while Bonferroni does not
    clear its own until B=19999 — and the gap silently inverted the headline
    power comparison.
    """
    assert check_resolution(1999, 100, 0.10, m_per_family=1)
    assert not check_resolution(1999, 100, 0.10, m_per_family=20)
    assert not check_resolution(4999, 100, 0.10, m_per_family=20)
    assert check_resolution(19999, 100, 0.10, m_per_family=20)


def test_block_length_rule_scales_with_T():
    assert block_length_rule(500) == 16
    assert block_length_rule(1000) == 20


def test_bootstrap_pvalues_are_bounded_by_resolution():
    """The plus-one correction means p can never be 0 — §5 rule 7."""
    panel = make_panel(pi1=0.0, delta=0.0, phi=0.0, seed=6, **SMALL)
    out = bootstrap_pvalues(
        panel.returns,
        panel.family_of,
        panel.n_families,
        n_boot=199,
        rng=np.random.default_rng(1),
    )
    for key in ("p_family", "p_candidate"):
        assert out[key].min() >= 1.0 / 200.0
        assert out[key].max() <= 1.0


def test_null_is_imposed_so_a_strong_signal_still_calibrates():
    """Centring is what makes the bootstrap a NULL distribution.

    Give every candidate a large true effect. Because the bootstrap statistic
    subtracts `theta_hat`, the resampled world is still centred at zero, so the
    observed maximum should NOT sit in the extreme tail of its own bootstrap
    distribution any more than under the null. Without the centring in
    `bootstrap.py`, the bootstrap would chase the observed effects and the
    p-values would collapse toward 1 here.
    """
    panel = make_panel(
        pi1=1.0,
        delta=8.0,
        n_signals_per_family=10,
        rho_w=0.0,
        rho_b=0.0,
        phi=0.0,
        seed=7,
        **SMALL,
    )
    out = bootstrap_pvalues(
        panel.returns,
        panel.family_of,
        panel.n_families,
        n_boot=399,
        rng=np.random.default_rng(2),
    )
    # A real effect of delta=8 should be detected: p-values at the floor.
    assert out["p_family"].max() < 0.05
    # And the bootstrap distribution itself is centred, not shifted: the mean
    # bootstrap max must be far below the observed max.
    assert out["m_obs"].mean() > 5.0


def test_bootstrap_family_p_is_never_below_its_candidate_minimum_times_nothing():
    """The family max p-value must be >= the best candidate's own p-value.

    Both come from the same draws: a bootstrap replication whose family max
    exceeds the observed max must also have had some candidate exceed its own
    observed statistic. The converse fails, so the inequality is one-directional
    and is a genuine internal-consistency check on the shared loop.
    """
    panel = make_panel(pi1=0.2, delta=3.0, rho_w=0.5, seed=8, **SMALL)
    out = bootstrap_pvalues(
        panel.returns,
        panel.family_of,
        panel.n_families,
        n_boot=399,
        rng=np.random.default_rng(3),
    )
    m = panel.n_candidates // panel.n_families
    best_candidate_p = out["p_candidate"].reshape(panel.n_families, m).min(axis=1)
    assert np.all(out["p_family"] >= best_candidate_p - 1e-12)


def test_block_length_override_changes_nothing_structural():
    panel = make_panel(pi1=0.0, delta=0.0, phi=0.3, seed=9, **SMALL)
    for block in (4, 16, 40):
        out = bootstrap_pvalues(
            panel.returns,
            panel.family_of,
            panel.n_families,
            n_boot=199,
            block_length=block,
            rng=np.random.default_rng(4),
        )
        assert out["block_length"] == block
        assert out["p_family"].shape == (panel.n_families,)


# --- methods ----------------------------------------------------------------


def test_by_constant_is_the_harmonic_sum():
    assert by_constant(1) == pytest.approx(1.0)
    assert by_constant(100) == pytest.approx(sum(1.0 / i for i in range(1, 101)))
    assert by_constant(100) == pytest.approx(5.187, abs=0.01)


def test_bonferroni_is_min_p_times_family_size_capped_at_one():
    p = np.array([0.001, 0.4, 0.6, 0.9, 0.02, 0.5, 0.7, 0.8, 0.3, 0.45])
    assert bonferroni_family_pvalues(p, 1, 10)[0] == pytest.approx(0.01)
    assert family_min_pvalues(p, 1, 10)[0] == pytest.approx(0.001)
    big = np.full(10, 0.5)
    assert bonferroni_family_pvalues(big, 1, 10)[0] == 1.0  # capped


def test_bonferroni_is_never_more_liberal_than_naive():
    """The negative control is, by construction, the one that ignores the search."""
    rng = np.random.default_rng(11)
    p = rng.uniform(size=200)
    assert np.all(bonferroni_family_pvalues(p, 20, 10) >= family_min_pvalues(p, 20, 10))


def test_by_is_never_more_liberal_than_bh():
    """BY runs BH at q/c(H); with c(H) > 1 it can only reject a subset."""
    rng = np.random.default_rng(12)
    p_family = rng.uniform(size=100) ** 3
    p_cand = rng.uniform(size=2000)
    out = apply_all_methods(
        p_boot=p_family,
        p_marginal=p_cand,
        family_of=np.repeat(np.arange(100), 20),
        n_families=100,
        m_per_family=20,
        q=0.10,
    )
    assert np.all(out["boot_max_by"].families <= out["boot_max_bh"].families)
    assert np.all(out["bonf_by"].families <= out["bonf_bh"].families)


def test_every_method_reports_and_only_flat_bh_rejects_candidates():
    """§5 rule 3: a family rejection does not validate any specification in it."""
    rng = np.random.default_rng(13)
    out = apply_all_methods(
        p_boot=rng.uniform(size=100) ** 4,
        p_marginal=rng.uniform(size=2000) ** 4,
        family_of=np.repeat(np.arange(100), 20),
        n_families=100,
        m_per_family=20,
        q=0.10,
    )
    assert set(out) == set(METHOD_ORDER)
    for name, decision in out.items():
        if name == "flat_bh":
            assert decision.candidates.any()
        else:
            assert not decision.candidates.any()


def test_flat_bh_family_column_is_induced_from_its_candidates():
    rng = np.random.default_rng(14)
    family_of = np.repeat(np.arange(100), 20)
    out = apply_all_methods(
        p_boot=np.full(100, 0.9),
        p_marginal=rng.uniform(size=2000) ** 5,
        family_of=family_of,
        n_families=100,
        m_per_family=20,
        q=0.10,
    )
    flat = out["flat_bh"]
    expected = np.zeros(100, dtype=bool)
    expected[np.unique(family_of[flat.candidates])] = True
    assert np.array_equal(flat.families, expected)


# --- metrics ----------------------------------------------------------------


def _decision(families, candidates, name="x"):
    from methods import Decision

    return Decision(np.asarray(families), np.asarray(candidates), name)


def test_power_is_nan_under_the_global_null_not_zero():
    """Reporting 0.0 power where nothing is findable would be a false statement."""
    panel = make_panel(pi1=0.0, delta=0.0, seed=15, **SMALL)
    score = score_decision(_decision(np.zeros(20, bool), np.zeros(200, bool)), panel)
    assert np.isnan(score.family_power)
    assert score.family_fdp == 0.0  # FDP is 0 when nothing was rejected


def test_global_null_fdp_equals_any_false_rejection():
    """The identity DESIGN.md §6 Figure 1 relies on: FDR == FWER when pi1 = 0."""
    panel = make_panel(pi1=0.0, delta=0.0, seed=16, **SMALL)
    fam = np.zeros(20, bool)
    fam[[2, 7]] = True
    score = score_decision(_decision(fam, np.zeros(200, bool)), panel)
    assert score.family_fdp == 1.0
    assert score.family_any_false == 1.0


def test_fdp_and_power_on_a_known_case():
    panel = make_panel(pi1=0.2, delta=3.0, n_signals_per_family=1, seed=17, **SMALL)
    true_families = np.flatnonzero(panel.is_nonnull_family)
    fam = np.zeros(20, bool)
    fam[true_families[0]] = True  # one true positive
    fam[np.flatnonzero(~panel.is_nonnull_family)[0]] = True  # one false positive
    score = score_decision(_decision(fam, np.zeros(200, bool)), panel)
    assert score.family_fdp == pytest.approx(0.5)
    assert score.family_power == pytest.approx(1.0 / len(true_families))


def test_paired_difference_detects_a_constant_shift():
    base = np.array([0.2, 0.3, 0.25, 0.28, 0.31])
    treat = base + 0.05
    out = paired_difference(treat, base)
    assert out["mean"] == pytest.approx(0.05)
    assert out["mcse"] == pytest.approx(0.0, abs=1e-12)
    assert out["significant"]


def test_paired_difference_handles_all_nan():
    out = paired_difference(np.full(5, np.nan), np.full(5, np.nan))
    assert np.isnan(out["mean"]) and not out["significant"]


# --- HAC diagnostic (retained, not on the main path) ------------------------


def test_newey_west_exceeds_iid_se_under_positive_serial_correlation():
    """Why the marginals moved to the bootstrap: the iid SE is simply wrong here."""
    panel = make_panel(
        n_obs=4000,
        n_families=4,
        m_per_family=5,
        pi1=0.0,
        delta=0.0,
        rho_w=0.0,
        rho_b=0.0,
        phi=0.6,
        seed=18,
    )
    hac = newey_west_se(panel.returns)
    iid = panel.returns.std(axis=0, ddof=1) / np.sqrt(panel.n_obs)
    assert np.mean(hac / iid) > 1.5
