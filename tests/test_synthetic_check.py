"""The synthetic panel, its scorer and the free random editor used by the step 3 check."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from capstone.factors.deepen import applicable, correlations
from capstone.factors.edits import RandomEditor
from capstone.factors.interpret import evaluate
from capstone.factors.memory import PRODUCIBLE_MOTIFS, edit_motif
from capstone.factors.synthetic import (
    PLANTED,
    PanelScorer,
    PanelSpec,
    forward_returns,
    icir,
    make_panel,
)
from capstone.factors.tree import FIELDS, identity_key, parse

DESIGN = tomllib.loads(
    (
        Path(__file__).resolve().parent.parent / "research" / "learned_memory_search_check.toml"
    ).read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def panels():
    p = DESIGN["panel"]
    spec = PanelSpec(n_stocks=p["n_stocks"], n_days=p["n_days"], beta=p["beta"], seed=p["seed"])
    return make_panel(spec), make_panel(PanelSpec(**{**spec.__dict__, "beta": 0.0}))


def test_the_panel_has_every_field_and_is_deterministic(panels):
    planted, _ = panels
    assert set(planted) == set(FIELDS)
    p = DESIGN["panel"]
    again = make_panel(PanelSpec(p["n_stocks"], p["n_days"], p["beta"], seed=p["seed"]))
    np.testing.assert_array_equal(again["returns"].to_numpy(), planted["returns"].to_numpy())


def test_planted_icir_is_about_0_3_and_null_is_small(panels):
    for panel, low, high in [(panels[0], 0.27, 0.33), (panels[1], -0.15, 0.15)]:
        fwd = forward_returns(panel["returns"], 20)
        assert low <= icir(evaluate(parse(PLANTED), panel), fwd, 5) <= high


def test_the_null_panel_shares_the_planted_panels_draws(panels):
    planted, null = panels
    np.testing.assert_array_equal(planted["volume"].to_numpy(), null["volume"].to_numpy())
    assert not np.allclose(planted["returns"].to_numpy(), null["returns"].to_numpy())


def test_the_scorer_caches_by_identity_and_samples_fixed_cells(panels):
    scorer = PanelScorer(panels[0], sample_cells=2_000)
    a = scorer(parse("ts_mean(returns, 5) + volume"))
    assert a is scorer(parse("volume + ts_mean(returns, 5)")) and scorer.computed == 1
    assert a.values.shape == (2_000,)
    np.testing.assert_array_equal(PanelScorer(panels[0], sample_cells=2_000).sample, scorer.sample)


def test_sampled_correlations_track_full_panel_ones(panels):
    scorer = PanelScorer(panels[0], sample_cells=20_000)
    for left, right in [
        ("ts_mean(returns, 5)", "ts_sum(returns, 10)"),
        ("volume", "ts_mean(volume, 21)"),
    ]:
        full = correlations(
            evaluate(parse(left), panels[0]).to_numpy().ravel(),
            evaluate(parse(right), panels[0]).to_numpy().ravel()[None, :],
        )[0]
        sampled = correlations(scorer(parse(left)).values, scorer(parse(right)).values[None, :])[0]
        assert sampled == pytest.approx(full, abs=0.02)


def test_the_random_editor_makes_the_asked_edit_at_least_95_percent():
    editor, walker = RandomEditor(0), RandomEditor(1)
    rng = np.random.default_rng(2)
    parents = [parse(e) for e in DESIGN["search"]["seeds"]]
    for e in DESIGN["search"]["seeds"][:5]:  # deeper parents, a few edits from a seed
        tree = parse(e)
        for _ in range(3):
            draft = walker.drafts(tree, PRODUCIBLE_MOTIFS[int(rng.integers(9))], 1, [], [])[0]
            tree = draft.tree or tree
        parents.append(tree)
    for motif in PRODUCIBLE_MOTIFS:
        hits = total = 0
        for parent in parents:
            if not applicable(parent, motif):
                continue
            for d in editor.drafts(parent, motif, 10, [], []):
                total += 1
                hits += d.tree is not None and edit_motif(parent, d.tree) == motif
        assert hits / total >= 0.95, motif


def test_the_random_editor_is_reproducible_and_never_returns_the_parent():
    parent = parse(DESIGN["search"]["seeds"][0])
    for motif in PRODUCIBLE_MOTIFS:
        a = RandomEditor(7).drafts(parent, motif, 3, [], [])
        b = RandomEditor(7).drafts(parent, motif, 3, [], [])
        assert [d.expression for d in a] == [d.expression for d in b]
        assert all(identity_key(d.tree) != identity_key(parent) for d in a if d.tree)
