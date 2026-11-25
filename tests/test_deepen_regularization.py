"""Opt-in parent-selection penalties in deepen: family crowding and formula size.

A parent's selection score is multiplied by (1 + c_f) ** -lambda * exp(-mu * nodes).
Off by default (nothing changes); the penalty reaches only the ranking, never the
reported or logged quality, pool admission or the memory context.

Fake editor, fake scorer, a temporary store and ledger: no model, no data.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from test_deepen import CONFIG, FakeScorer, ScriptedEditor
from test_deepen_family_cap import (
    EDIT_ROUNDS,
    FAMILY_SEEDS,
    PINNED_ASKED,
    families_store,
)

from capstone.factors import store
from capstone.factors.deepen import Deepener
from capstone.factors.tree import factor_id, node_count, parse


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


def two_families(path, seeds, *, scorer=None, **kwargs):
    """A store with one hypothesis per seed; returns (deepener, [factor id per seed])."""
    con = store.connect(path)
    store.add_paper(con, "paper:1", "A paper")
    for i, expression in enumerate(seeds):
        hid = store.add_hypothesis(con, store.Hypothesis("paper:1", f"h{i}", "k", "j", "s", "f"))
        store.add_factor(con, parse(expression), hid)
    d = Deepener(con, CONFIG, scorer or FakeScorer(), ScriptedEditor(), run_id="t", **kwargs)
    ids = [factor_id(parse(e)) for e in seeds]
    for fid in ids:
        assert d.add_parent(fid)
    return d, ids


EQUAL = ["ts_mean(returns, 21)", "ts_std(volume, 63)"]


def run_trace(path, **kwargs):
    con, names = families_store(path)
    editor = ScriptedEditor(*EDIT_ROUNDS)
    cfg = replace(CONFIG, memory_children_per_parent=2)
    d = Deepener(con, cfg, FakeScorer(), editor, run_id="test", seed=0, **kwargs)
    for expression in FAMILY_SEEDS.values():
        assert d.add_parent(factor_id(parse(expression)))
    for _ in range(6):
        d.step()
    return [(names[d.family(factor_id(p))], m, n) for p, m, n, *_ in editor.asked]


def test_zero_penalties_leave_selection_unchanged(tmp_path, ledger):
    default = run_trace(tmp_path / "a.db")
    zero = run_trace(tmp_path / "b.db", family_penalty=0.0, complexity_penalty=0.0)
    assert default == zero == PINNED_ASKED


def test_default_scale_is_exactly_one(tmp_path, ledger):
    d, ids = two_families(tmp_path / "a.db", EQUAL)
    assert [v.selection_scale for v in d._views()] == [1.0, 1.0]


def test_a_crowded_family_is_chosen_less(tmp_path, ledger):
    d, ids = two_families(tmp_path / "a.db", EQUAL, family_penalty=3.0)
    d.children_by_family[d.family(ids[0])] += 2
    for _ in range(5):  # whatever the rng does among motifs, never the crowded family
        assert d.choose()[0].id == ids[1]
    d.children_by_family[d.family(ids[1])] += 5
    assert d.choose()[0].id == ids[0]


def test_without_the_penalty_a_crowded_family_is_not_avoided(tmp_path, ledger):
    d, ids = two_families(tmp_path / "a.db", EQUAL)
    d.children_by_family[d.family(ids[0])] += 2
    assert [v.selection_scale for v in d._views()] == [1.0, 1.0]


def test_a_larger_parent_ranks_below_an_equal_smaller_one(tmp_path, ledger):
    small, big = "rank(close)", "rank(ts_mean(close, 5) / ts_std(volume, 21)) + rank(open)"
    assert node_count(parse(big)) > node_count(parse(small))
    d, ids = two_families(tmp_path / "a.db", [big, small], complexity_penalty=0.3)
    assert [c.view.id for c in d.candidates()][0] == ids[1]
    assert d.choose()[0].id == ids[1]
    plain, pids = two_families(tmp_path / "b.db", [big, small])
    assert [v.selection_scale for v in plain._views()] == [1.0, 1.0]


def test_scale_formula(tmp_path, ledger):
    d, ids = two_families(tmp_path / "a.db", EQUAL, family_penalty=2.0, complexity_penalty=0.1)
    d.children_by_family[d.family(ids[0])] = 3
    nodes = node_count(parse(EQUAL[0]))
    assert d.selection_scale(ids[0]) == pytest.approx(4**-2.0 * math.exp(-0.1 * nodes))
    assert d.selection_scale(ids[1]) == pytest.approx(math.exp(-0.1 * node_count(parse(EQUAL[1]))))


def test_agent_candidates_order_reflects_the_penalty(tmp_path, ledger):
    d, ids = two_families(tmp_path / "a.db", EQUAL, family_penalty=3.0)
    d.children_by_family[d.family(ids[0])] += 2
    assert [c.view.id for c in d.candidates()] == [ids[1], ids[0]]
    d.family_penalty = 0.0
    assert sorted(c.view.id for c in d.candidates()) == sorted(ids)


def test_penalty_never_changes_quality_context_or_pool(tmp_path, ledger):
    plain, ids = two_families(tmp_path / "a.db", EQUAL)
    pen, pids = two_families(tmp_path / "b.db", EQUAL, family_penalty=4.0, complexity_penalty=0.5)
    pen.children_by_family[pen.family(pids[0])] += 3
    for a, b in zip(plain._views(), pen._views(), strict=True):
        assert (a.id, a.quality, a.context, a.rho_max) == (b.id, b.quality, b.context, b.rho_max)
        assert b.selection_scale < 1.0
    assert plain.pool.quality == pen.pool.quality
    # the agent is shown the parent's own quality
    for c in pen.candidates():
        assert c.view.quality == 0.15


def test_children_are_counted_without_a_cap(tmp_path, ledger):
    con, names = families_store(tmp_path / "a.db")
    editor = ScriptedEditor(*EDIT_ROUNDS)
    cfg = replace(CONFIG, memory_children_per_parent=2)
    d = Deepener(con, cfg, FakeScorer(), editor, run_id="t", seed=0)
    for expression in FAMILY_SEEDS.values():
        d.add_parent(factor_id(parse(expression)))
    assert d.max_children_per_family is None
    for _ in range(3):
        d.step()
    assert sum(d.children_by_family.values()) == 6


@pytest.mark.parametrize("name", ["family_penalty", "complexity_penalty"])
@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True])
def test_penalties_refuse_bad_values(tmp_path, name, bad):
    con = store.connect(tmp_path / "a.db")
    with pytest.raises(ValueError, match=name):
        Deepener(con, CONFIG, FakeScorer(), None, run_id="t", **{name: bad})


def test_factor_config_and_digest_are_untouched():
    from dataclasses import fields

    from capstone.factors.llm import FactorConfig

    names = {f.name for f in fields(FactorConfig)}
    assert not {n for n in names if "penalty" in n}
