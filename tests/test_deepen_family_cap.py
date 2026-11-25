"""F-9: `[deepen] max_children_per_family`, a cap on the children deepen proposes per
family (a parent's hypothesis, the gate's family).

- unset (the default) deepen is unchanged: a pinned trace of parents, edits and
  outcomes, the same as with a cap no family reaches;
- with a cap, no family proposes more children than the cap, and a round asks only
  for what its family has left;
- the rounds a capped family can't use go to the other families; when every family
  is at the cap, deepen stops (`FamiliesCapped`).

The TOML key and the driver's handling of `FamiliesCapped` are follow-ups.

Fake editor, fake scorer, a temporary store and ledger: no model, no data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
from test_deepen import (  # noqa: F401 - the fixture is autouse here too
    CONFIG,
    AgentClient,
    FakeScorer,
    ScriptedEditor,
    _decision,
    edits_match_the_motif,
)

from capstone.factors import store
from capstone.factors.deepen import DeepenAgent, Deepener, FamiliesCapped, decide_prompt
from capstone.factors.tree import factor_id, parse

# One seed per family (hypothesis): three families.
FAMILY_SEEDS = {
    "h1": "ts_mean(returns, 21)",
    "h2": "ts_std(volume, 63)",
    "h3": "rank(close / open)",
}
# A fresh pair of edits each round, so children are new factors, not duplicates.
EDIT_ROUNDS = [[f"ts_mean(close, {3 + 2 * i})", f"rank(volume) * {i + 2}"] for i in range(50)]
ROUNDS = 6
CHILDREN = 2

# The uncapped trace on this fixture, per round: the parent's family, the edit and the
# children asked for; then each child's status. The trace depends on factor ids:
# candidates are ordered by id and the editor's picks are seeded. It is identical under
# PYTHONHASHSEED 0, 1 and 2, so a change here means deepen's behaviour changed.
PINNED_ASKED = [
    ("h2", "normalization", 2),
    ("h3", "normalization", 2),
    ("h1", "rank_switch", 2),
    ("h2", "nesting", 2),
    ("h3", "other", 2),
    ("h1", "normalization", 2),
]
PINNED_STATUSES = ["admitted"] + ["rejected"] * 11


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


def families_store(path):
    """A store with one hypothesis (family) per seed; returns (con, family of hypothesis)."""
    con = store.connect(path)
    store.add_paper(con, "paper:1", "A paper")
    names = {}
    for name, expression in FAMILY_SEEDS.items():
        hid = store.add_hypothesis(con, store.Hypothesis("paper:1", name, "k", "j", "s", "f"))
        store.add_factor(con, parse(expression), hid)
        names[hid] = name
    return con, names


def deepener(path, cap=None, *, children=CHILDREN, seed=0):
    con, names = families_store(path)
    editor = ScriptedEditor(*EDIT_ROUNDS)
    cfg = replace(CONFIG, memory_children_per_parent=children)
    d = Deepener(
        con, cfg, FakeScorer(), editor, run_id="test", seed=seed, max_children_per_family=cap
    )
    for expression in FAMILY_SEEDS.values():
        assert d.add_parent(factor_id(parse(expression)))
    return d, editor, names


def trace(path, cap=None, rounds=ROUNDS):
    """Run up to `rounds` rounds; per round (family, motif, n asked), every outcome."""
    d, editor, names = deepener(path, cap)
    outcomes = []
    for _ in range(rounds):
        try:
            outcomes.extend(d.step())
        except FamiliesCapped:
            break
    asked = [
        (names[d.family(factor_id(parent))], motif, n) for parent, motif, n, *_ in editor.asked
    ]
    return d, asked, outcomes


def test_the_default_is_the_pinned_uncapped_behaviour(tmp_path, ledger):
    _, asked, outcomes = trace(tmp_path / "a.db")
    assert asked == PINNED_ASKED
    assert [o.status for o in outcomes] == PINNED_STATUSES


def test_a_cap_no_family_reaches_changes_nothing(tmp_path, ledger):
    _, asked, outcomes = trace(tmp_path / "a.db")
    _, capped_asked, capped = trace(tmp_path / "b.db", cap=ROUNDS * CHILDREN)
    assert capped_asked == asked
    assert capped == outcomes


def test_the_cap_is_respected(tmp_path, ledger):
    d, asked, outcomes = trace(tmp_path / "a.db", cap=3, rounds=20)
    per_family = Counter()
    for family, _, n in asked:
        per_family[family] += n
    assert per_family == Counter({"h1": 3, "h2": 3, "h3": 3})
    assert all(n <= CHILDREN for _, _, n in asked)
    assert 1 in {n for _, _, n in asked}  # a round asks only for what is left
    assert len(outcomes) == 9
    assert max(d.children_by_family.values()) == 3


def test_the_rounds_left_go_to_the_other_families(tmp_path, ledger):
    # Cap 2 with 2 children a round: each family gets one round, then the next
    # rounds must pick another family until every family has had its share.
    d, asked, outcomes = trace(tmp_path / "a.db", cap=CHILDREN, rounds=ROUNDS)
    families = [family for family, _, _ in asked]
    assert sorted(families) == ["h1", "h2", "h3"]  # three rounds, one per family
    assert len(outcomes) == 3 * CHILDREN
    with pytest.raises(FamiliesCapped):
        d.step()


def test_a_capped_family_offers_no_candidates_to_the_agent(tmp_path, ledger):
    d, _, _ = deepener(tmp_path / "a.db", cap=CHILDREN)
    d.step()
    capped = next(f for f, n in d.children_by_family.items() if n >= CHILDREN)
    shown = {d.family(c.view.id) for c in d.candidates()}
    assert capped not in shown and len(shown) == 2


def test_candidates_put_a_losing_seed_last_when_quality_is_signed(tmp_path, ledger):
    d, _, _ = deepener(tmp_path / "a.db")
    d.config = replace(d.config, memory_signed_quality=True)
    losing = factor_id(parse(FAMILY_SEEDS["h1"]))
    d.pool.quality[losing] = -3.0
    for fid in d.pool.quality:
        if fid != losing:
            d.pool.quality[fid] = 1.5
    assert [c.view.id for c in d.candidates()][-1] == losing


def test_the_agent_is_asked_only_for_what_the_family_has_left(tmp_path, ledger):
    # Cap 1, two children a round. The agent answers two edits, the second broken: it is
    # asked for 1, so the broken extra costs no repair call and is not dropped unrecorded.
    d, _, names = deepener(tmp_path / "a.db", cap=1)
    parent = FAMILY_SEEDS["h3"]
    client = AgentClient(_decision(parent, "nesting", ["ts_mean(rank(close / open), 5)", "(("]))
    d.agent = DeepenAgent(client, d.config, max_calls=10, usage=Counter())
    assert {c.limit for c in d.candidates()} == {1}
    outcomes = d.step()
    prompt = client.calls[0]["messages"][0]["content"]
    assert "at most 1 edit" in prompt
    assert len(client.calls) == 1  # no repair spent on an edit the cap would drop
    assert len(outcomes) == 1
    assert d.children_by_family == Counter({d.family(factor_id(parse(parent))): 1})
    assert names[d.family(factor_id(parse(parent)))] == "h3"


def test_without_a_cap_the_agent_prompt_is_unchanged(tmp_path, ledger):
    d, _, _ = deepener(tmp_path / "a.db")
    assert {c.limit for c in d.candidates()} == {None}
    capped, _, _ = deepener(tmp_path / "b.db", cap=ROUNDS * CHILDREN)
    assert decide_prompt(capped.candidates(), CHILDREN) == decide_prompt(d.candidates(), CHILDREN)


def test_a_bad_cap_is_refused(tmp_path):
    con, _ = families_store(tmp_path / "a.db")
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_children_per_family"):
            Deepener(con, CONFIG, FakeScorer(), None, run_id="t", max_children_per_family=bad)


def test_deepen_ledger_entries_record_the_family(tmp_path, ledger):
    from capstone.runlog import read_entries

    d, _, names = deepener(tmp_path / "a.db")
    d.step()
    entries = read_entries()
    assert entries
    assert all(e["params"]["hypothesis_id"] in names for e in entries)
