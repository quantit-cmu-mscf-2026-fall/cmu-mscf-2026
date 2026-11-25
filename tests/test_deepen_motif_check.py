"""The deepen loop rejects a child whose realized edit is not the motif the round chose.

Before the check, a mismatch was screened, scored and counted under the realized motif:
credited to a motif nobody chose, never charged to the one that failed. Now it is
rejected ("motif_mismatch") before screening, never scored, and the memory counts it as
a failure of the chosen motif. Fake editor, fake scorer, temporary store: no model.
"""

from __future__ import annotations

from test_deepen import SEEDS, FakeScorer, ScriptedEditor, con, deepener, ledger  # noqa: F401

from capstone.factors.deepen import MOTIF_MISMATCH
from capstone.factors.memory import edit_motif, memory_events
from capstone.factors.tree import parse

PARENT = SEEDS[0]  # ts_mean(returns, 21)
RANK = "rank(ts_mean(returns, 21))"  # a rank_switch edit of PARENT
SHIFT = "delta(ts_mean(returns, 21), 5)"  # a temporal_shift edit of PARENT


def _round(con, editor, motif, scorer=None):  # noqa: F811
    d = deepener(con, editor, scorer, children=3)
    view = next(v for v in d._views() if d.trees[v.id] == parse(PARENT))
    d.choose = lambda: (view, motif)
    return d


def test_the_fixture_edits_are_the_motifs_the_tests_assume():
    assert edit_motif(parse(PARENT), parse(RANK)) == "rank_switch"
    assert edit_motif(parse(PARENT), parse(SHIFT)) == "temporal_shift"


def test_a_mismatched_edit_is_rejected_unscored_and_charged_to_the_chosen_motif(con, ledger):  # noqa: F811
    scorer = FakeScorer()
    d = _round(con, ScriptedEditor([SHIFT]), "rank_switch", scorer)
    scored_before = scorer.calls  # the parents
    (outcome,) = d.step()
    assert scorer.calls == scored_before  # never scored
    assert (outcome.status, outcome.reason) == ("rejected", MOTIF_MISMATCH)
    assert outcome.motif_intended == "rank_switch" and outcome.motif_realized == "temporal_shift"
    assert outcome.quality is None
    rows = con.execute(
        "SELECT status, reason FROM proposals WHERE edge_type = 'refined'"
    ).fetchall()
    assert [r["status"] for r in rows] == ["rejected"]
    assert rows[0]["reason"].startswith(MOTIF_MISMATCH)
    (event,) = memory_events(con, "test")
    assert event["status"] == "rejected" and event["motif_intended"] == "rank_switch"
    chosen = d.memory.stats(event["context"], "rank_switch")
    assert (chosen.attempts, chosen.failures) == (1, 1)  # a failure of the chosen motif
    assert d.memory.stats(event["context"], "temporal_shift").attempts == 0  # no credit


def test_a_matching_edit_still_proceeds(con, ledger):  # noqa: F811
    scorer = FakeScorer()
    d = _round(con, ScriptedEditor([RANK]), "rank_switch", scorer)
    scored_before = scorer.calls
    (outcome,) = d.step()
    assert scorer.calls == scored_before + 1
    assert outcome.reason != MOTIF_MISMATCH and outcome.quality is not None
    assert outcome.motif_realized == "rank_switch"


def test_a_mixed_round_rejects_only_the_mismatch(con, ledger):  # noqa: F811
    d = _round(con, ScriptedEditor([SHIFT, RANK]), "rank_switch")
    outcomes = d.step()
    assert [o.reason == MOTIF_MISMATCH for o in outcomes] == [True, False]


def test_the_family_cap_still_counts_a_mismatched_draft(con, ledger):  # noqa: F811
    d = deepener(con, ScriptedEditor([SHIFT, SHIFT]), children=2, parents=[PARENT])
    d.max_children_per_family = 5
    view = d._views()[0]
    d.choose = lambda: (view, "rank_switch")
    d.step()
    assert d.children_by_family[d.family(view.id)] == 2
    assert d.children_left(view.id) == 3
