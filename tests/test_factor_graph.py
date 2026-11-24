"""The strategy graph: bonds between stored strategies, their uniqueness, and moves.

No market data and no model: bonds are built from hand-written factors in a
temporary store.
"""

from __future__ import annotations

import pytest

from capstone.factors import graph, store
from capstone.factors.llm import FactorConfig
from capstone.factors.tree import parse

CONFIG = FactorConfig()
REVERSAL = "-ts_sum(returns, 5)"
TURNOVER = "ts_mean(volume / shares, 21)"
LOW_VOL = "-ts_std(returns, 21) * log(cap)"


@pytest.fixture
def con(tmp_path):
    con = store.connect(tmp_path / "factors.db")
    store.add_paper(con, "p", "Paper")
    yield con
    con.close()


@pytest.fixture
def hid(con):
    return store.add_hypothesis(
        con,
        store.Hypothesis(
            paper_key="p",
            observation="o",
            knowledge="k",
            justification="j",
            specification="s",
            falsification_condition="f",
            kind="market",
        ),
    )


def _factor(con, hid, expression):
    fid, _ = store.add_factor(con, parse(expression), hid)
    return fid


def _moves(con):
    return [dict(r) for r in con.execute("SELECT * FROM moves ORDER BY id")]


def test_a_bond_ties_its_members_trees_together_and_logs_the_move(con, hid):
    a, b = _factor(con, hid, REVERSAL), _factor(con, hid, TURNOVER)
    out = graph.bond(con, [a, b], CONFIG, rationale="reversal is stronger in liquid names")
    assert out.status == "stored"
    assert set(out.leaves) == {a, b} and out.member_share < 0.75 and out.bond_overlap == 0
    summary = graph.strategy(con, out.bond_id)
    assert summary["features"] == ["returns", "shares", "volume"]
    assert set(summary["leaves"]) == {a, b} and summary["members"] == sorted([a, b])
    (move,) = _moves(con)
    assert (move["action"], move["status"], move["result_id"]) == ("bond", "stored", out.bond_id)
    kinds = {r["member_kind"] for r in con.execute("SELECT member_kind FROM bond_members")}
    assert kinds == {"factor"}


def test_the_same_members_in_another_order_are_the_same_bond(con, hid):
    a, b = _factor(con, hid, REVERSAL), _factor(con, hid, TURNOVER)
    first = graph.bond(con, [a, b], CONFIG)
    again = graph.bond(con, [b, a], CONFIG)
    assert again.bond_id == first.bond_id and again.status == "duplicate"
    assert [m["status"] for m in _moves(con)] == ["stored", "duplicate"]


def test_a_member_inside_another_member_is_redundant(con, hid):
    inner = _factor(con, hid, "ts_mean(close, 5)")
    outer = _factor(con, hid, "rank(ts_mean(close, 5)) * log(cap)")
    out = graph.bond(con, [inner, outer], CONFIG)
    assert out.status == "rejected" and out.reason.startswith("redundant: 100%")
    assert out.member_share == 1.0 and out.nearest_members == (inner, outer)
    assert con.execute("SELECT COUNT(*) FROM bonds").fetchone()[0] == 0
    assert _moves(con)[0]["status"] == "rejected"


def test_bonds_join_bonds_and_a_repeat_molecule_is_not_original(con, hid):
    a, b, c = (_factor(con, hid, e) for e in (REVERSAL, TURNOVER, LOW_VOL))
    pair = graph.bond(con, [a, b], CONFIG)
    grown = graph.bond(con, [pair.bond_id, c], CONFIG)
    assert grown.status == "stored" and set(grown.leaves) == {a, b, c}
    assert grown.bond_overlap == pytest.approx(2 / 3) and grown.nearest_bond == pair.bond_id
    kinds = dict(
        con.execute(
            "SELECT member_id, member_kind FROM bond_members WHERE bond_id = ?", (grown.bond_id,)
        ).fetchall()
    )
    assert kinds == {pair.bond_id: "bond", c: "factor"}
    # The same three factors bonded directly: different members, same molecule.
    flat = graph.bond(con, [a, b, c], CONFIG)
    assert flat.status == "rejected" and flat.bond_overlap == 1.0
    assert flat.reason.startswith("not original")


def test_a_bond_over_the_leaf_limit_is_rejected(con, hid):
    ids = [_factor(con, hid, e) for e in (REVERSAL, TURNOVER, LOW_VOL, "ts_max(high, 63) / close")]
    out = graph.bond(con, ids, FactorConfig(max_bond_leaves=3))
    assert out.status == "rejected" and "too many factors: 4, limit 3" in out.reason


def test_bond_inputs_are_checked(con, hid):
    a = _factor(con, hid, REVERSAL)
    with pytest.raises(ValueError, match="two distinct"):
        graph.bond(con, [a, a], CONFIG)
    with pytest.raises(KeyError, match="no factor or bond"):
        graph.bond(con, [a, "nope"], CONFIG)
    with pytest.raises(ValueError, match="rule"):
        graph.bond(con, [a, _factor(con, hid, TURNOVER)], CONFIG, rule="max")


def test_deepen_and_switch_moves_are_logged_with_their_policy(con, hid):
    a = _factor(con, hid, REVERSAL)
    graph.record_move(con, "deepen", from_id=a, hypothesis_id=hid, reason="try 10 days", policy="p")
    graph.record_move(con, "switch", from_id=a, reason="no headroom left", policy="p")
    assert [(m["action"], m["policy"]) for m in _moves(con)] == [("deepen", "p"), ("switch", "p")]
    with pytest.raises(ValueError, match="action"):
        graph.record_move(con, "jump")


def test_a_factor_is_a_strategy_of_one_leaf(con, hid):
    a = _factor(con, hid, LOW_VOL)
    assert graph.leaves(con, a) == {a}
    assert graph.strategy(con, a)["features"] == ["cap", "returns"]


def test_shipped_config_has_the_bond_limits():
    from capstone.factors.llm import load_config

    config = load_config()
    assert (config.max_bond_leaves, config.max_member_share, config.max_bond_overlap) == (
        4,
        0.75,
        0.75,
    )
    assert graph.RULES == ("rank_mean",)
