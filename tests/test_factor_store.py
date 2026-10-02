"""The factor store keeps each factor once and every proposal of it."""

from __future__ import annotations

import pytest

from capstone.factors import store
from capstone.factors.tree import factor_id, parse


@pytest.fixture
def con(tmp_path):
    con = store.connect(tmp_path / "factors.db")
    store.add_paper(con, "arxiv:2502.16789", "AlphaAgent", url="https://arxiv.org/abs/2502.16789")
    yield con
    con.close()


def _hypothesis(paper_key="arxiv:2502.16789", **overrides):
    fields = dict(
        observation="Recent losers rebound over the following week.",
        knowledge="Short-term reversal from liquidity provision.",
        justification="Market makers are paid to absorb order imbalances.",
        specification="Negative 5-day return, ranked across stocks.",
        falsification_condition="No negative relation between 5-day and next-day returns.",
    )
    fields.update(overrides)
    return store.Hypothesis(paper_key=paper_key, **fields)


def test_default_path_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPSTONE_FACTOR_DB", str(tmp_path / "x.db"))
    assert store.default_path() == tmp_path / "x.db"
    monkeypatch.delenv("CAPSTONE_FACTOR_DB")
    assert store.default_path().parts[-2:] == ("experiments", "factors.db")


def test_hypothesis_id_is_content_based_and_fields_are_required():
    assert _hypothesis().id == _hypothesis().id
    assert _hypothesis().id != _hypothesis(knowledge="Something else.").id
    with pytest.raises(ValueError, match="knowledge"):
        _hypothesis(knowledge="  ")


def test_a_factor_proposed_twice_is_stored_once_with_two_proposals(con):
    first = store.add_hypothesis(con, _hypothesis())
    second = store.add_hypothesis(con, _hypothesis(specification="Rank of minus 5-day return."))

    fid, new = store.add_factor(con, parse("-(close / shift(close, 5) - 1)"), first)
    same, again = store.add_factor(
        con, parse("-(close / shift(close, 5) - 1)"), second, expression="-(close/shift(close,5)-1)"
    )

    assert (new, again) == (True, False)
    assert fid == same
    assert len(store.factors(con)) == 1
    assert [p["status"] for p in store.proposals(con)] == ["stored", "duplicate"]
    assert {row["hypothesis_id"] for row in store.lineage(con, fid)} == {first, second}


def test_commuted_expression_is_the_same_factor(con):
    hid = store.add_hypothesis(con, _hypothesis())
    store.add_factor(con, parse("rank(close) * volume"), hid)
    _, new = store.add_factor(con, parse("volume * rank(close)"), hid)
    assert not new


def test_rejections_are_kept_but_not_stored_as_factors(con):
    hid = store.add_hypothesis(con, _hypothesis())
    store.reject(con, hid, "ts_mean(close, 2.5)", "parse error: window must be an integer")
    node = parse("(close - open) / ((high - low) + 0.001)")
    store.reject(con, hid, "(close-open)/((high-low)+0.001)", "copies alpha_101", node=node)

    assert store.factors(con) == []
    rejected = store.proposals(con, status="rejected")
    assert len(rejected) == 2
    assert factor_id(node) in rejected[1]["reason"]


def test_lineage_reaches_the_paper(con):
    hid = store.add_hypothesis(con, _hypothesis(), model="claude-test", prompt_version="v1")
    parent, _ = store.add_factor(con, parse("close / shift(close, 5) - 1"), hid)
    child, _ = store.add_factor(
        con,
        parse("rank(close / shift(close, 5) - 1)"),
        hid,
        parent_factor_id=parent,
        edge_type="refined",
    )

    (edge,) = store.lineage(con, child)
    assert edge["paper_key"] == "arxiv:2502.16789"
    assert edge["parent_factor_id"] == parent
    assert edge["edge_type"] == "refined"


def test_unknown_edge_type_is_refused(con):
    hid = store.add_hypothesis(con, _hypothesis())
    with pytest.raises(ValueError, match="edge_type"):
        store.add_factor(con, parse("rank(close)"), hid, edge_type="mutated")


def test_a_hypothesis_needs_its_paper_first(con):
    with pytest.raises(Exception, match="FOREIGN KEY"):
        store.add_hypothesis(con, _hypothesis(paper_key="arxiv:not-added"))


def test_store_survives_reopening_and_trees_read_back(tmp_path):
    path = tmp_path / "factors.db"
    con = store.connect(path)
    store.add_paper(con, "p", "Paper")
    hid = store.add_hypothesis(con, _hypothesis(paper_key="p"))
    fid, _ = store.add_factor(con, parse("ts_corr(rank(high), rank(volume), 3) * 0.00001"), hid)
    con.close()

    reopened = store.connect(path)
    assert [name for name, _ in store.factor_trees(reopened)] == [fid]
    assert factor_id(store.factor_trees(reopened)[0][1]) == fid
