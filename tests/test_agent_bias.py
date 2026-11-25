"""Agent bias analysis on small hand-made inputs: no data files, no Drive."""

import json

import pandas as pd
import pytest

import research.agent_bias as cli
from capstone.analysis import agent_bias as ab

LEVEL = "(rank(-(shift(ts_sum(returns, 735), 21))) * rank(-(log(close))))"
CHANGE = "rank(-(delta(log(close), 21)))"
VOL = "-(rank(ts_mean((volume / shares), 252)))"
MOM = "rank(-(shift(ts_sum(returns, 231), 21)))"
FACTORS = {
    "a": ("f1", LEVEL),
    "b": ("f1", CHANGE),
    "c": ("f2", VOL),
    "d": ("f2", MOM),
    "e": ("f2", f"({LEVEL} * rank(ts_std(returns, 252)))"),
    "g": ("f2", f"({MOM} * rank(-(log(cap))))"),
}


def make_summary(scale=1.0):
    rows = []
    for fid, (fam, expr) in FACTORS.items():
        for k in range(2):
            rows.append(
                {
                    "variant_id": f"{fid}-{k}",
                    "factor_id": fid,
                    "hypothesis_id": fam,
                    "expression": expr,
                    "sharpe_net": scale * (0.5 + 0.1 * k),
                    "sharpe_gross": scale * (1.0 + 0.1 * k),
                }
            )
    return pd.DataFrame(rows).set_index("variant_id")


RUN = {
    "seed_factors": 4,
    "papers": [{"key": "p", "factors": 4}],
    "families": ["f1", "f2"],
    "deepen": {
        "children": {"admitted": 1, "rejected": 3},
        "decisions": [
            {"parent": "a", "motif": "interaction"},
            {"parent": "a", "motif": "interaction"},
            {"parent": "e", "motif": "rank_switch"},
        ],
    },
}


def attribution():
    rows = []
    for fid in ("a", "b", "e"):
        rows.append(dict(variant=f"{fid}-0", rebase=False, csu=False, strict=False, gross=1.4))
        rows.append(dict(variant=f"{fid}-0", rebase=True, csu=False, strict=False, gross=0.5))
    return pd.DataFrame(rows)


def test_lineage_rule_and_check():
    s = make_summary()
    lineage = ab.assign_lineage(s, RUN)
    assert list(lineage[lineage == "seed"].index) == ["a", "b", "c", "d"]
    assert list(lineage[lineage == "deepen_child"].index) == ["e", "g"]
    check = ab.lineage_check(s, RUN, lineage)
    assert (check["n_seed"], check["n_child"]) == (4, 2)
    assert check["children_in_one_family"] and check["seed_count_matches_papers"]
    assert check["n_parents_seed"] == 1 and check["n_parents_child"] == 1


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("close", True),
        ("log(close)", True),
        ("ts_mean(shift(low, 21), 63)", True),
        ("delta(log(close), 21)", False),
        ("close / ts_max(close, 252)", False),
        ("ts_sum(returns, 21)", False),
        ("log(cap)", False),
        ("close * volume", False),
    ],
)
def test_price_level_term(expression, expected):
    assert ab.price_level_term(expression) is expected


def test_factor_table_flags_and_scores():
    s = make_summary()
    t = ab.factor_table(s, make_summary(0.5), RUN)
    assert t.loc["a", "has_price_level_term"] and t.loc["a", "has_low_price_leg"]
    assert not t.loc["b", "has_price_level_term"]
    assert t.loc["a", "interaction"] and t.loc["a", "n_terms"] == 2
    assert t.loc["a", "best_net_old"] == pytest.approx(0.6)
    assert t.loc["a", "net_change"] == pytest.approx(-0.3)
    assert t.loc["e", "lineage"] == "deepen_child"


def test_archetype_counts_and_chi_square():
    t = ab.factor_table(make_summary(), make_summary(), RUN)
    counts = ab.counts_by(t, "archetype")
    assert counts.to_numpy().sum() == 6
    assert counts["seed"].sum() == 4 and counts["deepen_child"].sum() == 2
    chi = ab.chi_square(counts, "seed", "deepen_child")
    assert chi["small"] is True
    terms = ab.term_counts(t)
    assert terms.to_numpy().sum() == t["n_terms"].sum()


def test_hhi():
    assert ab.hhi([5, 5]) == pytest.approx(0.5)
    assert ab.hhi([10, 0, 0]) == pytest.approx(1.0)
    assert ab.hhi([1, 1, 1, 1]) == pytest.approx(0.25)


def test_concentration():
    s = make_summary()
    t = ab.factor_table(s, s, RUN)
    c = ab.concentration(t, s, RUN)
    assert c["variants_per_family"] == {"f1": 4, "f2": 8}
    assert c["largest_family_share"] == pytest.approx(8 / 12)
    assert c["children_per_family"] == {"f2": 2}
    assert c["motifs"] == {"interaction": 2, "rank_switch": 1}
    assert c["distinct_parents"] == 2
    assert c["hhi_variants"] == pytest.approx((4 / 12) ** 2 + (8 / 12) ** 2)


def test_rebase_effect_by_lineage_and_flag():
    s = make_summary()
    t = ab.factor_table(s, s, RUN)
    effect = ab.rebase_effect(attribution(), t)
    assert effect["drop"].tolist() == pytest.approx([0.9, 0.9, 0.9])
    r = ab.rebase_summary(effect)
    assert r["by_lineage"]["seed"]["n"] == 2 and r["by_lineage"]["deepen_child"]["n"] == 1
    assert r["by_price_level"]["True"]["n"] == 2
    assert r["all"]["median_rebase"] == pytest.approx(0.5)


def inputs():
    s = make_summary()
    return dict(
        summary=s,
        run_summary=RUN,
        new=make_summary(0.5),
        placebo=make_summary(0.1),
        attribution=attribution(),
    )


def test_analyse_statements_all_sourced():
    tables, stats, statements = ab.analyse(**inputs())
    assert statements
    assert all(st.source in stats for st in statements)
    assert all(st.caveat for st in statements)
    assert set(tables["archetype_real_vs_placebo"].columns) >= {"real_minus_placebo"}


def test_cli_logs_before_results_and_writes(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(cli.runlog, "log_run", lambda **kw: order.append(("log", kw)))
    real_read = pd.read_parquet

    def read(path, *a, **k):
        order.append(("read", None))
        return real_read(path, *a, **k)

    monkeypatch.setattr(cli.pd, "read_parquet", read)
    i = inputs()
    paths = {}
    for key in ("summary", "new", "placebo"):
        paths[key] = tmp_path / f"{key}.parquet"
        i[key].to_parquet(paths[key])
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(RUN))
    att = tmp_path / "rows.csv"
    i["attribution"].to_csv(att, index=False)
    out = cli.run(paths["summary"], run_path, paths["new"], paths["placebo"], att, tmp_path / "o")
    assert order[0][0] == "log"
    entry = order[0][1]
    assert entry["name"] == "agent-bias" and entry["tags"] == ["synthetic", "methodology"]
    assert set(entry["params"]) == {"summary", "run_summary", "new", "placebo", "attribution"}
    assert (out / "STATEMENTS.md").exists() and (out / "stats.json").exists()
    assert (out / "data" / "factors.parquet").exists()
