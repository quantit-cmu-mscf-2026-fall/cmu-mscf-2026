"""Funnel stage 1-3 distributions on synthetic family sets: planted families, a null, refusals."""

import json

import numpy as np
import pandas as pd
import pytest
from gate_helpers import N_BOOT, market
from scipy.stats import norm

from capstone import runlog
from capstone.analysis import funnel_distributions as fd
from capstone.analysis.outputs import write_outputs


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path / "ledger"))


def frames(m, *, through_holdout=False):
    returns = m.returns if through_holdout else m.returns.loc[m.returns.index <= m.search_end]
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    summary = pd.DataFrame(
        {
            "factor_id": [c.split("-")[0] for c in returns.columns],
            "hypothesis_id": m.family.reindex(returns.columns).to_numpy(),
            "sharpe_net": sharpe.to_numpy(),
            "pvalue_screen": norm.sf(sharpe.to_numpy() * np.sqrt(len(returns) / 252)),
            "mean_turnover": 0.01,
        },
        index=pd.Index(returns.columns, name="variant_id"),
    )
    return returns, summary


def planted():
    return market(seed=3, share_real=0.4, sharpe=1.5)


def test_planted_families_pass_stage1_more_than_null_ones():
    m = planted()
    returns, summary = frames(m)
    scored = fd.score_run(returns, summary, n_boot=N_BOOT)
    real = m.truth.reindex(scored.families.index)
    assert real.sum() > 0
    stage1 = scored.families["stage1"]
    assert stage1[real].mean() > stage1[~real].mean()
    assert (
        scored.families["p1_bootstrap"][real].median()
        < scored.families["p1_bootstrap"][~real].median()
    )
    assert scored.families["p1_bh_adjusted"].between(0, 1).all()


def test_tables_have_the_documented_columns():
    returns, summary = frames(planted())
    classes = pd.DataFrame({"factor_id": summary["factor_id"].unique(), "kind": "momentum"})
    scored = fd.score_run(returns, summary, n_boot=N_BOOT, classes=classes)
    assert {"sharpe_net", "pvalue_screen", "mean_turnover", "family", "class_kind"} <= set(
        scored.variants.columns
    )
    assert {"k", "best", "p1_bootstrap", "p1_bh_adjusted", "stage1", "stage2", "stage3"} <= set(
        scored.families.columns
    )
    assert "graded_psr" in scored.families.columns
    q = fd.quantile_table(scored.variants)
    assert ("ALL", "sharpe_net") in q.index
    assert q.loc[("ALL", "sharpe_net"), "q000"] <= q.loc[("ALL", "sharpe_net"), "q100"]


def test_null_world_comparison_computes_and_writes(tmp_path):
    real_m, null_m = planted(), market(seed=4, share_real=0.0)
    r, rs = frames(real_m)
    n, ns = frames(null_m)
    real = fd.score_run(r, rs, n_boot=N_BOOT)
    null = fd.score_run(n, ns, n_boot=N_BOOT)
    tables, stats, statements = fd.build_outputs(real, null=null, inputs=("m",), null_inputs=("n",))
    cmp = tables["comparison"]
    assert 0 <= cmp["ks_statistic"].iloc[0] <= 1
    assert (
        stats["stage1_survivors_real_vs_null"].value["real"]
        >= (stats["stage1_survivors_real_vs_null"].value["null"])
    )
    assert stats["sharpe_quantile_gaps"].value["q095"] > 0
    out = write_outputs(
        tmp_path / "o", title="t", tables=tables, stats=stats, statements=statements
    )
    assert (out / "STATEMENTS.md").exists()
    assert all(s.source in stats for s in statements)


def test_dates_past_search_end_are_refused():
    m = planted()
    returns, summary = frames(m, through_holdout=True)
    with pytest.raises(ValueError, match="past search_end"):
        fd.score_run(returns, summary, search_end=m.search_end, n_boot=N_BOOT)


def test_ledger_entry_precedes_the_results(tmp_path, monkeypatch):
    m = planted()
    returns, summary = frames(m)
    mp, sp = tmp_path / "m.parquet", tmp_path / "s.parquet"
    returns.to_parquet(mp)
    summary.to_parquet(sp)
    seen = []

    def spy(*a, **k):
        seen.append(runlog.read_ledger() if hasattr(runlog, "read_ledger") else _ledger_lines())
        raise RuntimeError("stop")

    monkeypatch.setattr(fd.funnel, "family_statistics", spy)
    with pytest.raises(RuntimeError, match="stop"):
        fd.run_analysis(mp, sp, run="r", n_boot=N_BOOT, search_end=str(m.search_end.date()))
    assert len(seen) == 1 and len(seen[0]) == 1
    assert _ledger_lines()[0]["name"] == "funnel-distributions"


def _ledger_lines():
    path = runlog.ledger_path()
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_run_analysis_end_to_end(tmp_path):
    m = planted()
    returns, summary = frames(m)
    mp, sp = tmp_path / "m.parquet", tmp_path / "s.parquet"
    returns.to_parquet(mp)
    summary.to_parquet(sp)
    tables, stats, statements = fd.run_analysis(
        mp,
        sp,
        run="r",
        null_matrix_path=mp,
        null_summary_path=sp,
        n_boot=N_BOOT,
        search_end=str(m.search_end.date()),
    )
    assert "comparison" in tables and "null_stage1_survivors" in stats
    entries = _ledger_lines()
    assert len(entries) == 1 and "synthetic" in entries[0]["tags"]
    assert entries[0]["params"]["n_boot"] == N_BOOT
    assert len(statements) >= 6


def test_a_variant_too_short_for_the_bootstrap_is_dropped_and_counted():
    returns, summary = frames(planted())
    short = returns.columns[0]
    returns = returns.copy()
    returns.loc[returns.index[100:], short] = np.nan  # trades for 100 days only
    scored = fd.score_run(returns, summary, n_boot=N_BOOT)
    assert scored.dropped == (short,)
    assert scored.n_variants == returns.shape[1] - 1
    _, stats, statements = fd.build_outputs(scored, null=None, inputs=("m",))
    assert stats["dropped_variants"].value == [short]
    assert any(s.source == "dropped_variants" for s in statements)


def _gross_frames(m, shift=0.0):
    returns, summary = frames(m)
    summary = summary.assign(sharpe_gross=summary["sharpe_net"] + shift)
    return returns, summary


def _fake(gross, stage1_n, n=20):
    """A Scored whose variant gross Sharpe is a ramp starting at `gross` and n-1 more up to +1."""
    v = pd.DataFrame(
        {
            "sharpe_net": np.linspace(-1, 1, n),
            "sharpe_gross": np.linspace(gross, gross + 1, n),
            "pvalue_screen": np.linspace(0.01, 0.9, n),
            "family": [f"f{i % 4}" for i in range(n)],
        }
    )
    f = pd.DataFrame(
        {
            "p1_bootstrap": np.linspace(0.01, 0.9, 4),
            "stage1": [True] * stage1_n + [False] * (4 - stage1_n),
            "stage2": [False] * 4,
            "stage3": [False] * 4,
        },
        index=[f"f{i}" for i in range(4)],
    )
    return fd.Scored(v, f, 0.5, n)


def test_band_counts_seeds_exceeded_on_a_constructed_case():
    real = _fake(1.0, 3)
    nulls = [_fake(0.0, 0), _fake(0.5, 1), _fake(2.0, 0)]  # real exceeds the first two on gross
    band = fd.null_band_table(real, nulls)
    g = band.loc[("variant_sharpe_gross", 0.5)]
    assert g["n_null"] == 3 and g["n_null_below_real"] == 2
    assert g["null_min"] == pytest.approx(0.5) and g["null_max"] == pytest.approx(2.5)
    assert g["share_null_below_real"] == pytest.approx(2 / 3)
    s = band.loc[("stage1_survivors", -1.0)]
    assert (s["real"], s["null_min"], s["null_max"], s["n_null_below_real"]) == (3, 0, 1, 3)
    _, stats, statements = fd.build_outputs(real, nulls=nulls, inputs=("m",), null_inputs=("n",))
    assert stats["null_band_gross_sharpe"].value["median"]["k"] == 2
    assert any("2 of 3 seeds" in st.text for st in statements)


def test_three_null_worlds_give_an_ordered_sourced_band(tmp_path):
    real = fd.score_run(*_gross_frames(planted(), 0.1), n_boot=N_BOOT)
    nulls = [
        fd.score_run(*_gross_frames(market(seed=s, share_real=0.0)), n_boot=N_BOOT)
        for s in (4, 5, 6)
    ]
    tables, stats, statements = fd.build_outputs(
        real, nulls=nulls, inputs=("m",), null_inputs=("n",)
    )
    b = tables["null_band"]
    assert (b["null_min"] <= b["null_median"]).all() and (b["null_median"] <= b["null_max"]).all()
    assert (b["n_null"] == 3).all() and len(tables["null_seeds"]) == 3
    assert all(s.source in stats for s in statements)
    assert any("net comparisons are dominated" in s.text for s in statements)
    assert stats["n_null_runs"].value == 3
    write_outputs(tmp_path / "o", title="t", tables=tables, stats=stats, statements=statements)


def test_single_null_in_a_list_equals_the_single_null_path():
    real = fd.score_run(*_gross_frames(planted()), n_boot=N_BOOT)
    null = fd.score_run(*_gross_frames(market(seed=4, share_real=0.0)), n_boot=N_BOOT)
    t1, s1, st1 = fd.build_outputs(real, null=null, inputs=("m",), null_inputs=("n",))
    t2, s2, st2 = fd.build_outputs(real, nulls=[null], inputs=("m",), null_inputs=("n",))
    assert "null_band" not in t2 and set(t1) == set(t2) and set(s1) == set(s2)
    assert [s.text for s in st1] == [s.text for s in st2]


def test_run_analysis_with_null_dirs_logs_one_entry_with_digests(tmp_path):
    m = planted()
    returns, summary = _gross_frames(m)
    mp, sp = tmp_path / "m.parquet", tmp_path / "s.parquet"
    returns.to_parquet(mp)
    summary.to_parquet(sp)
    dirs = []
    for i in range(2):
        d = tmp_path / f"seed{i}"
        d.mkdir()
        r, s = _gross_frames(market(seed=10 + i, share_real=0.0))
        r.to_parquet(d / "matrix.parquet")
        s.to_parquet(d / "summary.parquet")
        dirs.append(d)
    tables, stats, _ = fd.run_analysis(
        mp,
        sp,
        run="r",
        null_dirs=dirs,
        n_boot=N_BOOT,
        null_n_boot=N_BOOT // 2,
        search_end=str(m.search_end.date()),
    )
    assert "null_band" in tables and stats["n_null_runs"].value == 2
    entries = _ledger_lines()
    assert len(entries) == 1
    p = entries[0]["params"]
    assert p["null_n_boot"] == N_BOOT // 2
    assert set(p["null_dirs"]) == {str(d) for d in dirs}
    assert all(len(v["matrix_digest"]) == 16 for v in p["null_dirs"].values())
