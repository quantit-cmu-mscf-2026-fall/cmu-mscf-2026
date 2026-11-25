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
