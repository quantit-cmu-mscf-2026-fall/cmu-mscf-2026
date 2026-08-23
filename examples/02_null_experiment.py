"""Run the discovery pipeline on data with no real signal in it at all.

This is the cheapest test of any selection procedure: build a panel where the
honest answer is "nothing here", run the procedure, and see whether it says
so. A procedure that cannot return "nothing" is not a validation procedure —
it is a ranking.
"""

from __future__ import annotations

from capstone.backtest import sweep
from capstone.evaluate import benjamini_hochberg, bonferroni, sharpe_pvalue
from capstone.synth import make_panel


def main() -> None:
    panel = make_panel(n_dates=1260, n_assets=50, n_candidates=300, n_real=0, seed=0)
    print(panel)

    results = sweep(panel)
    n = len(results)

    best = results["sharpe"].max()
    p95 = results["sharpe"].quantile(0.95)
    mean = results["sharpe"].mean()

    print(f"\ncandidates: {n}")
    print(f"best Sharpe found: {best:.3f}")
    print(f"95th percentile Sharpe: {p95:.3f}")
    print(f"mean Sharpe: {mean:.3f}")

    # All candidates share the same date/asset grid, so n_obs is constant
    # across rows; sharpe_pvalue needs it to size the standard error.
    n_obs = int(results["n_obs"].iloc[0])
    pvalues = results["sharpe"].apply(lambda s: sharpe_pvalue(s, n_obs=n_obs))

    bh_rejected = benjamini_hochberg(pvalues, alpha=0.05)
    bonf_rejected = bonferroni(pvalues, alpha=0.05)
    naive_top10 = results["sharpe"].sort_values(ascending=False).head(10)

    print(f"\nBenjamini-Hochberg (alpha=0.05) rejections: {int(bh_rejected.sum())} / {n}")
    print(f"Bonferroni (alpha=0.05) rejections: {int(bonf_rejected.sum())} / {n}")
    print(f"naive 'top 10 by Sharpe' reports: {len(naive_top10)} discoveries (always 10)")

    print(
        "\nEvery one of these candidates was noise by construction: n_real=0, so "
        f"the ground truth is that nothing here predicts returns. The best Sharpe "
        f"above ({best:.2f}) is simply what looking {n} times at pure noise buys "
        "you — rank the candidates alone and it reads like a discovery. A "
        "procedure that cannot answer 'nothing here' when the honest answer is "
        "'nothing here' is not a validation procedure; it is a ranking."
    )


if __name__ == "__main__":
    main()
