"""Readable evidence for reference allocation; research and executable returns stay distinct."""

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    "equal_weight": "Equal weight",
    "shrunk_max_sharpe": "Shrunk maximum Sharpe",
    "median_window_sharpe": "Median-window Sharpe",
}


def report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output)
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for mode, label in LABELS.items():
        frame = pd.read_csv(
            root / "assumed_base" / mode / "monthly.csv", index_col="month", parse_dates=True
        )
        axes[0].plot(frame.index, frame.equity, label=label)
        wealth = np.r_[100000, frame.equity]
        axes[1].plot(frame.index, (wealth / np.maximum.accumulate(wealth) - 1)[1:] * 100)
    axes[0].set(
        title="Published reference pool: assumed base costs, 1996–2020",
        ylabel="Synthetic funded account ($)",
    )
    axes[0].legend()
    axes[1].set(title="Drawdowns, including initial capital", ylabel="Drawdown (%)")
    fig.savefig(root / "portfolio-results.png", dpi=150)
    plt.close(fig)
    correlation = pd.read_csv(root / "correlations.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set(
        xticks=range(5),
        yticks=range(5),
        xticklabels=correlation.columns,
        yticklabels=correlation.index,
        title="Reference correlations, 1996–2020",
    )
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center")
    for ax, mode in zip(axes[1:], list(LABELS)[1:], strict=True):
        frame = pd.read_csv(
            root / "assumed_base" / mode / "monthly.csv", index_col="month", parse_dates=True
        )
        columns = [n for n in frame if n.startswith("weight_")]
        ax.stackplot(
            frame.index,
            frame[columns].to_numpy().T,
            labels=[n.removeprefix("weight_") for n in columns],
        )
        ax.set(title=LABELS[mode], ylim=(0, 1.01), ylabel="Factor sleeve coefficients")
        ax.legend(fontsize=8, loc="upper left")
    fig.savefig(root / "relationships-and-weights.png", dpi=150)
    plt.close(fig)

    def number(value):
        return "undefined" if value is None else f"{value:.2f}"

    def table(period):
        rows = []
        for key, item in results["allocations"].items():
            scenario, mode = key.split("/")
            r = item[period]
            rows.append(
                f"<tr><td>{escape(scenario)}</td><td>{LABELS[mode]}</td>"
                f"<td>{r['cagr']:.2%}</td><td>{number(r['sharpe'])}</td>"
                f"<td>{r['annual_volatility']:.2%}</td>"
                f"<td>{r['maximum_drawdown']:.2%}</td>"
                f"<td>{r['mean_invested']:.1%}</td><td>{r['active_range']}</td></tr>"
            )
        return (
            "<table><tr><th>Cost view</th><th>Allocation rule</th><th>CAGR</th>"
            "<th>Excess Sharpe</th><th>Volatility</th><th>Max drawdown</th>"
            "<th>Mean coefficient sum</th><th>Active range</th></tr>" + "".join(rows) + "</table>"
        )

    uncertainty = ""
    contributions = ""
    for mode in list(LABELS)[1:]:
        item = results["allocations"]["assumed_base/" + mode]
        interval = item["paired_uncertainty_vs_equal"]["sharpe_difference_95_interval"]
        uncertainty += f"<li>{LABELS[mode]}: {escape(str(interval))}</li>"
        contributions += f"<h3>{LABELS[mode]}</h3><table><tr><th>Factor</th>"
        contributions += "<th>Mean coefficient</th><th>Factor P&amp;L before proxy costs</th></tr>"
        for name, c in item["contribution"].items():
            contributions += (
                f"<tr><td>{name}</td><td>{c['mean_coefficient']:.1%}</td>"
                f"<td>${c['factor_pnl_dollars_before_proxy_costs']:,.0f}</td></tr>"
            )
        contributions += "</table>"
    references = "".join(
        f"<tr><td>{name}</td><td>{r['cagr']:.2%}</td><td>{number(r['sharpe'])}</td>"
        f"<td>{r['maximum_drawdown']:.2%}</td></tr>"
        for name, r in results["standalone_gross"].items()
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Published strategies: portfolio allocation results</title>
<style>body{{font:16px/1.55 system-ui;max-width:1180px;margin:36px auto;padding:0 22px;
color:#172e42;background:#f7f9fb}}h1,h2,h3{{line-height:1.2}}table{{border-collapse:collapse;
width:100%;font-size:14px;background:white}}td,th{{padding:9px;border:1px solid #d6dfe7;
text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;height:auto}}
.note{{background:#e8eff6;padding:18px;border-left:4px solid #315e80}}a{{color:#175781}}</style>
<h1>Published strategies: portfolio allocation results</h1>
<p class="note">Five documented reference factors; variable strategy membership and weights.
This tests allocation on published return histories. It does not verify the agent's
stock construction, executable costs, future profitability, or formal admission.</p>
<h2>What ran</h2><p>HML (value), RMW (profitability), CMA (investment), Mom (momentum),
and U.S. BAB (defensive beta). Quarterly decisions use only completed prior months,
up to ten years; median-window optimization uses three-year training windows.
All rules use the same 10% estimated volatility ceiling, no coefficient leverage,
and zero-weight/cash permissions. This is a ceiling, not identical realized risk.</p>
<p>Maximum Sharpe uses 50% shrinkage of estimated mean returns toward zero and
Ledoit–Wolf covariance shrinkage. Median-window Sharpe uses the median of combined
portfolio training-window scores. Both use deterministic multi-start numerical search;
a global optimum is not guaranteed. Neither objective maximizes diversification.</p>
<h2>Full comparison: 1996–2020</h2>{table("full_1996_2020")}
<img src="portfolio-results.png" alt="Portfolio wealth and drawdowns">
<h2>Later slice: 2016–2020</h2><p>Shown separately after the cited methods were published.
These are revised, reused histories; this is not untouched out-of-sample certification.</p>
{table("later_2016_2020")}
<h2>Relationships and actual allocations</h2>
<img src="relationships-and-weights.png" alt="Factor correlations and changing weights">
<h2>Standalone references, published gross convention</h2><table><tr><th>Factor</th>
<th>Synthetic funded CAGR</th><th>Excess Sharpe</th><th>Max drawdown</th></tr>{references}</table>
<h2>Factor contributions under assumed base costs</h2>{contributions}
<p>Factor P&amp;L is the dollar contribution on the actual simulated allocation path,
before proxy costs and cash interest. It is not a leave-one-out causal estimate.</p>
<h2>Uncertainty</h2><p>Paired 95% bootstrap intervals for Sharpe difference versus equal
weighting, base assumptions, 1,999 draws of 12-month blocks, seed 404:</p><ul>{uncertainty}</ul>
<p>Intervals describe sampling uncertainty conditional on these allocations; they
do not correct the unknown historical factor search or certify an optimizer winner.</p>
<h2>Accounting and limits</h2><p>Account return equals French RF once plus weighted
published self-financing factor P&amp;L, minus proxy costs. Sharpe subtracts RF once.
Coefficients are not known stock gross exposures: BAB embeds variable leverage.
The base sensitivity assumes 10bp per invested coefficient per month plus 6bp
per dollar of coefficient switching; doubled costs use 20bp and 12bp.
These are assumptions, not measured stock turnover, impact, or borrow costs.
Published gross follows the vendors' embedded conventions and is not necessarily
a frictionless reconstruction. No constant stock cost is inferred from these files.</p>
<p>Sources are current-vintage reconstructed histories. Only 1990–2020 observations
are retained. No private holdout, licensed refresh, model calls, or broker connection.
No S&amp;P 500 adaptation inherits the references' validation.</p>
<p>Code revision: <code>{escape(str(manifest["git_commit"]))}</code>.
<a href="manifest.json">Frozen inputs and code hashes</a> ·
<a href="results.json">All metrics, periods, and contributions</a> ·
<a href="correlations.csv">Correlation data</a></p>
<p>Sources:
<a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html"
>French factors</a>,
<a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html">momentum</a>,
<a href="https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly"
>AQR BAB</a>.</p>
</html>"""
    (root / "RESULTS.html").write_text(html, encoding="utf-8")
