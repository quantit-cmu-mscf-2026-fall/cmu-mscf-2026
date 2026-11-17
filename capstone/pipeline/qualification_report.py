"""Readable qualification evidence. Missing performance is never plotted as zero."""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from capstone.pipeline.evidence import write_json


def _plots(root, summary, candidates, portfolios, benchmarks):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "capstone-plot-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.hashsalt": summary["study_hash"], "font.family": "DejaVu Sans"})
    groups = [
        ("Reference", "reference", None),
        ("Refinement", "alternate_weighting", None),
        ("Value", "fundamental", "value"),
        ("Profitability", "fundamental", "profitability"),
    ]
    evaluated, blocked = [], []
    for _, construction, method in groups:
        rows = [
            c
            for c in candidates
            if c["construction"] == construction and (not method or c["method"] == method)
        ]
        evaluated.append(sum(c["status"] == "evaluated" for c in rows))
        blocked.append(len(rows) - evaluated[-1])
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [g[0] for g in groups]
    ax.barh(labels, evaluated, color="#20796b", label="Evaluated")
    ax.barh(labels, blocked, left=evaluated, color="#bd664e", label="Blocked / incomplete")
    ax.set_xlabel("Reserved strategy/profile combinations")
    ax.set_title(
        "Synthetic engineering fixture"
        if summary["synthetic_only"]
        else "Historical study coverage"
    )
    ax.set_xlim(0, 13)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(root / "coverage.png", dpi=150)
    fig.savefig(root / "coverage.svg", metadata={"Date": None})
    plt.close(fig)
    complete = [
        r
        for r in portfolios
        if r.get("completed")
        and r.get("allocation") == "equal_family"
        and r.get("scenario") == "base"
    ]
    if complete:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for column, side in enumerate(("long_only", "long_short")):
            series = [(r["kind"], r["artifacts"]) for r in complete if r["profile"] == side]
            series += [
                (r["kind"], r["artifacts"])
                for r in benchmarks
                if r.get("completed")
                and r["profile"] == side
                and r["kind"] == "known_momentum_12_1"
            ]
            for label, artifact in series:
                daily = pd.read_parquet(root / artifact / "daily.parquet")
                wealth = (1 + daily.net).cumprod()
                axes[0, column].plot(wealth.index, wealth, label=label)
                path = pd.concat(
                    [pd.Series([1.0], index=[wealth.index[0] - pd.Timedelta(days=1)]), wealth]
                )
                axes[1, column].plot(path.index, (path / path.cummax() - 1) * 100, label=label)
            axes[0, column].set_title(side.replace("_", " "))
            axes[0, column].set_ylabel("Net wealth / initial capital")
            axes[1, column].set_ylabel("Drawdown (%)")
            axes[0, column].legend(fontsize=8)
        fig.suptitle(
            "SYNTHETIC — engineering evidence only"
            if summary["synthetic_only"]
            else "Qualified historical comparisons after declared costs"
        )
        fig.tight_layout()
        fig.savefig(root / "portfolio-results.png", dpi=150)
        fig.savefig(root / "portfolio-results.svg", metadata={"Date": None})
        plt.close(fig)
    if (root / "candidate-returns.parquet").exists():
        matrix = pd.read_parquet(root / "candidate-returns.parquet")
        corr = matrix.corr(min_periods=252)
        if any(pd.notna(corr.iloc[i, j]) for i in range(len(corr)) for j in range(i)):
            fig, ax = plt.subplots(figsize=(10, 9))
            plot = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
            mapping = {r.get("strategy_id"): r for r in candidates}
            labels = []
            for name in corr:
                row = mapping[name]
                labels.append(
                    f"{row['method']} {row['side']} {row['construction']} "
                    f"{row.get('tail', '')} {row.get('weighting', '')}"
                )
            ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
            prefix = "SYNTHETIC — engineering evidence only\n" if summary["synthetic_only"] else ""
            ax.set_title(prefix + "Net daily correlations — descriptive, never used for selection")
            fig.colorbar(plot, ax=ax, fraction=0.04)
            fig.tight_layout()
            fig.savefig(root / "correlations.png", dpi=160)
            fig.savefig(root / "correlations.svg", metadata={"Date": None})
            plt.close(fig)
        if (root / "correlations.json").exists():
            rows = json.loads((root / "correlations.json").read_text())["rolling_252_month_ends"]
            rows = [r for r in rows if r.get("median") is not None or r.get("p90") is not None]
            if rows:
                frame = pd.DataFrame(rows).set_index("as_of")
                frame.index = pd.to_datetime(frame.index)
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(frame.index, frame["median"], label="Median pair")
                ax.plot(frame.index, frame.p90, label="90th percentile pair")
                ax.axhline(0.8, color="#bd664e", linestyle="--", label="Grouping threshold")
                ax.set_ylim(-1, 1)
                prefix = (
                    "SYNTHETIC — engineering evidence only\n" if summary["synthetic_only"] else ""
                )
                ax.set_title(prefix + "Prior 252-session correlations at month ends")
                ax.legend()
                fig.tight_layout()
                fig.savefig(root / "rolling-correlations.png", dpi=150)
                plt.close(fig)
    write_json(root / "plot-runtime.json", {"matplotlib": version("matplotlib"), "backend": "Agg"})


def _number(value, *, percent=False):
    return "N/A" if value is None else (f"{value * 100:.2f}%" if percent else f"{value:.2f}")


def _html(text):
    """Render only the small, escaped Markdown vocabulary emitted by this report."""
    output, table = [], False
    for line in text.splitlines():
        escaped = html.escape(line)
        escaped = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"!\[([^]]*)\]\(([A-Za-z0-9._/-]+)\)", r"<img alt='\1' src='\2'>", escaped)
        escaped = re.sub(r"\[([^]]*)\]\(([A-Za-z0-9._/-]+)\)", r"<a href='\2'>\1</a>", escaped)
        if line.startswith("|"):
            if not table:
                output.append("<div class='table'><table>")
                table = True
            if set(line) <= set("|-: "):
                continue
            output.append(
                "<tr>"
                + "".join("<td>" + cell.strip() + "</td>" for cell in escaped.strip("|").split("|"))
                + "</tr>"
            )
            continue
        if table:
            output.append("</table></div>")
            table = False
        if line.startswith("## "):
            output.append("<h2>" + escaped[3:] + "</h2>")
        elif line.startswith("# "):
            output.append("<h1>" + escaped[2:] + "</h1>")
        elif line:
            output.append("<p>" + escaped + "</p>")
    if table:
        output.append("</table></div>")
    return "\n".join(output)


def render(root, summary, candidates, portfolios, benchmarks, preflight, *, legacy=None):
    root = Path(root)
    _plots(root, summary, candidates, portfolios, benchmarks)
    title = (
        "Synthetic engineering demonstration"
        if summary["synthetic_only"]
        else "Historical qualification report"
    )
    counts = Counter(row["status"] for row in candidates)
    lines = [
        f"# {title}",
        "",
        f"Code: `{summary['release']['commit']}`; study: `{summary['study_hash']}`.",
        "",
        "Synthetic prices cannot establish investment performance."
        if summary["synthetic_only"]
        else (
            "Historical evaluation is blocked. "
            "No corrected-strategy return estimates were produced."
            if not summary["evaluated_candidates"]
            else "These are reused discovery results, not an untouched out-of-sample evaluation."
        ),
        "",
        "Real strategy admission: **false**. Broker connection: **false**. "
        "Holdout opened: **false**.",
        "",
        f"Reserved candidates: **36**. Evaluated: **{summary['evaluated_candidates']}**. "
        f"Status counts: `{dict(counts)}`.",
        "Eligible research opportunities in this run's reconciled ledger: "
        f"**{summary.get('trial_count', 'unavailable')}**.",
        "",
        "![Study coverage](coverage.png)",
        "",
        "## Source qualification",
        "",
        "| Source | Qualified | Blockers |",
        "|---|---|---|",
    ]
    for name, row in preflight.get("sources", {}).items():
        lines.append(f"| {name} | {row['qualified']} | {', '.join(row['blockers']) or 'None'} |")
    if preflight.get("descriptor_present") is False:
        lines += [
            "",
            "The source descriptor `historical-sources.json` is missing. "
            "Unmapped sources are not evidence that the strategy code crashed. "
            "Bind verified existing sources before requesting additional exports.",
        ]
    if not summary["synthetic_only"] and not summary["historical_observations_loaded"]:
        lines += [
            "",
            "The backtest has not started. Price returns alone cannot establish when dividends "
            "became spendable cash or what shares/cash a merger delivered. "
            "Supply observed payment dates and corporate-action terms to run fixed-share replay. "
            "Current restated accounting also cannot establish original historical vintages.",
        ]
    if preflight.get("global_blockers"):
        lines += ["", "Global blockers: " + "; ".join(preflight["global_blockers"]) + "."]
    lines += [
        "",
        "## Corrected strategies",
        "",
        "CAGR, Sharpe and drawdown use the base cost view. N/A means unavailable, never zero.",
        "",
        "| Method | Profile | Variant | Candidate ID | Status | CAGR | Sharpe | Max drawdown |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in candidates:
        base = row.get("scenarios", {}).get("base", {})
        lines.append(
            f"| {row['method']} | {row['side']} | {row['construction']} "
            f"{row.get('tail', '')} {row.get('weighting', '')} | "
            f"{row['candidate_id']} | {row['status']} | "
            f"{_number(base.get('cagr'), percent=True)} | {_number(base.get('sharpe'))} | "
            f"{_number(base.get('maximum_drawdown'), percent=True)} |"
        )
    lines += [
        "",
        "## Portfolios",
        "",
        "Empty training selections hold cash. Cash returns do not establish an invested "
        "strategy's performance or formal admission.",
        "",
        "| Basket | Profile | Allocation | Cost view | Status | CAGR | Sharpe | Max drawdown |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in portfolios:
        lines.append(
            f"| {row['kind']} | {row['profile']} | {row.get('allocation', 'N/A')} | "
            f"{row.get('scenario', 'N/A')} | {row['status']} | "
            f"{_number(row.get('cagr'), percent=True)} | {_number(row.get('sharpe'))} | "
            f"{_number(row.get('maximum_drawdown'), percent=True)} |"
        )
    if (root / "portfolio-results.png").exists():
        lines += ["", "![Portfolio comparisons](portfolio-results.png)"]
    for filename, label in (
        ("correlations.png", "Strategy correlations"),
        ("rolling-correlations.png", "Rolling correlations"),
    ):
        if (root / filename).exists():
            lines += ["", f"![{label}]({filename})"]
    if (root / "candidate-returns.parquet").exists() and not (root / "correlations.png").exists():
        lines += [
            "",
            "No strategy pairs have 252 overlapping observed daily returns with "
            "defined correlation; correlation charts are unavailable.",
        ]
    if legacy:
        basket = legacy["basket"]
        lines += [
            "",
            "## Earlier exploratory basket — separate comparison",
            "",
            f"Saved discovery period: {legacy['common_start']} to {legacy['common_end']}. "
            f"CAGR {_number(basket['cagr'], percent=True)}, Sharpe {_number(basket['sharpe'])}, "
            f"maximum drawdown {_number(basket['max_drawdown'], percent=True)}.",
            "The old analysis selected in-sample representatives and used a different engine. "
            "Differences cannot be attributed entirely to the corrections.",
        ]
    lines += [
        "",
        "## Next steps and limits",
        "",
        "Supply checksum-bound exchange sessions, observed cash/conversion events, and qualified "
        "accounting/factor vintages through the shared-data workflow (`sd.load`). "
        "Both profiles use hypothetical $100,000 books. Borrowing/financing and execution costs "
        "are declared assumptions; locate availability is unverified. Unqualified fundamentals "
        "cannot support validation. Missing source data is never replaced by "
        "simulated observations.",
        "The research BH screen is separate from formal admission. HAC uncertainty is not "
        "selection-adjusted. Leave-one-family-out results reallocate capital and are not "
        "additive realized-fill attribution.",
        "",
        "[Summary](summary.json) · [Candidate evidence](candidates.json) · "
        "[Source preflight](preflight.json) · [Trial accounting](accounting.json) · "
        "[Benchmarks](benchmarks.json) · [Portfolios](portfolios.json) · "
        "[Archived rehearsals](rehearsals.json)",
        "",
    ]
    text = "\n".join(lines)
    (root / "REPORT.md").write_text(text, encoding="utf-8")
    # A standalone browser view preserves the same text and local charts without a server.
    (root / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Historical qualification</title>"
        "<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;"
        "padding:0 24px;background:#fafaf7;color:#172629}"
        "p{line-height:1.5}img{max-width:100%}.table{overflow:auto}"
        "table{border-collapse:collapse;width:100%}td{padding:8px;border-bottom:1px solid #ddd}"
        "tr:first-child{font-weight:bold;background:#e7eeea}code{overflow-wrap:anywhere}</style>"
        + _html(text),
        encoding="utf-8",
    )
    return root / "REPORT.md"
