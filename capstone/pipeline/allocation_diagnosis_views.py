"""Tables and scientific figures for saved allocation-diagnosis evidence."""

from html import escape

import numpy as np

from capstone.pipeline.reference_allocation import statistics

LABELS = {
    "equal_weight": "Equal weight",
    "shrunk_max_sharpe": "Shrunk maximum Sharpe",
    "median_window_sharpe": "Median-window Sharpe",
}
SCENARIOS = {
    "published_gross": "Published gross",
    "assumed_base": "Assumed base",
    "assumed_double": "Assumed double",
}
COLORS = ("#0072B2", "#D55E00", "#009E73")


def _number(value, kind="number"):
    if value is None or not np.isfinite(value):
        return "undefined"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "pp":
        return f"{value * 100:+.3f} pp"
    if kind == "money":
        return f"${value:,.0f}"
    return f"{value:.3f}"


def _table(caption, headers, rows):
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
    return (
        '<div class="table-wrap"><table><caption>'
        + escape(caption)
        + "</caption><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _figures(root, results, legacy, corrected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for color, (mode, label) in zip(COLORS, LABELS.items(), strict=True):
        for version, frame, style in (
            ("V1", legacy["assumed_base/" + mode], "--"),
            ("V2", corrected[mode], "-"),
        ):
            axes[0].plot(frame.index, frame.equity, style, color=color, label=f"{label}: {version}")
            wealth = np.r_[100000.0, frame.equity]
            dd = (wealth / np.maximum.accumulate(wealth) - 1)[1:]
            axes[1].plot(frame.index, dd * 100, style, color=color)
    axes[0].set(
        title="Preserved V1 and corrected V2: assumed base costs, 1996–2020",
        ylabel="Synthetic funded account ($)",
    )
    axes[0].legend(ncols=2, fontsize=8)
    axes[1].set(title="Drawdowns include initial $100,000", ylabel="Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(
        root / "version-comparison.png", dpi=150, metadata={"Software": "allocation diagnosis"}
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(3)
    for color, (mode, label) in zip(COLORS, LABELS.items(), strict=True):
        points = [
            results["frozen_target_costs"][mode][s]["full_1996_2020"]["cagr"] * 100
            for s in SCENARIOS
        ]
        axes[0].plot(x, points, "o-", color=color, label=label)
    axes[0].set(
        xticks=x,
        xticklabels=["Gross", "Base", "Double"],
        title="One frozen V1 base target schedule per rule",
        ylabel="CAGR (%)",
    )
    axes[0].legend(fontsize=8)
    old_vol = [statistics(legacy["assumed_base/" + m])["annual_volatility"] * 100 for m in LABELS]
    overlay = [
        results["prior_risk_overlay"][m]["full_1996_2020"]["annual_volatility"] * 100
        for m in LABELS
    ]
    axes[1].bar(x - 0.18, old_vol, 0.36, label="Original V1 base", color="#0072B2")
    axes[1].bar(x + 0.18, overlay, 0.36, label="Prior 5% ceiling", color="#E69F00")
    axes[1].set(
        xticks=x,
        xticklabels=["Equal", "Maximum", "Median"],
        title="Same prior ceiling does not equal realized risk",
        ylabel="Realized annual vol (%)",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(
        root / "cost-and-risk-controls.png", dpi=150, metadata={"Software": "allocation diagnosis"}
    )
    plt.close(fig)


def diagnostic_tables(results):
    forecast_rows, risk_forecast_rows, concentration_rows, identity_rows, solver_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for key, diagnostic in results["diagnostics"].items():
        version, mode = key.split("/")
        label = version.upper() + ": " + LABELS[mode]
        f = diagnostic["forecast"]
        rmse = f["quarter_arithmetic_rmse"]
        forecast_rows.append(
            [
                label,
                str(f["scored_quarters"]),
                *[_number(rmse[n], "percent") for n in ("raw", "half_shrunk", "zero_mean")],
            ]
        )
        eligible = [r for r in f["forecasts"] if r["status"] == "forecast_created"]
        scored = [r for r in f["following_quarter_scores"] if r["status"] == "scored"]
        if eligible and scored:
            risk_forecast_rows.append(
                [
                    label,
                    _number(
                        np.mean([r["sample_covariance_annual_volatility"] for r in eligible]),
                        "percent",
                    ),
                    _number(
                        np.mean([r["shrunk_covariance_annual_volatility"] for r in eligible]),
                        "percent",
                    ),
                    _number(
                        np.mean([r["actual_three_month_annualized_scatter"] for r in scored]),
                        "percent",
                    ),
                ]
            )
        c = diagnostic["concentration"]
        concentration_rows.append(
            [
                label,
                _number(c["mean_normalized_max_weight"], "percent"),
                _number(c["maximum_normalized_max_weight"], "percent"),
                _number(c["mean_effective_number"]),
                _number(c["BAB_signed_pnl_fraction"], "percent"),
                str(c["cash_only_months"]),
            ]
        )
        d = diagnostic["risk_sharpe_identity"]
        identity_rows.append(
            [
                label,
                _number(d["annual_excess_mean_difference"], "pp"),
                _number(d["volatility_term"], "pp"),
                _number(d["sharpe_term"], "pp"),
            ]
        )
        solver = diagnostic["solver"]
        if solver:
            gaps = [
                r["best_lattice_objective"] - r["best_local_objective"]
                for r in solver
                if r.get("best_lattice_objective") is not None
                and r.get("best_local_objective") is not None
            ]
            counts = sorted({r["lattice_points_evaluated"] for r in solver})
            solver_rows.append(
                [
                    label,
                    str(len(solver)),
                    escape(str(counts)),
                    str(sum(r.get("local_successes", 0) for r in solver))
                    + "/"
                    + str(sum(r.get("local_attempts", 0) for r in solver)),
                    _number(max(gaps) if gaps else None),
                    str(sum(r.get("feasible_prior_included", False) for r in solver)),
                ]
            )
    forecast_table = _table(
        "Following-quarter arithmetic factor-P&L forecast error",
        [
            "Version and rule",
            "Scored quarters",
            "Raw mean RMSE",
            "Half-shrunk mean RMSE",
            "Zero-mean RMSE",
        ],
        forecast_rows,
    )
    risk_forecast_table = _table(
        "Descriptive covariance forecast versus subsequent scatter",
        [
            "Version and rule",
            "Mean sample-cov annual vol forecast",
            "Mean shrunk-cov annual vol forecast",
            "Mean actual three-month annualized scatter",
        ],
        risk_forecast_rows,
    )
    concentration_table = _table(
        "Concentration and BAB contribution, base costs",
        [
            "Version and rule",
            "Mean largest normalized weight",
            "Largest normalized weight",
            "Mean effective sleeve count",
            "BAB signed factor-P&L fraction",
            "Cash-only months",
        ],
        concentration_rows,
    )
    identity_table = _table(
        "Exact annual arithmetic excess-mean identity versus original equal weight",
        ["Version and rule", "Excess mean difference", "Volatility term", "Sharpe term"],
        identity_rows,
    )
    solver_table = _table(
        "V2 declared bounded-search coverage",
        [
            "Rule",
            "Decisions",
            "Distinct lattice counts",
            "Successful local attempts / total",
            "Largest lattice minus local objective",
            "Feasible no-trade candidates",
        ],
        solver_rows,
    )

    return forecast_table, risk_forecast_table, concentration_table, identity_table, solver_table


def pool_tables(pool):
    pool_rows, period_rows = [], []
    for name, r in pool["overall"].items():
        interval = r["annual_excess_mean_95_marginal_interval"]
        pool_rows.append(
            [
                escape(name),
                _number(r["annual_excess_mean"], "percent"),
                " to ".join(_number(v, "percent") for v in interval),
                _number(r["sharpe"]),
                _number(r["annual_volatility"], "percent"),
            ]
        )
    for period, factors in pool["five_year_periods"].items():
        for name, r in factors.items():
            period_rows.append(
                [
                    escape(period),
                    escape(name),
                    _number(r["annual_excess_mean"], "percent"),
                    _number(r["sharpe"]),
                    " to ".join(
                        _number(v, "percent") for v in r["annual_excess_mean_95_marginal_interval"]
                    ),
                ]
            )
    pool_table = _table(
        "All five references: published excess returns, 1996–2020",
        [
            "Factor",
            "Annual arithmetic excess mean",
            "Marginal 95% mean interval",
            "Excess Sharpe",
            "Annual vol",
        ],
        pool_rows,
    )
    period_table = _table(
        "All fixed five-year factor slices; none removed after inspection",
        [
            "Period",
            "Factor",
            "Annual arithmetic excess mean",
            "Excess Sharpe",
            "Marginal 95% mean interval",
        ],
        period_rows,
    )
    return pool_table, period_table
