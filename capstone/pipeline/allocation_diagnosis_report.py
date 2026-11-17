"""Offline readable evidence for the frozen allocation diagnosis, without winner selection."""

import json
from html import escape
from pathlib import Path

import pandas as pd

from capstone.pipeline.allocation_diagnosis_views import (
    LABELS,
    SCENARIOS,
    _figures,
    _number,
    _table,
    diagnostic_tables,
    pool_tables,
)
from capstone.pipeline.reference_allocation import statistics


def _read_history(root, path):
    return pd.read_csv(
        root / path / "monthly.csv",
        index_col="month",
        parse_dates=True,
        float_precision="round_trip",
    )


def report(output):
    """Render existing evidence only; never run a replay, forecast, optimizer or download."""
    root = Path(output)
    results = json.loads((root / "results.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    legacy = {
        key: _read_history(root, "legacy-reproduction/" + key)
        for key in results["legacy_equivalence"]
    }
    corrected = {mode: _read_history(root, "corrected-v2/assumed_base/" + mode) for mode in LABELS}
    decisions = {
        mode: json.loads((root / "corrected-v2/assumed_base" / mode / "decisions.json").read_text())
        for mode in LABELS
    }
    _figures(root, results, legacy, corrected)

    version_rows, changes, evidence_rows = [], [], []
    for key, item in results["corrected_v2"].items():
        scenario, mode = key.split("/")
        current, previous = item["full_1996_2020"], statistics(legacy[key])
        version_rows.append(
            [
                SCENARIOS[scenario],
                LABELS[mode],
                _number(previous["cagr"], "percent"),
                _number(current["cagr"], "percent"),
                _number(current["cagr"] - previous["cagr"], "pp"),
                _number(current["sharpe"]),
                _number(current["annual_volatility"], "percent"),
                _number(current["maximum_drawdown"], "percent"),
                escape(str(current["active_range"])),
            ]
        )
        evidence_rows.append(
            [
                SCENARIOS[scenario] + ": " + LABELS[mode],
                f'<a href="corrected-v2/{escape(key)}/monthly.csv">Monthly evidence</a>',
                f'<a href="corrected-v2/{escape(key)}/decisions.json">Saved decisions</a>',
            ]
        )
        if scenario == "assumed_base":
            changes.append(
                f"{LABELS[mode]}: {_number(previous['cagr'], 'percent')} → "
                f"{_number(current['cagr'], 'percent')} CAGR"
            )
    version_table = _table(
        "All nine explicitly versioned V2 comparisons",
        [
            "Costs",
            "Rule",
            "V1 CAGR",
            "V2 CAGR",
            "Change",
            "V2 excess Sharpe",
            "V2 annual vol",
            "V2 max drawdown",
            "V2 active range",
        ],
        version_rows,
    )

    frozen_rows = []
    for mode in LABELS:
        for scenario, item in results["frozen_target_costs"][mode].items():
            r = item["full_1996_2020"]
            frozen_rows.append(
                [
                    LABELS[mode],
                    SCENARIOS[scenario],
                    _number(r["cagr"], "percent"),
                    _number(r["sharpe"]),
                    _number(r["maximum_drawdown"], "percent"),
                    _number(r["assumed_cost_dollars"], "money"),
                ]
            )
    frozen_table = _table(
        "Nine frozen-target cost controls: no reoptimization",
        [
            "Original base target schedule",
            "Cost charge",
            "CAGR",
            "Excess Sharpe",
            "Max drawdown",
            "Proxy cost dollars",
        ],
        frozen_rows,
    )
    risk_rows = []
    for mode, item in results["prior_risk_overlay"].items():
        r = item["full_1996_2020"]
        risk_rows.append(
            [
                LABELS[mode],
                _number(r["cagr"], "percent"),
                _number(r["sharpe"]),
                _number(r["annual_volatility"], "percent"),
                _number(r["maximum_drawdown"], "percent"),
                _number(r["mean_invested"], "percent"),
            ]
        )
    risk_table = _table(
        "Three prior-only 5% risk-ceiling controls, base charges",
        [
            "Original base target schedule",
            "CAGR",
            "Excess Sharpe",
            "Realized annual vol",
            "Max drawdown",
            "Mean coefficient sum",
        ],
        risk_rows,
    )

    (forecast_table, risk_forecast_table, concentration_table, identity_table, solver_table) = (
        diagnostic_tables(results)
    )
    pool_table, period_table = pool_tables(results["pool_evidence"])

    hindsight_rows = []
    for mode, item in results["hindsight_fixed_mix"].items():
        r = item["full_1996_2020"]
        hindsight_rows.append(
            [
                "NONDEPLOYABLE: " + LABELS[mode],
                _number(r["cagr"], "percent"),
                _number(r["sharpe"]),
                _number(r["maximum_drawdown"], "percent"),
                escape(str(item["fit"]["weights"])),
            ]
        )
    hindsight_table = _table(
        "Two hindsight fixed mixes: use future information",
        ["Comparator", "CAGR", "Excess Sharpe", "Max drawdown", "Full-sample fitted coefficients"],
        hindsight_rows,
    )
    intervals = []
    for mode in LABELS:
        interval = results["corrected_v2"]["assumed_base/" + mode]["paired_vs_old_equal"][
            "sharpe_difference_95_interval"
        ]
        text = "undefined" if interval is None else " to ".join(_number(v) for v in interval)
        intervals.append(f"<li>{LABELS[mode]}: {text}</li>")
    preserved = sum(results["legacy_equivalence"].values())
    equivalence = f"{preserved} of {len(results['legacy_equivalence'])}"
    decision_count = ", ".join(f"{LABELS[m]}: {len(d)}" for m, d in decisions.items())
    accounting = escape(json.dumps(manifest["registration"]["trial_accounting"], indent=2))
    evidence_table = _table(
        "All corrected monthly records and saved decisions",
        ["Comparison", "History", "Decisions"],
        evidence_rows,
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Allocation diagnosis: defects, forecasts, costs and candidate evidence</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#182d40;background:#f7f9fb;
max-width:1200px;margin:32px auto;padding:0 20px}}h1,h2,h3{{line-height:1.25}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff}}
caption{{text-align:left;font-weight:700;padding:10px 0}}th,td{{border:1px solid #ccd7e0;
padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{background:#eaf0f5}}.table-wrap{{overflow-x:auto;margin:20px
0}}img{{width:100%;height:auto}}
.note{{padding:18px;background:#e7eff6;border-left:4px solid #245d83}}
.warning{{padding:18px;background:#fff1d6;border-left:4px solid #875500}}
a{{color:#07588c}}code,pre{{overflow-wrap:anywhere;white-space:pre-wrap}}
details{{margin:18px 0}}summary{{cursor:pointer;font-weight:700}}li{{margin:7px 0}}
</style></head><body><main>
<h1>Allocation diagnosis</h1>
<p class="note">This investigation separates demonstrated implementation defects from
forecast weakness, different risk, proxy costs and candidate quality. It preserves the original
study and reports every frozen comparison. It does not choose a new winner or certify
profitability.</p>
<h2>What was actually wrong</h2>
<ol><li><strong>Scored trades could differ from returned trades.</strong> V1 scored a unit-sum
mix
against actual risk-scaled prior coefficients, then scaled its selected mix afterward. An
independent
one-sleeve example falsely sent a feasible, unchanged, positive-edge position to cash because it
charged
a trade that would not occur. This is a demonstrated correctness defect; it does not establish
how much
of the historical performance it explains.</li>
<li><strong>The median search could miss a better feasible solution.</strong> Independent
synthetic
seed 108 found objective 0.23584 from the original local search versus 0.25142 at an explicit
feasible
grid point. These are monthly training-objective units, not realized annual Sharpes. A better
training
score does not establish better subsequent returns.</li></ol>
<p>V2 scores the actual feasible coefficients, includes a feasible unchanged-position candidate,
and
adds a frozen bounded lattice with local refinement. It remains opt-in. It searches the declared
risk-cap surface plus a feasible prior, <strong>not the full joint domain of every cash level
and mix</strong>.
It is not a globally certified optimizer. Training switching costs remain an amortized
approximation;
replay charges exact post-fee-NAV coefficient trades.</p>
<h2>Did correction change the historical outcome?</h2>
<p>{escape("; ".join(changes))}. These are version comparisons on reused discovery histories,
not independent confirmations or a reason to select whichever result looks best.</p>
<p><strong>{equivalence} original comparisons reproduced exactly</strong>
in monthly financial and saved-decision bytes. Historical policies and identities remain intact.
Saved V2 base decision counts: {escape(decision_count)}.</p>
{version_table}
<figure><img src="version-comparison.png" alt="Preserved V1 dashed and V2 solid account values
and drawdowns for all three rules">
<figcaption>Assumed base costs. Each line uses the same input histories, initial $100,000 and
accounting convention. An estimated 10% target-risk ceiling does not equalize realized
risk.</figcaption></figure>
<h3>Uncertainty remains</h3><p>Paired 95% Sharpe-difference intervals versus the original
equal-weight
base account, with 1,999 paired 12-month moving-block draws and seed
404:</p><ul>{"".join(intervals)}</ul>
<p>These intervals are not adjusted for the unknown historical factor search or this reused
diagnostic
analysis. They do not certify an optimizer winner.</p>
<h2>Costs and risk: separate controls</h2>
<p>The nine cost controls retain each original base target schedule and vary only the registered
proxy charges: gross 0/0, base 6bp switching plus 10bp per invested coefficient per month,
doubled
12bp plus 20bp. No cost-view-specific reoptimization occurs. NAV and coefficient drift can still
differ
as charges compound. These are declared mechanical charges, not observed stock spread, borrow,
market impact or turnover.</p>{frozen_table}
<p>The three risk controls shrink the original base targets, without increasing leverage, using
only the prior covariance and a common 5% annual ceiling. They are <strong>not equal
realized-risk
portfolios</strong>; within-quarter drift and forecast error remain.</p>{risk_table}
<figure><img src="cost-and-risk-controls.png" alt="Frozen-schedule CAGR across three proxy cost
charges and original versus prior-risk-overlay realized volatility">
<figcaption>Cost sensitivity holds target schedules fixed. The risk chart compares realized
volatility
after applying the same prior ceiling; it is not a claim of exact risk
matching.</figcaption></figure>
<details><summary>Arithmetic excess-mean decomposition</summary>{identity_table}
<p>Identity: Δmean = Sharpe_equal × (vol_candidate − vol_equal) +
(Sharpe_candidate − Sharpe_equal) × vol_candidate. This algebra is exact for annual arithmetic
excess means when both Sharpes exist. It is not causal attribution and does not decompose
CAGR.</p></details>
<h2>Can prior estimates forecast subsequent factor P&amp;L?</h2>
<p>Saved weights are audited using at most 120 prior completed months, with a 60-month minimum.
Raw and half-shrunk monthly mean forecasts are separately compared with following-quarter raw
factor
returns at the same fixed saved coefficients, against zero-mean forecasts. Lower RMSE means
smaller errors in
this particular scoring comparison; it does not prove predictable profits. We do not compare a
shrunk
training objective directly with realized annual Sharpe.</p>{forecast_table}
<details><summary>Covariance forecasts and realized scatter</summary>{risk_forecast_table}
<p>Actual scatter uses only three subsequent monthly observations per quarter and is very noisy.
Displayed averages are descriptive; they are not a calibrated risk-test statistic. Both
forecasts and
primary scores freeze decision coefficients: fees and intraperiod drift cannot change their
RMSE.
Scores sum arithmetic weighted factor returns; they are not compounded total-account returns.
Actual-path P&amp;L is separately labeled in machine evidence and is not the primary forecast
score.</p></details>
<h2>Is the candidate pool persistently profitable?</h2>
<p>Published methods do not guarantee persistent or executable premiums. All five candidates and
all
five fixed periods remain visible. Marginal mean intervals use 1,999 noncircular 12-month
moving-block
draws, seed 404. They are neither multiplicity- nor selection-adjusted and confer no
admission.</p>
{pool_table}<details><summary>All 25 factor-period
observations</summary>{period_table}</details>
<h3>Is one candidate carrying the portfolio?</h3>{concentration_table}
<p>Normalized concentration excludes cash and measures reference sleeve coefficients, not stock
gross
exposures. Effective count is 1 / sum(normalized weight²). BAB uses variable embedded leverage.
Its signed cumulative dollar factor-P&amp;L fraction can be negative or exceed 100%; it is
undefined when
total factor P&amp;L is zero. Contributions depend on the NAV path and exclude costs and cash
interest.
They are not causal leave-one-out benefits.</p>
<h2>What the bounded solver checked</h2>{solver_table}
<p>The frozen lattice has tenth-increment simplex spacing, at most 10,000 points, plus fixed
local
starts/refinements and a feasible exact prior. With five distinct sleeves this lattice has 1,001
points.
The objective-gap column is the best checked lattice score minus the best successful local
score,
in monthly training-objective units. A nonpositive gap does not prove a global optimum. Equal
weighting
has no search. No starts or policy parameters were tuned using these financial outcomes.</p>
<h2>Hindsight diagnostic: explicitly NONDEPLOYABLE</h2>
<p class="warning">Both fixed mixes below use the full 1996–2020 sample to fit weights before
replaying
that same sample. They use future information and could not have been deployed that way. They
are
diagnostic comparators, <strong>not mathematical upper bounds</strong> and not investment
recommendations.</p>
{hindsight_table}
<h2>What remains unresolved</h2>
<p>The histories are current-vintage and may be revised; historical availability has not been
verified.
The funded account earns RF once plus weighted self-financing factor returns minus proxy
charges.
Underlying stock gross exposures, constituent construction, borrow availability and executable
costs
remain unqualified. The 2016–2020 and five-year allocation metrics are retained in the machine
evidence;
none is untouched out-of-sample certification. Correct software can still produce weak economic
results.</p>
<p>No new candidates, source downloads, licensed refresh, paid model calls, broker connection,
formal admission or 2021–2025 private holdout access occurred in this diagnostic protocol.</p>
<h2>Evidence and comparison accounting</h2><p>Code revision:
<code>{escape(manifest["git_commit"])}</code>. The registered comparisons are dependent
controls;
we do not assert that their count is an independent statistical trial
count.</p><pre>{accounting}</pre>
<p><a href="manifest.json">Frozen registration, inputs and code hashes</a> ·
<a href="results.json">Complete metrics and diagnostics</a> ·
<a href="trial-accounting.json">Trial accounting</a> ·
<a href="comparisons.jsonl">Comparison ledger</a></p>
{evidence_table}
</main></body></html>"""
    (root / "RESULTS.html").write_text(html, encoding="utf-8")
