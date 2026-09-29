"""The four figures DESIGN.md §6 requires, plus their tidy plotting tables.

Run:  PYTHONPATH=. python make_figures.py

Reads `results.json` — the numbers the run actually logged — so a figure can
never drift from its ledger entry. §6 asks for the underlying plotting data to
be saved alongside the rendered images, so each figure also writes a tidy CSV.

    fig01_global_null_fdr_fwer.png              + fig01_data.csv
    fig02_empirical_fdr_vs_q.png                + fig02_data.csv
    fig03_power_vs_within_family_correlation.png+ fig03_data.csv
    fig04_bootstrap_minus_bonferroni_power.png  + fig04_data.csv

Shared visual grammar
---------------------
§6 requires the same method colours, labels, ordering and axis limits across
every figure, so all of that lives in this module's constants and nothing is
decided per-figure.

Colours are slots 1-6 of the reference categorical palette in fixed order, never
cycled by rank. Marker shape is a second, redundant encoding of method identity,
so a reader who cannot separate two hues can still separate two series.

§6 also requires flat BH to be marked as a different-target method and naive
minimum-p as a negative control: both are drawn with dashed, visually muted
lines, and both carry the marking in the legend text rather than relying on the
reader remembering.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from methods import METHOD_LABEL, METHOD_ORDER, METHOD_ROLE

# No display: write files directly. Safe after importing pyplot because no
# figure has been created yet.
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "results.json"

# --- design tokens ----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#11181c"
INK_2 = "#38454a"
MUTED = "#6c7679"
FAINT = "#98a0a3"
GRID = "#e4e6e2"
BASELINE = "#c7ccc6"

# Reference categorical palette, slots 1-6, fixed order.
METHOD_COLOR = {
    "boot_max_bh": "#2a78d6",
    "boot_max_by": "#eb6834",
    "bonf_bh": "#1baf7a",
    "bonf_by": "#eda100",
    "flat_bh": "#e87ba4",
    "naive_min_bh": "#008300",
}
METHOD_MARKER = {
    "boot_max_bh": "o",
    "boot_max_by": "s",
    "bonf_bh": "^",
    "bonf_by": "D",
    "flat_bh": "v",
    "naive_min_bh": "X",
}
# §6: flat BH targets a different estimand; naive min-p is an invalid control.
MUTED_METHODS = {"flat_bh", "naive_min_bh"}
SHORT_LABEL = {
    "boot_max_bh": "Boot max + BH",
    "boot_max_by": "Boot max + BY",
    "bonf_bh": "Bonferroni + BH",
    "bonf_by": "Bonferroni + BY",
    "flat_bh": "Flat BH  (different target)",
    "naive_min_bh": "Naive min-p  (negative control)",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "legend.frameon": False,
    }
)


def load_results() -> tuple[dict, dict]:
    if not RESULTS_JSON.exists():
        raise SystemExit(
            f"{RESULTS_JSON.name} not found — run `PYTHONPATH=. python run_experiment.py` first."
        )
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return payload["results"], payload["meta"]


def _style(ax, *, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)


def _titles(fig, title: str, subtitle: str, y=0.975, sub_y=0.945) -> None:
    fig.text(0.03, y, title, fontsize=12.5, color=INK, ha="left", va="top")
    fig.text(0.03, sub_y, subtitle, fontsize=8, color=FAINT, ha="left", va="top")


def _legend(fig, methods, y=0.012, ncol=3) -> None:
    handles = [
        plt.Line2D(
            [],
            [],
            color=METHOD_COLOR[m],
            marker=METHOD_MARKER[m],
            markersize=6,
            linestyle="--" if m in MUTED_METHODS else "-",
            alpha=0.55 if m in MUTED_METHODS else 1.0,
            linewidth=1.8,
        )
        for m in methods
    ]
    fig.legend(
        handles,
        [SHORT_LABEL[m] for m in methods],
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, y),
        labelcolor=INK_2,
        fontsize=8,
        handletextpad=0.6,
        columnspacing=1.9,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scenarios(results: dict, group: str) -> list[tuple[str, dict]]:
    return [(k, v) for k, v in results.items() if v["scenario"]["group"] == group]


# --- Figure 1 ---------------------------------------------------------------


def figure1(results: dict, meta: dict) -> Path:
    """Global-null FDR/FWER by method, rho_W, phi and q.

    Under `pi1 = 0` every rejection is false, so `FDP = 1{any rejection}` and
    hypothesis-level FDR coincides with FWER — the identity §6 Figure 1 is built
    on. The y-axis deliberately extends past `q` rather than truncating there,
    because the point of including the negative control is to *see* how far it
    overshoots.
    """
    cells = _scenarios(results, "global_null")
    if not cells:
        raise SystemExit("no global_null scenarios in results.json")

    phis = sorted({c["scenario"]["phi"] for _, c in cells})
    rhos = sorted({c["scenario"]["rho_w"] for _, c in cells})
    qs = sorted(meta["q_levels"])
    rows: list[dict] = []

    fig, axes = plt.subplots(
        len(phis),
        len(qs),
        figsize=(11.0, 2.6 * len(phis) + 2.0),
        squeeze=False,
        sharey=True,
    )
    top = 0.0
    for _, cell in cells:
        for q in qs:
            for m in METHOD_ORDER:
                top = max(top, cell["by_q"][str(q)]["methods"][m]["family_any_false_hi"])

    for r, phi in enumerate(phis):
        for c, q in enumerate(qs):
            ax = axes[r][c]
            for mi, method in enumerate(METHOD_ORDER):
                xs, ys, los, his = [], [], [], []
                for ri, rho in enumerate(rhos):
                    match = [
                        x
                        for x in cells
                        if x[1]["scenario"]["phi"] == phi and x[1]["scenario"]["rho_w"] == rho
                    ]
                    if not match:
                        continue
                    agg = match[0][1]["by_q"][str(q)]["methods"][method]
                    x = mi + (ri - (len(rhos) - 1) / 2) * 0.17
                    xs.append(x)
                    ys.append(agg["family_any_false"])
                    los.append(agg["family_any_false"] - agg["family_any_false_lo"])
                    his.append(agg["family_any_false_hi"] - agg["family_any_false"])
                    rows.append(
                        {
                            "phi": phi,
                            "q": q,
                            "rho_w": rho,
                            "method": method,
                            "fwer": agg["family_any_false"],
                            "mcse": agg["family_any_false_mcse"],
                            "lo": agg["family_any_false_lo"],
                            "hi": agg["family_any_false_hi"],
                        }
                    )
                alpha = np.linspace(0.42, 1.0, len(xs))
                for xi, yi, lo, hi, a in zip(xs, ys, los, his, alpha, strict=True):
                    ax.errorbar(
                        xi,
                        yi,
                        yerr=[[lo], [hi]],
                        fmt=METHOD_MARKER[method],
                        markersize=5,
                        color=METHOD_COLOR[method],
                        alpha=a,
                        elinewidth=1.0,
                        capsize=2.0,
                        markeredgecolor=SURFACE,
                        markeredgewidth=0.6,
                    )
            ax.axhline(q, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels(
                [SHORT_LABEL[m].split("  ")[0] for m in METHOD_ORDER],
                rotation=28,
                ha="right",
                fontsize=7.2,
            )
            ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)
            ax.set_ylim(0, max(top * 1.08, max(qs) * 2))
            if r == 0:
                ax.set_title(f"q = {q}", fontsize=9.5, color=INK, pad=8)
            if c == 0:
                _style(ax, ylabel=f"φ = {phi}\nglobal-null FDR = FWER")
            else:
                _style(ax)

    _titles(
        fig,
        "Figure 1 — Global-null error control",
        f"π₁ = 0, δ = 0, T = 500, ρ_B = 0 · R = {meta['n_reps']}, B = {meta['n_boot']} · "
        f"marker opacity = ρ_W ∈ {rhos} (faint → dark) · dashed line = nominal q · "
        "bars are 95% Monte Carlo",
    )
    _legend(fig, METHOD_ORDER, y=0.008, ncol=3)
    fig.subplots_adjust(top=0.885, bottom=0.175, left=0.085, right=0.98, hspace=0.42, wspace=0.08)
    out = HERE / "fig01_global_null_fdr_fwer.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    _write_csv(HERE / "fig01_data.csv", rows)
    return out


# --- Figure 2 ---------------------------------------------------------------


def figure2(results: dict, meta: dict) -> Path:
    """Empirical hypothesis-level FDR against nominal q, with the y = x reference.

    A valid method sits on or below the 45-degree line within Monte Carlo error.
    Far below is conservatism; materially above is failure to control FDR.
    """
    cells = _scenarios(results, "calibration")
    if not cells:
        raise SystemExit("no calibration scenario in results.json")
    key, cell = cells[0]
    qs = sorted(meta["q_levels"])
    rows: list[dict] = []

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    lim = max(qs) * 1.9
    ax.plot([0, lim], [0, lim], color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(lim * 0.97, lim * 0.97, "y = x", fontsize=8, color=INK, ha="right", va="bottom")

    for method in METHOD_ORDER:
        ys, los, his = [], [], []
        for q in qs:
            agg = cell["by_q"][str(q)]["methods"][method]
            ys.append(agg["family_fdp"])
            los.append(agg["family_fdp"] - agg["family_fdp_lo"])
            his.append(agg["family_fdp_hi"] - agg["family_fdp"])
            rows.append(
                {
                    "q": q,
                    "method": method,
                    "family_fdr": agg["family_fdp"],
                    "mcse": agg["family_fdp_mcse"],
                    "lo": agg["family_fdp_lo"],
                    "hi": agg["family_fdp_hi"],
                }
            )
        muted = method in MUTED_METHODS
        ax.errorbar(
            qs,
            ys,
            yerr=[los, his],
            marker=METHOD_MARKER[method],
            markersize=7,
            color=METHOD_COLOR[method],
            linewidth=1.8,
            linestyle="--" if muted else "-",
            alpha=0.55 if muted else 1.0,
            elinewidth=1.0,
            capsize=3,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            zorder=3,
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    _style(ax, xlabel="nominal FDR level  q", ylabel="empirical hypothesis-level FDR")

    s = cell["scenario"]
    _titles(
        fig,
        "Figure 2 — FDR calibration away from the global null",
        f"T = {s['T']}, π₁ = {s['pi1']}, δ = {s['delta']}, ρ_W = {s['rho_w']}, "
        f"ρ_B = {s['rho_b']}, φ = {s['phi']} · R = {meta['n_reps']}, B = {meta['n_boot']} · "
        "on or below y = x is valid",
        y=0.972,
        sub_y=0.935,
    )
    _legend(fig, METHOD_ORDER, y=0.01, ncol=2)
    fig.subplots_adjust(top=0.855, bottom=0.235, left=0.115, right=0.97)
    out = HERE / "fig02_empirical_fdr_vs_q.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    _write_csv(HERE / "fig02_data.csv", rows)
    return out


# --- Figure 3 ---------------------------------------------------------------


def figure3(results: dict, meta: dict, q: float = 0.10) -> Path:
    """Hypothesis-level power against within-family correlation.

    §6: interpret power only for methods whose FDR is controlled in the matching
    scenario. The muted dashed styling of flat BH and naive min-p is the visual
    carrier of that instruction.
    """
    cells = _scenarios(results, "power_grid")
    if not cells:
        raise SystemExit("no power_grid scenarios in results.json")

    lengths = sorted({c["scenario"]["T"] for _, c in cells})
    deltas = sorted({c["scenario"]["delta"] for _, c in cells})
    rhos = sorted({c["scenario"]["rho_w"] for _, c in cells})
    rows: list[dict] = []

    fig, axes = plt.subplots(
        len(lengths),
        len(deltas),
        figsize=(3.4 * len(deltas) + 0.8, 3.2 * len(lengths) + 2.0),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for r, n_obs in enumerate(lengths):
        for c, delta in enumerate(deltas):
            ax = axes[r][c]
            for method in METHOD_ORDER:
                xs, ys, los, his = [], [], [], []
                for rho in rhos:
                    match = [
                        x
                        for x in cells
                        if x[1]["scenario"]["T"] == n_obs
                        and x[1]["scenario"]["delta"] == delta
                        and x[1]["scenario"]["rho_w"] == rho
                    ]
                    if not match:
                        continue
                    agg = match[0][1]["by_q"][str(q)]["methods"][method]
                    xs.append(rho)
                    ys.append(agg["family_power"])
                    los.append(agg["family_power"] - agg["family_power_lo"])
                    his.append(agg["family_power_hi"] - agg["family_power"])
                    rows.append(
                        {
                            "T": n_obs,
                            "delta": delta,
                            "rho_w": rho,
                            "q": q,
                            "method": method,
                            "power": agg["family_power"],
                            "mcse": agg["family_power_mcse"],
                            "lo": agg["family_power_lo"],
                            "hi": agg["family_power_hi"],
                            "family_fdr": agg["family_fdp"],
                        }
                    )
                muted = method in MUTED_METHODS
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[los, his],
                    marker=METHOD_MARKER[method],
                    markersize=6,
                    color=METHOD_COLOR[method],
                    linewidth=1.7,
                    linestyle="--" if muted else "-",
                    alpha=0.5 if muted else 1.0,
                    elinewidth=0.9,
                    capsize=2.5,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.7,
                )
            ax.set_ylim(0, 1.0)
            ax.set_xticks(rhos)
            if r == 0:
                ax.set_title(f"δ = {delta:g}", fontsize=9.5, color=INK, pad=8)
            _style(
                ax,
                ylabel=(f"T = {n_obs}\nhypothesis-level power" if c == 0 else None),
                xlabel=("within-family correlation  ρ_W" if r == len(lengths) - 1 else None),
            )

    _titles(
        fig,
        "Figure 3 — Power against within-family correlation",
        f"π₁ = 0.1, ρ_B = 0, φ = 0.2, q = {q} · R = {meta['n_reps']}, B = {meta['n_boot']} · "
        "read power only where the matching FDR is controlled (Figure 2)",
    )
    _legend(fig, METHOD_ORDER, y=0.008, ncol=3)
    fig.subplots_adjust(top=0.875, bottom=0.185, left=0.095, right=0.98, hspace=0.18, wspace=0.1)
    out = HERE / "fig03_power_vs_within_family_correlation.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    _write_csv(HERE / "fig03_data.csv", rows)
    return out


# --- Figure 4 ---------------------------------------------------------------


def figure4(results: dict, meta: dict, q: float = 0.10) -> Path:
    """Paired power difference, bootstrap max + BH minus Bonferroni + BH.

    The primary comparison of the whole experiment, because these two target the
    same family-level null. Positive favours bootstrap.

    Cells whose paired 95% Monte Carlo interval contains zero are hatched: the
    difference there is not distinguishable from simulation noise, and §6 asks
    for exactly that marking so a reader cannot mistake noise for a result.
    """
    cells = _scenarios(results, "power_grid")
    if not cells:
        raise SystemExit("no power_grid scenarios in results.json")

    lengths = sorted({c["scenario"]["T"] for _, c in cells})
    deltas = sorted({c["scenario"]["delta"] for _, c in cells})
    rhos = sorted({c["scenario"]["rho_w"] for _, c in cells})
    rows: list[dict] = []

    grids, sig_grids = {}, {}
    for n_obs in lengths:
        grid = np.full((len(deltas), len(rhos)), np.nan)
        sig = np.zeros((len(deltas), len(rhos)), dtype=bool)
        for di, delta in enumerate(deltas):
            for ri, rho in enumerate(rhos):
                match = [
                    x
                    for x in cells
                    if x[1]["scenario"]["T"] == n_obs
                    and x[1]["scenario"]["delta"] == delta
                    and x[1]["scenario"]["rho_w"] == rho
                ]
                if not match:
                    continue
                d = match[0][1]["by_q"][str(q)]["boot_minus_bonf_power"]
                grid[di, ri] = d["mean"]
                sig[di, ri] = d["significant"]
                rows.append(
                    {
                        "T": n_obs,
                        "delta": delta,
                        "rho_w": rho,
                        "q": q,
                        "delta_power": d["mean"],
                        "mcse": d["mcse"],
                        "lo": d["lo"],
                        "hi": d["hi"],
                        "significant": d["significant"],
                    }
                )
        grids[n_obs], sig_grids[n_obs] = grid, sig

    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    span = max(float(np.abs(finite).max()), 1e-6) if finite.size else 1e-6

    fig, axes = plt.subplots(
        1, len(lengths), figsize=(4.9 * len(lengths) + 1.0, 4.9), squeeze=False
    )
    for c, n_obs in enumerate(lengths):
        ax = axes[0][c]
        grid, sig = grids[n_obs], sig_grids[n_obs]
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-span, vmax=span, aspect="auto", origin="lower")
        for di in range(len(deltas)):
            for ri in range(len(rhos)):
                if not np.isfinite(grid[di, ri]):
                    continue
                if not sig[di, ri]:
                    ax.add_patch(
                        plt.Rectangle(
                            (ri - 0.5, di - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="////",
                            edgecolor=SURFACE,
                            linewidth=0.0,
                            alpha=0.75,
                        )
                    )
                shade = INK if abs(grid[di, ri]) < span * 0.55 else SURFACE
                ax.text(
                    ri,
                    di,
                    f"{grid[di, ri]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=shade,
                )
        ax.set_xticks(range(len(rhos)), [f"{r:g}" for r in rhos])
        ax.set_yticks(range(len(deltas)), [f"{d:g}" for d in deltas])
        ax.set_title(f"T = {n_obs}", fontsize=9.5, color=INK, pad=8)
        ax.set_xlabel("within-family correlation  ρ_W", color=INK_2, fontsize=8.5)
        if c == 0:
            ax.set_ylabel("standardised effect  δ", color=INK_2, fontsize=8.5)
        ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.035, pad=0.03)
    cbar.set_label("Δ power  (bootstrap − Bonferroni)", color=INK_2, fontsize=8.5)
    cbar.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    cbar.outline.set_visible(False)

    _titles(
        fig,
        "Figure 4 — Does the bootstrap beat Bonferroni on the same target?",
        f"π₁ = 0.1, ρ_B = 0, φ = 0.2, q = {q} · R = {meta['n_reps']}, B = {meta['n_boot']} · "
        "paired differences · hatched = 95% Monte Carlo interval contains zero",
        y=0.965,
        sub_y=0.918,
    )
    fig.subplots_adjust(top=0.80, bottom=0.115, left=0.075, right=0.90)
    out = HERE / "fig04_bootstrap_minus_bonferroni_power.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    _write_csv(HERE / "fig04_data.csv", rows)
    return out


def main() -> None:
    results, meta = load_results()
    for builder in (figure1, figure2, figure3, figure4):
        print(f"wrote {builder(results, meta).name}")
    print("tidy plotting tables: fig01_data.csv … fig04_data.csv")
    _ = METHOD_LABEL, METHOD_ROLE  # re-exported for the write-up


if __name__ == "__main__":
    main()
