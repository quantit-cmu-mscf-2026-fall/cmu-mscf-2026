"""Step 3 of learned memory search: the small pre-registered check.

    python research/learned_memory_search_check.py            # every run, then the summary
    python research/learned_memory_search_check.py --design research/<design>.toml
    python research/learned_memory_search_check.py --budget 50 --runs 1   # time one short run

The design (default research/learned_memory_search_check.toml) is written
before any run; each design's outputs go to their own folder. Each run is
the deepen move (capstone.factors.deepen) over a fresh factor store, with
the free random editor, on the planted or the null panel, with the memory
on or off. Every scored child is a ledger trial, written by
the deepen move as it happens. Stores, per-run results and the summary go to
experiments/learned_memory_search_check/ (gitignored); the numbers belong in
the PR description.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tomllib
from dataclasses import replace
from pathlib import Path

from capstone.factors import store
from capstone.factors.deepen import LEDGER_TAGS, Deepener, correlations
from capstone.factors.edits import RandomEditor
from capstone.factors.llm import load_config
from capstone.factors.synthetic import PLANTED, PanelScorer, PanelSpec, make_panel
from capstone.factors.tree import factor_id, parse

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "research" / "learned_memory_search_check.toml"
OUT = ROOT / "experiments" / "learned_memory_search_check"
ADMITTED = ("admitted", "high_quality")


def scorers(design: dict) -> dict[str, PanelScorer]:
    p = design["panel"]
    out = {}
    for name, beta in (("planted", p["beta"]), ("null", 0.0)):
        spec = PanelSpec(n_stocks=p["n_stocks"], n_days=p["n_days"], beta=beta, seed=p["seed"])
        out[name] = PanelScorer(
            make_panel(spec), p["horizon"], ic_step=p["ic_step"], sample_cells=p["sample_cells"]
        )
    return out


def run_one(
    design: dict,
    scorer: PanelScorer,
    panel: str,
    condition: str,
    seed: int,
    budget: int,
    out: Path,
    label: str,
):
    config = replace(load_config(), alignment_model="")
    if condition == "no_memory":
        config = replace(config, memory_lambda_max=0.0, memory_veto_confidence=1.0)
    run_id = f"{label}-{panel}-{condition}-{seed}"
    path = out / "stores" / f"{run_id}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = store.connect(path)
    con.execute("PRAGMA synchronous = OFF")  # a disposable per-run store
    try:
        store.add_paper(con, "synthetic:check", "Synthetic panel, one planted factor")
        hid = store.add_hypothesis(
            con,
            store.Hypothesis(
                "synthetic:check",
                "A synthetic panel with one planted factor.",
                "Planted by construction (capstone.factors.synthetic).",
                "Checks whether the deepen move finds it.",
                "Any factor; scored by search-period |ICIR|.",
                "No planted factor recovered within the budget.",
            ),
        )
        for e in design["search"]["seeds"]:
            store.add_factor(con, parse(e), hid)
        d = Deepener(
            con,
            config,
            scorer,
            RandomEditor(seed),
            run_id=run_id,
            seed=seed,
            ledger_tags=[*LEDGER_TAGS, "synthetic-check", panel, condition],
        )
        for e in design["search"]["seeds"]:
            d.add_parent(factor_id(parse(e)))
        target = scorer(parse(PLANTED)).values[None, :]
        first = None
        statuses: dict[str, int] = {}
        while d.evaluations < budget:
            start = d.evaluations
            for i, o in enumerate(d.step(), 1):
                statuses[o.status] = statuses.get(o.status, 0) + 1
                if first is None and o.status in ADMITTED:
                    corr = abs(correlations(scorer(parse(o.expression)).values, target)[0])
                    if corr >= design["search"]["recovery_corr"]:
                        first = start + i
        return {
            "run_id": run_id,
            "panel": panel,
            "condition": condition,
            "seed": seed,
            "evaluations": d.evaluations,
            "first_recovery": first,
            "admitted": statuses.get("admitted", 0) + statuses.get("high_quality", 0),
            "statuses": statuses,
            "fallbacks": d.fallbacks,
            "best_quality": max(d.pool.quality.values()),
        }
    finally:
        con.close()


def summarize(results: list[dict], budget: int) -> str:
    def median(xs):
        return statistics.median(xs) if xs else float("nan")

    lines = [
        "| Panel | Condition | Runs | Evals to recovery, median [min, max] | Recovered "
        "| Admitted/run, median [min, max] | Best Q, median | Fallbacks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    table = {}
    for panel in ("planted", "null"):
        for condition in ("memory", "no_memory"):
            rs = [r for r in results if r["panel"] == panel and r["condition"] == condition]
            if not rs:
                continue
            rec = [r["first_recovery"] or budget + 1 for r in rs]
            adm = [r["admitted"] for r in rs]
            table[(panel, condition)] = (median(rec), median(adm))
            rec_text = f"{median(rec):.0f} [{min(rec)}, {max(rec)}]"
            if median(rec) > budget:
                rec_text = f"not reached [{min(rec)}, {max(rec)}]"
            lines.append(
                f"| {panel} | {condition} | {len(rs)} | {rec_text} "
                f"| {sum(r['first_recovery'] is not None for r in rs)}/{len(rs)} "
                f"| {median(adm):.0f} [{min(adm)}, {max(adm)}] "
                f"| {median([r['best_quality'] for r in rs]):.3f} "
                f"| {sum(r['fallbacks'] for r in rs)} |"
            )
    lines += ["", f"Runs that never recover count as {budget + 1}.", ""]
    if ("planted", "memory") in table and ("planted", "no_memory") in table:
        m, n = table[("planted", "memory")][0], table[("planted", "no_memory")][0]
        lines.append(
            f"1. Planted: memory recovers sooner (median {m:.0f} vs {n:.0f}): "
            f"**{'met' if m < n else 'not met'}**"
        )
    if ("null", "memory") in table and ("null", "no_memory") in table:
        m, n = table[("null", "memory")][1], table[("null", "no_memory")][1]
        lines.append(
            f"2. Null: memory admits no more (median {m:.0f} vs {n:.0f}): "
            f"**{'met' if m <= n else 'not met'}**"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--design", type=Path, default=DESIGN, help="the design file")
    parser.add_argument("--budget", type=int, help="override the design's budget (timing only)")
    parser.add_argument("--runs", type=int, help="run only the first N runs (timing only)")
    args = parser.parse_args()
    design = tomllib.loads(args.design.read_text(encoding="utf-8"))
    label = args.design.stem.removeprefix("learned_memory_search_")  # check, check_v2
    out = OUT if label == "check" else OUT / label  # v1's outputs stay where they ran
    check = design["check"]
    budget = args.budget or check["budget"]
    plan = [
        (panel, condition, seed)
        for panel in check["panels"]
        for condition in check["conditions"]
        for seed in check["run_seeds"]
    ][: args.runs]
    t0 = time.time()
    built = scorers(design)
    print(f"panels built in {time.time() - t0:.1f}s; {len(plan)} runs of {budget} evaluations")
    results = []
    for panel, condition, seed in plan:
        t = time.time()
        r = run_one(design, built[panel], panel, condition, seed, budget, out, label)
        results.append(r)
        print(
            f"{r['run_id']}: {time.time() - t:.0f}s, recovery at {r['first_recovery']}, "
            f"admitted {r['admitted']}",
            flush=True,
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summary = summarize(results, budget)
    (out / "results.md").write_text(summary + "\n", encoding="utf-8")
    print(f"\n{summary}\n\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
