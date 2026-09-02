from __future__ import annotations

import pandas as pd

from capstone.backtest import sweep
from capstone.evaluate import compare_rejection_rules
from capstone.runlog import log_run
from capstone.synth import make_panel


def main() -> None:
    seed = 0
    panel = make_panel(n_dates=2520, n_assets=100, n_candidates=200, n_real=8, seed=seed)
    summary = sweep(panel, cost_bps=0.0, freq="daily")
    comparison = compare_rejection_rules(summary, panel.truth, alpha=0.05)

    entry = log_run(
        "candidate_correction_compare",
        params={
            "n_dates": panel.returns.shape[0],
            "n_assets": panel.returns.shape[1],
            "n_candidates": panel.n_candidates,
            "n_real": panel.n_true,
            "alpha": 0.05,
            "freq": "daily",
            "cost_bps": 0.0,
        },
        metrics={
            "bh_rejections": int(comparison.loc["benjamini_hochberg", "rejections"]),
            "bh_power": float(comparison.loc["benjamini_hochberg", "power"]),
            "bh_fdr": float(comparison.loc["benjamini_hochberg", "fdr"]),
            "bonf_rejections": int(comparison.loc["bonferroni", "rejections"]),
            "bonf_power": float(comparison.loc["bonferroni", "power"]),
            "bonf_fdr": float(comparison.loc["bonferroni", "fdr"]),
        },
        seed=seed,
        tags=["sweep", "compare", "multiple-testing", "synthetic"],
        notes="Compare BH and Bonferroni on a planted-signal synthetic panel.",
    )

    print("side-by-side multiple-testing comparison")
    print(comparison.to_string())
    print(f"ledger_entry={entry['name']}:{entry['metrics']}")


if __name__ == "__main__":
    main()
