from __future__ import annotations

import pandas as pd

from capstone.acceptance import strategy_accepts
from capstone.backtest import run_backtest, sweep
from capstone.evaluate import benjamini_hochberg, sharpe_pvalue
from capstone.ml_candidates import generate_ml_candidates
from capstone.runlog import _read_entries, log_run
from capstone.sample_data import load_sample_prices
from capstone.synth import SyntheticPanel


def main() -> None:
    prices = load_sample_prices()
    candidates = generate_ml_candidates(prices, lookback=20, seed=0)
    truth = pd.Series(False, index=[candidate.name for candidate in candidates])
    signal_panel = pd.concat({candidate.name: candidate.signal for candidate in candidates}, axis=1)
    panel = SyntheticPanel(
        returns=prices,
        signals=signal_panel,
        truth=truth,
        params={
            "source": "load_sample_prices",
            "family": "regularized_linear",
            "lookback": 20,
            "seed": 0,
            "n_candidates": len(candidates),
        },
    )

    for candidate in candidates:
        log_run(
            "ml_sample_candidate",
            params={
                "name": candidate.name,
                "family": candidate.family,
                "features": candidate.features,
                "lookback": candidate.lookback,
                "seed": candidate.seed,
            },
            metrics={
                "signal_shape": [int(candidate.signal.shape[0]), int(candidate.signal.shape[1])],
                "signal_std": float(candidate.signal.stack().std(ddof=1)) if not candidate.signal.empty else 0.0,
            },
            seed=0,
            tags=["ml-sample", "candidate", "pre-screen"],
            notes="candidate generated before reading any screening results",
        )

    summary = sweep(panel, cost_bps=0.0, freq="daily")
    pvalues = pd.Series(
        [
            sharpe_pvalue(float(row["sharpe"]), int(row["n_obs"]), periods_per_year=252)
            for _, row in summary.iterrows()
        ],
        index=summary.index,
    )
    rejected = benjamini_hochberg(pvalues, alpha=0.05)

    pass_names: list[str] = []
    nothing_names: list[str] = []
    for candidate in candidates:
        strategy_returns = run_backtest(candidate.signal, prices, cost_bps=0.0, demean=True, gross=1.0)
        decision = strategy_accepts(strategy_returns, n_trials=len(candidates), freq="daily")
        if bool(rejected.get(candidate.name, False)) and decision == "pass":
            pass_names.append(candidate.name)
        else:
            nothing_names.append(candidate.name)

    print(f"candidate_count={len(candidates)}")
    print(f"pass={len(pass_names)}")
    print(f"nothing={len(nothing_names)}")
    for name in pass_names:
        print(f"pass {name}")
    for name in nothing_names:
        print(f"nothing {name}")

    ledger_entries = [entry for entry in _read_entries() if entry.get("name") == "ml_sample_candidate"]
    ledger_names = {entry["params"]["name"] for entry in ledger_entries if "name" in entry.get("params", {})}
    missing = [candidate.name for candidate in candidates if candidate.name not in ledger_names]
    print(f"ledger_hits={len(ledger_entries)}")
    print(f"missing_from_ledger={missing}")
    if missing:
        raise RuntimeError(f"missing candidate logs: {missing}")


if __name__ == "__main__":
    main()
