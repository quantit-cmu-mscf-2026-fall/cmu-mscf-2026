from __future__ import annotations

from capstone.acceptance import strategy_accepts
from capstone.backtest import run_backtest
from capstone.forecaster_momentum import momentum_signal
from capstone.runlog import log_run
from capstone.synth import make_panel


def main() -> None:
    seed = 0
    panel = make_panel(n_dates=2520, n_assets=100, n_candidates=200, n_real=0, seed=seed)
    signal = momentum_signal(panel.returns, lookback=20, skip=1)
    strategy_returns = run_backtest(signal, panel.returns, cost_bps=0.0, demean=True, gross=1.0)
    decision = strategy_accepts(strategy_returns, n_trials=panel.n_candidates)

    entry = log_run(
        "momentum_null_gate",
        params={
            "lookback": 20,
            "skip": 1,
            "n_dates": panel.returns.shape[0],
            "n_assets": panel.returns.shape[1],
            "n_candidates": panel.n_candidates,
            "n_real": panel.n_true,
        },
        metrics={
            "decision": decision,
            "mean_return": float(strategy_returns.dropna().mean()),
            "sharpe": float(strategy_returns.dropna().std(ddof=1) or 0.0),
        },
        seed=seed,
        tags=["null", "momentum", "acceptance"],
        notes="Null-panel smoke test: momentum forecaster should not pass the acceptance gate.",
    )

    print(f"decision={decision}")
    print(f"signal_shape={signal.shape}")
    print(f"ledger_entry={entry['name']}:{entry['metrics']['decision']}")


if __name__ == "__main__":
    main()
