from capstone.acceptance import strategy_accepts
from capstone.backtest import run_backtest
from capstone.synth import candidate_frames, make_panel


def test_null_synth_panel_never_passes():
    panel = make_panel(n_dates=2520, n_assets=100, n_candidates=200, n_real=0, seed=0)
    passes = []

    for name, signal, _is_real in candidate_frames(panel):
        returns = run_backtest(signal, panel.returns, cost_bps=0.0, demean=True, gross=1.0)
        if strategy_accepts(returns, n_trials=panel.n_candidates) == "pass":
            passes.append(name)

    assert not passes
