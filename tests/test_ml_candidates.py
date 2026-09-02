from __future__ import annotations

import pandas as pd

from capstone.ml_candidates import generate_ml_candidates, screen_ml_candidates
from capstone.synth import make_panel


def test_generate_ml_candidates_has_metadata_and_shape():
    panel = make_panel(n_dates=300, n_assets=10, n_candidates=20, n_real=0, seed=0)
    candidates = generate_ml_candidates(panel.returns, panel.returns, panel.returns, seed=7)

    assert candidates
    first = candidates[0]
    assert first.family == "regularized_linear"
    assert first.lookback > 0
    assert first.seed == 7
    assert set(first.metadata) >= {"family", "features", "lookback", "seed"}
    assert isinstance(first.signal, pd.DataFrame)
    assert first.signal.shape[0] == panel.returns.shape[0]
    assert first.signal.shape[1] == panel.returns.shape[1]


def test_screen_ml_candidates_on_null_panel_returns_nothing():
    panel = make_panel(n_dates=600, n_assets=25, n_candidates=80, n_real=0, seed=0)
    result = screen_ml_candidates(panel, alpha=0.05, seed=0)

    assert result["decision"] == "nothing"
    assert result["selected"] == []
    assert result["screened"][0] == "nothing"
