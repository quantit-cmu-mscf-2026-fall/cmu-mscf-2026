"""Shared fixtures for the Alpha-GPT harness tests. Nothing here calls Claude."""

from __future__ import annotations

import json

import pytest

from capstone.alpha_gpt.synth_ohlcv import make_ohlcv_panel


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Point the run ledger at a temp dir; returns a reader for its entries."""
    monkeypatch.setenv("CAPSTONE_LEDGER_DIR", str(tmp_path))
    path = tmp_path / "runs.jsonl"

    def read() -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    return read


@pytest.fixture(scope="session")
def reversal_panel():
    return make_ohlcv_panel(n_dates=500, n_assets=60, pattern="reversal", strength=0.06, seed=0)


@pytest.fixture(scope="session")
def null_panel():
    return make_ohlcv_panel(n_dates=500, n_assets=60, pattern="none", seed=0)
