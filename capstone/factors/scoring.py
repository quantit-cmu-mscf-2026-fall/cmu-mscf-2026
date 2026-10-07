"""Evaluation as the deepen move's scorer (QUANTIT-114).

`EvaluationScorer` scores a factor tree the way evaluation trades it: one
variant (the trading spec in `config/deepen_crsp.toml`) on the CRSP discovery
period, net of half the quoted spread. Its quality is the funnel's stage 1
statistic for that variant, as a z-score:

    Q = Phi^-1(1 - p),  p = `evaluation.screen_pvalue(net returns)`,

the one-sided HAC test of a Sharpe ratio above zero. z rather than p because
the memory averages quality differences between children and their parents,
and z is additive where p saturates near 0 and 1. Negative z means the
variant loses after costs, so the memory bins quality with its sign
(`memory_signed_quality`).

The values the pool correlates are the variant's daily net returns: two
strategies that earn on the same days are near-duplicates whatever their
formulas look like.

The scorer never logs. The deepen move logs each score as a trial, with the
scorer's `params` (data version, period, spec, variant id) and `metrics`
(net and gross Sharpe, p-value, turnover, cost). It sees only the discovery
period (the panel refuses the holdout), and stages 2-4 of the funnel never
reach it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import stats

from capstone.factors import store
from capstone.factors.deepen import Scored
from capstone.factors.evaluation import (
    DISCOVERY_END,
    DISCOVERY_START,
    Panel,
    TradingSpec,
    factor_signal,
    returns_from_weights,
    screen_pvalue,
    signal_weights,
)
from capstone.factors.llm import FactorConfig
from capstone.factors.tree import Node, factor_id

CRSP_SETTINGS = Path(__file__).resolve().parents[2] / "config" / "deepen_crsp.toml"
# p is clipped before Phi^-1 so a perfect series gives a finite z (|z| <= ~7.9).
P_CLIP = 1e-15


@dataclass
class EvaluationScorer:
    """Score a tree as one evaluated variant on the discovery period."""

    panel: Panel
    spec: TradingSpec

    def __call__(self, tree: Node) -> Scored | None:
        try:
            signal = factor_signal(tree, self.panel)
        except (ValueError, ArithmeticError):
            return None
        weights = signal_weights(signal, self.panel, self.spec.neutral)
        result = returns_from_weights(weights, self.panel, self.spec.hold)
        p = screen_pvalue(result.net)
        if not np.isfinite(p):
            return None
        quality = float(stats.norm.isf(np.clip(p, P_CLIP, 1 - P_CLIP)))
        fid = factor_id(tree)
        return Scored(
            quality,
            result.net.to_numpy(dtype=float),
            params={
                "scorer": "evaluation",
                "variant_id": store.variant_id(fid, self.spec.as_dict()),
                "spec": self.spec.as_dict(),
                "data_version": self.panel.data_version,
                "period": [DISCOVERY_START, DISCOVERY_END],
                "costs": "half quoted spread per unit traded",
            },
            metrics={
                "pvalue_screen": p,
                "sharpe_net": result.sharpe("net"),
                "sharpe_gross": result.sharpe("gross"),
                "mean_turnover": float(result.turnover.mean()),
                "mean_cost": float(result.cost.mean()),
                "days": int(result.net.notna().sum()),
            },
        )


def crsp_settings(
    config: FactorConfig, path: str | Path = CRSP_SETTINGS
) -> tuple[FactorConfig, TradingSpec]:
    """`config` with the CRSP scorer's quality thresholds, and the spec it trades.

    The file's [memory] keys replace the config's (they are in z units, not
    the default |ICIR|); [scorer] gives the trading spec. Unknown keys raise.
    """
    with open(path, "rb") as handle:
        values = tomllib.load(handle)
    unknown = set(values) - {"scorer", "memory"}
    if unknown:
        raise ValueError(f"unknown sections in {path}: {sorted(unknown)}")
    memory = values.get("memory", {})
    bad = [k for k in memory if not k.startswith("memory_") or not hasattr(config, k)]
    if bad:
        raise ValueError(f"unknown [memory] keys in {path}: {sorted(bad)}")
    spec = TradingSpec(**values.get("scorer", {}))
    return replace(config, **memory), spec
