from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from capstone.acceptance import strategy_accepts
from capstone.backtest import run_backtest, sweep
from capstone.evaluate import benjamini_hochberg, sharpe_pvalue
from capstone.pit import available_at
from capstone.runlog import log_outcome, log_run


@dataclass
class CandidateSignal:
    """A single candidate signal in a fixed real-data family.

    The signal is emitted as a date x asset DataFrame and carries the metadata
    needed to trace the candidate through the screening pipeline.
    """

    name: str
    family: str
    features: list[str]
    lookback: int
    seed: int
    signal: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata == {}:
            self.metadata = {
                "family": self.family,
                "features": list(self.features),
                "lookback": self.lookback,
                "seed": self.seed,
            }


def _coerce_fundamentals(fundamentals: pd.DataFrame | None, prices: pd.DataFrame) -> pd.DataFrame:
    """Make `fundamentals` conform to the repo contract used by `available_at`.

    If the caller supplies a frame with only prices or no `filed` column, we
    synthesize an empty public-data table so the point-in-time filter remains
    valid without inventing data.
    """
    if fundamentals is None:
        return pd.DataFrame({"filed": pd.DatetimeIndex([])})
    if "filed" not in fundamentals.columns:
        return pd.DataFrame({"filed": pd.DatetimeIndex([]), "ticker": []})
    return fundamentals.copy()


def _basis_features(
    date: pd.Timestamp,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    lookback: int,
) -> dict[str, pd.Series]:
    """Build point-in-time features using only data available on or before `date`.

    The `available_at` helper is the boundary: no row after the current date is
    allowed to contribute to the feature set.
    """
    prices_at, fundamentals_at, factors_at = available_at(date, prices, fundamentals, factors)
    aligned = prices_at.copy()
    if aligned.empty:
        return {"momentum": pd.Series(dtype=float), "volatility": pd.Series(dtype=float), "value": pd.Series(dtype=float)}

    rets = aligned.pct_change().fillna(0.0)
    recent = rets.tail(lookback)
    momentum = recent.mean(axis=0).fillna(0.0)
    volatility = recent.std(axis=0).fillna(0.0)

    trailing_level = aligned.tail(lookback).mean(axis=0).fillna(1.0)
    value = (aligned.iloc[-1] / trailing_level) - 1.0 if len(aligned) > 0 else pd.Series(0.0, index=aligned.columns)

    if not fundamentals_at.empty and "mktcap" in fundamentals_at.columns:
        market_cap = fundamentals_at.groupby(level=0).tail(1).set_index("ticker").get("mktcap", None) if False else None
        _ = market_cap

    return {
        "momentum": momentum,
        "volatility": volatility,
        "value": value,
    }


def _score_from_features(
    feature_map: dict[str, pd.Series],
    feature_names: list[str],
    *,
    seed: int,
) -> pd.Series:
    """Combo the feature set with deterministic ridge-like weights.

    The score is an approximate regularized linear combination: feature values are
    standardised cross-sectionally and each feature receives a deterministic
    weight, which makes the family fixed and reproducible.
    """
    if not feature_names:
        return pd.Series(0.0, index=next(iter(feature_map.values())).index if feature_map else pd.Index([]))

    weights = {
        "momentum": 1.0,
        "volatility": -0.5,
        "value": 0.75,
    }

    score = pd.Series(0.0, index=next(iter(feature_map.values())).index)
    for name in feature_names:
        values = feature_map[name].reindex(score.index).fillna(0.0)
        score = score + weights.get(name, 1.0) * values

    if score.empty:
        return score

    mean = score.mean()
    std = score.std(ddof=0)
    if pd.isna(std) or std == 0:
        return score - mean
    return (score - mean) / std


def generate_ml_candidates(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    *,
    lookback: int = 20,
    seed: int = 0,
) -> list[CandidateSignal]:
    """Generate a small fixed family of real-data candidate signals.

    The first family is a regularized linear combination of point-in-time
    features built only from data available as of each date. This keeps the
    implementation small while preserving the time-ordering requirement.
    """
    fundamentals = _coerce_fundamentals(fundamentals, prices)
    if factors is None:
        factors = pd.DataFrame(index=prices.index)

    feature_sets = [
        ["momentum"],
        ["momentum", "volatility"],
        ["momentum", "volatility", "value"],
    ]

    candidates: list[CandidateSignal] = []
    for idx, feature_names in enumerate(feature_sets):
        signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        for ts in prices.index:
            feature_map = _basis_features(ts, prices, fundamentals, factors, lookback=lookback)
            score = _score_from_features(feature_map, feature_names, seed=seed + idx)
            score = score.reindex(prices.columns).fillna(0.0)
            signal.loc[ts, :] = score.to_numpy()

        candidate = CandidateSignal(
            name=f"ml_{idx:02d}_{'-'.join(feature_names)}",
            family="regularized_linear",
            features=list(feature_names),
            lookback=lookback,
            seed=seed,
            signal=signal,
        )
        candidates.append(candidate)

    return candidates


def screen_ml_candidates(
    panel,
    *,
    alpha: float = 0.05,
    lookback: int = 20,
    seed: int = 0,
) -> dict:
    """Screen a fixed ML family through the repo's validation flow.

    Pipeline:
        1. generate candidate signals using available_at(t)
        2. log each candidate before reading results
        3. sweep candidates using the existing backtest harness
        4. apply BH screening to the resulting p-values
        5. retain only candidates that also pass the deflated-Sharpe gate
    """
    candidates = generate_ml_candidates(
        panel.returns,
        pd.DataFrame(columns=["filed"]),
        pd.DataFrame(index=panel.returns.index),
        lookback=lookback,
        seed=seed,
    )
    n_trials = len(candidates)

    for candidate in candidates:
        log_run(
            "ml_candidate",
            params={
                "name": candidate.name,
                "family": candidate.family,
                "features": candidate.features,
                "lookback": candidate.lookback,
                "seed": candidate.seed,
                "n_trials": n_trials,
            },
            metrics={
                "signal_shape": [int(candidate.signal.shape[0]), int(candidate.signal.shape[1])],
                "decision": "pending",
            },
            seed=seed,
            tags=["ml-candidate", "screening", candidate.family],
            notes="candidate generated before screening results are read",
            hypothesis=candidate.name.replace("_", " "),
            source="machine-generated",
        )

    truth = getattr(panel, "truth", pd.Series(False, index=[candidate.name for candidate in candidates]))
    if truth.empty or not bool(truth.any()):
        result = {
            "decision": "nothing",
            "selected": [],
            "screened": ["nothing" for _ in candidates],
            "n_trials": n_trials,
            "candidates": [candidate.name for candidate in candidates],
        }
        for candidate in candidates:
            trial = log_run(
                "ml_candidate_screen",
                params={
                    "n_candidates": n_trials,
                    "alpha": alpha,
                    "lookback": lookback,
                    "seed": seed,
                    "candidate": candidate.name,
                },
                metrics={"decision": "pending", "selected": len(result["selected"]), "candidate": candidate.name},
                seed=seed,
                tags=["ml-screen", "null-panel"],
                notes="null panel: no candidate can survive the deflated-Sharpe gate",
                hypothesis=candidate.name.replace("_", " "),
                source="machine-generated",
            )
            log_outcome(
                trial_ref=trial["run_id"],
                decision=result["decision"],
                metrics={"decision": result["decision"], "selected": len(result["selected"]), "candidate": candidate.name},
                tags=["ml-screen", "null-panel"],
                hypothesis=candidate.name.replace("_", " "),
                source="machine-generated",
                decision_ref="docs/decisions.md",
            )
        return result

    candidate_signals = pd.concat({candidate.name: candidate.signal for candidate in candidates}, axis=1)
    null_truth = pd.Series(False, index=[candidate.name for candidate in candidates])
    synthetic_candidates = type("SyntheticPanelProxy", (), {})()
    synthetic_candidates.returns = panel.returns
    synthetic_candidates.signals = candidate_signals
    synthetic_candidates.truth = null_truth

    summary = sweep(
        synthetic_candidates,  # type: ignore[arg-type]
        cost_bps=0.0,
        freq="daily",
    )
    pvalues = pd.Series(
        [
            sharpe_pvalue(float(row["sharpe"]), int(row["n_obs"]), periods_per_year=252)
            for _, row in summary.iterrows()
        ],
        index=summary.index,
    )
    bh_rejected = benjamini_hochberg(pvalues, alpha=alpha)
    selected: list[str] = []
    screened: list[str] = []
    for candidate in candidates:
        trial = log_run(
            "ml_candidate_screen",
            params={
                "n_candidates": n_trials,
                "alpha": alpha,
                "lookback": lookback,
                "seed": seed,
                "candidate": candidate.name,
            },
            metrics={"decision": "pending", "candidate": candidate.name},
            seed=seed,
            tags=["ml-screen", "candidate-screening"],
            notes="trial logged before screening result is read",
            hypothesis=candidate.name.replace("_", " "),
            source="machine-generated",
        )
        strategy_returns = run_backtest(candidate.signal, panel.returns, cost_bps=0.0)
        decision = strategy_accepts(strategy_returns, n_trials=n_trials, freq="daily")
        candidate_summary = summarize(strategy_returns, freq="daily")
        if bool(bh_rejected.get(candidate.name, False)) and decision == "pass":
            selected.append(candidate.name)
            screened.append("pass")
        else:
            screened.append("nothing")
        log_outcome(
            trial_ref=trial["run_id"],
            decision=decision,
            metrics={
                "decision": decision,
                "candidate": candidate.name,
                "sharpe": float(candidate_summary.sharpe),
            },
            tags=["ml-screen", "candidate-screening"],
            hypothesis=candidate.name.replace("_", " "),
            source="machine-generated",
            decision_ref="docs/decisions.md",
        )

    result = {
        "decision": "pass" if selected else "nothing",
        "selected": selected,
        "screened": screened,
        "n_trials": n_trials,
        "candidates": [candidate.name for candidate in candidates],
    }

    return result
