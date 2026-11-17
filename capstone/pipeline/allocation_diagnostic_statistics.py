"""Bounded descriptive diagnostics; no allocation selection or strategy admission."""

import numpy as np
import pandas as pd

from capstone.pipeline.reference_inputs import POOL
from capstone.strategy.robust_allocation import shrunk_covariance


def _monthly(frame, columns):
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.has_duplicates
        or frame.index.hasnans
        or frame.columns.has_duplicates
        or not len(frame)
        or not set(columns) <= set(frame)
    ):
        raise ValueError("unique dated monthly observations and required columns needed")
    result = frame.loc[:, sorted(columns)].sort_index().astype(float)
    expected = pd.date_range(result.index[0], result.index[-1], freq="MS")
    if not result.index.equals(expected) or not np.isfinite(result.to_numpy()).all():
        raise ValueError("complete finite month-start observations needed")
    return result


def _moments(values):
    values = np.asarray(values, dtype=float)
    annual_mean = float(values.mean() * 12)
    volatility = float(values.std(ddof=1) * np.sqrt(12)) if len(values) > 1 else None
    return {
        "months": len(values),
        "annual_excess_mean": annual_mean,
        "annual_volatility": volatility,
        "sharpe": annual_mean / volatility if volatility and volatility > 1e-12 else None,
    }


def risk_sharpe_decomposition(candidate_history, equal_history):
    """Exact excess-mean identity, not an attribution of causal economic effects."""
    candidate = _monthly(candidate_history, ["excess"])
    equal = _monthly(equal_history, ["excess"])
    if not candidate.index.equals(equal.index):
        raise ValueError("decomposition requires identical observation months")
    c, e = _moments(candidate.excess), _moments(equal.excess)
    delta = c["annual_excess_mean"] - e["annual_excess_mean"]
    defined = c["sharpe"] is not None and e["sharpe"] is not None
    risk = e["sharpe"] * (c["annual_volatility"] - e["annual_volatility"]) if defined else None
    sharpe = (c["sharpe"] - e["sharpe"]) * c["annual_volatility"] if defined else None
    return {
        "candidate": c,
        "equal_weight": e,
        "annual_excess_mean_difference": delta,
        "volatility_term": risk,
        "sharpe_term": sharpe,
        "identity_residual": delta - risk - sharpe if defined else None,
        "status": "defined" if defined else "undefined_zero_or_insufficient_variation",
        "interpretation": "arithmetic identity, not causal attribution; does not decompose CAGR",
    }


def _pool_period(frame):
    x = frame.to_numpy()
    n = len(x)
    rng = np.random.default_rng(404)
    starts = rng.integers(0, n - 12 + 1, size=(1999, int(np.ceil(n / 12))))
    indices = (starts[:, :, None] + np.arange(12)).reshape(1999, -1)[:, :n]
    means = x[indices].mean(axis=1) * 12
    intervals = np.quantile(means, [0.025, 0.975], axis=0)
    return {
        name: {
            **_moments(x[:, i]),
            "annual_excess_mean_95_marginal_interval": intervals[:, i].tolist(),
        }
        for i, name in enumerate(frame)
    }


def pool_evidence(monthly_excess_panel):
    """Published factor excess returns, without an admission or persistence claim."""
    frame = _monthly(monthly_excess_panel, POOL).loc["1996-01-01":"2020-12-01"]
    if not frame.index.equals(pd.date_range("1996-01-01", "2020-12-01", freq="MS")):
        raise ValueError("pool diagnostics require the complete 1996-2020 evaluation window")
    periods = {}
    for year in range(1996, 2021, 5):
        periods[f"{year}-{year + 4}"] = _pool_period(frame.loc[str(year) : str(year + 4)])
    return {
        "overall": _pool_period(frame),
        "five_year_periods": periods,
        "return_basis": "published self-financing factor excess returns before added proxy costs",
        "interval": {
            "draws": 1999,
            "seed": 404,
            "block_months": 12,
            "method": "noncircular moving-block percentile interval for arithmetic excess mean",
            "multiplicity_adjusted": False,
            "selection_adjusted": False,
        },
        "formal_admission": False,
        "limitation": "marginal intervals are not proof of persistence, profitability or execution",
    }


def concentration_diagnostics(history):
    """Concentration excludes cash; contribution uses signed cumulative dollar P&L."""
    weight_columns = sorted(c for c in history if c.startswith("weight_"))
    pnl_columns = sorted(c for c in history if c.startswith("factor_pnl_"))
    if not weight_columns or not pnl_columns:
        raise ValueError("saved sleeve weights and factor P&L columns required")
    if {c[7:] for c in weight_columns} != {c[11:] for c in pnl_columns}:
        raise ValueError("weight and P&L factor identities must agree")
    frame = _monthly(history, [*weight_columns, *pnl_columns])
    weights = frame[weight_columns].to_numpy()
    if (weights < 0).any() or (weights.sum(axis=1) > 1 + 1e-8).any():
        raise ValueError("nonnegative sleeve coefficients summing to at most one required")
    total = weights.sum(axis=1)
    normalized = np.divide(
        weights, total[:, None], out=np.zeros_like(weights), where=total[:, None] > 0
    )
    squares = np.square(normalized).sum(axis=1)
    rows = [
        {
            "month": str(at.date()),
            "invested_coefficient": float(total[i]),
            "normalized_max_weight": float(normalized[i].max()) if total[i] > 0 else None,
            "effective_number": float(1 / squares[i]) if squares[i] > 0 else None,
        }
        for i, at in enumerate(frame.index)
    ]
    totals = {c[11:]: float(frame[c].sum()) for c in pnl_columns}
    pnl_total = sum(totals.values())
    bab = totals.get("BAB")
    maxima = [r["normalized_max_weight"] for r in rows if r["normalized_max_weight"] is not None]
    effective = [r["effective_number"] for r in rows if r["effective_number"] is not None]
    return {
        "monthly": rows,
        "mean_normalized_max_weight": float(np.mean(maxima)) if maxima else None,
        "maximum_normalized_max_weight": max(maxima) if maxima else None,
        "mean_effective_number": float(np.mean(effective)) if effective else None,
        "cash_only_months": int((total == 0).sum()),
        "factor_pnl_dollars": totals,
        "total_factor_pnl_dollars": pnl_total,
        "BAB_signed_pnl_fraction": bab / pnl_total
        if bab is not None and abs(pnl_total) > 1e-12
        else None,
        "limitations": [
            "normalized concentration excludes cash; sleeve coefficients are not stock gross",
            "signed P&L fractions can be negative or exceed one; undefined at zero total P&L",
            "dollar contributions depend on the NAV path; costs and cash interest are excluded",
        ],
    }


def forecast_audit(monthly_excess_panel, decisions, history):
    """Forecast from strictly prior months; inspect future data only for separate scoring."""
    panel = _monthly(monthly_excess_panel, POOL)
    actual = _monthly(history, ["equity", "return", "factor_pnl"])
    if (actual["return"] <= -1).any() or (actual.equity <= 0).any():
        raise ValueError("valid positive historical NAV required")
    ordered = sorted(decisions, key=lambda d: pd.Timestamp(d["as_of"]))
    dates = [pd.Timestamp(d["as_of"]) for d in ordered]
    if len(set(dates)) != len(dates) or any(at.day != 1 for at in dates):
        raise ValueError("unique month-start decisions required")
    forecasts, scores = [], []
    for decision, at in zip(ordered, dates, strict=True):
        names = list(panel)
        if set(decision["weights"]) != set(names):
            raise ValueError("decision weights must identify the complete reference pool")
        weights = np.array([decision["weights"][name] for name in names], dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() > 1 + 1e-8:
            raise ValueError("invalid saved decision weights")
        past = panel.loc[panel.index < at].tail(120)
        row = {
            "as_of": str(at.date()),
            "history_end": str(past.index[-1].date()) if len(past) else None,
            "training_observations": len(past),
        }
        if len(past) < 60:
            forecasts.append({**row, "status": "insufficient_training_history"})
            continue
        x = past.to_numpy()
        raw_mean = float(x.mean(axis=0) @ weights)
        shrunk, intensity = shrunk_covariance(x)
        sample_variance = max(float(weights @ np.cov(x, rowvar=False, ddof=1) @ weights), 0)
        shrunk_variance = max(float(weights @ shrunk @ weights), 0)
        forecasts.append(
            {
                **row,
                "status": "forecast_created",
                "raw_monthly_mean": raw_mean,
                "half_shrunk_monthly_mean": raw_mean * 0.5,
                "sample_covariance_annual_volatility": float(np.sqrt(sample_variance * 12)),
                "shrunk_covariance_annual_volatility": float(np.sqrt(shrunk_variance * 12)),
                "covariance_shrinkage": float(intensity),
            }
        )
        future = pd.date_range(at, periods=3, freq="MS")
        if not future.isin(panel.index).all():
            scores.append({"as_of": str(at.date()), "status": "incomplete_following_quarter"})
            continue
        gross = panel.loc[future, names].to_numpy() @ weights
        realized = float(gross.sum())
        actual_path = None
        if future.isin(actual.index).all():
            observed = actual.loc[future]
            opening_nav = observed.equity / (1 + observed["return"])
            actual_path = float((observed.factor_pnl / opening_nav).sum())
        scores.append(
            {
                "as_of": str(at.date()),
                "status": "scored",
                "months": [str(date.date()) for date in future],
                "actual_quarter_arithmetic_factor_pnl_fraction": realized,
                "actual_path_quarter_arithmetic_factor_pnl_fraction": actual_path,
                "raw_quarter_mean_prediction": raw_mean * 3,
                "half_shrunk_quarter_mean_prediction": raw_mean * 1.5,
                "raw_squared_error": float((realized - raw_mean * 3) ** 2),
                "half_shrunk_squared_error": float((realized - raw_mean * 1.5) ** 2),
                "zero_mean_squared_error": realized**2,
                "sample_predicted_quarter_variance": sample_variance * 3,
                "shrunk_predicted_quarter_variance": shrunk_variance * 3,
                "actual_three_month_annualized_scatter": float(gross.std(ddof=1) * np.sqrt(12)),
            }
        )
    scored = [row for row in scores if row["status"] == "scored"]
    return {
        "forecasts": forecasts,
        "following_quarter_scores": scores,
        "scored_quarters": len(scored),
        "quarter_arithmetic_rmse": {
            name: float(np.sqrt(np.mean([row[f"{name}_squared_error"] for row in scored])))
            if scored
            else None
            for name in ("raw", "half_shrunk", "zero_mean")
        },
        "forecast_rule": "saved weights; last 120 completed months before as_of; minimum 60",
        "score_basis": "sum of raw following-quarter factor returns at fixed saved coefficients",
        "limitations": [
            "primary scores freeze saved coefficients; no fees, cash interest or intraperiod drift",
            "separate actual_path scores include fee-reduced NAV and cost-dependent weight drift",
            "three-month scatter is noisy and descriptive, not a calibrated risk test",
            "quarter predictions and scores are arithmetic, not compounded portfolio returns",
            "training Sharpe is not compared with realized Sharpe",
            "reused historical evidence; no formal admission or new holdout evaluation",
        ],
    }
