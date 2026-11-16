"""Prior-only research allocation of explicitly self-financing excess-return sleeves."""

from dataclasses import asdict, dataclass
from math import comb

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from capstone.strategy.research import digest

MODES = {"equal_weight", "shrunk_max_sharpe", "median_window_sharpe"}


@dataclass(frozen=True)
class AllocationPolicy:
    periods: int = 252
    lookback: int = 2520
    minimum: int = 1260
    window: int = 756
    step: int = 252
    mean_shrink: float = 0.5
    annual_risk_limit: float = 0.10
    ongoing_bps: float = 0
    switch_bps: float = 0
    holding_periods: int = 63

    def __post_init__(self):
        for name in ("periods", "lookback", "minimum", "window", "step", "holding_periods"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"invalid {name}")
        if not 2 <= self.window <= self.minimum <= self.lookback:
            raise ValueError("window <= minimum <= lookback required")
        if not 0 < self.mean_shrink <= 1 or not 0 < self.annual_risk_limit <= 1:
            raise ValueError("invalid shrinkage or risk limit")
        if not all(np.isfinite(x) and 0 <= x <= 100 for x in (self.ongoing_bps, self.switch_bps)):
            raise ValueError("invalid cost assumptions")


@dataclass(frozen=True)
class AllocationPolicyV2(AllocationPolicy):
    """Opt-in feasible-coefficient objective and bounded deterministic search.

    Searches the risk-capped composition surface plus the feasible prior point;
    this is not a full joint optimization of cash and sleeve coefficients.
    """

    policy_version: int = 2

    def __post_init__(self):
        super().__post_init__()
        if type(self.policy_version) is not int or self.policy_version != 2:
            raise ValueError("allocation policy version must be 2")


def shrunk_covariance(values):
    """Ledoit-Wolf scaled-identity shrinkage, using ML empirical covariance.

    Beta estimates the sum of covariance-entry variances from centered outer
    products; clipping beta at delta keeps the shrinkage intensity in [0, 1].
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("finite two-dimensional observations required")
    x = x - x.mean(axis=0)
    n, p = x.shape
    sample = x.T @ x / n
    target = np.eye(p) * np.trace(sample) / p
    delta = float(np.square(sample - target).sum())
    beta = max(
        0.0, float((np.square(np.square(x).sum(axis=1)).mean() - np.square(sample).sum()) / n)
    )
    shrink = min(1.0, beta / delta) if delta > 0 else 0.0
    return (1 - shrink) * sample + shrink * target, shrink


def optimize_allocation(returns, as_of, *, mode, policy=None, previous=None):
    """Choose weights without a cardinality requirement; failed optimizations stay cash.

    Input returns MUST be excess returns per explicitly documented sleeve
    coefficient. This function neither qualifies a strategy nor assumes its gross
    stock exposure. Missing history is a blocker, never imputed or dropped.
    """
    policy = policy or AllocationPolicy()
    if mode not in MODES or not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("known allocation mode and datetime observations required")
    if (
        returns.index.has_duplicates
        or returns.index.hasnans
        or returns.columns.has_duplicates
        or not len(returns.columns)
        or not all(isinstance(n, str) for n in returns.columns)
    ):
        raise ValueError("unique nonempty strategy columns and observation dates required")
    past = returns.loc[returns.index < pd.Timestamp(as_of)].sort_index().sort_index(axis=1)
    past = past.tail(policy.lookback).astype(float)
    if not np.isfinite(past.to_numpy()).all():
        raise ValueError("missing or nonfinite prior inputs block allocation")
    names = list(past)
    previous = {} if previous is None else previous
    if set(previous) - set(names):
        raise ValueError("previous coefficients identify unknown strategies")
    prev = pd.Series(previous, dtype=float).reindex(names, fill_value=0).to_numpy()
    if not np.isfinite(prev).all() or (prev < 0).any() or prev.sum() > 1 + 1e-8:
        raise ValueError("invalid previous coefficients")
    result = {
        "schema_version": 3,
        "mode": mode,
        "policy": asdict(policy),
        "as_of": str(pd.Timestamp(as_of).date()),
        "observations": len(past),
        "history_end": str(past.index[-1].date()) if len(past) else None,
        "weights": dict.fromkeys(names, 0.0),
        "cash_weight": 1.0,
        "return_basis": "self_financing_excess",
        "formal_admission": False,
    }
    if isinstance(policy, AllocationPolicyV2):
        return _optimize_v2(past, mode, policy, prev, result)
    if len(past) < policy.minimum:
        result["status"] = "insufficient_history_cash"
    else:
        # Exact return copies cannot obtain extra optimization degrees of freedom.
        representatives, copies = [], {}
        for name in names:
            representative = next(
                (r for r in representatives if np.array_equal(past[r], past[name])), name
            )
            if representative == name:
                representatives.append(name)
            copies.setdefault(representative, []).append(name)
        x = past[representatives].to_numpy()
        covariance, shrink = shrunk_covariance(x)
        p = len(representatives)
        previous_group = np.array(
            [sum(prev[names.index(n)] for n in copies[r]) for r in representatives]
        )
        windows = [
            x[end - policy.window : end] for end in range(len(x), policy.window - 1, -policy.step)
        ]
        estimates = [shrunk_covariance(w)[0] for w in windows]

        def score(w):
            switching = policy.switch_bps / 10000 * abs(w - previous_group).sum()
            drag = policy.ongoing_bps / 10000 + switching / policy.holding_periods
            if mode == "median_window_sharpe":
                ratios = [
                    (policy.mean_shrink * z.mean(axis=0) @ w - drag)
                    / max(float(np.sqrt(max(w @ c @ w, 0))), 1e-12)
                    for z, c in zip(windows, estimates, strict=True)
                ]
                return float(np.median(ratios))
            variance = max(float(w @ covariance @ w), 0)
            return float(
                (policy.mean_shrink * x.mean(axis=0) @ w - drag) / max(np.sqrt(variance), 1e-12)
            )

        starts = [np.full(p, 1 / p), *np.eye(p)]
        if mode == "equal_weight":
            chosen, success = starts[0], True
        else:
            solutions = [
                minimize(
                    lambda w: -score(w),
                    start,
                    method="SLSQP",
                    bounds=[(0, 1)] * p,
                    constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                    options={"ftol": 1e-10, "maxiter": 250},
                )
                for start in starts
            ]
            valid = [
                s.x
                for s in solutions
                if s.success
                and np.isfinite(s.x).all()
                and abs(s.x.sum() - 1) < 1e-7
                and s.x.min() >= -1e-8
            ]
            success = bool(valid)
            chosen = max(valid, key=score) if valid else starts[0]
        chosen = np.maximum(chosen, 0)
        chosen[chosen < 1e-8] = 0
        chosen /= chosen.sum()
        value = score(chosen)
        result.update(
            shrinkage=shrink,
            training_objective=value,
            window_ends=[
                str(past.index[end - 1].date())
                for end in range(len(x), policy.window - 1, -policy.step)
            ],
            duplicate_groups=list(copies.values()),
        )
        if not success:
            result["status"] = "optimizer_failed_cash"
        elif mode != "equal_weight" and value <= 0:
            result["status"] = "no_positive_training_edge_cash"
        else:
            volatility = float(np.sqrt(max(chosen @ covariance @ chosen, 0) * policy.periods))
            scale = min(1.0, policy.annual_risk_limit / max(volatility, 1e-12))
            for representative, weight in zip(representatives, chosen * scale, strict=True):
                for name in copies[representative]:
                    result["weights"][name] = float(weight / len(copies[representative]))
            result.update(
                status="allocated",
                estimated_annual_volatility=volatility * scale,
                cash_weight=float(1 - scale),
            )
    result["id"] = digest(result)
    return result


def _simplex_lattice(parts, units=10):
    """Integer simplex compositions in a fixed lexicographic order."""
    if parts == 1:
        yield (units,)
    else:
        for first in range(units + 1):
            for rest in _simplex_lattice(parts - 1, units - first):
                yield (first, *rest)


def _optimize_v2(past, mode, policy, previous, result):
    result.update(schema_version=4, training_objective=None)
    solver = {
        "scope": "risk_cap_surface_and_feasible_prior",
        "global_optimum_guaranteed": False,
        "lattice_step": 0.10,
        "maximum_lattice_points": 10000,
        "lattice_points_evaluated": 0,
        "lattice_status": "not_run",
        "lattice_refinement_starts": 4,
        "cost_model": "invested_ongoing_plus_actual_coefficient_switch_amortization",
        "cash_edge_rule": "positive_feasible_objective_required_except_fixed_equal_weight",
    }
    result["solver"] = solver
    result["cash_comparison"] = {
        "sharpe": None,
        "status": "undefined_zero_volatility",
        "amortized_liquidation_cost": float(
            policy.switch_bps / 10000 * previous.sum() / policy.holding_periods
        ),
    }
    if len(past) < policy.minimum:
        result["status"] = "insufficient_history_cash"
        result["id"] = digest(result)
        return result

    names = list(past)
    representatives, copies = [], {}
    for name in names:
        representative = next(
            (r for r in representatives if np.array_equal(past[r], past[name])), name
        )
        if representative == name:
            representatives.append(name)
        copies.setdefault(representative, []).append(name)
    p = len(representatives)
    # Compression keeps identical series from changing covariance shrinkage.
    membership = np.array([[float(n in copies[r]) for r in representatives] for n in names])
    expansion = membership.T / membership.sum(axis=0)[:, None]
    x = past[representatives].to_numpy()
    covariance, shrink = shrunk_covariance(x)
    ends = list(range(len(x), policy.window - 1, -policy.step))
    windows = [x[end - policy.window : end] for end in ends]
    estimates = [shrunk_covariance(z)[0] for z in windows]
    result.update(
        shrinkage=shrink,
        window_ends=[str(past.index[end - 1].date()) for end in ends],
        duplicate_groups=list(copies.values()),
    )

    def surface(compositions):
        q = np.atleast_2d(compositions)
        volatility = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", q, covariance, q), 0) * policy.periods
        )
        scale = np.minimum(1, policy.annual_risk_limit / np.maximum(volatility, 1e-12))
        return (q * scale[:, None]) @ expansion

    def scores(coefficients):
        w = np.atleast_2d(coefficients)
        grouped = w @ membership
        drag = (
            policy.ongoing_bps / 10000 * w.sum(axis=1)
            + policy.switch_bps / 10000 * abs(w - previous).sum(axis=1) / policy.holding_periods
        )
        pairs = (
            zip(windows, estimates, strict=True)
            if mode == "median_window_sharpe"
            else [(x, covariance)]
        )
        ratios = []
        for z, c in pairs:
            risk = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", grouped, c, grouped), 0))
            ratios.append(
                (policy.mean_shrink * grouped @ z.mean(axis=0) - drag) / np.maximum(risk, 1e-12)
            )
        return np.median(ratios, axis=0)

    starts = [np.full(p, 1 / p), *np.eye(p)]
    candidates = [(surface(starts[0])[0], "equal_weight_start")]
    previous_grouped = previous @ membership
    prior_risk = float(
        np.sqrt(max(previous_grouped @ covariance @ previous_grouped, 0) * policy.periods)
    )
    prior_eligible = previous.sum() > 0 and prior_risk <= policy.annual_risk_limit + 1e-12
    solver["feasible_prior_included"] = bool(prior_eligible and mode != "equal_weight")
    if mode == "equal_weight":
        solver.update(lattice_status="fixed_equal_weight", local_attempts=0, local_successes=0)
    else:
        candidates.extend((surface(q)[0], "vertex_start") for q in np.eye(p))
        if prior_eligible:
            candidates.append((previous.copy(), "prior_no_trade"))
            starts.append(previous_grouped / previous_grouped.sum())
        lattice_size = comb(p + 9, p - 1)
        solver["lattice_points_required"] = lattice_size
        if lattice_size <= solver["maximum_lattice_points"]:
            lattice = np.asarray(list(_simplex_lattice(p)), dtype=float) / 10
            feasible = surface(lattice)
            values = scores(feasible)
            order = np.argsort(-values, kind="stable")
            best = int(order[0])
            candidates.append((feasible[best].copy(), "lattice"))
            starts.extend(lattice[i] for i in order[:4])
            solver.update(
                lattice_status="evaluated",
                lattice_points_evaluated=len(lattice),
                lattice_candidate_coverage="all_feasible_points_scored_best_retained",
                best_lattice_objective=float(values[best]),
            )
        else:
            solver["lattice_status"] = "skipped_exceeds_point_bound"
        solutions = [
            minimize(
                lambda q: -float(scores(surface(q))[0]),
                start,
                method="SLSQP",
                bounds=[(0, 1)] * p,
                constraints={"type": "eq", "fun": lambda q: q.sum() - 1},
                options={"ftol": 1e-10, "maxiter": 250},
            )
            for start in starts
        ]
        valid = [
            s.x
            for s in solutions
            if s.success
            and np.isfinite(s.x).all()
            and abs(s.x.sum() - 1) < 1e-7
            and s.x.min() >= -1e-8
        ]
        for solution in valid:
            q = np.maximum(solution, 0)
            q[q < 1e-8] = 0
            candidates.append((surface(q / q.sum())[0], "local_refinement"))
        solver.update(local_attempts=len(solutions), local_successes=len(valid))
        solver["best_local_objective"] = max(
            (float(scores(w)[0]) for w, origin in candidates if origin == "local_refinement"),
            default=None,
        )

    finite = [
        (w, origin, float(scores(w)[0])) for w, origin in candidates if np.isfinite(scores(w)[0])
    ]
    if not finite:
        result["status"] = "optimizer_failed_cash"
    else:
        chosen, origin, value = max(finite, key=lambda item: item[2])
        solver.update(
            best_feasible_objective=value,
            selected_candidate=origin,
            feasible_candidates_compared=len(finite),
        )
        if mode != "equal_weight" and value <= 0:
            result["status"] = "no_positive_training_edge_cash"
        else:
            grouped = chosen @ membership
            result.update(
                status="allocated",
                weights=dict(zip(names, map(float, chosen), strict=True)),
                training_objective=value,
                estimated_annual_volatility=float(
                    np.sqrt(max(grouped @ covariance @ grouped, 0) * policy.periods)
                ),
                cash_weight=float(1 - chosen.sum()),
            )
    result["id"] = digest(result)
    return result
