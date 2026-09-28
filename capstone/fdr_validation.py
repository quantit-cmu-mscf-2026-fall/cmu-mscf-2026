"""
fdr_validation.py
-----------------
Validate a batch of backtested strategies with FDR control:

  * BH  - Benjamini & Hochberg (1995). Valid for independent or PRDS
          (positively dependent) test statistics.
  * BY  - Benjamini & Yekutieli (2001). Valid under ANY dependence;
          BH thresholds are divided by c(m) = sum_{i=1..m} 1/i.

Typical use
-----------
    import pandas as pd
    from fdr_validation import validate_strategies

    # returns: DataFrame, rows = time (e.g. daily), columns = strategies
    # values  = periodic (excess) returns of each strategy
    res = validate_strategies(returns, q=0.05, periods_per_year=252)
    survivors_bh = res[res["reject_bh"]].index
    survivors_by = res[res["reject_by"]].index

Important
---------
* Include EVERY strategy you tried (not just the good ones) -- m must be the
  true number of hypotheses, otherwise FDR control is meaningless.
* FDR corrects selection bias only. Still confirm survivors out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Core FDR procedures
# --------------------------------------------------------------------------- #
def fdr_stepup(pvals, q: float = 0.05, method: str = "bh"):
    """
    Step-up FDR procedure on a vector of p-values.

    Parameters
    ----------
    pvals  : array-like of raw p-values (length m)
    q      : target FDR level
    method : "bh" (Benjamini-Hochberg) or "by" (Benjamini-Yekutieli)

    Returns
    -------
    reject   : boolean np.ndarray, same order as input
    p_adj    : np.ndarray of adjusted p-values (q-values), same order as input
    threshold: largest raw p-value that was rejected (nan if none)
    """
    p = np.asarray(pvals, dtype=float)
    if np.any(np.isnan(p)):
        raise ValueError("pvals contains NaN; drop or fix those strategies first.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvals must lie in [0, 1].")

    m = p.size
    method = method.lower()
    if method == "bh":
        c_m = 1.0
    elif method == "by":
        c_m = np.sum(1.0 / np.arange(1, m + 1))  # harmonic number ~ ln(m)+0.577
    else:
        raise ValueError("method must be 'bh' or 'by'.")

    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)

    # Rejection rule: largest k with p_(k) <= k/(m*c_m) * q
    crit = ranks / (m * c_m) * q
    below = p_sorted <= crit
    if below.any():
        k = np.max(np.nonzero(below)[0])  # 0-based index of the largest such i
        reject_sorted = np.zeros(m, dtype=bool)
        reject_sorted[: k + 1] = True
        threshold = p_sorted[k]
    else:
        reject_sorted = np.zeros(m, dtype=bool)
        threshold = np.nan

    # Adjusted p-values: p_adj(i) = min_{j>=i} min(1, m*c_m/j * p_(j))
    adj_sorted = np.minimum.accumulate((p_sorted * m * c_m / ranks)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    reject = np.empty(m, dtype=bool)
    p_adj = np.empty(m, dtype=float)
    reject[order] = reject_sorted
    p_adj[order] = adj_sorted
    return reject, p_adj, threshold


# --------------------------------------------------------------------------- #
# Turning strategy returns into p-values
# --------------------------------------------------------------------------- #
def _newey_west_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """t-stat for H0: mean(x) = 0 using a Newey-West (Bartlett) HAC variance."""
    x = np.asarray(x, dtype=float)
    T = x.size
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, T - 1))

    gamma0 = np.dot(e, e) / T
    lrv = gamma0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma_l = np.dot(e[lag:], e[:-lag]) / T
        lrv += 2.0 * w * gamma_l
    lrv = max(lrv, 1e-18)
    return mu / np.sqrt(lrv / T)


def strategy_pvalues(
    returns: pd.DataFrame,
    hac: bool = True,
    lags: int | None = None,
    one_sided: bool = True,
) -> pd.Series:
    """
    Raw p-value per strategy for H0: mean return <= 0 (one-sided) or = 0 (two-sided).

    hac=True uses Newey-West to allow autocorrelation / heteroskedasticity.
    Uses the t distribution with T-1 d.o.f.
    """
    pvals = {}
    for col in returns.columns:
        x = returns[col].dropna().to_numpy()
        T = x.size
        if T < 10:
            pvals[col] = np.nan
            continue
        if hac:
            t = _newey_west_tstat(x, lags)
        else:
            sd = x.std(ddof=1)
            t = x.mean() / (sd / np.sqrt(T)) if sd > 0 else 0.0
        if one_sided:
            p = stats.t.sf(t, df=T - 1)
        else:
            p = 2.0 * stats.t.sf(abs(t), df=T - 1)
        pvals[col] = p
    return pd.Series(pvals, name="p_raw")


# --------------------------------------------------------------------------- #
# One-call validation
# --------------------------------------------------------------------------- #
def validate_strategies(
    returns: pd.DataFrame | None = None,
    pvalues: pd.Series | None = None,
    q: float = 0.05,
    periods_per_year: int = 252,
    hac: bool = True,
    lags: int | None = None,
    one_sided: bool = True,
) -> pd.DataFrame:
    """
    Run BH and BY on a family of strategies.

    Provide EITHER `returns` (DataFrame: time x strategies, periodic excess
    returns) OR `pvalues` (Series indexed by strategy name, e.g. from a
    bootstrap / reality-check procedure you ran yourself).

    Returns a DataFrame sorted by raw p-value with columns:
        [mean_ret_ann, vol_ann, sharpe_ann, t_stat*, p_raw,
         p_bh, p_by, reject_bh, reject_by]
      (*performance columns only when `returns` is given)
    p_bh / p_by are FDR-adjusted p-values (compare directly to q).
    """
    if (returns is None) == (pvalues is None):
        raise ValueError("Pass exactly one of `returns` or `pvalues`.")

    if returns is not None:
        p_raw = strategy_pvalues(returns, hac=hac, lags=lags, one_sided=one_sided)
        mean = returns.mean()
        vol = returns.std(ddof=1)
        out = pd.DataFrame(
            {
                "mean_ret_ann": mean * periods_per_year,
                "vol_ann": vol * np.sqrt(periods_per_year),
                "sharpe_ann": (mean / vol) * np.sqrt(periods_per_year),
                "n_obs": returns.count(),
            }
        )
        out["p_raw"] = p_raw
    else:
        out = pd.DataFrame({"p_raw": pd.Series(pvalues, dtype=float)})

    valid = out["p_raw"].notna()
    if not valid.any():
        raise ValueError("No valid p-values to test.")

    rej_bh, adj_bh, thr_bh = fdr_stepup(out.loc[valid, "p_raw"].values, q, "bh")
    rej_by, adj_by, thr_by = fdr_stepup(out.loc[valid, "p_raw"].values, q, "by")

    out["p_bh"] = np.nan
    out["p_by"] = np.nan
    out["reject_bh"] = False
    out["reject_by"] = False
    out.loc[valid, "p_bh"] = adj_bh
    out.loc[valid, "p_by"] = adj_by
    out.loc[valid, "reject_bh"] = rej_bh
    out.loc[valid, "reject_by"] = rej_by

    out.attrs.update(
        {
            "q": q,
            "m": int(valid.sum()),
            "threshold_bh": thr_bh,
            "threshold_by": thr_by,
            "n_reject_bh": int(rej_bh.sum()),
            "n_reject_by": int(rej_by.sum()),
        }
    )
    return out.sort_values("p_raw")


def summarize(res: pd.DataFrame) -> str:
    a = res.attrs
    return (
        f"Strategies tested (m): {a['m']} | target FDR q = {a['q']}\n"
        f"BH survivors: {a['n_reject_bh']}  (raw p cutoff = {a['threshold_bh']})\n"
        f"BY survivors: {a['n_reject_by']}  (raw p cutoff = {a['threshold_by']})\n"
        f"Naive p<{a['q']} (no correction): {int((res['p_raw'] < a['q']).sum())}"
    )


# --------------------------------------------------------------------------- #
# Demo / self-test with simulated data
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    T, m, m_true = 1260, 200, 20  # 5y daily, 200 strategies, 20 with real alpha
    daily_sigma = 0.01
    alpha_daily = 0.0010  # ~25% annualized true edge (Sharpe ~1.6) for the real ones

    data = rng.normal(0, daily_sigma, size=(T, m))
    data[:, :m_true] += alpha_daily
    rets = pd.DataFrame(
        data,
        columns=[f"real_{i}" if i < m_true else f"noise_{i}" for i in range(m)],
    )

    res = validate_strategies(returns=rets, q=0.05, periods_per_year=252)
    print(summarize(res))
    print()
    print(res.head(10).round(4).to_string())

    fd_bh = res[res.reject_bh].index.str.startswith("noise").sum()
    fd_by = res[res.reject_by].index.str.startswith("noise").sum()
    print(f"\nFalse discoveries -> BH: {fd_bh}, BY: {fd_by}")
