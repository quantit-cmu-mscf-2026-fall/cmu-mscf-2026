"""Literal transcription of Harvey & Liu's `Haircut_SR.m`, as a reference oracle.

Source (downloaded 2026-10-05):
    https://people.duke.edu/~charvey/backtesting/Haircut_SR.m
    https://people.duke.edu/~charvey/backtesting/sample_random_multests.m
for Harvey, C. R. and Y. Liu (2015), "Backtesting", Journal of Portfolio
Management 41(1), 13-28.

This file exists to be *obviously* a transcription, not good Python: it keeps the
authors' variable names, their loop structure, their `mvnrnd` over the full
`Nsim_tests` columns, and their inconsistencies (see NOTES). It is the oracle
`tests/test_evaluate.py::TestHaircutReferenceComparison` compares
`evaluate.haircut_sharpe` against, so it must stay independent of `capstone`:
nothing here imports from the package.

NOTES on the authors' code, reproduced deliberately:

1. **Bonferroni uses M; Holm and BHY use M + 1.** `p_BON = min(M*p_val,1)` while
   `m_vec = 1:(M+1)` and the Holm/BHY loops run to `M+1`. The observed strategy
   is appended to M simulated ones, so the family really is M + 1 and the
   Bonferroni line is inconsistent with the other two. Kept as-is.
2. **p-values are two-sided throughout**, from a t distribution with N-1 degrees
   of freedom for the observed strategy (`2*(1-tcdf(T,N-1))`) but from a *normal*
   for the simulated ones (`2*(1-normcdf(t_value,0,1))`). Kept as-is.
3. **N is a monthly observation count**, and a daily year is 360 days
   (`N = floor(num_obs*12/360)`), not 252.
4. **The RHO >= 0.8 branch extrapolates.** It reuses the [0.6, 0.8) formula, so
   at RHO = 0.9 the parameters are `-0.5*row4 + 1.5*row5`. Kept as-is.
5. **`ind_an == 0 && ind_aut == 1` cannot run in MATLAB**: the branch reads a
   lowercase `sr` that is never assigned. Transcribed with the obviously
   intended `SR`.
6. **A non-positive Sharpe gives p_val > 1.** `p_val = 2*(1-tcdf(T,N-1))` with
   T < 0 exceeds 1, because the authors use T and not |T|. Their method assumes a
   positive reported Sharpe; this transcription returns NaN there rather than
   propagating a number that is not a p-value.

Run it to print the reference table the tests pin against:

    python scripts/haircut_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

PARAMS = Path(__file__).with_suffix(".json")

# Harvey, Liu and Zhu (2014) parameters, verbatim from Haircut_SR.m:
# columns are [rho, m_tot, p_0, lambda].
PARA0 = np.array(
    [
        [0.0, 1295, 3.9660 * 0.1, 5.4995 * 0.001],
        [0.2, 1377, 4.4589 * 0.1, 5.5508 * 0.001],
        [0.4, 1476, 4.8604 * 0.1, 5.5413 * 0.001],
        [0.6, 1773, 5.9902 * 0.1, 5.5512 * 0.001],
        [0.8, 3109, 8.3901 * 0.1, 5.5956 * 0.001],
    ]
)

# sm_fre in the authors' code: 1..5 = daily, weekly, monthly, quarterly, annual.
FREQUENCIES = {"daily": 1, "weekly": 2, "monthly": 3, "quarterly": 4, "annual": 5}
PERIODS_PER_YEAR = {1: 360, 2: 52, 3: 12, 4: 4, 5: 1}


def interpolate_parameters(RHO: float) -> np.ndarray:
    """`para_inter`, including the authors' extrapolating RHO >= 0.8 branch."""
    if 0.0 <= RHO < 0.2:
        return ((0.2 - RHO) / 0.2) * PARA0[0] + ((RHO - 0.0) / 0.2) * PARA0[1]
    if 0.2 <= RHO < 0.4:
        return ((0.4 - RHO) / 0.2) * PARA0[1] + ((RHO - 0.2) / 0.2) * PARA0[2]
    if 0.4 <= RHO < 0.6:
        return ((0.6 - RHO) / 0.2) * PARA0[2] + ((RHO - 0.4) / 0.2) * PARA0[3]
    if 0.6 <= RHO < 0.8:
        return ((0.8 - RHO) / 0.2) * PARA0[3] + ((RHO - 0.6) / 0.2) * PARA0[4]
    if 0.8 <= RHO < 1.0:
        # NOTE 4: the authors' own branch, reused unchanged.
        return ((0.8 - RHO) / 0.2) * PARA0[3] + ((RHO - 0.6) / 0.2) * PARA0[4]
    return PARA0[1]  # "Set at the preferred level if RHO is misspecified"


def sample_random_multests(
    rho: float,
    m_tot: int,
    p_0: float,
    lambda_: float,
    M_simu: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """`sample_random_multests.m`, including the full m_tot x m_tot `mvnrnd`."""
    sigma = 0.15 / np.sqrt(12)  # assumed level of monthly vol
    N = 240  # number of time-series

    sig_vec = np.concatenate([[1.0], rho * np.ones(m_tot - 1)])
    SIGMA = np.array([[sig_vec[abs(i - j)] for j in range(m_tot)] for i in range(m_tot)])
    MU = np.zeros(m_tot)
    shock_mat = rng.multivariate_normal(MU, SIGMA * (sigma**2 / N), M_simu)

    prob_vec = rng.uniform(0.0, 1.0, (M_simu, m_tot))
    mean_vec = rng.exponential(lambda_, (M_simu, m_tot))
    m_indi = prob_vec > p_0
    mu_nul = m_indi * mean_vec  # Null-hypothesis
    return np.abs(mu_nul + shock_mat) / (sigma / np.sqrt(N))


def annualised_sharpe(SR: float, sm_fre: int, ind_an: int, ind_aut: int, rho: float) -> float:
    """`sr_annual`: annualise, and correct for autocorrelation, as the authors do."""
    p = PERIODS_PER_YEAR[sm_fre]
    if ind_aut == 1 and sm_fre != 5:
        factor = (
            1.0 + (2.0 * rho / (1.0 - rho)) * (1.0 - ((1.0 - rho**p) / (p * (1.0 - rho))))
        ) ** -0.5
    else:
        factor = 1.0
    scale = 1.0 if ind_an == 1 else np.sqrt(p)
    # NOTE 5: the ind_an == 0 && ind_aut == 1 branch reads an unassigned `sr`.
    return SR * scale * factor


def monthly_observations(num_obs: int, sm_fre: int) -> int:
    """`N`: the observation count converted to months. A daily year is 360 days."""
    return int(np.floor(num_obs * 12 / PERIODS_PER_YEAR[sm_fre]))


def haircut_sr(
    sm_fre: int,
    num_obs: int,
    SR: float,
    ind_an: int,
    ind_aut: int,
    rho: float,
    num_test: int,
    RHO: float,
    *,
    WW: int = 2000,
    seed: int | None = 0,
) -> dict:
    """`Haircut_SR.m`. `num_test` is the authors' M: the count of OTHER trials."""
    rng = np.random.default_rng(seed)

    sr_annual = annualised_sharpe(SR, sm_fre, ind_an, ind_aut, rho)
    N = monthly_observations(num_obs, sm_fre)
    M = num_test

    m_vec = np.arange(1, M + 2)  # 1:(M+1)
    c_const = float(np.sum(1.0 / m_vec))  # c(M+1)

    para_inter = interpolate_parameters(RHO)

    # Nsim_tests = (floor(M/para_inter(2)) + 1)*floor(para_inter(2)+1)
    Nsim_tests = (int(M // int(para_inter[1])) + 1) * int(np.floor(para_inter[1] + 1))
    t_sample = sample_random_multests(
        para_inter[0], Nsim_tests, para_inter[2], para_inter[3], WW, rng
    )

    sr = sr_annual / np.sqrt(12)  # Sharpe ratio, monthly
    T = sr * np.sqrt(N)
    p_val = float(2.0 * (1.0 - stats.t.cdf(T, N - 1)))
    if not 0.0 <= p_val <= 1.0:  # NOTE 6
        return {"sr_annual": sr_annual, "N": N, "p_val": p_val, "defined": False}

    p_holm = np.ones(WW)
    p_bhy = np.ones(WW)

    for ww in range(WW):
        yy = t_sample[ww, :M]
        t_value = yy
        p_val_sub = 2.0 * (1.0 - stats.norm.cdf(t_value, 0.0, 1.0))

        p_val_all = np.append(p_val_sub, p_val)
        p_val_order = np.sort(p_val_all)

        # Holm
        p_holm_vec = np.empty(M + 1)
        for i in range(1, M + 2):
            p_new = [(M + 1 - j + 1) * p_val_order[j - 1] for j in range(1, i + 1)]
            p_holm_vec[i - 1] = min(max(p_new), 1.0)
        k = int(np.searchsorted(p_val_order, p_val, side="left"))
        p_holm[ww] = p_holm_vec[k]

        # BHY
        p_bhy_vec = np.empty(M + 1)
        p_0 = np.nan
        for i in range(1, M + 2):
            kk = (M + 1) - (i - 1)
            if kk == (M + 1):
                p_new = p_val_order[-1]
            else:
                p_new = min(((M + 1) * c_const / kk) * p_val_order[kk - 1], p_0)
            p_bhy_vec[kk - 1] = p_new
            p_0 = p_new
        p_bhy[ww] = p_bhy_vec[k]

    p_BON = min(M * p_val, 1.0)  # NOTE 1: M, not M + 1
    p_HOL = float(np.median(p_holm))
    p_BHY = float(np.median(p_bhy))
    p_avg = (p_BON + p_HOL + p_BHY) / 3.0

    out = {"sr_annual": sr_annual, "N": N, "p_val": p_val, "defined": True}
    for name, p in (("BON", p_BON), ("HOL", p_HOL), ("BHY", p_BHY), ("avg", p_avg)):
        z = float(stats.t.ppf(1.0 - p / 2.0, N - 1))
        sr_adj = (z / np.sqrt(N)) * np.sqrt(12)
        out[f"p_{name}"] = p
        out[f"sr_{name}"] = sr_adj
        out[f"hc_{name}"] = (sr_annual - sr_adj) / sr_annual
    return out


def main() -> None:
    p = json.loads(PARAMS.read_text())
    print(f"{'case':34} {'p_raw':>9} {'BON':>7} {'HOL':>7} {'BHY':>7}   (haircut Sharpe, then %)")
    for case in p["cases"]:
        res = haircut_sr(
            FREQUENCIES[case["frequency"]],
            case["num_obs"],
            case["sharpe"],
            1 if case["annualized"] else 0,
            1 if case["autocorr"] else 0,
            case.get("rho", 0.0),
            case["num_test"],
            case["avg_correlation"],
            WW=p["WW"],
            seed=p["seed"],
        )
        label = case["label"]
        if not res["defined"]:
            print(f"{label:34} undefined (p_val = {res['p_val']:.3f} > 1)")
            continue
        print(
            f"{label:34} {res['p_val']:9.6f} "
            f"{res['sr_BON']:7.3f} {res['sr_HOL']:7.3f} {res['sr_BHY']:7.3f}   "
            f"{res['hc_BON'] * 100:5.1f}% {res['hc_HOL'] * 100:5.1f}% "
            f"{res['hc_BHY'] * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
