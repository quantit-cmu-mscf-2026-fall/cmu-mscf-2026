# Hypothesis-Level Multiple-Testing Experiment

**Status:** Initial design draft (v0.1)  
**Current scope:** Research question, testing targets, experimental workflow, and references.  
**Next revision:** Add the full data-generating process (DGP), simulation grid, and performance-analysis specification.

## 1. Objective

This project compares multiple-testing procedures for an agentic alpha-research setting. The agent first proposes economic hypotheses. Within each economic hypothesis, it generates multiple signal expressions and hyperparameter configurations. The statistical problem therefore has two levels:

1. **Hypothesis level:** Which economic hypotheses contain at least one genuinely predictive implementation?
2. **Specification level:** Which individual expressions or hyperparameter configurations appear predictive?

The primary target of the experiment is **hypothesis-level discovery**, not selection of every individually valid specification.

For hypothesis/family \(h=1,\ldots,H\), let

\[
\mathcal F_h=\{(e,\theta):\text{expression and hyperparameter choices generated for hypothesis }h\},
\]

and let \(m_h=|\mathcal F_h|\). Index the members of this family by \(j=1,\ldots,m_h\). Let \(\theta_{hj}\) denote the population performance of candidate \(j\), measured using one pre-specified primary endpoint, such as mean long-short return or risk-adjusted alpha.

The family-level null and alternative are

\[
H_h:\theta_{hj}\le 0\quad\text{for every }j\in\mathcal F_h,
\]

versus

\[
H_h^A:\theta_{hj}>0\quad\text{for at least one }j\in\mathcal F_h.
\]

Thus, rejecting \(H_h\) means that the economic hypothesis has at least one implementation whose apparent performance cannot be explained by the internal expression/hyperparameter search alone.

## 2. Core Questions

The experiment should answer the following questions.

1. Does the proposed **bootstrap max omnibus + BH/BY** procedure control hypothesis-level false discovery rate at the nominal level?
2. Relative to a Bonferroni family-level test, how much power is gained by preserving and using the dependence among related specifications?
3. How does hypothesis-level testing differ from applying BH directly to all individual specifications?
4. How severely does the naive minimum-\(p\) approach fail when the number of searched specifications grows?
5. How do the conclusions change with within-family correlation, cross-family correlation, unequal family size, time-series dependence, signal sparsity, and effect size?

## 3. Methods to Compare

Use the same simulated dataset, candidate statistics, random seed, and one-sided alternative for every method within a Monte Carlo replication. This paired design isolates differences caused by the multiple-testing procedure.

### 3.1 Main method: bootstrap max omnibus + outer BH/BY

For every candidate, compute a studentized statistic

\[
T_{hj}=\frac{\sqrt{T}\,\widehat\theta_{hj}}{\widehat\sigma_{hj}},
\]

and define the observed family maximum

\[
M_h^{\mathrm{obs}}=\max_{1\le j\le m_h}T_{hj}.
\]

For bootstrap replication \(b=1,\ldots,B\):

1. Draw one common set of time blocks and apply it jointly to all hypotheses and specifications. Do not bootstrap each candidate independently.
2. Recompute each candidate estimate and standard error on the resampled data.
3. Impose the null by centering the bootstrap statistic:

   \[
   T_{hj}^{*(b)}=
   \frac{\sqrt T\left(\widehat\theta_{hj}^{*(b)}-\widehat\theta_{hj}\right)}
   {\widehat\sigma_{hj}^{*(b)}}.
   \]

4. Repeat the internal search in that bootstrap sample:

   \[
   M_h^{*(b)}=\max_{1\le j\le m_h}T_{hj}^{*(b)}.
   \]

Estimate the family-level omnibus p-value using the finite-simulation correction

\[
\widehat p_h^{\mathrm{boot}}
=
\frac{1+\sum_{b=1}^{B}\mathbf 1\{M_h^{*(b)}\ge M_h^{\mathrm{obs}}\}}
{B+1}.
\]

Apply either BH or BY to

\[
\widehat p_1^{\mathrm{boot}},\ldots,\widehat p_H^{\mathrm{boot}}.
\]

Report BH and BY as separate variants. BH is the higher-power default when the dependence assumptions are plausible; BY is the conservative robustness procedure under arbitrary dependence.

This construction is a **single max-statistic omnibus test within each family**. It is motivated by the resampling/max-\(T\) literature, but it should not be labeled the full Romano-Wolf stepdown procedure, because the experiment is not using stepdown inference to identify individual rejected specifications inside a family.

### 3.2 Same-target baseline: Bonferroni omnibus + outer BH/BY

First obtain a marginally valid one-sided p-value \(p_{hj}\) for every candidate using the same studentization and time-series inference assumptions as the main method. Construct

\[
p_h^{\mathrm{Bonf}}
=
\min\left(1,m_h\min_{1\le j\le m_h}p_{hj}\right).
\]

Then apply BH or BY across the \(H\) family-level p-values.

This is the fairest baseline because it tests the same family-level null as the bootstrap omnibus method. It does not use the dependence among specifications and will often be conservative when many candidate signals are strongly correlated.

### 3.3 Different-target baseline: flat BH over all specifications

Pool every candidate p-value across all families:

\[
\{p_{hj}:h=1,\ldots,H;\;j=1,\ldots,m_h\},
\]

and apply BH once to the full collection.

This procedure controls **candidate/specification-level FDR** under the relevant BH assumptions. It does not automatically control hypothesis-level FDR if a hypothesis is called discovered whenever at least one of its specifications is rejected. Consequently, report its candidate-level results and its induced hypothesis-level results separately.

Flat BH is useful for showing how the estimand changes when numerous highly similar expressions and hyperparameters are treated as distinct discoveries. It should not be described as an equivalent replacement for the family-level methods.

### 3.4 Negative control: naive minimum-p + outer BH

Define

\[
p_h^{\mathrm{naive}}=\min_{1\le j\le m_h}p_{hj}
\]

without correcting for the internal search, and apply BH across families.

This p-value is generally not valid under the family null because searching more candidates mechanically makes the minimum p-value smaller. This method is intentionally included as a negative control to demonstrate false-discovery inflation, not as a recommended procedure.

## 4. Experimental Workflow

### Phase A: Freeze the statistical specification

Before running the main simulation:

1. Fix the one-sided family null and the definition of a true non-null family.
2. Select one primary performance parameter \(\theta_{hj}\). Do not search over Sharpe ratio, IC, alpha, and mean return and then report whichever is most significant. If multiple metrics are searched, metric choice must itself be included in \(\mathcal F_h\).
3. Pre-specify the test level \(q\), primary block-bootstrap method, block-length rule, number of bootstrap replications \(B\), and number of Monte Carlo replications \(R\).
4. Fix the exact variants to compare. At minimum:
   - Bootstrap max omnibus + BH;
   - Bootstrap max omnibus + BY;
   - Bonferroni omnibus + BH;
   - Flat BH over all specifications;
   - Naive minimum-p + BH.
5. Pre-specify any secondary variants, such as Bonferroni + BY, flat BY, or Simes omnibus + BH.

### Phase B: Generate one Monte Carlo dataset

For replication \(r=1,\ldots,R\):

1. Generate a time series for \(H\) hypothesis families with \(m_h\) candidate specifications in each family.
2. Store the true candidate effects \(\theta_{hj}\).
3. Label family \(h\) as truly non-null if and only if at least one \(\theta_{hj}>0\).
4. Preserve the same generated dataset for every comparison method in that replication.

#### Current DGP parameter configuration

The first implementation uses the following parameter grid. The simulated outcome is the candidate signal's excess return net of transaction costs.

| Parameter | Symbol | Current values |
| --- | ---: | ---: |
| Number of hypothesis families | \(H\) | \(100\) |
| Candidate specifications per family | \(m_h\) | \(20\) |
| Time-series length | \(T\) | \(500, 1000\) |
| Proportion of non-null families | \(\pi_1\) | \(0, 0.1, 0.3\) |
| Within-family correlation | \(\rho_W\) | \(0, 0.3, 0.7, 0.9\) |
| Between-family correlation | \(\rho_B\) | \(0, 0.1\) |
| Serial correlation | \(\phi\) | \(0, 0.2, 0.5\) |
| Standardized signal strength | \(\delta\) | \(0, 2, 3, 4\) |
| Target FDR level | \(q\) | \(0.05, 0.10\) |
| Bootstrap replications | \(B\) | pilot: \(999\); final: \(4{,}999\) |
| Monte Carlo replications | \(R\) | pilot: \(200\); final: at least \(1{,}000\) |

Use

\[
\delta_{hj}=\frac{\sqrt T\,\theta_{hj}}{\sigma_{hj}}
\]

as the standardized effect-size parameter. When a common marginal volatility \(\sigma\) is used, set a nonzero candidate mean to

\[
\theta_{hj}=\frac{\delta\sigma}{\sqrt T}.
\]

This local-alternative parameterization keeps the statistical difficulty comparable when \(T\) changes. A later robustness experiment may instead hold annualized Sharpe ratio fixed, in which case power should increase with \(T\).

Apply the following validity rules when constructing the grid:

1. Only use correlation pairs satisfying \(0\leq\rho_B\leq\rho_W\leq1\). In particular, when \(\rho_W=0\), require \(\rho_B=0\); do not run the invalid pair \((\rho_W,\rho_B)=(0,0.1)\).
2. Use \(\delta=0\) for null-family calibration. A global-null scenario has \(\pi_1=0\) and \(\delta=0\).
3. Alternative scenarios require \(\pi_1>0\) and \(\delta\in\{2,3,4\}\). Do not label a family non-null when all of its candidate effects are zero.
4. Use \(B=999\) and \(R=200\) only for pilot runs and code validation. Use \(B=4{,}999\) and \(R\geq1{,}000\) for final reported results.
5. Do not run the full Cartesian grid until the global-null and a small alternative scenario pass the sanity checks in Phase F.

The mathematical return-generating equation will be finalized in the next revision. It must allow the experiment to vary:

- the proportion of non-null families;
- sparse versus dense signals within a non-null family;
- equal versus unequal family sizes;
- within-family dependence;
- between-family dependence;
- serial dependence and volatility persistence;
- sample size and signal strength.

### Phase C: Compute common candidate-level inputs

1. Estimate \(\widehat\theta_{hj}\), \(\widehat\sigma_{hj}\), \(T_{hj}\), and marginal p-values \(p_{hj}\) for all candidates.
2. Use identical estimated statistics as inputs to all methods whenever possible.
3. Generate one shared collection of bootstrap index sequences and reuse it across candidates. This preserves the joint dependence structure and reduces irrelevant Monte Carlo noise between methods.
4. Record diagnostic information, including effective family size, correlation of candidate statistics, convergence failures, and bootstrap Monte Carlo standard errors.

### Phase D: Apply every testing method

On the same dataset:

1. Compute bootstrap max omnibus p-values and apply outer BH and BY.
2. Compute Bonferroni family p-values and apply outer BH (and BY if included).
3. Apply flat BH to all candidate p-values.
4. Compute naive family minimum p-values and apply outer BH.
5. Store both family-level and candidate-level rejection indicators for every method.

### Phase E: Aggregate across Monte Carlo replications

The next revision will provide formal definitions and code-level formulas. At minimum, evaluate:

- hypothesis-level FDR and realized false discovery proportion;
- hypothesis-level power;
- candidate-level FDR and power;
- familywise probability of at least one false discovery;
- sensitivity to family size and dependence;
- stability across block lengths;
- computational time and bootstrap Monte Carlo error.

The primary comparison is:

> **Bootstrap max omnibus + BH/BY versus Bonferroni omnibus + BH/BY**, because these methods target the same family-level scientific question.

Flat BH is a secondary comparison used to demonstrate the difference between hypothesis-level and candidate-level discovery. Naive minimum-p + BH is a calibration failure check.

### Phase F: Sanity checks before interpreting power

1. Run a global-null scenario first. A method that fails type-I error/FDR calibration under the global null should not be compared on power without clearly flagging the failure.
2. Verify that increasing \(m_h\) does not inflate rejection probability for null families under the valid omnibus procedures.
3. Check that the naive minimum-p method becomes increasingly anti-conservative as family size grows.
4. Under candidate independence, verify that bootstrap-max and Bonferroni results are directionally consistent.
5. Under strong positive within-family dependence, check whether bootstrap max gains power relative to Bonferroni by recognizing that many searched specifications are redundant.
6. Confirm that results are not driven by one arbitrary block length or too few bootstrap draws.

## 5. Important Interpretation Rules

1. **Do not compare raw rejection counts alone.** More rejections are desirable only when the relevant error rate remains controlled.
2. **Do not treat flat BH as a hypothesis-level procedure.** Candidate-level FDR and hypothesis-level FDR are different quantities.
3. **Do not call the winning in-family specification validated merely because its family omnibus test rejects.** The omnibus rejection establishes evidence that at least one implementation works; it does not provide selection-adjusted inference for the identity or effect size of the winner.
4. **Separate discovery from translation.** After a family is discovered, selecting a deployable signal requires a pre-specified rule, selective inference, sample splitting, or an untouched validation sample.
5. **Preserve joint resampling.** Independent resampling of each specification destroys the dependence structure that gives the bootstrap max procedure its potential power advantage.
6. **Use studentized statistics.** Raw performance metrics with different scales should not be maximized together.
7. **Treat estimated p-values as Monte Carlo quantities.** Use the plus-one correction and enough bootstrap draws for the intended rejection threshold.

## 6. Required Performance Figures

Generate the following four figures from the Monte Carlo results. Save the underlying plotting data as tidy tables in addition to saving the rendered figures. Use the same method colors, labels, ordering, and axis limits across all figures.

### Common performance definitions

For Monte Carlo replication \(r\), let \(\mathcal D_H^{(r)}\) be the set of rejected hypothesis families, \(\mathcal H_0\) the set of true-null families, and \(\mathcal H_1\) the set of true non-null families. Define

\[
V_H^{(r)}=|\mathcal D_H^{(r)}\cap\mathcal H_0|,
\qquad
S_H^{(r)}=|\mathcal D_H^{(r)}\cap\mathcal H_1|,
\]

\[
\operatorname{FDP}_H^{(r)}
=
\frac{V_H^{(r)}}{\max(|\mathcal D_H^{(r)}|,1)},
\qquad
\operatorname{Power}_H^{(r)}
=
\frac{S_H^{(r)}}{|\mathcal H_1|}.
\]

Estimate hypothesis-level FDR and power by

\[
\widehat{\operatorname{FDR}}_H
=
\frac1R\sum_{r=1}^R\operatorname{FDP}_H^{(r)},
\qquad
\widehat{\operatorname{Power}}_H
=
\frac1R\sum_{r=1}^R\operatorname{Power}_H^{(r)}.
\]

For any Monte Carlo average \(\widehat m=R^{-1}\sum_r m^{(r)}\), report

\[
\operatorname{MCSE}(\widehat m)
=
\frac{\operatorname{sd}(m^{(1)},\ldots,m^{(R)})}{\sqrt R},
\]

and plot 95% Monte Carlo error bars \(\widehat m\pm1.96\operatorname{MCSE}(\widehat m)\). These intervals quantify simulation uncertainty, not sampling uncertainty within one generated dataset.

Use the following method order throughout:

1. Bootstrap max omnibus + BH;
2. Bootstrap max omnibus + BY;
3. Bonferroni omnibus + BH;
4. Bonferroni omnibus + BY, if implemented;
5. Flat BH over all specifications, mapped to family discovery when at least one candidate is rejected;
6. Naive minimum-p + BH.

Clearly mark flat BH as a different-target method and naive minimum-p + BH as a negative control.

### Figure 1: Global-null empirical FDR/FWER

**Purpose:** Determine whether each procedure controls false discoveries when every family is null.

Use the global-null scenario

\[
\pi_1=0,
\qquad
\delta=0.
\]

Under the global null,

\[
\operatorname{FDP}_H^{(r)}
=
\mathbf 1\{|\mathcal D_H^{(r)}|>0\},
\]

so hypothesis-level FDR equals the probability of at least one false rejection, which is also FWER:

\[
\widehat{\operatorname{FDR}}_H
=
\widehat{\operatorname{FWER}}_H
=
\frac1R\sum_{r=1}^R
\mathbf 1\{|\mathcal D_H^{(r)}|>0\}.
\]

Plot specification:

- x-axis: testing method;
- y-axis: empirical global-null FDR/FWER;
- point estimate with 95% Monte Carlo error bar;
- horizontal reference line at the nominal \(q\);
- distinguish \(\rho_W\in\{0,0.3,0.7,0.9\}\) using color or grouped points;
- facet columns by \(q\in\{0.05,0.10\}\);
- facet rows by \(\phi\in\{0,0.2,0.5\}\);
- primary figure: \(T=500\), \(\rho_B=0\); place \(T=1000\) in the appendix or a second version.

Use a common y-axis beginning at zero. Extend the upper limit enough to reveal inflation from the naive method rather than truncating values at \(q\).

Suggested filename: `fig01_global_null_fdr_fwer.png`.

### Figure 2: Empirical FDR versus nominal q

**Purpose:** Compare error-rate calibration outside the global null.

Primary scenario:

\[
T=500,
\quad
\pi_1=0.1,
\quad
\delta=3,
\quad
\rho_W=0.7,
\quad
\rho_B=0.1,
\quad
\phi=0.2.
\]

Plot specification:

- x-axis: nominal FDR level \(q\);
- y-axis: empirical hypothesis-level FDR;
- one line and point series per method;
- add the 45-degree reference line \(y=x\);
- show 95% Monte Carlo error bars;
- use the current primary grid \(q\in\{0.05,0.10\}\);
- optionally add \(q\in\{0.01,0.025\}\) as diagnostic values to make the calibration curve more informative.

A valid method should lie on or below the reference line within Monte Carlo uncertainty. Being far below the line indicates conservatism; being materially above it indicates failure to control FDR.

Suggested filename: `fig02_empirical_fdr_vs_q.png`.

### Figure 3: Hypothesis-level power versus within-family correlation

**Purpose:** Measure whether the bootstrap max procedure gains power by recognizing dependence and redundancy among related specifications.

Primary scenario:

\[
\pi_1=0.1,
\quad
\rho_B=0,
\quad
\phi=0.2,
\quad
q=0.10.
\]

Setting \(\rho_B=0\) allows the full grid \(\rho_W\in\{0,0.3,0.7,0.9\}\), including \(\rho_W=0\), while respecting \(\rho_B\leq\rho_W\).

Plot specification:

- x-axis: within-family correlation \(\rho_W\);
- y-axis: empirical hypothesis-level power;
- one line and point series per method;
- show 95% Monte Carlo error bars;
- facet columns by \(\delta\in\{2,3,4\}\);
- facet rows by \(T\in\{500,1000\}\);
- use a common y-axis from 0 to 1.

Interpret power only for methods whose hypothesis-level FDR is adequately controlled in the corresponding scenario. Use a dashed or visually muted line for flat BH because it has a different primary error-rate target, and for naive minimum-p because it is an invalid negative control.

Suggested filename: `fig03_power_vs_within_family_correlation.png`.

### Figure 4: Power improvement heatmap, bootstrap minus Bonferroni

**Purpose:** Summarize where dependence-aware bootstrap inference provides a meaningful power improvement over the same-target Bonferroni baseline.

The primary comparison is

\[
\text{Bootstrap max omnibus + BH}
\quad\text{versus}\quad
\text{Bonferroni omnibus + BH}.
\]

For each Monte Carlo replication, calculate the paired difference

\[
d^{(r)}
=
\operatorname{Power}_{\mathrm{boot+BH}}^{(r)}
-
\operatorname{Power}_{\mathrm{Bonf+BH}}^{(r)}.
\]

Then report

\[
\widehat{\Delta\operatorname{Power}}
=
\frac1R\sum_{r=1}^R d^{(r)},
\qquad
\operatorname{MCSE}(\widehat{\Delta\operatorname{Power}})
=
\frac{\operatorname{sd}(d^{(1)},\ldots,d^{(R)})}{\sqrt R}.
\]

Use paired differences because every method is applied to the same generated dataset in each replication.

Primary scenario:

\[
\pi_1=0.1,
\quad
\rho_B=0,
\quad
\phi=0.2,
\quad
q=0.10.
\]

Plot specification:

- x-axis: \(\rho_W\in\{0,0.3,0.7,0.9\}\);
- y-axis: \(\delta\in\{2,3,4\}\);
- fill color: \(\widehat{\Delta\operatorname{Power}}\);
- facet columns by \(T\in\{500,1000\}\);
- use a diverging color scale centered at zero;
- print the estimated power difference inside each cell;
- visually mark cells whose paired 95% Monte Carlo interval includes zero;
- use the same symmetric color limits across facets.

Positive values favor bootstrap max; negative values favor Bonferroni. Do not interpret a positive difference as an improvement if bootstrap max fails FDR control in that same scenario.

Suggested filename: `fig04_bootstrap_minus_bonferroni_power.png`.

## 7. Planned Additions

The next version should add:

1. A mathematical DGP with separate common time, family, and idiosyncratic shocks.
2. Sparse, dense, smooth-parameter-surface, and global-null alternatives.
3. A complete simulation parameter grid.
4. Exact formulas for hypothesis-level and candidate-level FDR and power.
5. Confidence intervals or Monte Carlo standard errors for all reported performance measures.
6. Additional secondary tables and diagnostic figures.
7. Pseudocode and a proposed Python module/test structure.

## 8. References

### False discovery rate and multiple testing

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *The Annals of Statistics*, 29(4), 1165–1188. <https://doi.org/10.1214/aos/1013699998>
- Simes, R. J. (1986). An improved Bonferroni procedure for multiple tests of significance. *Biometrika*, 73(3), 751–754. <https://doi.org/10.1093/biomet/73.3.751>
- Benjamini, Y., & Bogomolov, M. (2014). Selective inference on multiple families of hypotheses. *Journal of the Royal Statistical Society: Series B*, 76(1), 297–318. <https://doi.org/10.1111/rssb.12028>

### Resampling, block bootstrap, and max-statistic methods

- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7(1), 1–26. <https://doi.org/10.1214/aos/1176344552>
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241. <https://doi.org/10.1214/aos/1176347265>
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313. <https://doi.org/10.1080/01621459.1994.10476870>
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment*. Wiley.
- Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237–1282. <https://doi.org/10.1111/j.1468-0262.2005.00615.x>

### Data snooping and finance motivation

- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126. <https://doi.org/10.1111/1468-0262.00152>
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365–380. <https://doi.org/10.1198/073500105000000063>
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *The Review of Financial Studies*, 29(1), 5–68. <https://doi.org/10.1093/rfs/hhv059>
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *The Journal of Portfolio Management*, 40(5), 94–107. <https://doi.org/10.3905/jpm.2014.40.5.094>

## 9. Working Principle for Future Implementation

When adding code or revising the design, preserve the following hierarchy:

\[
\text{candidate statistics}
\longrightarrow
\text{family-level omnibus p-values}
\longrightarrow
\text{outer BH/BY decisions}
\longrightarrow
\text{separate signal-selection/validation stage}.
\]

Do not silently change the error-rate target, pool specifications across families, optimize tuning choices on reported Monte Carlo results, or interpret family rejection as validation of the in-sample winner.
