# Hypothesis-level multiple testing — implementation

Implements [`DESIGN.md`](DESIGN.md): does a **bootstrap max omnibus + BH/BY**
procedure control hypothesis-level FDR, and does it gain power over a
**Bonferroni omnibus** baseline by exploiting dependence among related
specifications?

```bash
cd paper_validation/validation1_hypothesis_level_FDR
PYTHONPATH=. python run_experiment.py          # pilot: B=1999, R=120, ~35 min
PYTHONPATH=. python make_figures.py            # the four §6 figures + tidy CSVs
PYTHONPATH=. pytest . -q                       # 25 tests, ~2 s
```

`PYTHONPATH=.` is required — the modules import each other as siblings and the
directory is not an installed package.

| File | Role |
|---|---|
| `DESIGN.md` | The specification. Source of truth; this code serves it. |
| `dgp.py` | Data-generating process, HAC standard errors, candidate statistics |
| `bootstrap.py` | Circular block bootstrap; family max-omnibus **and** per-candidate p-values |
| `methods.py` | The six procedures of §3, plus BH/BY primitives |
| `metrics.py` | §6 measures at both levels, MCSE, paired differences |
| `run_experiment.py` | Monte Carlo driver, scenario grid, CLI, ledger logging |
| `make_figures.py` | The four §6 figures and their tidy plotting tables |
| `test_validation1.py` | 25 tests, weighted toward the DGP and the bootstrap's null |

---

## The DGP, which DESIGN.md left open

§4 Phase B says the return-generating equation "will be finalized in the next
revision"; §7 asks for separate common-time, family, and idiosyncratic shocks.
The model implemented is

```
r[h,j,t] = theta[h,j] + sigma * z[h,j,t]

z[h,j,t] = sqrt(rho_B)         * c[t]       common across all families
         + sqrt(rho_W - rho_B) * f[h,t]     shared within family h
         + sqrt(1 - rho_W)     * e[h,j,t]   idiosyncratic
```

each component a unit-variance AR(1), `x[t] = phi*x[t-1] + sqrt(1-phi^2)*eta[t]`.

The weights make `Var(z) = 1`, within-family correlation exactly `rho_W` and
between-family exactly `rho_B`. The middle weight is real only when
`rho_W >= rho_B` — so **§4 validity rule 1 falls out of the model** rather than
being imposed on top of it. Effects use the local-alternative parameterisation
`theta = delta*sigma/sqrt(T)`, and a family is non-null **iff** at least one of
its candidates is (§4 Phase B.3), derived rather than assigned.

---

## Three decisions worth reviewing

Each was a judgement call where the spec was silent, or where measurement
contradicted the obvious choice. Full reasoning lives in the relevant module
docstring.

### 1. Marginal p-values come from the bootstrap, not from HAC

The first implementation gave the Bonferroni / flat / naive baselines
Newey–West marginal p-values. At the global null they were anti-conservative,
worsening with serial dependence:

| `P(p <= 0.01)`, target 0.010 | φ=0 | φ=0.2 | φ=0.5 |
|---|---|---|---|
| Newey–West | 0.0121 | 0.0147 | 0.0232 |

Taking a minimum over 20 candidates compounds it, and **Bonferroni's global-null
FWER reached 0.40** — an over-rejection caused by marginal inference, not by
multiplicity, which would have confounded the one comparison the experiment
exists to make. §3.2 requires *marginally valid* p-values "using the same
time-series inference assumptions as the main method", so the same block
bootstrap now calibrates every candidate. Bonferroni is correctly conservative
afterwards. `dgp.newey_west_se` is kept for diagnostics.

### 2. `B` must exceed `H/q`

Outer BH tests its most extreme rank at `q/H` — 0.0005 at H=100, q=0.05 — but a
bootstrap p-value can never fall below `1/(B+1)`. The originally planned pilot
`B=499` could not resolve that threshold, which would have silently destroyed
power rather than failing loudly. `bootstrap.check_resolution` enforces the
condition and `run_experiment.py` refuses to start when `--boot` is too small.
This is §5 rule 7 made executable.

### 3. Block length `ceil(2*T^(1/3))` = 16 at T=500, chosen by measurement

§4 Phase A.3 wants a pre-specified rule and Phase F.6 wants evidence the
conclusions do not hinge on an arbitrary one. Marginal calibration at the global
null, `P(p <= 0.01)`:

| block | 8 | 16 | 25 | 50 |
|---|---|---|---|---|
| φ=0.0 | 0.0121 | 0.0132 | 0.0159 | 0.0227 |
| φ=0.2 | 0.0141 | 0.0144 | 0.0169 | 0.0234 |
| φ=0.5 | 0.0198 | 0.0172 | 0.0189 | 0.0245 |

Short blocks sever serial dependence at block boundaries and understate the
long-run variance; long blocks leave too few distinct blocks and the resampled
variance itself turns noisy. The optimum sits near 16 at T=500.

**A residual anti-conservatism of roughly 1.3–1.7× at the 1% level survives at
every block length.** That is a finite-sample property of the studentised block
bootstrap, not a defect here, and it applies identically to every method — so
absolute levels are mildly optimistic while the comparison stays fair. Figure 1
measures the family-level consequence directly.

---

## Scope of the delivered run

Not the full Cartesian grid, and deliberately so. §4 validity rule 5: *"Do not
run the full Cartesian grid until the global-null and a small alternative
scenario pass the sanity checks in Phase F."* The default scope is that path:

| group | scenarios | feeds |
|---|---|---|
| `global_null` | π₁=0, δ=0, over ρ_W × φ (12) | Figure 1 |
| `calibration` | the §6 Figure 2 primary scenario (1) | Figure 2 |
| `power_grid` | π₁=0.1, over ρ_W × δ (12) | Figures 3, 4 |

`--scope full` adds T=1000; `--boot 4999 --reps 1000` reaches DESIGN.md's final
settings. `q` is free — it only re-thresholds p-values already computed — so
both levels are evaluated from one run, which is what makes Figure 2 a
calibration *curve*.

### Performance

A naive bootstrap gathers `X[idx]`, moving `B*T*N` floats. A circular block sum
is a difference of prefix sums, so only `B*n_blocks*N` need move — an `l`-fold
reduction. **B=1999 now runs faster than B=499 did before the rewrite**
(≈0.7 s/replication), which is what makes the final `B=4999` reachable rather
than theoretical.

---

## Reading the results

Three rules from §5 that the code encodes rather than merely documents:

1. **Flat BH is not a hypothesis-level procedure.** It controls candidate-level
   FDR; its family column is *induced* (a family counts as found if any of its
   specifications was). Both are reported; only the first is a claim.
2. **A family rejection does not validate any specification inside it.** Every
   family-level method therefore records an all-False candidate vector rather
   than "the winner" — §5 rule 3 made structural.
3. **More rejections is not better.** Read power only where the matching FDR is
   controlled; Figures 3 and 4 mute the different-target and negative-control
   series for exactly this reason.

Every invocation of `run_experiment.py` appends one entry to
`experiments/runs.jsonl`, logged before any result is printed. `make_figures.py`
deliberately does not log — it re-renders a trial already recorded.
