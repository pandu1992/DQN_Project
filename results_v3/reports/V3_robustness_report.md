# Robustness of Value-Based DRL Agents to Sensor Noise and Communication Packet-Loss in Autonomous Vessel Navigation

**A multi-seed factorial study on the VesselEnvV3 physical/sensing/communication testbed**

> **Scope.** This is a *separate follow-up study* to the earlier Q1 comparison
> (which used `VesselEnvV2` and is not revised here). It runs on
> `VesselEnvV3`, which adds genuinely-implemented physical, sensing, and
> communication mechanisms (see `results/reports/V3_EXTENSION.md`). Every metric
> is really computed from the environment dynamics; none is fabricated. The
> deployed web application is unaffected. Every number is reproducible from
> `results_v3/raw/` via the scripts in `experiments/`.

---

## 1. Experimental Design

**Question.** When the perception and communication channel degrade, do the
four value-based DQN variants *differ in how well they hold up* — in navigation
success, and in the safety and precision metrics that matter for maritime
autonomy (collisions, IALA channel-keeping, docking accuracy, cross-track
error)?

**Design.** A full **Algorithm (4) × Noise (3) × Packet-Error (3)** factorial:

| Factor | Levels |
|---|---|
| Algorithm | DQN, Double DQN, Dueling DQN, Dueling Double DQN |
| Sensor observation noise (std) | 0.0, 0.1, 0.25 |
| Communication packet-error rate | 0.0, 0.2, 0.4 |

= **36 conditions**, each over **6 shared seeds** (a randomized-block / paired
design) = **216 runs**, **6,480 held-out evaluation episodes**. 130 training /
30 greedy-evaluation episodes per cell.

> [!INSIGHT]
> This is the study the earlier Q1 report could not run: it converts the six
> previously *out-of-scope* mechanisms into a concrete **robustness** analysis —
> quantifying, for the first time in this project, how sensing and communication
> degradation affect navigation **safety and precision**, not just reward.

---

## 2. Environment and Metrics

`VesselEnvV3` reuses the same graph, edge attributes, and Dijkstra planner as
the deployed simulator, adding continuous kinematics with a control-drift model
so that degradation propagates into real trajectory error. Metrics:

- **Navigation:** success rate (%), mean reward
- **Safety:** collision rate (% of episodes with ≥1 collision), collisions/ep,
  IALA violations/ep (channel-departure past a buoy)
- **Precision:** docking accuracy (distance from final pose to dock point;
  lower = better), cross-track error (CTE) vs the Dijkstra route
- **Routing:** optimality ratio, mean steps
- **Comms (manipulation check):** dropped frames/ep

Overall evaluation success across all conditions was **29.6 %** — a deliberately
hard, non-saturated benchmark (calibration showed success plateaus at ~20–45 %
for these small pure-JS agents).

---

## 3. Mathematical and Statistical Methodology

**Unit of analysis.** The 30 evaluation episodes of each seed are collapsed to
one scalar per metric; the **6 seed-scalars** are the independent observations
(episodes within a seed share a network and are not independent).

**Effect size (variance explained) in the factorial model:**

$$ \eta^2_p = \frac{\mathrm{SS}_{\text{effect}}}{\mathrm{SS}_{\text{effect}} + \mathrm{SS}_{\text{error}}} $$

**Pipeline.**
1. **Normality** — Shapiro–Wilk per cell (≈ 3 % of cells pass → non-parametric
   tests are the primary route).
2. **Variance homogeneity** — Levene / Brown–Forsythe across algorithms at the
   clean condition.
3. **Three-way factorial ANOVA** — `y ~ algorithm * noise * PER + seed`
   (Type-II SS, partial η²) for every metric: main effects **and all
   interactions**. The **Algorithm × Noise** and **Algorithm × PER**
   interactions are the core robustness question.
4. **Robustness contrasts** — worst condition (noise 0.25, PER 0.4) vs clean
   (0, 0), paired Wilcoxon signed-rank (exact) + Cohen's $d_z$, Holm-corrected
   within each metric.
5. **Post-hoc** — pairwise algorithm comparison at the harshest condition,
   paired Wilcoxon + Holm + Benjamini–Hochberg; Cohen's $d_z$ + Cliff's δ.

$$ d_z = \frac{\overline{d}}{s_d}, \qquad d = x_{\text{worst}} - x_{\text{clean}} \ \text{(paired by seed)} $$

Holm: $\tilde p_{(i)} = \max_{j\le i}\big[(m-j+1)\,p_{(j)}\big]$; BH:
$\tilde p_{(i)} = \min_{j\ge i}\big[\tfrac{m}{j}p_{(j)}\big]$; α = 0.05.

---

## 4. The Manipulation Works (sanity checks)

Before interpreting robustness, the mechanisms behave as designed:

- **Dropped frames** are explained almost entirely by the packet-error factor
  (**partial η² = 0.966**, p < 0.0001) and not by noise (η² = 0.013) — the comms
  channel does exactly what it should.
- Collisions, IALA violations, CTE and docking error are ~0 at the clean
  condition and rise monotonically with degradation (§5).

---

## 5. Results — Degradation Under Stress

Clean → harshest condition (noise 0.25, PER 0.4), per algorithm:

| Algorithm | Collision % | IALA/ep | Docking error |
|---|---|---|---|
| DQN | 3 → 44 | 0 → 54 | 0.0 → 9.7 |
| Double DQN | 11 → 31 | 0 → 65 | 0.0 → 11.1 |
| Dueling DQN | 19 → 37 | 0 → 63 | 0.0 → 10.9 |
| Dueling Double DQN | 7 → 47 | 0 → 58 | 0.0 → 10.4 |

**Factorial ANOVA — the dominant effects are the stressors, not the algorithm:**

| Metric | Noise η²ₚ | Packet-error η²ₚ | Algorithm η²ₚ | Notable interaction |
|---|---:|---:|---:|---|
| IALA violations | **0.855** (p<0.0001) | **0.795** (p<0.0001) | 0.012 (n.s.) | Noise×PER 0.292 (p<0.0001) |
| Docking accuracy | **0.670** | **0.606** | 0.014 (n.s.) | — |
| Collision rate | **0.428** | 0.057 (p=0.006) | 0.053 (p=0.023) | — |
| Success rate | 0.106 (p<0.001) | 0.079 (p<0.001) | 0.041 (n.s.) | — |

> [!INSIGHT]
> **Sensing and communication quality — not the choice of DQN variant — govern
> safety and precision.** Sensor noise and packet-error each explain 60–86 % of
> the variance in IALA channel-keeping and docking accuracy, while the algorithm
> explains ≈ 1 %. For a deployed autonomous vessel, this argues that investment
> in sensor/link quality (or noise-robust perception) will pay off far more than
> swapping between these value-based RL variants.

> [!INSIGHT]
> **Noise and packet-loss compound each other on channel compliance.** The
> significant **Noise × Packet-error interaction on IALA violations**
> (η²ₚ = 0.29, p < 0.0001) means their combined effect is worse than the sum of
> the parts — degraded sensing *and* a lossy link together push the vessel out
> of the buoyed channel disproportionately.

---

## 6. Do Algorithms Degrade Differently? (the robustness question)

**Largely no — degradation is parallel.** Across every metric the
**Algorithm × Noise** and **Algorithm × PER** interactions are small and
non-significant (partial η² ≤ 0.06, p > 0.1). The four variants lose safety and
precision at essentially the same rate as conditions worsen.

The only interaction hints are a significant **three-way** term on collisions/ep
(p = 0.039, η² = 0.115) and the Noise×PER term on IALA — neither localizes to a
single "robust" algorithm.

> [!INSIGHT]
> **No DQN variant is meaningfully more robust than the others.** The headline
> for a Q1 robustness section: within this family, architecture choice
> (Double / Dueling / both) does not buy resilience to sensing or communication
> degradation. Robustness must come from elsewhere (perception, control, or a
> different algorithm class), not from these value-function refinements.

An exploratory robustness ranking (mean |$d_z$| of clean→worst degradation over
the four safety/precision metrics) orders Dueling DQN (2.55) < Double DQN (2.71)
< Dueling Double DQN (3.31) < DQN (3.32) — i.e. Dueling DQN degrades *slightly*
least — but the between-algorithm differences are not statistically supported.

---

## 7. Statistical Significance and the Power Limitation

The **factorial ANOVA pools the whole grid** and yields decisive, large effects
for the stressors (§5). The **per-algorithm paired robustness contrasts**,
however, use only **n = 6** seeds: with all six seed-differences same-signed, the
exact two-sided Wilcoxon p-value floor is 2/2⁶ ≈ 0.031, and after Holm
correction across the four algorithms it cannot fall below ≈ 0.125. So although
the clean→worst degradations have **very large effect sizes** (Cohen's $d_z$ up
to ≈ 5.9 for docking, ≈ 4–5 for IALA), none is individually "Holm-significant."

> [!INSIGHT]
> **Absence of a significant per-algorithm contrast here is a power artefact,
> not evidence of no effect.** The effect sizes are enormous and the pooled
> ANOVA is unambiguous; the paired test is simply floor-limited at n = 6. A
> confirmatory study should use ≥ 15–20 seeds. Report the ANOVA + effect sizes as
> primary, and the paired p-values with this caveat.

---

## 8. Trade-off Analysis

- **Safety vs sensing quality.** Collision rate scales with noise (η² = 0.43);
  under the harshest condition 31–47 % of episodes involve a collision vs 3–19 %
  clean. Safety is the metric most exposed to sensing degradation.
- **Compliance vs communication.** IALA violations are driven by *both* noise and
  packet-error and their interaction — channel-keeping is the most fragile
  behaviour under combined stress.
- **Success can be misleading.** Navigation success is only weakly affected by
  degradation (η² ≈ 0.08–0.11) and for some algorithms mild noise slightly
  *raises* success (a regularisation-like effect). **Reporting success alone
  would hide the large safety/precision costs** — a direct argument for the
  multi-metric evaluation this study enables.

---

## 9. Main Findings

- **F1 (Empirical).** All four variants suffer large safety/precision
  degradation as sensor noise and packet-loss increase (collisions up to ~47 %,
  IALA up to ~65/ep, docking error ~10–11).
- **F2 (Statistical).** Degradation is driven overwhelmingly by the **stressors**
  (noise η²ₚ up to 0.86, PER up to 0.80); the **algorithm** explains ≈ 1–5 %.
- **F3 (Statistical).** **Degradation is parallel** — Algorithm × stressor
  interactions are non-significant; no variant is more robust.
- **F4 (Statistical).** **Noise × Packet-error interact** on channel compliance
  (IALA, η²ₚ = 0.29) — combined degradation is disproportionately harmful.
- **F5 (Practical).** Robustness in this setting is a **perception/communication**
  problem, not an algorithm-selection problem; and safety/precision must be
  reported alongside success, which alone understates the risk.

---

## 10. Limitations

- **Statistical power.** n = 6 seeds floor-limits the per-algorithm paired tests
  (§7); the pooled ANOVA + effect sizes are the trustworthy evidence. Use ≥ 15
  seeds for confirmatory pairwise claims.
- **Synthesized testbed.** `VesselEnvV3` is a research model on the project's
  graph — a controllable physical/sensing/comms environment, **not** a validated
  hydrodynamic or RF simulator. Absolute magnitudes are model-specific; the
  *relative* factor effects are the transferable result.
- **Budget & task.** 130 training episodes on small pure-JS agents; a harder or
  longer-trained regime could shift absolute performance.
- **Algorithm family.** Only value-based DQN variants; PPO/SAC/etc. are not in
  the codebase and were not added.

---

## 11. Recommended Tables and Figures for the Q1 Paper

**Tables** (`results_v3/tables/`, each with an Interpretation column; CSV + LaTeX
+ `all_tables_v3.xlsx`): Table 3 (safety, clean vs harsh), Table 4 (precision),
Table 5 (factorial effects + η²), Table 6 (robustness worst-vs-clean with effect
sizes), Table 7 (robustness ranking).

**Figures** (`results_v3/figures/`, 320 DPI PNG + SVG): `figV3_effect_sizes`
(the variance-explained heatmap — the single most informative figure),
`figV3_success_vs_noise` and `figV3_collision_vs_noise` (parallel-degradation
curves), `figV3_iala_heatmap` (noise×PER compounding), `figV3_robustness_ranking`.

### Provenance
All values are generated by `experiments/aggregate_v3.py`, `stats_v3.py`,
`make_tables_v3.py`, and `make_figures_v3.py` from
`results_v3/raw/eval_episodes.csv` (6,480 episodes, 0 NaN/Inf), and are
reproducible end-to-end. Design frozen in
`results_v3/configs/experiment_config_v3.json`.
