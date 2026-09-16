# Study 3 — A COLREGs-Aware Rule-Based Baseline and Confirmatory Replication

**Is a classical channel-following controller more robust to sensing and communication degradation than learned value-based policies? A 15-seed confirmatory study on VesselEnvV3.**

> **Scope.** Study 3 extends the robustness analysis of Study 2 in two ways:
> (1) it adds a non-learning **COLREGs-aware rule-based controller** as a
> classical baseline against the four value-based DQN variants, and (2) it
> raises the seed count to **15** (from Study 2's 6) so the paired post-hoc
> tests have the statistical power that was previously floor-limited. It runs
> on the same `VesselEnvV3` physical/sensing/comms testbed; every metric is
> genuinely computed. All numbers are reproducible from `results_study3/raw/`.
> The deployed web application is unaffected.

---

## 1. Motivation

Study 2 established that sensor noise and communication packet-loss sharply
degrade the safety and precision of value-based DQN agents, but two questions
remained:

1. **Is that degradation avoidable?** A learned policy might simply be a poor
   choice under degradation — a classical controller could do better. We need a
   non-RL reference point.
2. **Were the null pairwise results real?** Study 2's per-algorithm paired
   Wilcoxon tests were floor-limited at n = 6 (minimum Holm-adjusted p ≈ 0.125),
   so *no* pairwise contrast could reach significance regardless of effect size.

Study 3 answers both: it introduces a **rule-based COLREGs-aware baseline** and
uses **15 seeds** so the paired tests can cross significance.

> [!INSIGHT]
> This turns the robustness question from "how much do the DQN variants
> degrade?" (Study 2) into "**is learning even the right tool here, or does a
> classical charted-channel controller hold up better?**" — a question a Q1
> reviewer will expect a maritime-autonomy paper to answer.

---

## 2. The Rule-Based Baseline

`RuleBasedAgent` (`js/rulebased.js`) is a deterministic, non-learning controller
that mirrors the RL agent interface so the harness drives it identically. Its
policy combines maritime rules of the road:

1. **Charted-channel following** — its primary guide is the Dijkstra-planned
   route (the surveyed safe passage); it prefers the edge that advances the
   plan, staying in marked water.
2. **Goal-seeking** — a secondary preference toward the **sensed** goal bearing.
3. **COLREGs collision avoidance** — penalises edges passing near a detected
   obstacle; hard-avoids edges crossing an obstacle disc.
4. **Channel/IALA keeping** — prefers lower navigation-cost / lower-difficulty
   (safer, marked) edges.
5. **Anti-stall** — avoids immediately revisiting the previous waypoint.

> [!INSIGHT]
> **Fair comparison:** the baseline reads the *sensed* goal bearing from the
> observation vector, so it is subject to the **same sensor noise and stale-comms
> degradation** as the DQN agents. It is not given privileged ground-truth — the
> only asymmetry is that it uses the chart (as any classical planner would),
> which is exactly the point of the comparison.

---

## 3. Design

| Factor | Levels |
|---|---|
| Agent | **Rule-based (COLREGs)**, DQN, Double DQN, Dueling DQN, Dueling Double DQN |
| Condition | clean (0, 0), mid (0.1, 0.2), harsh (0.25, 0.4) — (noise std, packet-error) |

= 15 cells × **15 shared seeds** = **225 runs**, **6,750 evaluation episodes**.
Per-seed scalars (n = 15) are the unit of analysis (paired/randomized-block).
RL agents train 130 episodes; the rule-based controller does not train.

---

## 4. Statistical Methodology

Same rigor as Studies 1–2: Shapiro–Wilk normality and Levene/Brown–Forsythe
homogeneity checks; a **two-way factorial ANOVA** `y ~ agent * condition + seed`
(Type-II SS, partial η²); and the **primary contrast — rule-based vs each DQN
variant, per condition** — via paired Wilcoxon (exact) + paired t, **Holm and
Benjamini–Hochberg** corrected within each (metric, condition) family, with
Cohen's $d_z$ and Cliff's δ effect sizes. Robustness is the clean→harsh paired
change per agent. α = 0.05.

$$ \eta^2_p=\frac{\mathrm{SS}_{\text{effect}}}{\mathrm{SS}_{\text{effect}}+\mathrm{SS}_{\text{error}}}, \qquad d_z=\frac{\overline{d}}{s_d} $$

---

## 5. Results

### 5.1 Navigation success

| Agent | clean | mid | harsh |
|---|---:|---:|---:|
| **Rule-based** | 47.3% | 47.3% | 47.3% |
| DQN | 20.0% | 19.1% | 32.0% |
| Double DQN | 19.3% | 22.7% | 23.8% |
| Dueling DQN | 32.2% | 31.8% | 32.7% |
| Dueling Double DQN | 17.8% | 33.8% | 29.6% |

The agent main effect on success is significant (partial η² = 0.26, p < 0.0001).

> [!INSIGHT]
> **The rule-based baseline's success is invariant to degradation (47.3% at
> every condition)** and is the highest of all agents. Because it follows the
> chart, sensor noise and packet-loss do not change *whether* it reaches the
> goal — only *how cleanly* it does so (§5.2). The DQN agents' success is lower
> and condition-dependent.

### 5.2 Safety — IALA channel compliance (the headline result)

IALA violations per episode (channel departures):

| Agent | clean | mid | harsh |
|---|---:|---:|---:|
| **Rule-based** | 0.0 | 7.4 | **24.3** |
| DQN | 0.0 | 17.6 | 57.7 |
| Double DQN | 0.0 | 17.9 | 63.2 |
| Dueling DQN | 0.0 | 16.8 | 59.4 |
| Dueling Double DQN | 0.0 | 15.9 | 57.9 |

The factorial ANOVA shows a **significant agent × condition interaction on IALA**
(partial η² = 0.491, p < 0.0001) on top of the huge condition (η² = 0.93) and
agent (η² = 0.49) main effects. The pairwise contrasts confirm it: **at the
harshest condition, the rule-based baseline commits ~2× fewer channel violations
than every DQN variant, and all four contrasts are Holm-significant**
(p < 0.001; Cohen's $d_z$ from −1.9 to −2.4, all *large*).

> [!INSIGHT]
> **Headline finding.** The classical COLREGs-aware controller keeps the buoyed
> channel far better than the learned policies as sensing/comms degrade. This is
> the opposite of "parallel degradation": the significant interaction means the
> DQN variants' channel-keeping collapses much faster than the baseline's. For a
> maritime-autonomy paper, this is a concrete, significant argument that a
> chart-following classical layer confers a robustness advantage the learned
> value functions do not.

### 5.3 Safety — collisions

Collision rate is dominated by the condition (η² = 0.64, p < 0.0001); the agent
main effect is not significant (η² = 0.04). In means the rule-based baseline has
fewer harsh-condition collisions per episode (0.53 vs 1.0–1.4 for the DQN
variants), but per-seed variance is high and the individual contrasts are not
Holm-significant. **We report this honestly:** the baseline's collision
advantage is suggestive but not statistically established here.

### 5.4 Precision — CTE and docking

Cross-track error shows a significant agent effect (η² = 0.08, p = 0.002): the
DQN agents track the *optimal* route more tightly when they do reach the goal,
whereas the rule-based controller trades some path-tightness for channel-keeping.
Docking accuracy is governed almost entirely by the condition (η² = 0.86) with no
agent difference — degraded localisation blurs the final approach for everyone.

### 5.5 The power problem is solved

Holm-significant rule-based-vs-DQN contrasts by condition: **clean 6/32, mid
19/32, harsh 21/32** (46/96 overall). With n = 15 the paired tests now resolve
real differences — directly addressing Study 2's n = 6 floor limitation.

---

## 6. Main Findings

- **F1 (Empirical).** The rule-based baseline reaches the goal as often or more
  often than the DQN variants (47% vs 18–33%), and its success is invariant to
  degradation.
- **F2 (Statistical).** **Rule-based commits ~2× fewer IALA channel violations
  than every DQN variant under stress, all Holm-significant (large effect).** A
  significant agent × condition interaction (η² = 0.49) shows the DQN agents'
  channel-keeping degrades far faster.
- **F3 (Statistical).** With 15 seeds, 46/96 baseline-vs-DQN contrasts survive
  Holm correction — the confirmatory power that Study 2 lacked.
- **F4 (Practical).** A classical charted-channel controller is a strong,
  robust baseline for this task; the learned value-based policies do not beat it
  on safety-critical channel compliance under sensing/comms degradation.

---

## 7. Limitations

- **Synthesized testbed.** `VesselEnvV3` is a research model, not a validated
  hydrodynamic/RF simulator; absolute magnitudes are model-specific, the
  *relative* baseline-vs-DQN pattern is the transferable result.
- **Baseline design.** The rule-based controller is one reasonable
  COLREGs-inspired heuristic; a differently-tuned controller could shift the
  numbers. Its advantage comes largely from chart-following, which presumes an
  accurate chart.
- **DQN budget.** 130 training episodes on small pure-JS agents; longer training
  or larger networks could narrow the gap.
- **Collision significance.** The baseline's fewer-collisions trend is not
  statistically established (high per-seed variance).
- **Not a COLREGs certification.** "COLREGs-aware" denotes rule-of-the-road
  *inspired* behaviour (channel-keeping, give-way avoidance), not formal
  compliance with the full regulations.

---

## 8. Recommended Tables and Figures for the Q1 Paper

**Tables** (`results_study3/tables/`): Table 2 (navigation), Table 3 (safety
clean vs harsh), Table 4 (factorial effects), **Table 5 (rule-based vs DQN at
harsh, with significance + effect sizes — the key contrast)**, Table 6
(robustness).

**Figures** (`results_study3/figures/`): **`figS3_iala_interaction`** (the
fan-out interaction plot — the single clearest figure), `figS3_iala_harsh`
(bar comparison at harsh), `figS3_effect_matrix` (per-condition effect-size grid
with significance marks), `figS3_success_vs_condition`.

### Provenance
Generated by `experiments/aggregate_study3.py`, `stats_study3.py`,
`make_tables_study3.py`, `make_figures_study3.py` from
`results_study3/raw/eval_episodes.csv` (6,750 episodes, 0 NaN/Inf); design frozen
in `results_study3/configs/experiment_config_study3.json`.
