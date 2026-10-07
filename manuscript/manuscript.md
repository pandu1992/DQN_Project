# What Makes Autonomous Vessel Navigation Robust? A Controlled Factor-Hierarchy Study of Deep Reinforcement Learning under Degraded Sensing, Communication, and Multi-Vessel Interaction

**A reproducible experimental investigation on a Synthetic Port channel-navigation testbed**

> **Reproducibility statement.** Every quantitative claim in this manuscript is
> computed from committed experiment data (`results/`, `results_v3/`,
> `results_study3/`, `results_study4/`, `results_study5/`, `results_chart/`,
> `results_maps/`) by the scripts in `experiments/`, and is regenerable
> end-to-end. No metric is hand-set or fabricated. All environments reuse the
> same synthesized port graph, edge attributes, and Dijkstra planner; the
> deployed browser simulation is unchanged by any of this analysis.

> **Scope statement.** This is a *controlled experimental* study on a synthesized
> channel-navigation testbed, not a validated hydrodynamic/radio-frequency
> simulator or a deployment study. Our claims concern the **relative** influence
> of design factors under explicitly manipulated conditions; absolute magnitudes
> are model-specific. We report null and negative results, power limitations, and
> external-validity caveats as faithfully as positive findings.

---

## Abstract

Deep reinforcement learning (DRL) is widely proposed for autonomous maritime
navigation, and a large literature reports learned policies that navigate and
avoid collisions. Far less is established about **where robust, complete,
safe navigation actually comes from**: how the influence of algorithmic choice
compares with that of perception/communication quality, control architecture,
multi-vessel interaction, and mission structure — under a common testbed, a fair
baseline, and disciplined statistics. We address this with a programme of linked,
controlled experiments on a reproducible Synthetic Port channel-navigation
testbed, treating the **random seed as the unit of inference** and reporting
effect sizes with multiple-comparison correction and a complementary
mixed-effects (random-seed) model.

A **diagnostic** experiment shows the four value-based DRL variants (DQN, Double,
Dueling, Dueling-Double, with Prioritized Replay and Noisy Nets) are
**statistically indistinguishable** on task success; run-to-run **seed** variance
dominates (partial $\eta^2\approx0.20$). Under graded **sensor noise and
communication packet-loss**, safety and precision are governed **overwhelmingly
by the stressors** (partial $\eta^2$ up to 0.80–0.86 on channel compliance) and
only $\approx$1–5% by the algorithm. A **COLREGs-aware rule-based baseline** keeps
the buoyed channel markedly better than every DRL variant under degradation — but
a controlled **chart-fairness** experiment shows this advantage is **largely an
information-access effect**: giving the learner the same Dijkstra-plan prior the
baseline uses **closes the success gap to non-significance** (arm partial
$\eta^2=0.30$). With **two independent vessels**, the **control pairing** dominates
inter-vessel safety (partial $\eta^2$ up to 0.70 for closest-approach): learned
vessels keep roughly twice the separation of two classical controllers but
complete far fewer missions — a safety-versus-completion tradeoff. Finally,
**mission structure** matters most of all: a full inbound→dock→outbound **round
trip roughly halves success** relative to a one-way transit. Replication across
**three distinct map geometries** shows the geometry *per se* is not the driver
(map main effect $\eta^2=0.018$, n.s.), though the baseline gap is geometry-
dependent.

Taken together, the influence on safe, complete autonomous vessel navigation is
**ordered**: *mission structure and multi-vessel interaction $\gtrsim$ control
architecture / navigation prior $\gtrsim$ perception & communication quality
$\gg$ value-based algorithmic refinement*. We argue that robustness in this domain
is a representation/architecture and evaluation-design problem rather than a
value-function-refinement problem, and that safety-relevant, end-to-end,
multi-vessel metrics must be reported alongside single-vessel task success.

**Keywords:** deep reinforcement learning, autonomous surface vessels, robustness,
sensor noise, communication reliability, COLREGs, navigation priors, multi-vessel
collision avoidance, mission completion, reproducibility, mixed-effects.

---

## 1. Introduction

Autonomous surface-vessel navigation is a safety-critical control problem: an
agent must reach a berth or waypoint while remaining inside marked channels,
avoiding static hazards and other vessels, and respecting the international rules
of the road (COLREGs). Deep reinforcement learning (DRL) has become a popular tool
for such sequential decision problems, and a mature literature reports learned
policies for USV/MASS navigation and COLREGs-aware collision avoidance
([§2](#related)). That literature establishes that DRL *can* navigate and avoid
collisions; it is far less clear **which factors actually determine whether such
navigation is robust, safe, and complete** once the idealized assumptions of a
single vessel, perfect sensing, perfect communication, and a one-way transit are
relaxed.

This manuscript asks a single, overarching question:

> **Under controlled degradation and increasing operational complexity, which
> factors determine the safety, reliability, and completion of DRL-based
> autonomous vessel navigation — and how do they rank?**

We decompose it into four research questions, each addressed by a controlled
experiment on one shared, reproducible testbed:

- **RQ1 (algorithm — diagnostic).** Does algorithmic refinement within the
  value-based DRL family materially affect performance once seed variability is
  controlled?
- **RQ2 (perception/communication).** How do sensor noise and communication
  packet-loss affect safety and precision relative to algorithmic variation?
- **RQ3 (control architecture / navigation prior).** Is a classical baseline's
  robustness advantage inherent to classical control, or an artifact of the
  chart/plan information it is given — i.e. does supplying the learner the same
  navigation prior close the gap?
- **RQ4 (interaction & mission structure).** How do multi-vessel encounters and
  round-trip / two-way-traffic missions change the conclusions one would draw
  from single-vessel, one-way benchmarks?

**Contributions.** We contribute **methodology and controlled evidence**, not a
new algorithm: (a) a reproducible, deliberately *discriminative* channel-
navigation testbed with genuinely computed safety/precision metrics; (b) a
factorial robustness protocol over jointly manipulated sensing and communication
degradation; (c) a **fair-baseline** design that makes chart/navigation-prior
access an *explicit experimental factor*, disentangling classical control from its
information advantage; (d) a **multi-vessel** extension (two independent agents)
and (e) **operational-realism** scenarios (round trip, two-way traffic); and
(f) a **seed-level, effect-size-first, mixed-effects** statistical treatment that
reports null, power-limited, and negative results faithfully. The synthesis is a
**factor hierarchy** that reframes the design question from "which DRL variant
wins" to "what level of the design stack confers robust, complete, multi-vessel
behaviour".

> [!INSIGHT]
> **Headline.** On this testbed the influence on safe, complete navigation is
> ordered — *mission structure & multi-vessel interaction $\gtrsim$ control
> architecture / navigation prior $\gtrsim$ perception & communication quality
> $\gg$ value-based algorithmic refinement.* The ordering, not any single
> algorithm result, is the contribution.

![Overview of the experiments](assets/schematics/study_comparison.png)
**Figure 1 (Overview).** The experimental programme on one testbed, progressively
richer: a clean **algorithm** comparison (diagnostic); **sensing/communication**
degradation with real safety metrics; a **COLREGs rule-based baseline** and the
**chart-fairness** control; **two independent vessels** that must avoid each
other; and **round-trip + two-way-traffic** missions — all on the same synthesized
chart and statistical protocol.

---

<a name="related"></a>

## 2. Related Work and Positioning

**Maritime DRL for navigation and COLREGs collision avoidance** is a mature, active
area spanning value-based, actor–critic, and distributional RL, including
COLREG-compliant USV avoidance ([Meyer et al. 2020](https://arxiv.org/html/2006.09540v1); [risk-based COLREGs 2021](https://arxiv.org/html/2112.00115v1)), multi-vessel decision-making ([Zhang et al. 2024](https://www.mdpi.com/2077-1312/12/3/372/xml); [USV-fleet planning 2023](https://www.mdpi.com/2077-1312/11/12/2334)), decentralized multi-ASV via distributional RL ([2024](https://arxiv.org/pdf/2402.11799v2)), and safety-layer / dual-mode designs ([2026](https://www.nature.com/articles/s41598-026-62506-2)). These typically report *nominal* performance.

**Robustness to imperfect perception** is emerging but usually studied in
isolation: benchmarks under sensor noise/denial ([2024](https://arxiv.org/html/2410.14616)), robustness evaluations for autonomous shipping ([2024](https://arxiv.org/html/2411.04915v1)), and distributionally-robust mitigation of observational noise ([2025](https://arxiv.org/html/2512.00030)) — generally one stressor at a time, rarely with channel-compliance/docking safety metrics.

**Communication failure** is almost absent from maritime RL; where modelled, it
appears in connected-vehicle / platoon MARL (packet loss, delay) ([DCT-MARL 2025](https://arxiv.org/abs/2508.12633v1); [MTCC](https://arxiv.org/pdf/2311.11281.pdf)), and a 2026 stress-test notes cooperative embodied-AI is "almost universally evaluated under idealized communication" ([2026](https://arxiv.org/html/2603.20285v1)).

**Classical priors improve learned navigation.** Outside maritime, guiding RL with
classical planners/heuristics improves sample efficiency, generalization and
safety ([regularized RL + classical planning 2024](https://arxiv.org/html/2403.18524v1); [heuristics as dense rewards 2021](https://arxiv.org/html/2109.14830v2); [hybrid classical/RL planner 2024](https://arxiv.org/html/2410.03066v1)) — motivating the *navigation prior* as a first-class factor (our RQ3).

**RL methodology** work demands what most applied maritime papers omit: multiple
seeds, significance tests, and effect sizes, with the seed treated as a unit of
variance ([Henderson et al. 2018](https://arxiv.org/html/1709.06560); [Colas et al. 2018](https://arxiv.org/html/1806.08295v2); [Agarwal/"Hitchhiker's Guide" 2019](https://arxiv.org/html/1904.06979v2); [Patterson et al. 2023](https://arxiv.org/html/2304.01315v1)). A 2026 maritime review independently flags "inconsistent benchmarks, limited cross-scenario generalization, and insufficient full-scale validation" as persistent barriers ([2026 review](https://www.mdpi.com/2077-1312/14/16/1477/xml)).

**Gap.** Prior work establishes that DRL *can* navigate and *can* be brittle to
isolated perturbations; it has not systematically quantified the **relative**
influence of algorithm, perception/communication degradation, control architecture
/ navigation prior, multi-vessel interaction, and mission structure on safe,
complete navigation — on a common testbed, with a baseline whose **information** is
controlled, and with seed-level statistics. A full capability-comparison table
appears in [Appendix A.0](#appendix-lit). (*Related-work content was rephrased for
compliance with source licensing; see cited originals.*)

---

## 3. Methods

### 3.1 Testbed and environments

All experiments run on a synthesized port channel graph (a constructed testbed,
not a real-world nautical chart): two access channels (North L02, South L08)
feeding a shared harbour lane (L01), with buoys, virtual shortcut branches, and
per-edge attributes (distance, risk, traffic, weather, current, travel time,
energy cost, navigation cost, difficulty). A Dijkstra planner over navigation cost
provides the optimal charted route. We reuse the *same graph, attributes, and
planner* across all environments so the experiments are commensurable:

- **`VesselEnvV2`** (RQ1): a genuine graph-navigation MDP — the agent chooses among
  the current node's real outgoing edges. Varied missions (115 start–goal pairs
  per seed). Observation dim 25, four action slots, 60-step budget.
- **`VesselEnvV3`** (RQ2–RQ3): adds a *physical/sensing/comms* layer — continuous
  kinematics with a control-drift model, Gaussian observation noise, a packet-error
  channel (dropped frames held stale), a seeded obstacle field with collision
  detection, docking accuracy, cross-track error (CTE), and IALA channel-departure
  detection. Observation dim 28.
- **`VesselEnvV3Chart`** (RQ3, chart-fairness): `VesselEnvV3` with an optional
  per-slot **`onPlan`** feature (1 if an outgoing edge is the next Dijkstra-plan
  segment), giving a learner the *same* charted-route prior the rule-based
  controller uses (obs 28→32); byte-identical to `VesselEnvV3` when disabled.
- **`VesselEnvV3ChartMap`** + `mapgen.js` (external validity): the above over a
  **family of map geometries** (map 0 = the canonical anchor; maps 1–4 distinct
  layouts), varying lane lengths, curvature, separation, berth position and
  shortcuts.
- **`MultiVesselEnvV4`** (RQ4): composes *two* `VesselEnvV3` vessels over the same
  world; synchronized continuous sub-steps measure inter-vessel **collision**
  (hull separation < 18 units), **near-miss** (< 40), and **closest-point-of-
  approach (CPA)**; distinct-event counting avoids saturation from locked
  single-file encounters. Each vessel senses the partner (obs 28→31) through the
  same noise/comms pipeline; a COLREGs give-way role is computed from real bearing
  for attribution.
- **`TwoPhaseEnvV5` / `TwoWayEnvV5`** (RQ4): `TwoPhaseEnvV5` requires a full port
  call (inbound→dock→outbound) in one episode (success = the full cycle);
  `TwoWayEnvV5` places an inbound and an outbound vessel on one lane with
  overlapping paths, forcing head-on encounters.

> [!INSIGHT]
> **The benchmark had to be made discriminative before it could be studied.**
> The originally deployed environment is *saturated* — every agent trivially
> reaches ~100% success — making comparison vacuous. Reconstructing it as a
> genuine routing MDP (V2) and adding a real physical/sensing/comms layer (V3) is
> the pivotal design step; everything else is built on the *same* chart.

### 3.2 Agents

Four value-based DRL agents share an MLP torso ($\text{obs}\to128\to128$, ReLU), a
target network with periodic hard updates, Huber loss, and a hand-written Adam
optimizer. **DQN** uses the bootstrap target $y_t=r_t+\gamma(1-d_t)\max_{a'}
Q_{\theta^-}(s_{t+1},a')$. **Double DQN** decouples selection/evaluation; **Dueling
DQN** uses value+advantage streams with mean-advantage subtraction; **Dueling
Double DQN** combines both. **Prioritized Experience Replay** and **Noisy Nets**
are optional enhancements. A fifth, non-learning **COLREGs-aware rule-based
controller** follows the Dijkstra-charted channel, prefers edges reducing distance
to the *sensed* goal, applies give-way obstacle avoidance, and keeps to lower-cost
marked water. It reads the sensed goal from the same noise/dropout-affected
observation the DRL agents receive; **its one asymmetry is explicit use of the
chart — which RQ3 turns into a controlled factor rather than a confound.**

### 3.3 Evaluation metrics and episode outcomes

The unit of analysis is the **per-seed scalar**. Metrics: navigation **success
rate** and **mean reward**; safety **collision rate / collisions-per-episode** and
**IALA violations/episode**; precision **docking accuracy** and **cross-track
error (CTE)**; routing **optimality ratio**; and a communication manipulation
check, **dropped frames**.

Three measurement properties are stated explicitly because they shaped how we
report (established by an internal validity audit; see
[Appendix A.0](#appendix-audit)):

1. **Success is goal-reaching only.** Episodes terminate on reaching the goal or on
   timeout; collisions do **not** terminate an episode and there is no off-channel
   termination. Hence *timeout is the sole failure mode*, and collisions/IALA are
   **within-episode safety metrics orthogonal to the success outcome**. Under harsh
   degradation ~35–47% of *successful* episodes nonetheless incur ≥1 collision —
   success does **not** imply a safe trajectory, which is precisely why we report
   safety separately.
2. **CTE is reported by outcome, never as a single aggregate.** CTE drift scales
   with degradation, so a *successful* clean episode has CTE$\approx$0 by
   construction; an all-episode CTE mean is a success-rate-weighted mixture of a
   near-zero (successful) and a large (failed) sub-population and is **non-monotone
   for the wrong reason**. We therefore report **CTE conditioned on success** (the
   true tracking-precision signal, which rises monotonically with degradation),
   alongside the failed-episode value and the success rate that sets the mixture.
3. **Precision metrics are solved-only.** Docking accuracy and optimality ratio are
   defined only for successful episodes; we always report them next to the success
   rate (and $n_\text{solved}$) and avoid cross-condition claims that ignore the
   differing success bases.

### 3.4 Statistical protocol

We report means with 95% CIs (Student-$t$, cross-checked by 10,000-sample
bootstrap). We test normality (Shapiro–Wilk) and variance homogeneity (Levene),
run **factorial ANOVA** with Type-II sums of squares and partial $\eta^2$,

$$ \eta^2_p=\frac{\mathrm{SS}_{\text{effect}}}{\mathrm{SS}_{\text{effect}}+\mathrm{SS}_{\text{error}}}, $$

and perform paired post-hoc tests (Wilcoxon signed-rank, exact; paired $t$
secondary) with **Holm** and **Benjamini–Hochberg** correction, Cohen's $d_z$ and
Cliff's $\delta$. The shared-seed design makes contrasts within-block (paired).
We **complement** the ANOVA with a linear **mixed-effects** model that treats
**seed as a random effect**, $y \sim \text{(fixed factors)} + (1\,|\,\text{seed})$,
and report the seed/residual variance components and the intraclass correlation
(ICC) — the share of variance absorbed by run-to-run seed variability
([Appendix A.0](#appendix-mixed)). $\alpha=0.05$ throughout.

> **On sample size and power.** Across the programme we evaluate **31,150
> episodes/rows**. This is a measure of **computational replication and
> reproducibility, not statistical power**: the unit of inference is the per-seed
> scalar, so the effective $n$ is the **seed count** (6–15 depending on the study).
> We therefore treat the 6-seed robustness study as providing **limited evidence**
> (large effect sizes, power-limited significance) and the 15-seed study as the
> **confirmatory** analysis for seed-sensitive claims.

> **On evaluation.** Agents are evaluated greedily on *fresh draws from the same
> mission distribution they trained on* — i.e. within-distribution generalization.
> This is **not a held-out test split**: the mission space is small (~120 start–goal
> pairs) and train/eval missions overlap. We state this plainly and treat the
> generalization claim accordingly.

![Shared experimental workflow](assets/schematics/workflow.png)
**Figure 2 (Workflow).** The shared pipeline — seeded environment → train (RL) or
run (rule-based) → greedy within-distribution evaluation → per-episode real metrics
→ per-seed aggregation → factorial + mixed-effects statistics → tables and figures.
Only the environment build, factors, seeds and metrics differ across experiments.

---

<a name="rq1"></a>

## 4. RQ1 (Diagnostic) — Algorithmic Refinement Is a Weak Lever

**Design.** Four algorithms × PER{off,on} × Noisy{off,on} = 16 configurations,
10 shared seeds, 150/40 train/eval episodes (6,400 evaluation episodes) on
`VesselEnvV2`.

**Results.** Base success rates were DQN 49.8±18.1%, Double 43.2±19.3%, Dueling
40.2±11.8%, Dueling-Double 45.5±17.9% (mean ± 95% CI) — heavily overlapping. The
factorial ANOVA found **no significant algorithm effect** on success (partial
$\eta^2=0.008$, $p=0.78$); instead the **random seed dominated** (partial
$\eta^2=0.200$, $p=0.0003$), exceeding every design factor. **No pairwise algorithm
difference survived** Holm correction. The only design signal was a small Noisy-Nets
effect on route optimality ($\eta^2=0.061$, $p=0.005$). Full results:
[Appendix A.1](#appendix-a1).

> [!INSIGHT]
> **The four DRL variants are effectively interchangeable for reaching the goal;
> run-to-run variance swamps algorithm choice.** We therefore treat the algorithm
> comparison as a *diagnostic* that sets the floor of the factor hierarchy, not as
> a contribution in itself. The practical question becomes "how many seeds before a
> winner can be claimed" — here, more than 10.

<a name="fig-s1-var"></a>
![RQ1 — variance dominance](results/figures/fig8_variance_dominance.png)
**Figure 3.** RQ1: partial $\eta^2$ from the factorial ANOVA. The Seed block
explains the most variance in success and reward ($\eta^2\approx0.20$), exceeding
all design factors.

---

<a name="rq2"></a>

## 5. RQ2 (Perception/Communication) — Degradation Dominates Safety

**Design.** Four algorithms × sensor-noise{0,0.1,0.25} × packet-error{0,0.2,0.4}
= 36 conditions, 6 shared seeds, 130/30 episodes (6,480 evaluation episodes) on
`VesselEnvV3`. Manipulation check: dropped frames are explained almost entirely by
the packet-error factor (partial $\eta^2=0.966$) and not noise ($\eta^2=0.013$).

**Results.** Degradation sharply worsened safety and precision, driven by the
*stressors*, not the algorithm:

| Metric | Noise $\eta^2_p$ | Packet-error $\eta^2_p$ | Algorithm $\eta^2_p$ |
|---|---:|---:|---:|
| IALA violations | **0.855** | **0.795** | 0.012 (n.s.) |
| Docking accuracy | **0.670** | **0.606** | 0.014 (n.s.) |
| Collision rate | 0.428 | 0.057 | 0.053 |
| Success rate | 0.106 | 0.079 | 0.041 (n.s.) |

Noise and packet-loss each explained **60–86%** of the variance in channel
compliance and docking accuracy; the algorithm $\approx$1–5%. Noise and
packet-error **interacted** on compliance (IALA Noise×PER $\eta^2=0.292$). The
**algorithm×stressor interactions were small and non-significant** ($\eta^2\le0.06$):
the four variants degrade *in parallel*. With 6 seeds the paired contrasts had
large effect sizes but were power-limited — motivating the confirmatory study
([§6](#rq3)). The mixed-effects model confirms the picture: noise and packet-error
are highly significant fixed effects (both $p<0.001$) and the seed random effect is
negligible once the stressors are in the model (ICC $\approx 0$). Full effects:
[Appendix A.2](#appendix-a2).

> [!INSIGHT]
> **Sensing and communication quality — not the DRL variant — govern safety and
> precision.** Investment in sensor/link quality (or noise-robust perception) pays
> off far more than swapping between these value-based refinements. Reporting
> practice matters here: task success was nearly insensitive to degradation even as
> collisions and channel violations climbed — a success-only paper would have
> concluded the agents were robust.

<a name="fig-s2-eff"></a>
![RQ2 — effect sizes](results_v3/figures/figV3_effect_sizes.png)
**Figure 4.** RQ2: partial $\eta^2$ per factor and metric. Noise and packet-error
dominate the safety/precision metrics; the algorithm and its interactions are small.

<a name="fig-cte"></a>
![CTE by outcome](results_audit/figures/figA_cte_outcome_correction.png)
**Figure 5.** Cross-track error must be read by outcome (the measurement
correction of [§3.3](#methods)). *Left:* the all-episode CTE mean is non-monotone
across degradation — a mixture artifact (clean successful episodes have CTE$\approx$0
by construction). *Right:* CTE **conditioned on success** rises monotonically with
degradation for every controller — the true tracking-precision loss.

---

<a name="rq3"></a>

## 6. RQ3 (Control Architecture / Navigation Prior)

### 6.1 A classical baseline is more robust — at higher power

**Design.** Five agents (rule-based + four DRL) × three conditions (clean, mid,
harsh) × **15 shared seeds** (6,750 evaluation episodes) on `VesselEnvV3`.

**Results.** Navigation success showed a significant agent effect (partial
$\eta^2=0.261$): the **rule-based baseline's success was invariant to degradation
(47.3% at every condition)** and highest overall. On channel compliance, IALA
violations/episode at harsh were **24.3 (rule-based) vs 57.7–63.2 (DRL)**, with a
significant **agent×condition interaction** (partial $\eta^2=0.491$); every
rule-based-vs-DRL IALA contrast at harsh was Holm-significant ($d_z$ from $-1.9$ to
$-2.4$, all *large*). Reading CTE by outcome ([§3.3](#methods)), CTE-on-success
rises monotonically for all agents (rule-based 0.0→5.9→13.2; DQN 0.4→7.3→17.8,
clean/mid/harsh). We note honestly that the baseline's *fewer-collisions* trend is
**not** statistically established at this seed count. Full contrasts:
[Appendix A.3](#appendix-a3).

### 6.2 But the advantage is largely chart access, not classical control

The baseline's one asymmetry is that it reads the Dijkstra plan; the DRL
observation carries only a *greedy geometric* "toward-goal" edge flag, not the
planner's "advances the charted route" flag. We make this an **explicit factor** on
`VesselEnvV3Chart`: **DRL-no-chart** (28-dim), **DRL-with-chart** (32-dim, +4
`onPlan` flags — the same prior the baseline uses), and **Rule-based**. Three arms
× three conditions × 8 shared seeds (2,160 evaluation episodes).

**Results.** Success rate (mean over 8 seeds):

| Arm | clean | mid | harsh |
|---|---|---|---|
| DRL (no chart) | 18.3% | 17.1% | 35.0% |
| **DRL (+chart prior)** | **35.8%** | **42.9%** | **43.7%** |
| Rule-based (has chart) | 47.1% | 47.1% | 47.1% |

The **arm** factor dominates success (partial $\eta^2=0.30$, $p=0.0001$; reward
0.38; optimality 0.30). The *confounded* DRL-no-chart-vs-rule-based gap is large
and Holm-significant (clean $d_z=-1.57$, mid $-1.13$) — this is the "classical
wins" result. But **DRL-with-chart-vs-rule-based is non-significant in every
condition** ($d_z$ from $-0.10$ to $-0.53$): once the learner is given the same
navigation prior, the gap closes. Adding the prior to the *same* learner yields
medium–large gains ($d_z$ up to $-0.99$; at $n=8$ the within-DRL Holm-$p\approx0.05$
— a power limitation we report). The mixed-effects model confirms DRL-with-chart as
a significant fixed effect ($p<0.001$).

> [!INSIGHT]
> **The rule-based advantage is substantially an information-access effect, not an
> inherent property of classical control.** The operative lever is whether the
> policy has a structured **navigation prior** — a representation/architecture
> question — which argues directly for **hybrid** designs that feed a classical
> planner's route into a learned controller, and strengthens the factor hierarchy.

<a name="fig-chart"></a>
![RQ3 — chart-fairness](results_chart/figures/figChart_success_gap_closes.png)
**Figure 6.** RQ3 chart-fairness (n=8, 95% CI). The DRL agent denied the charted
prior (red) sits below the rule-based baseline (black), reproducing the "classical
is more robust" result; giving the same agent the chart prior (blue) closes the gap
to non-significance. The lever is the navigation prior, not learning-vs-classical.

### 6.3 External validity across map geometries

We replicated the three arms across **three distinct maps** (canonical, compact-
curved, asymmetric), clean+harsh, 6 seeds (3,240 evaluation episodes), with a
three-way factorial (arm × condition × map).

**Results.** The **map main effect is small and non-significant** (partial
$\eta^2=0.018$, $p=0.46$) — the headline results are *not* an artifact of one
geometry. The **chart-prior direction is preserved on every map** (DRL-with-chart
$\ge$ DRL-no-chart success on all three). However, the *magnitude* relative to the
baseline is **geometry-dependent** (significant arm×map interaction, $\eta^2=0.123$,
$p=0.024$): on the canonical and compact maps the prior closes the gap, whereas on
the asymmetric map the rule-based controller remains clearly stronger. We report
this rather than average it away.

<a name="fig-maps"></a>
![RQ3 — external validity](results_maps/figures/figMaps_success_across_maps.png)
**Figure 7.** RQ3 external validity: success per arm across three geometries
(clean/harsh). The arm ordering and the chart-prior effect are preserved across
maps; the map factor adds variance without overturning the ordering.

---

<a name="rq4"></a>

## 7. RQ4 (Interaction & Mission Structure)

### 7.1 Two independent vessels — the pairing dominates safety

**Design.** Two **independent** vessels (each its own policy and mission, *no*
central controller) share the channel on `MultiVesselEnvV4`; the safety-critical
event is an *inter-vessel* collision. Four pairings (`DQN×DQN`,
`DuelingDouble×DuelingDouble`, `DQN×Rule`, `Rule×Rule`) × three conditions × 8
seeds (2,880 encounters / 5,760 per-vessel rows).

**Results — a safety-versus-completion tradeoff.** Two classical vessels complete
the most missions (~48%, invariant) but collide most (38–41% of encounters) and
pass closest (CPA ≈ 118) — the baseline models only *static* hazards. The learned
pairings keep vessels much farther apart (CPA ≈ 190–233) and collide less (14–30%),
at a large cost to completion (14–27%). The **pairing dominates** every
inter-vessel outcome (closest-approach partial $\eta^2=0.70$, collision rate 0.52,
success 0.69; all $p<0.001$); degradation contributes a smaller real effect on
collisions ($\eta^2\approx0.11$–0.14). **49/90** reference contrasts are
Holm-significant; at harsh, the learned pairings' larger CPA ($d_z\approx-1.5$ to
$-1.7$) and lower success ($d_z\approx2.0$) are both large and significant. The
mixed-effects model shows a substantial seed random effect here (ICC $\approx0.38$).

> [!INSIGHT]
> **Two learned vessels genuinely avoid each other — roughly doubling the closest
> approach — but pay for it in completion.** The classical controller completes
> reliably yet is blind to the moving partner. The *control architecture*, not the
> degradation level, governs multi-vessel safety.

<a name="fig-s4"></a>
![RQ4 — tradeoff](results_study4/figures/figS4_tradeoff_scatter.png)
**Figure 8.** RQ4 two-vessel safety–completion tradeoff. The classical Rule×Rule
pairing (upper-right) completes most but collides most; learned/mixed pairings sit
lower-left — fewer collisions, lower completion.

### 7.2 Mission structure — the strongest factor

**Design.** `TwoPhaseEnvV5`: a single vessel must complete a full port call
(inbound→dock→outbound; success = full cycle), rule-based and DQN, 48 cells.
`TwoWayEnvV5`: inbound and outbound vessels share a lane (head-on), three pairings,
72 cells. 8 seeds; 5,760 rows.

**Results.** The round trip succeeds only about **half** as often as docking alone
(rule-based dock 43.3% → full-cycle **21.7%**, invariant, $d_z\approx1.25$; the DQN
docks occasionally but rarely completes the cycle, ≤5.4%). In two-way traffic the
**pairing again dominates** (success $\eta^2=0.70$, collision rate 0.55,
closest-approach 0.56; significant pairing×condition interaction on head-on events,
$\eta^2=0.20$): two rule-based vessels transit and meet head-on (42.5% collide, CPA
≈ 92, success 45.2%), while the DQN pairings barely transit — their low collision
counts are a **stall artefact**, not learned avoidance.

> [!INSIGHT]
> **A full port call is materially harder than a one-way transit, and classical
> control is what actually carries two-way traffic here.** The value-based DQN does
> not master the harbour transit, the round trip, or two-way traffic on this
> testbed; we verified this is a genuine property of the harbour-goal mission (a
> plain `VesselEnvV3` DQN also fails it), not a wiring bug — an honest negative
> result consistent with RQ1 and RQ3.

<a name="fig-s5"></a>
![RQ4 — two-phase](results_study5/figures/figS5_twophase_dock_vs_full.png)
**Figure 9.** RQ4 mission structure: reaching the dock (light) succeeds far more
often than completing the FULL inbound→dock→outbound cycle (dark). The outbound leg
roughly halves the rule-based controller's success and all but eliminates the DQN's.

---

## 8. Discussion — A Factor Hierarchy

Reading the four research questions together yields a consistent ordering of what
governs safe, complete navigation on this testbed:

$$
\underbrace{\text{mission structure, multi-vessel interaction}}_{\eta^2_p\ \approx\ 0.5\text{–}0.7}
\ \gtrsim\
\underbrace{\text{control architecture / navigation prior}}_{\approx\ 0.3}
\ \gtrsim\
\underbrace{\text{perception \& communication quality}}_{\approx\ 0.1\text{–}0.86\ \text{(metric-dependent)}}
\ \gg\
\underbrace{\text{value-based algorithmic refinement}}_{\approx\ 0.01\text{–}0.05}.
$$

This is not a universal law but a repeatable pattern in the data: **higher-level
design factors dominate lower-level algorithmic ones.** Four consequences:

- **Algorithm choice is a weak lever** (RQ1); seed variance dominates it, so claims
  of one value-based variant beating another require far more seeds than typical.
- **Robustness is a perception/communication problem** (RQ2): sensing and comms
  quality — not the algorithm — determine safety and precision, with a compounding
  interaction. This locates the bottleneck in the sensing/comms pipeline and the
  control layer that consumes it.
- **The classical-vs-learned debate is really about information** (RQ3): a
  rule-based controller's robustness advantage largely reflects its access to a
  charted-route prior; granting the learner the same prior closes the gap. The
  productive question is *architectural* — how to supply structured navigation
  priors to a learned policy — motivating hybrid classical–learned control.
- **Interaction and mission structure dominate** (RQ4): single-vessel one-way
  success substantially overstates end-to-end capability; two-vessel safety is
  governed by the control architecture, and a full round trip roughly halves
  success. Benchmarks must include multi-vessel encounters and complete missions.

**Reporting practice** threads through all four: success-only, single-vessel,
one-way evaluation would have painted a far rosier and misleading picture than the
multi-metric, outcome-decomposed, multi-vessel evaluation supports.

---

## 9. Limitations

- **Synthesized testbed.** The environments are research models on synthesized port
  graphs, not validated hydrodynamic/RF simulators; absolute magnitudes are
  model-specific and the **relative** factor effects are the transferable results.
  External validity is probed across three geometries ([§6.3](#rq3)) but not beyond
  synthetic maps.
- **Within-distribution evaluation.** Evaluation uses fresh draws from the training
  mission distribution, not a held-out split; the generalization claim is scoped
  accordingly ([§3.4](#methods)).
- **Statistical power.** Inference is seed-level; the 6-seed studies are
  power-limited for pairwise contrasts (we rely on the pooled ANOVA, effect sizes,
  the mixed-effects model, and the 15-seed confirmatory study). The within-DRL
  chart-prior contrast is large but borderline-significant at $n=8$.
- **Algorithm family.** Only value-based DRL was studied; policy-gradient /
  actor–critic methods (PPO, SAC) are not implemented and were not fabricated. A
  single policy-gradient comparator is the most valuable next addition.
- **DRL failure scope.** The DQN's failure on the harbour transit / round trip /
  two-way traffic is a property of this testbed *at this budget*; it is reported as
  an honest negative result, not a general claim about value-based RL for maritime
  autonomy.
- **Baseline and metrics.** "COLREGs-aware" denotes rule-of-the-road *inspired*
  behaviour, not certified compliance; IALA compliance is an *unoptimized*
  evaluation metric (the agents are not rewarded for it); inter-vessel separation
  uses a first-order interpolation between macro-step poses.

---

## 10. Conclusion

Across a programme of controlled experiments (**31,150 evaluation episodes/rows**,
analyzed at the per-seed level with effect sizes, multiple-comparison correction,
and a mixed-effects model) we find that, for autonomous channel navigation on a
Synthetic Port testbed, **what determines safe, complete navigation is ordered**:
mission structure and multi-vessel interaction dominate, followed by control
architecture / navigation prior, then perception and communication quality, with
the choice among value-based DRL variants a distant last. A COLREGs-aware classical
baseline is more robust than the learned variants — but *because of the chart
information it uses*, which a learner can be given to close the gap. We recommend
that DRL navigation studies (i) budget enough seeds to separate algorithm from seed
variance and report seed-level, effect-size statistics; (ii) evaluate under
degraded perception/communication; (iii) always report safety-relevant metrics by
episode outcome alongside success; (iv) make baseline information (chart/prior) a
controlled factor rather than a confound; and (v) test multi-vessel encounters and
full round-trip / two-way missions rather than single-vessel one-way transits
alone. Promising future work: hybrid classical–learned controllers with explicit
navigation priors, policy-gradient baselines, higher-density traffic, and
validation on higher-fidelity simulators.

---

## Data and Code Availability

All raw per-episode data, aggregated statistics, tables, figures, frozen
configurations, and analysis scripts are committed under `results/` (RQ1),
`results_v3/` (RQ2), `results_study3/` (RQ3 baseline), `results_chart/` (RQ3
chart-fairness), `results_maps/` (RQ3 external validity), `results_study4/` and
`results_study5/` (RQ4), `results_audit/` (validity audit, outcome decomposition,
mixed-effects, literature positioning), and `experiments/`. Interactive dashboards
for each study and a downloadable PDF of this manuscript are available from the
project site.

*(Appendix follows: validity audit & measurement notes, mixed-effects summary,
literature capability table, and full per-study configuration, performance,
factorial, and pairwise-significance tables.)*
