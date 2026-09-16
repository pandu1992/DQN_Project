# Robustness of Value-Based Deep Reinforcement Learning for Autonomous Vessel Navigation: A Three-Study Comparison Against a Classical COLREGs-Aware Baseline

**A reproducible experimental investigation on a Bintulu-Port channel-navigation testbed**

> **Reproducibility statement.** Every quantitative claim in this manuscript is
> computed from committed experiment data (`results/`, `results_v3/`,
> `results_study3/`) by the scripts in `experiments/`, and is regenerable
> end-to-end. No metric is hand-set or fabricated. All environments reuse the
> same synthesized port graph, edge attributes, and Dijkstra planner; the
> deployed browser simulation is unchanged by any of this analysis.

---

## Abstract

Deep reinforcement learning (DRL) is increasingly proposed for autonomous
maritime navigation, yet its *robustness* to imperfect perception and
communication — and its standing relative to classical rule-based control — is
rarely quantified with statistical rigor. We present three linked experiments on
a reproducible Bintulu-Port channel-navigation testbed. **Study 1** compares four
value-based DRL agents (DQN, Double DQN, Dueling DQN, Dueling Double DQN) with
Prioritized Experience Replay and Noisy Nets in a clean environment
(4×2×2 factorial, 10 seeds, 6,400 evaluation episodes). **Study 2** subjects the
same four agents to graded sensor-observation noise and communication
packet-loss (4×3×3 factorial, 6 seeds, 6,480 episodes), measuring genuinely
computed safety and precision metrics (collisions, IALA channel compliance,
docking accuracy, cross-track error). **Study 3** adds a non-learning
COLREGs-aware rule-based controller and raises the seed count to 15 for
confirmatory power (5 agents × 3 conditions × 15 seeds, 6,750 episodes).

Three findings emerge. (i) In clean conditions the four DRL variants are
**statistically indistinguishable** on task success; the dominant source of
variance is the **random seed** (partial $\eta^2 \approx 0.20$), not the
algorithm. (ii) Under degradation, safety and precision collapse and are governed
**overwhelmingly by the stressors** — sensor noise and packet-loss explain
60–86% of the variance in channel compliance and docking accuracy, while the
choice of DRL variant explains ≈1–5%. (iii) A simple **chart-following
COLREGs-aware baseline keeps the navigable channel roughly twice as well as every
DRL variant under harsh degradation** (all four contrasts significant after Holm
correction, large effect sizes), a difference the 15-seed design resolves where a
6-seed design could not. We argue that, for this class of task, robustness is
primarily a *perception/communication and control-architecture* problem rather
than a value-function-refinement problem, and that safety-relevant metrics must
be reported alongside task success, which alone conceals the risk.

**Keywords:** deep reinforcement learning, autonomous surface vessels, robustness,
sensor noise, communication reliability, COLREGs, reproducibility.

---

## 1. Introduction

Autonomous surface-vessel navigation is a safety-critical control problem: an
agent must reach a berth or waypoint while remaining inside marked channels,
avoiding obstacles, and respecting the international rules of the road (COLREGs).
Deep reinforcement learning has become a popular tool for such sequential
decision problems, and a large literature reports learned policies that match or
exceed hand-designed controllers on nominal performance. Three questions,
however, are under-examined in that literature and are decisive for deployment:

1. **Do the popular value-based DRL refinements actually differ** from one
   another on the task, once run-to-run variability is accounted for?
2. **How gracefully do learned policies degrade** when perception (sensor noise)
   and communication (packet-loss) are imperfect — the normal condition at sea?
3. **Is learning even the right tool**, or does a classical rule-based controller
   remain competitive, particularly on safety-relevant behaviour?

This manuscript answers all three with a single, reproducible testbed and a
consistent statistical protocol. Our contributions are: (a) a controlled
comparison of four value-based DRL variants with two Rainbow-style enhancements
that isolates the effect of algorithm choice from seed variance
([Study 1](#study1)); (b) a factorial *robustness* analysis under genuinely
implemented sensing and communication degradation, with real safety and precision
metrics ([Study 2](#study2)); and (c) a confirmatory study introducing a
COLREGs-aware rule-based baseline at higher statistical power
([Study 3](#study3)). Throughout, we prioritize honest reporting: we state where
effects are absent, where power is limited, and where the testbed is a synthesized
model rather than a validated simulator.

> [!INSIGHT]
> **Framing.** The three studies form a deliberate arc: *do the algorithms
> differ?* (largely no), *how do they fail under stress?* (badly, and in
> parallel), and *can a classical controller do better?* (yes, significantly, on
> channel compliance). This reframes the design question from "which DRL variant
> wins" to "what confers robustness in this domain".

![Overview of the three studies](assets/schematics/study_comparison.png)
**Figure 1 (Overview).** The three studies on one testbed, progressively richer:
**Study 1** compares four DQN variants on a clean channel-navigation task;
**Study 2** adds sensor noise, communication packet-loss, obstacles/collisions,
IALA channel markers, docking accuracy and cross-track error; **Study 3** adds a
COLREGs-aware rule-based baseline (which follows the charted channel) versus the
DQN variants (which deviate under stress), at higher statistical power.

---

## 2. Related Work and Positioning

Value-based DRL for navigation builds on the Deep Q-Network and its widely used
refinements — Double DQN (decoupled action selection/evaluation to curb
over-estimation), Dueling networks (separate state-value and advantage streams),
Prioritized Experience Replay (sampling high-TD-error transitions), and Noisy
Nets (learnable parameter-noise exploration). These are typically benchmarked on
nominal return. Robustness studies in RL more broadly show that learned policies
can be brittle to observation perturbations; and the maritime-autonomy community
has long relied on rule-based, COLREGs-informed controllers whose behaviour is
predictable and certifiable. Our work does not propose a new algorithm; it
contributes **methodology and evidence** — a reproducible testbed, a factorial
robustness protocol with real safety metrics, an explicit classical baseline, and
a statistically disciplined comparison that reports null and negative results as
faithfully as positive ones.

---

## 3. Methods

### 3.1 Testbed and environments

All experiments run on a synthesized Bintulu-Port channel graph: two access
channels (North L02, South L08) feeding a shared harbour lane (L01), with buoys,
virtual shortcut branches, and per-edge attributes (distance, risk, traffic,
weather, current, travel time, energy cost, navigation cost, difficulty). A
Dijkstra planner over navigation cost provides the optimal charted route.

The originally deployed environment is a *saturated* benchmark — a two-mission,
index-advancing task on which every agent trivially reaches ~100% success with
near-zero variance, making comparison vacuous. We therefore use two research
builds that reuse the *same graph, attributes, and planner*:

- **`VesselEnvV2`** (Study 1): a genuine graph-navigation MDP. The agent chooses
  among the current node's real outgoing edges (routing decisions at junctions
  and virtual shortcuts). Missions are varied (115 distinct start–goal pairs per
  seed; optimal-cost range 5.3–151.5). Observation dimension 25, four action
  slots, 60-step budget.
- **`VesselEnvV3`** (Studies 2–3): `VesselEnvV2` plus a *physical/sensing/comms*
  layer — continuous kinematics with a control-drift model, Gaussian observation
  noise, a communication packet-error channel (dropped frames held stale), a
  seeded obstacle field with collision detection, docking accuracy, cross-track
  error, and IALA channel-departure detection. Observation dimension 28.

> [!INSIGHT]
> **The benchmark had to be made discriminative before it could be studied.**
> Reconstructing the task as a genuine routing MDP (V2) and then adding a real
> physical/sensing/comms layer (V3) is the pivotal design step that makes both
> the algorithm comparison and the robustness analysis meaningful — and it is
> done on the *same* underlying chart, so the three studies are commensurable.

### 3.2 Agents

Four value-based DRL agents share an MLP torso ($\text{obs} \to 128 \to 128$,
ReLU), a target network with periodic hard updates, Huber loss, and a
hand-written Adam optimizer. Study 3 adds a fifth, non-learning agent.

**DQN.** Q-values $Q_\theta(s,a)$; bootstrap target and Huber loss ($\delta=1$):

$$ y_t = r_t + \gamma\,(1-d_t)\,\max_{a'} Q_{\theta^-}(s_{t+1},a'), \qquad
\mathcal{L}(\theta)=\mathbb{E}\big[\ell_\delta\!\big(Q_\theta(s_t,a_t)-y_t\big)\big]. $$

**Double DQN.** Online net selects, target net evaluates:

$$ a^\* = \arg\max_{a'} Q_\theta(s_{t+1},a'), \qquad y_t = r_t + \gamma\,(1-d_t)\,Q_{\theta^-}(s_{t+1}, a^\*). $$

**Dueling DQN.** Value + advantage streams with mean-advantage subtraction:

$$ Q(s,a) = V(s) + \Big(A(s,a) - \tfrac{1}{|\mathcal{A}|}\textstyle\sum_{a'} A(s,a')\Big). $$

**Dueling Double DQN** combines both. **Prioritized Experience Replay** samples
transition $i$ with probability $P(i)=p_i^\alpha / \sum_k p_k^\alpha$,
$p_i=|\delta_i|+\epsilon$, and corrects the bias with importance weights
$w_i = (N\,P(i))^{-\beta}/\max_j w_j$. **Noisy Nets** replace head weights with
$W=\mu_W+\sigma_W\odot\varepsilon$ (factorized Gaussian noise), disabling
$\epsilon$-greedy.

**COLREGs-aware rule-based baseline** (Study 3). A deterministic controller that
follows the Dijkstra-charted channel, prefers edges reducing distance to the
*sensed* goal, applies give-way collision avoidance (penalising/forbidding edges
near obstacles), keeps to lower-cost, lower-difficulty (marked) water, and avoids
back-tracking. Crucially, it reads the **sensed** goal bearing from the same
(noise- and dropout-affected) observation the DRL agents receive, so the
comparison under degradation is fair; its only asymmetry is use of the chart, as
any classical planner would have.

### 3.3 Evaluation metrics

The unit of analysis is the **per-seed scalar**: within a seed, the evaluation
episodes are collapsed to one value per metric; the seed-scalars are the
independent observations. Metrics: navigation **success rate** and **mean
reward**; safety **collision rate**, **collisions/episode**, and **IALA
violations/episode** (channel departures); precision **docking accuracy**
(distance of final pose to the dock point) and **cross-track error** (deviation
from the planned route); routing **optimality ratio** (route cost ÷ Dijkstra
optimum, $\ge 1$); and a communication manipulation-check, **dropped frames**.
Definitional formulas:

$$ \mathrm{SR}=\frac{N_\text{success}}{N_\text{episodes}}\times100\%, \qquad
\bar R = \frac1N\sum_i R_i, \qquad \rho=\frac{\text{cost}_\text{route}}{\text{cost}_\text{Dijkstra}}. $$

### 3.4 Statistical protocol

The same protocol applies across studies. We report descriptive means with 95%
confidence intervals (Student-$t$, cross-checked by 10,000-sample percentile
bootstrap):

$$ \mathrm{CI}_{95\%}=\bar x \pm t_{0.975,\,n-1}\,\frac{s}{\sqrt n}. $$

We test normality (Shapiro–Wilk) and variance homogeneity (Levene /
Brown–Forsythe), run **factorial ANOVA** with Type-II sums of squares and partial
$\eta^2$ effect sizes,

$$ \eta^2_p=\frac{\mathrm{SS}_{\text{effect}}}{\mathrm{SS}_{\text{effect}}+\mathrm{SS}_{\text{error}}}, $$

and perform paired post-hoc tests (Wilcoxon signed-rank, exact; paired $t$ as a
secondary) with **Holm** and **Benjamini–Hochberg** multiple-comparison
correction. Paired effect size is Cohen's $d_z=\bar d/s_d$ (with Cliff's $\delta$
for ordinal robustness). The shared-seed design makes all contrasts within-block
(paired). $\alpha=0.05$ throughout.

![Shared experimental workflow](assets/schematics/workflow.png)
**Figure 2 (Workflow).** The shared experimental pipeline — seeded environment →
train (RL) or run (rule-based) → held-out greedy evaluation → per-episode real
metrics → per-seed aggregation → statistics → tables and figures — and how each
study specializes it (environment build, factors, seed count, and metrics). Only
these differ across the three studies; the graph, planner, agents, and
statistical protocol are shared.

<a name="study1"></a>

## 4. Study 1 — Algorithm Comparison in Clean Conditions

**Design.** Four algorithms × PER {off,on} × Noisy {off,on} = 16 configurations,
10 shared seeds, 150 training / 40 evaluation episodes each (160 runs, 6,400
evaluation episodes) on `VesselEnvV2`.

**Results.** Baseline (no enhancements) success rates were DQN 49.8±18.1%,
Double DQN 43.2±19.3%, Dueling DQN 40.2±11.8%, Dueling Double DQN 45.5±17.9%
(mean ± 95% CI). Confidence intervals overlap heavily
([Fig. 1](#fig-s1-overall), [Fig. 3](#fig-s1-ci)). The factorial ANOVA found **no
significant algorithm effect** on success (partial $\eta^2=0.008$, $p=0.78$) or
reward; instead the **random seed dominated the variance** (partial
$\eta^2=0.200$, $p=0.0003$) — larger than every design factor combined
([Fig. 4](#fig-s1-var)). After Holm correction, **no pairwise algorithm
difference survived** on any metric ([Fig. 2](#fig-s1-sig)). The only design
signal was on route optimality, where Noisy Nets had a small significant effect
($\eta^2=0.061$, $p=0.005$) and an algorithm×Noisy interaction ($\eta^2=0.082$,
$p=0.014$). Full per-configuration results and tests are in
[Appendix A.1](#appendix-a1).

> [!INSIGHT]
> **In clean conditions the four DRL variants are effectively interchangeable for
> reaching the goal; run-to-run variance swamps algorithm choice.** This reframes
> the reproducibility question from "which algorithm wins" to "how many seeds are
> needed before a winner can even be claimed" — here, more than the 10 used.

<a name="fig-s1-overall"></a>
![Study 1 — overall performance](results/figures/fig1_overall_performance.png)
**Figure 1.** Study 1: navigation success and mean reward for the four base DRL
variants (mean ± 95% CI, 10 seeds). Overlapping intervals indicate no significant
algorithm effect.

<a name="fig-s1-var"></a>
![Study 1 — variance dominance](results/figures/fig8_variance_dominance.png)
**Figure 2.** Study 1: partial $\eta^2$ from the factorial ANOVA. The Seed block
explains the most variance in success and reward ($\eta^2\approx0.20$), exceeding
all design factors.

<a name="study2"></a>

## 5. Study 2 — Robustness to Sensing and Communication Degradation

**Design.** Four algorithms × sensor-noise {0, 0.1, 0.25} × packet-error
{0, 0.2, 0.4} = 36 conditions, 6 shared seeds, 130/30 episodes (216 runs, 6,480
evaluation episodes) on `VesselEnvV3`.

**Manipulation check.** Dropped frames were explained almost entirely by the
packet-error factor (partial $\eta^2=0.966$, $p<0.0001$) and not by noise
($\eta^2=0.013$) — the communication channel behaves as designed.

**Results.** Degradation sharply worsened safety and precision, and the effect was
driven by the *stressors*, not the algorithm ([Fig. 3](#fig-s2-eff)):

| Metric | Noise $\eta^2_p$ | Packet-error $\eta^2_p$ | Algorithm $\eta^2_p$ |
|---|---:|---:|---:|
| IALA violations | **0.855** | **0.795** | 0.012 (n.s.) |
| Docking accuracy | **0.670** | **0.606** | 0.014 (n.s.) |
| Collision rate | 0.428 | 0.057 | 0.053 |
| Success rate | 0.106 | 0.079 | 0.041 (n.s.) |

Sensor noise and packet-loss each explained **60–86%** of the variance in channel
compliance and docking accuracy; the algorithm explained ≈1–5%. Noise and
packet-error also **interacted** on channel compliance (IALA Noise×PER
$\eta^2=0.292$, $p<0.0001$): combined degradation is disproportionately harmful.
Crucially, the **algorithm×stressor interactions were small and non-significant**
($\eta^2\le0.06$, $p>0.1$): the four variants degrade *in parallel*, so none is
meaningfully more robust than another. With 6 seeds the paired robustness
contrasts had large effect sizes but were power-limited (Holm-adjusted $p$ floored
near 0.125); this motivated Study 3. Full effects and per-condition tables:
[Appendix A.2](#appendix-a2).

> [!INSIGHT]
> **Sensing and communication quality — not the choice of DRL variant — govern
> safety and precision.** For a deployed vessel this argues that investment in
> sensor/link quality (or noise-robust perception) pays off far more than swapping
> between these value-based refinements.

<a name="fig-s2-eff"></a>
![Study 2 — effect sizes](results_v3/figures/figV3_effect_sizes.png)
**Figure 3.** Study 2: partial $\eta^2$ per factor and metric (factorial ANOVA
$y\sim\text{algorithm}\times\text{noise}\times\text{PER}+\text{seed}$). Noise and
packet-error dominate the safety/precision metrics; the algorithm and its
interactions with the stressors are small.

<a name="study3"></a>

## 6. Study 3 — A COLREGs-Aware Rule-Based Baseline and Confirmatory Replication

**Design.** Five agents (the rule-based baseline + four DRL variants) × three
conditions (clean, mid = noise 0.1/PER 0.2, harsh = noise 0.25/PER 0.4) × **15
shared seeds** (225 runs, 6,750 evaluation episodes) on `VesselEnvV3`. The higher
seed count restores the statistical power Study 2 lacked.

**Results.** Navigation success showed a significant agent effect (partial
$\eta^2=0.261$, $p<0.0001$): the **rule-based baseline's success was invariant to
degradation (47.3% at every condition)** — because it follows the chart — and was
the highest of all agents ([Fig. 4](#fig-s3-succ)). The headline result is on
channel compliance. IALA violations per episode at the harshest condition were
**24.3 for the rule-based baseline versus 57.7–63.2 for the DRL variants**; a
significant **agent×condition interaction** (partial $\eta^2=0.491$, $p<0.0001$)
shows the DRL agents' compliance collapses far faster ([Fig. 5](#fig-s3-iala)).
Every rule-based-vs-DRL IALA contrast at harsh was **significant after Holm
correction** ($p<0.001$; Cohen's $d_z$ from $-1.9$ to $-2.4$, all *large*).
Across all metrics and conditions, **46 of 96** baseline-vs-DRL contrasts survived
Holm correction — the confirmatory power the 6-seed Study 2 could not achieve.

We report honestly that the baseline's *fewer collisions* trend (0.53 vs 1.0–1.4
per episode at harsh) is **not** statistically established (high per-seed
variance), and that DRL agents track the optimal route more tightly (lower
cross-track error) when they do reach the goal. Full contrasts and effect sizes:
[Appendix A.3](#appendix-a3).

> [!INSIGHT]
> **A simple chart-following COLREGs-aware controller keeps the buoyed channel
> roughly twice as well as every learned policy under degradation — significantly
> so.** For this task, a classical control layer confers a safety-robustness
> advantage that the value-based refinements do not; learning is not, on this
> evidence, the right tool for channel compliance under imperfect sensing/comms.

> [!INSIGHT]
> **Cross-study nuance.** Study 2 found *parallel* degradation among the DRL
> variants; Study 3 finds a *significant* agent×condition interaction once the
> rule-based baseline is included. The two are consistent: the DRL variants
> degrade similarly to each other, but all of them degrade much faster than the
> classical baseline.

<a name="fig-s3-succ"></a>
![Study 3 — success vs condition](results_study3/figures/figS3_success_vs_condition.png)
**Figure 4.** Study 3: navigation success vs degradation condition, per agent
(mean ± 95% CI, 15 seeds; rule-based in bold black). The baseline's success is
invariant to degradation.

<a name="fig-s3-iala"></a>
![Study 3 — IALA interaction](results_study3/figures/figS3_iala_interaction.png)
**Figure 5.** Study 3: IALA channel-departure violations vs condition. The lines
fan out (significant agent×condition interaction, partial $\eta^2=0.49$): the DRL
variants' violations rise far more steeply than the rule-based baseline's.

---

## 7. Discussion

The three studies converge on a coherent, and somewhat cautionary, picture for
DRL-based vessel autonomy on this class of task.

**Algorithm choice is a weak lever.** Neither in clean conditions (Study 1) nor
under degradation (Study 2) did the choice among DQN, Double, Dueling, or
Dueling-Double meaningfully change task success. The refinements that the
literature emphasizes did not translate into a robustness advantage here; the
only reliable algorithmic signal was a small Noisy-Nets effect on route
optimality (Study 1).

**Robustness is a perception/communication problem.** The variance in the
safety-critical metrics under Study 2 was overwhelmingly attributable to the
sensor-noise and packet-loss factors, with a compounding interaction on channel
compliance. This locates the robustness bottleneck in the sensing and
communication pipeline (and the control layer that consumes it), not in the value
function.

**Classical control remains a strong, safer baseline.** Study 3's rule-based
controller — a modest heuristic that follows the charted channel — was not only
competitive on success but *significantly* better at IALA channel compliance under
stress, with large effect sizes and adequate power. This is a concrete argument
for hybrid designs in which a classical, chart- and rules-aware layer guards
safety-critical behaviour while learning is reserved for aspects where it
demonstrably helps.

**Reporting practice.** Task success alone was nearly insensitive to degradation
even as collisions and channel violations climbed steeply. A paper reporting only
success would have concluded the agents were robust; the multi-metric evaluation
shows they are not. Safety-relevant metrics are not optional.

---

## 8. Limitations

- **Synthesized testbed.** `VesselEnvV2/V3` are research models on a synthesized
  port graph, not validated hydrodynamic or radio-frequency simulators. Absolute
  magnitudes are model-specific; the **relative** factor effects and
  baseline-vs-DRL patterns are the transferable results.
- **Statistical power.** Study 1 (10 seeds) and especially Study 2 (6 seeds) are
  power-limited for pairwise contrasts; we relied on the pooled factorial ANOVA
  and effect sizes there, and Study 3 (15 seeds) for confirmatory pairwise claims.
- **Algorithm family.** Only value-based DRL variants were studied; policy-gradient
  and actor–critic methods (PPO, SAC) are not implemented in the codebase and were
  not added rather than fabricated.
- **Baseline design.** The rule-based controller is one reasonable COLREGs-inspired
  heuristic; its advantage derives substantially from chart-following, which
  presumes an accurate chart. "COLREGs-aware" denotes rule-of-the-road *inspired*
  behaviour, not formal regulatory compliance.
- **Training budget.** Small pure-in-browser networks trained for 130–150 episodes;
  larger models or longer training could shift absolute performance.

---

## 9. Conclusion

Across three reproducible experiments (19,630 evaluation episodes in total) we
find that, for autonomous channel navigation on a Bintulu-Port testbed, the choice
among four value-based DRL variants is not a statistically meaningful lever;
run-to-run seed variance dominates in clean conditions, and sensing/communication
degradation dominates safety and precision under stress, affecting all variants in
parallel. A simple COLREGs-aware, chart-following rule-based controller matches or
exceeds the learned policies on success and is *significantly* more compliant with
channel-keeping under degradation. We recommend that DRL navigation studies (i)
budget enough seeds to separate algorithm from seed variance, (ii) evaluate under
degraded perception/communication, (iii) always report safety-relevant metrics
alongside success, and (iv) include a classical baseline. Promising future work
includes hybrid classical–learned controllers, policy-gradient baselines, and
validation on higher-fidelity simulators.

---

## Data and Code Availability

All raw per-episode data, aggregated statistics, tables, figures, frozen
configurations, and analysis scripts are committed to the project repository under
`results/` (Study 1), `results_v3/` (Study 2), `results_study3/` (Study 3), and
`experiments/`. Interactive dashboards for each study, and a downloadable PDF of
this manuscript, are available from the project site.

*(Appendix tables follow: full per-study configuration, performance, factorial,
and pairwise-significance tables. See the Tables tab / Appendix.)*
