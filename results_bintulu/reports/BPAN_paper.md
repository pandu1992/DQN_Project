# Bintulu Port Autonomous Navigation (BPAN): A Chart-Grounded Case Study of Value-Based Deep Reinforcement Learning under Degraded Sensing and Communication

**A reproducible, chart-grounded investigation on the Bintulu Port approach channels**

> **Reproducibility statement.** Every quantitative claim is computed from
> `results_bintulu/raw/eval_episodes.csv` (2,880 evaluation episodes) by
> `experiments/analyze_bintulu.py`; nothing is hand-set or fabricated. The
> waypoint graph is placed on the native pixel frame of the Bintulu Port approach
> chart (`assets/bintulu/bintulu_chart.png`), which is also the background of the
> live browser simulation (`bintulu.html`).

> **Scope & relationship to prior work.** This is a **standalone case study on the
> real Bintulu Port chart**, deliberately kept separate from the author's
> *Synthetic Port* programme (Studies 1–5, which use a constructed graph). It
> reuses the same physics/sensing/communication and metric machinery for
> comparability, but the geometry here is Bintulu-specific. The dynamics are a
> *faithful approximation* grounded in the chart, not a surveyed hydrographic or
> radio-frequency simulator.

---

## Abstract

Bintulu Port, on the coast of Sarawak, is approached through two buoyed access
channels (North and South) that converge toward the Inner Harbours and the
Southern Jetty / Container Terminal. We ask whether a value-based deep
reinforcement learning (DRL) agent can learn to transit these channels and berth,
and — more importantly — **what governs its safety and reliability** when
perception and communication are imperfect. We build a browser-based environment
whose waypoint graph is placed directly on the Bintulu approach chart (the chart's
lateral buoys NO 1'–NO 9'/N1'–N9' on the North channel and R2'–R10'/G2'–G10' on
the South channel), extending the charted channels into the harbour berths so a
vessel completes a full sea-to-berth mission. Four DRL variants (DQN, Double DQN,
Dueling DQN, Dueling Double DQN) are evaluated across graded sensor noise and
communication packet-loss (3 conditions × 8 shared seeds; 2,880 evaluation
episodes), with genuinely computed safety and precision metrics (collisions, IALA
channel compliance, cross-track error, docking accuracy) and seed-level statistics.

Two findings stand out. First, an agent **can** learn to navigate the Bintulu
approaches, but **the choice among the four value-based variants is immaterial**
— the algorithm factor explains ≈1–3% of the variance in every outcome
(all non-significant). Second, **sensing and communication degradation dominate
safety and compliance**: moving from clean to harsh conditions raises IALA
channel-departure violations from ≈0 to ≈23 per episode and collisions from ≈0.2
to ≈1.6 per episode (condition partial η² ≈ 0.63 for both; p < 0.001), while
cross-track error on successful transits grows from ≈3 to ≈25. We conclude that,
on the Bintulu approaches, robust autonomy is primarily a perception/communication
problem rather than a value-function-refinement problem, and that safety-relevant
metrics must be reported alongside berthing success. The chart-grounded testbed
and live simulation are released for reproducibility.

**Keywords:** Bintulu Port, autonomous surface vessel, deep reinforcement learning,
COLREGs/IALA channel-keeping, sensor noise, communication reliability, nautical
chart, reproducibility.

---

## 1. Introduction

Bintulu Port is one of Malaysia's busiest deep-water ports, and its approaches are
constrained: a vessel must stay within the buoyed North or South Access Channel,
avoid shoals and traffic, and berth precisely at the Inner Harbours or Southern
Jetty. Automating this transit is attractive, and deep reinforcement learning
(DRL) is a natural candidate for the sequential decision problem. Yet most DRL
navigation studies report *nominal* success under idealized perfect sensing and a
single clean trajectory, which says little about behaviour at sea, where sensors
are noisy and communication links drop frames.

This case study poses two concrete questions on the **real Bintulu approach
chart**:

1. **Can a value-based DRL agent learn to transit the Bintulu access channels and
   berth — and does the choice among common DQN variants matter?**
2. **How do sensor noise and communication packet-loss affect the safety and
   precision of that navigation, relative to the choice of algorithm?**

We answer both with a reproducible, chart-grounded testbed and a disciplined,
seed-level statistical protocol, and we report safety-relevant metrics (collision
rate, IALA channel compliance, cross-track error, docking accuracy) — not success
alone.

![The Bintulu Port approach chart used as the testbed and live-simulation background.](assets/bintulu/bintulu_chart.png)
**Figure 1.** The Bintulu Port approach chart. The North Access Channel (lateral
buoys NO 1'–NO 9' / N1'–N9') and the South Access Channel (R2'–R10' / G2'–G10')
converge toward the Inner Harbours, Southern Jetty and Container Terminal. The
study's waypoint graph is placed on this chart and extended into the berths.

---

## 2. Related Work and Positioning

DRL for maritime navigation and COLREGs-aware collision avoidance is an active
field, spanning value-based, actor–critic, and distributional methods on
unmanned surface vehicles and maritime autonomous surface ships. Robustness to
imperfect perception is beginning to receive attention (benchmarks under sensor
noise/denial; distributionally-robust formulations), but communication failure is
rarely modelled for a single learned vessel, and studies seldom report
channel-compliance or docking safety metrics alongside success. Methodological
work in RL further stresses multiple seeds, effect sizes, and significance tests
with the seed treated as a unit of variance.

This case study's contribution is **not a new algorithm** but **chart-grounded,
statistically-disciplined evidence** for one real port: a reproducible Bintulu
testbed with a live chart-overlay simulation, jointly manipulated sensing and
communication degradation, genuinely computed safety metrics, four value-based
variants under a shared-seed design, and honest reporting of null and
power-limited results. It complements the author's broader *Synthetic Port*
programme, which generalizes these questions across a constructed benchmark; here
the point is specifically to ground the phenomenon on the **actual Bintulu
chart**.

---

## 3. Methods

### 3.1 Chart-grounded environment

The environment (`js/environmentBintulu.js`, `BintuluEnv`) places a waypoint graph
on the Bintulu approach chart's native 1536×1024 pixel frame, so every waypoint and
buoy lands on the corresponding chart feature and the chart can serve as the live
simulation background. The topology follows the chart:

- **North Access Channel** — a centreline between the red "NO" buoys and the green
  "N" buoys, running from the sea entrance to the Southern Jetty / Container
  Terminal / Inner Harbour 2 berths.
- **South Access Channel** — a centreline between the red "R" and green "G" buoys,
  running from the sea entrance to the Inner Harbour 1 berth.
- A short **harbour connector** links the two channel mouths through the Southern
  Jetty / Inner Harbour 2 area.

The chart's magenta channel markings stop in open water; because an autonomous
vessel must actually reach a berth, we **extend** the navigable graph past the last
buoys into the harbour berths. The geometry is thus a faithful approximation of the
Bintulu approaches rather than a surveyed reproduction — adequate for a controlled
DRL case study and transparent about its limits.

On top of the graph the environment implements the same **physical / sensing /
communication** layer as the author's other work: continuous kinematics with a
control-drift model, Gaussian observation noise (std `noiseStd`), a communication
packet-error channel (dropped frames held stale, rate `packetErrorRate`), a seeded
obstacle field with collision detection, docking accuracy, cross-track error (CTE),
and IALA channel-departure detection from the buoy geometry. Observation dimension
28; four discrete edge-selection actions; a 60-step episode budget.

### 3.2 Agents and training

Four value-based DRL agents share an MLP torso (obs → 128 → 128, ReLU), a target
network with periodic hard updates, Huber loss and a hand-written Adam optimizer:
**DQN**, **Double DQN** (decoupled selection/evaluation), **Dueling DQN**
(value+advantage streams), and **Dueling Double DQN**. Each is trained for 150
episodes and then evaluated greedily for 30 episodes.

### 3.3 Metrics, evaluation, and statistics

Metrics: navigation **success rate** (berth reached), **mean reward**, routing
**optimality ratio**; safety **collision rate / collisions-per-episode** and **IALA
violations/episode**; precision **docking accuracy** and **cross-track error
(CTE)**; and a communication manipulation check, **dropped frames**.

Three reporting rules follow the author's validity audit. (i) *Success is
goal-reaching only*; collisions do not terminate an episode, so collisions and IALA
are within-episode safety metrics orthogonal to success. (ii) *CTE is reported on
successful episodes* (CTE drift scales with degradation, so an all-episode mean is
a success-rate-weighted mixture; the clean successful CTE is near zero by
construction). (iii) *Docking/optimality are solved-only* and reported with the
success base. Evaluation uses greedy roll-outs on fresh draws from the same mission
distribution the agent trained on — within-distribution generalization, **not** a
held-out split.

The unit of inference is the **per-seed scalar** (8 shared seeds; `env_seed = 42 +
seed`, agent RNG seeded separately). We report means with 95% CIs, run a two-way
factorial ANOVA (algorithm × condition + seed; Type-II SS; partial η²), and test
clean→harsh robustness per algorithm with paired Wilcoxon signed-rank tests, Holm
correction, and Cohen's $d_z$. The **4 × 3 × 8 = 96 cells / 2,880 episodes** index
computational replication; the effective sample size is the 8 seeds.

---

## 4. Results

### 4.1 The agent learns to transit — but the variant does not matter

All four variants learn to berth at non-trivial rates, and their success rates are
statistically indistinguishable. In the two-way factorial ANOVA the **algorithm
factor is negligible and non-significant on every outcome** (success partial
η² = 0.016, p = 0.75; collision rate η² = 0.007; IALA η² = 0.026; all algorithm
contrasts n.s.), with no meaningful algorithm×condition interaction. Mean success
by variant and condition:

| Agent | clean | mid | harsh |
|---|---|---|---|
| DQN | 21.7% | 17.1% | 28.3% |
| Double DQN | 11.7% | 17.5% | 35.4% |
| Dueling DQN | 21.2% | 25.8% | 27.5% |
| Dueling Double DQN | 16.7% | 20.8% | 27.1% |

> [!INSIGHT]
> **For transiting the Bintulu approaches, the four value-based DQN variants are
> effectively interchangeable.** Effort spent choosing among them is not where the
> performance comes from.

### 4.2 Sensing and communication degradation dominate safety and compliance

Degradation sharply worsens safety and precision, and the **condition factor
dominates** every safety/precision metric (all p < 0.001):

| Metric (mean over variants) | clean | mid | harsh | Condition partial η² |
|---|---|---|---|---|
| IALA violations / episode | 0.0 | — | 22.7 | **0.63** |
| Collisions / episode | 0.19 | — | 1.63 | **0.64** |
| Cross-track error (on success) | 2.9 | — | 25.2 | 0.57 |
| Docking accuracy error | — | — | — | **0.79** |
| Navigation success rate | 17.8% | — | 29.6% | 0.18 |

The communication manipulation check behaves as designed: dropped frames rise
0.0 → 10.4 → 19.8 across clean/mid/harsh. Channel-keeping and docking are the most
sensitive: IALA violations go from essentially zero in clean water to ~23 per
episode under harsh noise+packet-loss, and docking-accuracy variance is almost
entirely explained by the condition (η² = 0.79). The algorithm explains ≈1–3% of
the variance in these same metrics.

![Variance explained: the condition dominates, not the algorithm.](results_bintulu/figures/figBPAN_anova_eta2.png)
**Figure 2.** Partial η² per factor and metric (factorial ANOVA). The degradation
**condition** (orange) explains the large majority of the variance in safety and
precision; the **algorithm** (blue) and the interaction are small.

![IALA channel-departure violations vs condition, per variant.](results_bintulu/figures/figBPAN_iala.png)
**Figure 3.** IALA channel-departure violations per episode rise steeply from clean
to harsh for all four variants, which track each other closely — degradation, not
the algorithm, drives the loss of channel compliance.

### 4.3 Robustness (clean → harsh)

Per-algorithm clean→harsh paired contrasts show large, consistent increases in
collisions and IALA violations (and a success rate that is noisy and modest
throughout). At n = 8 seeds these per-cell contrasts are power-limited for
significance after Holm correction, so the **pooled factorial ANOVA carries the
inference** — exactly the pattern the degradation η² values report. We state this
rather than over-claim per-cell significance.

> [!INSIGHT]
> **Success alone would mislead.** Navigation success is modest and only weakly
> tied to the condition, even as collisions and channel-departure violations climb
> steeply. A success-only evaluation of a Bintulu autonomy stack would conclude it
> is "about as good" in harsh conditions, while the safety metrics show the
> trajectories have become markedly less compliant and more collision-prone.

---

## 5. Discussion

On the real Bintulu approaches, the evidence is consistent and actionable:

- **Algorithm choice is a weak lever.** The four value-based DQN variants are
  interchangeable for this transit; refining the value-function family is not where
  robustness is gained.
- **Robustness is a perception/communication problem.** Sensor noise and
  packet-loss, not the algorithm, govern channel compliance, collision avoidance,
  cross-track precision, and docking. For a Bintulu deployment this argues that
  investment in sensor and link quality (or noise-robust perception and
  communication-aware control) pays off far more than swapping DRL variants.
- **Report safety, not just berthing success.** Because collisions do not end an
  episode and IALA departures are unrewarded, success conceals the safety cost of
  degradation; the multi-metric, outcome-aware evaluation reveals it.

These conclusions mirror, on a real chart, what the author's broader Synthetic Port
programme found on a constructed benchmark — strengthening the case that the
pattern is a property of the task class, not of one particular synthetic map.

---

## 6. Limitations

- **Chart-grounded, not hydrographically surveyed.** Waypoints and buoys are placed
  on the chart image and the channels are extended into the berths; the dynamics are
  a faithful approximation, not a validated hydrodynamic/RF simulator. Absolute
  magnitudes are model-specific; the **relative** factor effects are the
  transferable result.
- **Within-distribution evaluation** (fresh draws from the training mission
  distribution), not a held-out split.
- **Seed-level power.** n = 8 gives the pooled ANOVA good power but leaves per-cell
  paired robustness contrasts power-limited.
- **Value-based DRL only;** policy-gradient / actor–critic methods and an explicit
  classical COLREGs baseline are natural next additions (as explored in the
  Synthetic Port programme).
- **Unoptimized safety metrics.** IALA compliance is measured but not rewarded, so
  it reflects emergent rather than trained behaviour.

---

## 7. Conclusion

We presented **BPAN**, a chart-grounded case study of value-based deep
reinforcement learning for autonomous navigation of the Bintulu Port approach
channels, with a live simulation that runs the learned vessel directly over the
real chart. Across 2,880 evaluation episodes analysed at the per-seed level, an
agent **can** learn to transit the channels and berth, but **which value-based
variant is used does not matter**, whereas **sensor noise and communication
packet-loss dominate** safety and precision — raising IALA channel-departure
violations from near-zero to ~23 per episode and collisions from ~0.2 to ~1.6 per
episode from clean to harsh conditions. For autonomous navigation of a real port
like Bintulu, robustness is primarily a perception and communication problem, and
safety-relevant metrics must be reported alongside berthing success. We release the
chart-grounded environment, the experiment pipeline, and the browser simulation for
reproduction and extension (classical COLREGs baselines, policy-gradient agents, and
higher-fidelity dynamics).

---

## Data and Code Availability

Raw per-episode data (`results_bintulu/raw/eval_episodes.csv`, 2,880 rows),
aggregated statistics, factorial ANOVA, robustness tests, tables, figures, the
frozen configuration, the environment (`js/environmentBintulu.js`), the experiment
scripts (`experiments/*bintulu*`), and the live simulation (`bintulu.html`,
`js/bintulu_main.js`) are committed to the project repository. The interactive
Bintulu simulation and a downloadable PDF of this paper are available from the
project site.
