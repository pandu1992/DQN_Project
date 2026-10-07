# Bintulu Port Autonomous Navigation (BPAN): What Level of the Design Stack Confers Robust, Complete, Multi-Vessel Autonomy? A Chart-Grounded Factor-Hierarchy Study

**A reproducible, chart-grounded investigation on the Bintulu Port approach channels — from single-vessel transit to shared-water and round-trip operations**

> **Reproducibility statement.** Every quantitative claim is computed from the
> committed per-episode data by the analysis scripts; nothing is hand-set or
> fabricated. The core single-vessel study draws on
> `results_bintulu/raw/eval_episodes.csv` (2,880 episodes,
> `experiments/analyze_bintulu.py`); the multi-vessel extension on
> `results_bintulu/raw_multi/` (54 cells, `analyze_bintulu_multi.py`); the
> operational-realism extension on `results_bintulu/raw_ops/` (72 cells,
> `analyze_bintulu_ops.py`); and the seed-level mixed-effects treatment on
> `mixed_effects_bpan.py`. The waypoint graph is placed on the native pixel frame
> of the Bintulu Port approach chart (`assets/bintulu/bintulu_chart.png`), which is
> also the background of the live browser simulation (`bintulu.html`).

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
metrics must be reported alongside berthing success.

We then extend the study beyond the single one-way transit along three axes and
synthesise the results into a **design-stack factor hierarchy**. **(d) Multi-vessel:**
with two independent vessels sharing the channels, the *pairing of control
architectures* dominates (success partial η² = 0.74, near-misses 0.59) — a classical
follower completes most but collides most, a learned vessel gives way but completes
less — while the DRL variant and even the degradation condition are minor on the
inter-vessel outcomes. **(e) Operational realism:** requiring a full inbound-dock-
outbound *round trip* makes completion a control-architecture capability (agent
partial η² = 0.90: the rule-based planner closes the cycle 100% of the time, the
naive learner ≈16%), while a *two-way* head-on in one surveyed channel makes
collision **structural** (≈1 per episode regardless of algorithm or condition) — an
honest null that only a mission-geometry change (traffic separation) could remove.
**(f)** A seed-level mixed-effects model ($y\sim\text{fixed}+(1\mid\text{seed})$)
reproduces every conclusion under a conservative random-seed treatment and reports
the seed's intraclass correlation (≤ 0.45, and ≈ 0 where the outcome is structurally
fixed). The **synthesis** is a factor hierarchy: *mission & interaction structure ≳
control architecture & navigation prior > perception & communications ≫ learning
algorithm.* Robust, complete, multi-vessel autonomy is conferred by the upper levels
of the design stack; the choice of value-based algorithm sits at the bottom and
moves nothing. The chart-grounded testbed, the three extensions, the animations, and
the live simulation are released for reproducibility.

**Keywords:** Bintulu Port, autonomous surface vessel, deep reinforcement learning,
multi-vessel interaction, COLREGs/IALA channel-keeping, operational realism,
mixed-effects models, factor hierarchy, sensor noise, communication reliability,
nautical chart, reproducibility.

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

The study proceeds as a single sequential pipeline — from the real chart, through
a shared environment contract, into four study blocks, down to a per-seed unit of
inference, and up through a three-tier statistical treatment to a factor-hierarchy
synthesis. Figure M lays out that flow in full; each stage below corresponds to a
box in the figure.

![BPAN methodology flow, from the real chart to the factor-hierarchy synthesis.](results_bintulu/figures/figBPAN_methodology.png)
**Figure M.** The end-to-end BPAN methodology. (1) the real Bintulu chart grounds
(2) a waypoint/buoy/berth graph that feeds (3) the shared environment contract
(physics, sensor noise, comms loss, obstacles, IALA, docking, CTE). Four study
blocks run on that contract — (A) the core 4-variant comparison, (B) the
DQN-improvement arms, (C) the multi-vessel extension *(d)*, and (D) the
operational-realism extension *(e)*. Every cell is reduced to (4) a per-seed
scalar, the unit of inference, with cross-track error taken on **successful**
episodes only. The scalars drive a three-tier statistical treatment — (5a)
factorial ANOVA with partial η², (5b) paired clean→harsh Wilcoxon/Holm/Cohen
$d_z$ contrasts, and (5c) the seed-level mixed-effects model *(f)* — which
together support (6) the factor-hierarchy synthesis.

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

### 3.2 Mathematical formulation

This section states the model explicitly. All symbols and constants are exactly
those used in `js/environmentBintulu.js` and `js/dqn.js`; nothing below is
illustrative.

#### 3.2.1 Markov decision process

We model Bintulu navigation as a finite-horizon, deterministically-structured but
stochastically-perturbed **Markov decision process**
$\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma,H)$. The navigable space is a
directed graph $G=(V,E)$ whose nodes $v\in V$ are chart waypoints with pixel
coordinates $\mathbf{p}_v=(x_v,y_v)\in[0,W]\times[0,H]$, $W=1536$, $H=1024$. At
step $t$ the agent occupies a current waypoint $c_t\in V$ with a continuous pose
$\mathbf{z}_t=(X_t,Y_t)$ and must reach a goal berth $g\in V$. The action
$a_t\in\mathcal{A}=\{0,1,2,3\}$ selects one of up to $N_\text{slot}=4$ outgoing
edges of $c_t$, ordered by ascending navigation cost (Eq. 2). An episode runs for
at most $H=60$ steps and terminates on arrival ($c_t=g$) or timeout. The discount
is $\gamma=0.99$.

#### 3.2.2 Edge attributes and planning cost

Each directed edge $e=(u,v)\in E$ carries a Euclidean length
$d_e=\lVert\mathbf{p}_u-\mathbf{p}_v\rVert_2$ and seeded random attributes
risk $\rho_e$, traffic $\tau_e$, weather $w_e$ and current $\kappa_e$. The along-edge
speed, travel time and energy are

$$ s_e=\max\!\big(2,\; 8\,(1-\kappa_e)\big),\qquad
   t_e=\frac{d_e}{s_e},\qquad \varepsilon_e = 0.01\,d_e . \tag{1}$$

The **navigation cost** that defines the charted route and the action ordering, and
the normalised **difficulty**, are

$$ \mathrm{nav}(e)=t_e+\varepsilon_e+2\tau_e+3\rho_e,\qquad
   \mathrm{dif}(e)=\min\!\big(1,\;0.35\rho_e+0.25\tau_e+0.2 w_e+0.2|\kappa_e|\big). \tag{2}$$

The **planned (charted) route** $\pi^\star(u\!\to\!g)$ is the minimum-cost path under
$\mathrm{nav}(\cdot)$, obtained by Dijkstra's algorithm; its node sequence
$\mathcal{P}=(p_0,\dots,p_L)$ and optimal cost
$C^\star=\sum_i \mathrm{nav}(p_i,p_{i+1})$ are used both as the reference track for
cross-track error (Eq. 6) and, in the improvement study, as the chart prior.

#### 3.2.3 Continuous kinematics with degradation-driven drift

Choosing edge $e=(c_t,v)$ moves the vessel along the segment in $K=6$ sub-steps
$k=1,\dots,K$ with $\theta_k=k/K$. Let $\hat{\mathbf{n}}_e$ be the unit normal to
the segment. The realised pose is the nominal interpolation plus a lateral
**control drift** whose amplitude grows with the sensing/communication degradation:

$$ \mathbf{z}(\theta_k)=\mathbf{p}_u+\theta_k(\mathbf{p}_v-\mathbf{p}_u)
   \;+\;\hat{\mathbf{n}}_e\,\underbrace{\sigma_\text{dr}\,\xi_k\,\psi(\theta_k)}_{\text{drift}},
   \qquad \xi_k\sim\mathcal{N}(0,1), \tag{3}$$

with drift scale and shape

$$ \sigma_\text{dr}=90\,\sigma_\text{obs}+45\,p_\text{err},\qquad
   \psi(\theta)=\sin(\pi\theta)\,(1-h)+h\,\theta\,\mathbb{1}[v=g],\;\; h=\tfrac12\mathbb{1}[v=g], \tag{4}$$

where $\sigma_\text{obs}=$`noiseStd` and $p_\text{err}=$`packetErrorRate` are the
condition parameters. The $\sin(\pi\theta)$ shape makes drift vanish at both
waypoints on through-edges (the vessel is "pinned" to nodes) and lets residual
offset persist at a berth, modelling imperfect station-keeping.

#### 3.2.4 Obstacles, IALA channel-keeping, cross-track error, docking

A seeded obstacle field $\mathcal{O}=\{(\mathbf{o}_j,r_j)\}$, $r_j=22$, is placed
near edges. A **collision** at sub-step $k$ occurs iff the hull enters a disc,

$$ \mathrm{coll}_k=\mathbb{1}\!\Big[\min_j \lVert\mathbf{z}(\theta_k)-\mathbf{o}_j\rVert_2\le r_j\Big]. \tag{5}$$

Let $\mathrm{seg}(\mathbf z,\mathcal P)$ be the distance from a pose to the nearest
planned segment. The per-step **cross-track error** sample and the episode mean on
the $M$ recorded sub-steps are

$$ \mathrm{CTE}_k=\mathrm{seg}\big(\mathbf{z}(\theta_k),\mathcal{P}\big),\qquad
   \overline{\mathrm{CTE}}=\frac{1}{M}\sum_{k} \mathrm{CTE}_k. \tag{6}$$

For **IALA channel-keeping** we use the signed side of a segment,
$\mathrm{sgn\text{-}side}(\mathbf z,u,v)=(x_v\!-\!x_u)(Y\!-\!y_u)-(y_v\!-\!y_u)(X\!-\!x_u)$.
A lateral buoy $b$ within the gate ($\le 60$ px of the segment) is **violated** when
the vessel passes on the buoy's side *and farther out* than the buoy, i.e. with
perpendicular offsets $d^\perp_{\text{vessel}}>d^\perp_b$ on the same side; the
episode count $\mathrm{IALA}=\sum_b \mathbb{1}[\text{violated}_b]$ (each buoy once).
On arrival the **docking accuracy** is the terminal berth miss distance
$\delta=\lVert\mathbf{z}_T-\mathbf{p}_g\rVert_2$.

#### 3.2.5 Observation and sensing/communication model

The *true* observation $\mathbf{o}^\text{true}_t\in\mathbb{R}^{28}$ concatenates a
9-D navigational block, a $4\times4$ action-slot block, and a 3-D safety block:

$$ \mathbf{o}^{\text{true}}_t=\Big[\tfrac{X_t}{W},\tfrac{Y_t}{H},\tfrac{x_g}{W},\tfrac{y_g}{H},
   \tfrac{x_g-X_t}{W},\tfrac{y_g-Y_t}{H},\tfrac{d_g}{1500},\tfrac{\ell(c_t)}{2},\tfrac{t}{H}\;\big\Vert\;
   \{\mathbb{1}_a,\hat c_a,\mathrm{dif}_a,\beta_a\}_{a=0}^{3}\;\big\Vert\;
   \widehat{\mathrm{CTE}},\widehat{\mathrm{obs}},\lambda_t\Big], \tag{7}$$

where $d_g=\lVert\mathbf z_t-\mathbf p_g\rVert_2$, $\ell(\cdot)$ encodes the lane,
$\hat c_a=\min(1,\mathrm{nav}(e_a)/40)$, $\beta_a=\mathbb{1}[\text{edge }a\text{ reduces goal distance}]$,
and $\lambda_t$ flags a collision on the previous step. The agent never sees ground
truth: a **Gaussian sensor model** corrupts every non-indicator entry,

$$ \mathbf{o}^\text{sens}_t = \mathbf{o}^\text{true}_t + \sigma_\text{obs}\,\boldsymbol{\eta}_t,
   \qquad \boldsymbol{\eta}_t\sim\mathcal{N}(\mathbf 0,\mathbf I), \tag{8}$$

followed by a **packet-loss channel** that, with probability $p_\text{err}$, drops
the frame and re-delivers the last received observation (stale hold):

$$ \mathbf{o}_t=\begin{cases}\mathbf{o}_{t-1}, & \text{w.p. } p_\text{err}\ \ (\text{dropped frame}),\\[2pt]
   \mathbf{o}^\text{sens}_t, & \text{w.p. } 1-p_\text{err}.\end{cases} \tag{9}$$

The three evaluated conditions are $(\sigma_\text{obs},p_\text{err})\in\{(0,0),\,(0.1,0.2),\,(0.25,0.4)\}$
for clean / mid / harsh.

#### 3.2.6 Reward

The one-step reward aggregates a step penalty, a cost-shaped movement term, event
penalties, and terminal bonuses:

$$ R_t = -0.2\;\underbrace{-5\,\mathbb{1}[\text{invalid}]}_{\text{illegal slot}}
   \;\underbrace{-0.1\,\mathrm{nav}(e_t)}_{\text{movement}}
   \;\underbrace{-\,\mathbb{1}[\text{revisit}]}_{-1}
   \;\underbrace{-\,25\,\mathrm{coll}_t}_{\text{collision}}
   \;-\,0.005\,\overline{\mathrm{CTE}} \;+\; R^\text{term}_t, \tag{10}$$

$$ R^\text{term}_t=\begin{cases}
   100 + 20\max\!\big(0,\,1-\delta/40\big), & c_t=g\ (\text{berth, docking bonus}),\\
   -30, & t=H,\ c_t\neq g\ (\text{timeout}),\\
   0, & \text{otherwise.}\end{cases} \tag{11}$$

The agent maximises the expected discounted return
$\mathbb{E}\!\big[\sum_{t=0}^{H-1}\gamma^{t}R_t\big]$.

#### 3.2.7 Value-based agents

All four agents learn an action-value function $Q_\phi(\mathbf o,a)$ with an MLP
torso ($28\!\to\!128\!\to\!128$, ReLU) and target network $Q_{\phi^-}$ synchronised
every $1000$ steps. Transitions are stored in a replay buffer and sampled in
minibatches of $64$; training uses Adam ($\eta=10^{-3}$) on the **Huber loss**
$\mathcal{L}_\kappa$ with $\kappa=1$. The temporal-difference target differs by
variant:

$$ y_t=r_t+\gamma(1-\mathrm{done})\cdot
   \begin{cases}
   \displaystyle\max_{a'} Q_{\phi^-}(\mathbf o_{t+1},a'), & \text{DQN / Dueling-DQN},\\[6pt]
   Q_{\phi^-}\!\big(\mathbf o_{t+1},\,\arg\max_{a'}Q_{\phi}(\mathbf o_{t+1},a')\big), & \text{Double / Dueling-Double}.
   \end{cases} \tag{12}$$

$$ \delta_t=Q_\phi(\mathbf o_t,a_t)-y_t,\qquad
   \mathcal{L}_\kappa(\delta_t)=\begin{cases}\tfrac12\delta_t^2, & |\delta_t|\le\kappa,\\[2pt]
   \kappa\big(|\delta_t|-\tfrac12\kappa\big), & |\delta_t|>\kappa.\end{cases} \tag{13}$$

The **dueling** variants factor the value through a state value and a
mean-centred advantage,

$$ Q_\phi(\mathbf o,a)=V_\phi(\mathbf o)+\Big(A_\phi(\mathbf o,a)-\tfrac{1}{|\mathcal A|}\sum_{a'}A_\phi(\mathbf o,a')\Big). \tag{14}$$

Exploration during training is $\varepsilon$-greedy with $\varepsilon$ annealed
linearly from $1.0$ to $0.05$ over the first $20\%$ of the step budget; evaluation
is greedy ($\varepsilon=0$).

#### 3.2.8 Inter-vessel geometry (extensions d, e)

For the two-vessel scenarios (§3.5–3.6), hull separation is evaluated continuously
on the $K=6$ sub-steps between the vessels $A,B$:
$s_k=\lVert\mathbf z^A(\theta_k)-\mathbf z^B(\theta_k)\rVert_2$. A step registers a
**collision** if $\min_k s_k\le R_\text{coll}=26$ px and a **near-miss** if
$R_\text{coll}<\min_k s_k\le R_\text{near}=60$ px (hysteretic, counted once per
encounter), and the episode **closest-point-of-approach** is
$\mathrm{CPA}=\min_t\min_k s_k$. A COLREGs-style **role** is assigned from the
relative bearing $\beta$ (bearing to the other vessel minus own heading) and the
heading difference $\Delta h$: *head-on* if $|\,|\Delta h|-\pi\,|<\pi/8$ and
$|\beta|<\pi/8$; *give-way* if $0<\beta<\pi/2$; *stand-on* if $-\pi/2<\beta<0$.

#### 3.2.9 Statistical estimators

The unit of inference is the **per-seed scalar** $y_{ijk}$ for factor level $i$,
condition $j$, seed $k$ (the mean of a metric over a cell's evaluation episodes;
CTE/docking averaged over *successful* episodes only). We fit a two-way
fixed-effects model with the seed as a block,

$$ y_{ijk}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+s_k+\epsilon_{ijk}, \tag{15}$$

and report, from Type-II sums of squares, the **partial eta-squared** effect size
$\eta^2_p=\mathrm{SS}_\text{effect}/(\mathrm{SS}_\text{effect}+\mathrm{SS}_\text{resid})$.
Robustness is the clean$\to$harsh paired contrast per level, tested with the
Wilcoxon signed-rank statistic, Holm-corrected across metrics, with the paired
effect size $d_z=\overline{D}/\mathrm{sd}(D)$, $D_k=y^\text{harsh}_k-y^\text{clean}_k$.
As a conservative complement (§3.7) we refit each metric as a linear mixed model
with the seed as a **random** effect,
$y=\mathbf{X}\boldsymbol\gamma+u_k+\epsilon,\;u_k\sim\mathcal N(0,\sigma^2_s)$,
reporting the intraclass correlation
$\mathrm{ICC}=\sigma^2_s/(\sigma^2_s+\sigma^2_\epsilon)$.

### 3.3 Agents and training

Four value-based DRL agents instantiate the $Q$-learning template of §3.2.7 —
**DQN**, **Double DQN** (decoupled selection/evaluation, Eq. 12), **Dueling DQN**
(value+advantage streams, Eq. 14), and **Dueling Double DQN** — sharing the MLP
torso, target network, Huber loss and Adam optimizer above. Each is trained for 150
episodes and then evaluated greedily for 30 episodes.

### 3.4 Metrics, evaluation, and statistics

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
seed`, agent RNG seeded separately) defined in §3.2.9. We report means with 95%
CIs, run the two-way factorial ANOVA of Eq. (15) (algorithm × condition + seed;
Type-II SS; partial $\eta^2_p$), and test clean→harsh robustness per algorithm with
paired Wilcoxon signed-rank tests, Holm correction, and Cohen's $d_z$ (§3.2.9). The
**4 × 3 × 8 = 96 cells / 2,880 episodes** index computational replication; the
effective sample size is the 8 seeds.

### 3.5 Extension (d): multi-vessel interaction

To test behaviour when the port is **shared**, `js/environmentBintuluMulti.js`
(`MultiVesselBintulu`) instantiates **two independent `BintuluEnv` vessels** on the
same chart, each with its own policy, its own mission (decorrelated RNG streams so
the two receive different start/berth assignments), and a 3-value **partner
channel** appended to each observation (sensed relative bearing `dx, dy` and
`range` to the other vessel, obs 28 → 31). The partner channel is corrupted by the
*same* Gaussian noise as the rest of the observation, so degradation also blinds a
vessel to its neighbour. Inter-vessel geometry is evaluated continuously: at every
one of the 6 kinematic sub-steps the hull–hull separation is measured, a
**collision** is counted below 26 px and a **near-miss** below 60 px (hysteretic,
so one encounter is counted once), the running **minimum closest-point-of-approach
(CPA)** is tracked, and a COLREGs-style role (head-on / give-way / stand-on) is
assigned from the relative bearing and heading difference.

We cross three **pairings** — `Rule/Rule`, `DQN/Rule`, `DQN/DQN` — with the three
degradation conditions and 6 shared seeds (**3 × 3 × 6 = 54 cells**, 30 eval
episodes per cell × 2 vessels). The rule-based controller is the project's
COLREGs-aware channel-follower (`js/rulebased.js`), reused unchanged. The pairing
factor isolates **the control architecture of the interaction partner**: does
pairing a learner with a classical controller, or two learners together, change
who completes and who collides?

### 3.6 Extension (e): operational realism

Real port traffic is not a single one-way transit. `js/environmentBintuluOps.js`
adds two scenarios on the chart:

- **Round trip (`TwoPhaseBintulu`).** A single vessel must transit *inbound* from
  the sea entrance to the inner berth of an access channel, dock, then **retarget**
  and transit *outbound* back to the entrance. Success is the **full cycle**; a
  phase flag and a cycle-progress scalar are appended to the observation
  (obs 28 → 30) and the step budget is doubled (120). This probes whether a stack
  that can reach a berth can also *complete an operation*.
- **Two-way traffic (`TwoWayBintulu`).** An **inbound** and an **outbound** vessel
  share one access channel in opposing directions, producing genuine head-on
  encounters on a single surveyed centreline. The inter-vessel collision / near-miss
  / CPA / COLREGs-role machinery is exactly the geometry formalised in §3.2.8.

Both scenarios enforce an **operational invariant that the committed core
environment deliberately omits**: a working port keeps its surveyed channels
navigable. The core single-leg study averages over many missions, so a seed that
happens to spawn a static obstacle across a berth's final approach edge is simply
absorbed into the (already low) mean success; but a *round-trip* or *opposed-traffic*
mission that **must** reach one specific berth would then be impossible, confounding
the operational signal with a cartographic accident. We therefore nudge any
obstacle that would fully seal a lane centreline segment just clear of that corridor
(perpendicular to the segment, preserving its side); obstacles in open water are
untouched, so drift under degradation can still carry a vessel into them. This
invariant is applied to the operational and multi-vessel environments **only** — the
committed core environment and its published results are left byte-identical.

Scenarios run as **agent {Rule, DQN} × condition × 6 seeds** (round trip, 36 cells)
and **pairing {Rule/Rule, DQN/Rule} × condition × 6 seeds** (two-way, 36 cells).

### 3.7 Extension (f): a seed-level mixed-effects treatment

The factorial ANOVAs above enter the shared seed as a fixed block. As a
complementary, effect-size-first check we refit each key metric as a **linear
mixed-effects model** that treats the seed as a *random* effect,
$y \sim \text{(design factors, fixed)} + (1\mid\text{seed})$
(`experiments/mixed_effects_bpan.py`, REML). This (i) propagates seed-to-seed
variability into the fixed-effect inference honestly rather than conditioning it
away, and (ii) yields the **intraclass correlation**
$\text{ICC}=\sigma^2_{\text{seed}}/(\sigma^2_{\text{seed}}+\sigma^2_{\text{resid}})$,
the share of residual outcome variance attributable to the seed alone. We fit the
model for the core, multi-vessel, round-trip and two-way sub-studies and report the
variance partition plus whether the design-factor fixed effects remain significant
under the random-seed model.

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

### 4.4 Qualitative behaviour (animations)

To make the learned behaviour concrete, we render trained-DQN greedy episodes
directly over the Bintulu chart. Because animations cannot be embedded in a static
manuscript, each scenario is presented as a **three-panel still-frame plate** whose
panels are selected *automatically from the trajectory data* at the analytically
meaningful instants (peak off-channel deviation, first collision, minimum
closest-point-of-approach); the full-resolution animated versions are on the
project site. In **clean** conditions (Figure 3a) the vessel tracks the planned
channel and berths cleanly; under **harsh** sensing/comms (Figure 3b) the realised
track drifts off the buoyed centreline (panel b, cross-track error annotated),
crosses channel markers (IALA violations), and grazes an obstacle (panel c,
collision) — the visual counterpart of §4.2.

![Clean-conditions key frames: departure, mid-channel transit, berthing.](results_bintulu/figures/figBPAN_keyframes_clean.png)
**Figure 3a.** Clean conditions, key frames auto-selected from the trajectory:
(a) departure from the sea entrance, (b) mid-channel transit along the buoyed
North channel, (c) berthing at the harbour.

![Harsh-degradation key frames: on-centreline, peak drift (IALA), collision.](results_bintulu/figures/figBPAN_keyframes_harsh.png)
**Figure 3b.** Harsh degradation, key frames auto-selected: (a) the vessel still
on the centreline, (b) the instant of **peak off-channel drift** (cross-track error
annotated — the IALA-departure moment), (c) the frame of the **obstacle collision**
(red burst). The still plate makes the §4.2 safety collapse legible in a form that
can be printed in the paper.

### 4.5 Improving the DQN — the navigation prior is the dominant lever

Given that algorithm choice does not help ([§4.1](#rq1)), we asked **what does**.
We evaluated two principled, generic improvements as explicit arms (4 arms × 3
conditions × 6 seeds = 72 cells / 2,160 episodes), each isolated against the
baseline:

- **Chart prior** — append to the observation a per-outgoing-edge flag indicating
  the next segment of the Dijkstra charted route (a *structured navigation prior*;
  the lever found decisive on the Synthetic Port chart). Obs 28 → 32.
- **Reward shaping** — potential-based shaping toward the goal,
  $F=\gamma\,\phi(s')-\phi(s)$ with $\phi=-\text{dist}_{\text{goal}}$, which is
  policy-invariant (Ng et al., 1999) and so can only speed/stabilise learning.

**Results (navigation success rate, mean over 6 seeds):**

| Arm | clean | mid | harsh |
|---|---|---|---|
| Baseline DQN | 15.0% | 17.2% | 25.0% |
| + reward shaping | 20.0% | 36.7% | 45.6% |
| + chart prior | 35.6% | 42.8% | 56.1% |
| **+ chart + shaping** | **30.0%** | **50.0%** | **65.0%** |

A two-way factorial ANOVA makes the **improvement arm the dominant factor** on
success (partial $\eta^2 = 0.41$, $p < 0.001$) — larger than the degradation
condition ($\eta^2 = 0.32$) — with a small, non-significant arm×condition
interaction. The chart prior alone roughly **doubles to triples** baseline success
at every condition (clean 15→36%, harsh 25→56%; Cohen's $d_z$ up to 2.6 at harsh);
reward shaping helps on its own (harsh 25→46%); and the two **combine** for the
best harsh-condition success (**65%**). At n = 6 the per-condition paired contrasts
are power-limited after Holm correction (adjusted $p \approx 0.09$–0.20), so — as
elsewhere — the pooled ANOVA and the large effect sizes carry the inference.

![Improving the DQN on Bintulu: the chart prior is the dominant lever.](results_bintulu/figures/figBPAN_improvements.png)
**Figure 4.** Navigation success by improvement arm across conditions (n = 6
seeds, 95% CI). The baseline (grey) sits far below the three improved arms;
appending the charted-route prior (blue) is the single most effective change, and
combining it with potential-based shaping (purple) is best under harsh degradation.

> [!INSIGHT]
> **The way to improve DQN on the Bintulu approaches is not a better value-function
> variant but a better *input*: give the policy the charted route.** This mirrors
> the broader finding that structured navigation priors — not algorithmic
> refinement — are the operative lever, and it argues for hybrid designs that feed
> the port's Dijkstra/charted plan into the learned controller.

### 4.6 Multi-vessel: the interaction partner dominates, not the algorithm (ext. d)

When two vessels share the Bintulu channels, **the pairing of control
architectures — not the degradation condition — governs who completes and who
collides.** Mean outcomes (6 seeds; the per-seed scalar pools both vessels of a
cell):

| Pairing | Success clean→harsh | Inter-vessel collisions/ep | Near-misses/ep | Min CPA (px) |
|---|---|---|---|---|
| Rule / Rule | 84% → 84% | 1.24 → 1.08 | 0.00 → 0.16 | 73 |
| DQN / Rule | 55% → 59% | 0.31 → 0.73 | 0.51 → 0.30 | 83–88 |
| DQN / DQN | 32% → 37% | 0.61 → **1.89** | 0.49 → 0.54 | 59–93 |

The factorial ANOVA is unambiguous: **pairing explains the large majority of the
variance** — success partial η² = **0.74** (p < 10⁻¹¹), near-misses η² = **0.59**
(p < 10⁻⁷), vessel-collision rate η² = **0.60** (p < 10⁻⁸) — while the degradation
**condition** is small-to-moderate on these inter-vessel outcomes
(success η² = 0.01, n.s.; near-miss η² < 0.01, n.s.) and the pairing×condition
interaction is non-significant for success and near-misses. The one place
degradation bites hard is the *static-obstacle* collision rate (condition
η² = 0.70) and cross-track error (condition η² = 0.91) — i.e. the single-vessel
degradation story of §4.2 persists underneath, but the *inter-vessel* story is
dominated by who the two controllers are.

The pattern is interpretable rather than a simple ranking. The classical
`Rule/Rule` pair **completes** most missions (84%) because each vessel rigidly
follows its charted channel — but for exactly that reason it **collides most**
(1.24/ep): neither vessel yields. Pairing a learner with the rule-follower
(`DQN/Rule`) **halves–quarters** the collision rate (0.31–0.73/ep) and keeps the
largest separation (CPA ≈ 83–88 px) at the cost of lower completion (55–59%), and
two learners together (`DQN/DQN`) complete least and, under harsh degradation when
the partner channel is noisiest, collide **most** (1.89/ep) — the one condition
where degradation and the pairing compound.

![Multi-vessel variance explained: the interaction pairing dominates.](results_bintulu/multi/figures/figMULTI_anova_eta2.png)
**Figure 5.** Partial η² per factor for the multi-vessel study. The **pairing**
(blue) dominates success, near-misses and inter-vessel collisions; the degradation
**condition** (orange) and the interaction are comparatively small on the
inter-vessel outcomes.

![Multi-vessel success by pairing and condition.](results_bintulu/multi/figures/figMULTI_success.png)
**Figure 6.** Transit success by pairing across conditions. The three pairings
separate cleanly and are essentially flat across degradation — the gap between them
is the interaction structure, not the noise.

![Multi-vessel key frames: approach, closest encounter (min CPA), after passing.](results_bintulu/multi/figures/figMULTI_keyframes.png)
**Figure 6b.** Two independent DQN vessels sharing the port, key frames
auto-selected from the trajectory at the **minimum closest-point-of-approach**:
(a) the two vessels approach on their separate missions, (b) the closest encounter
(the inter-vessel link and separation are annotated; a red ring marks a contact),
(c) the vessels after passing. This is the inter-vessel interaction whose *pairing*
dominates the statistics above.

> [!INSIGHT]
> **In shared water, the operative design variable is the *composition of
> controllers*, not the DRL variant.** A classical follower completes but will not
> give way; a learned vessel gives way (fewer collisions, larger CPA) but completes
> less; two learners interact worst under heavy degradation. Safe multi-vessel
> operation is a question of *who shares the channel*, decided a level above the
> value-function family.

### 4.7 Operational realism: completing an operation is a control-architecture problem; a shared channel makes collision structural (ext. e)

**Round trip.** Requiring the *full* inbound-dock-outbound cycle exposes a sharp
split by control architecture. The classical follower completes the whole round
trip in **100%** of episodes at every condition; the naive DQN **docks** (reaches
the inner berth) much of the time but **closes the full cycle only ≈16–19%** of the
time, because retargeting for the return leg is a second composite task it was never
reliably trained for:

| Agent | Leg-1 dock (clean) | Full cycle clean | Full cycle mid | Full cycle harsh |
|---|---|---|---|---|
| Rule-based | 100% | **100%** | **100%** | **100%** |
| DQN | 100% | 17% | 19% | 16% |

Here the ANOVA cleanly separates two levels of the stack onto two different
outcomes: **agent (control architecture) dominates full-cycle success**
(partial η² = **0.90**, p < 10⁻¹⁰; condition negligible, η² < 0.01), while
**degradation dominates the static-collision rate** (condition η² = **0.80**,
p < 10⁻⁷; agent η² ≈ 0.00). *Whether you finish the operation* is set by the
controller; *how safely you move while doing it* is set by the sensing/comms
condition.

**Two-way traffic.** When an inbound and an outbound vessel are forced down one
surveyed centreline, a non-cooperative controller **cannot** avoid the oncoming
vessel — the collision is **structural**:

| Pairing | Success clean→harsh | Vessel collisions/ep | Head-on events/ep clean→harsh |
|---|---|---|---|
| Rule / Rule | 100% → 100% | 1.00 (all conditions) | 1.00 → 1.00 |
| DQN / Rule | 90% → 80% | ≈1.00–1.06 | 0.84 → 0.75 |

The pairing significantly governs **transit success** (η² = 0.44, p < 10⁻³) and
**head-on event rate** (η² = 0.37, p < 10⁻³): the learned inbound vessel reduces
head-on *events* (1.00 → 0.75) by learning to **hold** (temporal give-way, letting
the oncoming vessel pass) and takes a modest success hit under degradation
(100% → 80%). But the **collision count itself is ≈1/episode regardless of pairing
or condition** (pairing η² = 0.09, n.s.; seed ICC ≈ 0, §4.8) — on a single shared
centreline there is no lateral room to pass, so two opposing vessels that both
insist on transiting must meet. This is a faithful **null/structural result**: the
lever that removes two-way collisions is not the algorithm or the sensors but the
*mission geometry* (a traffic-separation scheme / one-way scheduling), a level
above everything measured here.

![Two-way head-on event rate by pairing.](results_bintulu/ops/figures/figOPS_twoway_headon.png)
**Figure 7.** Head-on encounter events per episode. The learned inbound vessel
(blue) cuts head-on events below the rule/rule structural baseline (red ≈ 1.0),
evidence of emergent temporal give-way; the residual inter-vessel collision,
however, is structural on a single centreline.

![Two-way key frames: opposing approach, head-on encounter (min CPA), outcome.](results_bintulu/ops/figures/figOPS_twoway_keyframes.png)
**Figure 7b.** Opposing traffic in one access channel, key frames auto-selected at
the **head-on minimum closest-point-of-approach**: (a) the inbound (blue, DQN) and
outbound (orange, rule-based) vessels approach from opposite ends of the same
centreline, (b) the head-on encounter (red ring = contact), (c) the outcome after
the meeting. The plate makes the *structural* nature of the single-channel
collision — the §4.7 null — directly visible.

![Round-trip full-cycle success by agent.](results_bintulu/ops/figures/figOPS_twophase_full.png)
**Figure 8.** Full round-trip success. The classical follower closes the cycle
every time; the naive learner docks but rarely returns — a control-architecture
gap, flat across degradation.

> [!INSIGHT]
> **Operational completeness and shared-channel safety sit at *different* levels of
> the stack.** Finishing a round trip is a control-architecture capability (the
> classical planner has it, the naive learner does not); avoiding a head-on in a
> single channel is a mission-geometry problem no controller in our set can design
> away. Neither is an algorithm-selection problem.

### 4.8 Mixed-effects confirmation and the role of the seed (ext. f)

Refitting each metric with the seed as a **random** effect (REML,
$y\sim\text{fixed}+(1\mid\text{seed})$) reproduces every ANOVA conclusion and adds
a variance partition. The design-factor fixed effects behave exactly as the
factor-hierarchy story predicts: in the **core** study **0 of 9** algorithm-level
fixed effects reach p < 0.05 (the algorithm is immaterial), whereas **4 of 6**
pairing effects (multi-vessel), the **agent** effect on round-trip success, and
**2 of 3** pairing effects (two-way) are significant. The intraclass correlations
show the seed carries real but secondary weight — moderate for success-type metrics
(core success ICC = 0.45; multi-vessel success ICC = 0.43; round-trip collisions
ICC = 0.28) and **essentially zero where the outcome is structurally fixed**
(two-way vessel collisions ICC = 0.00, head-on events ICC = 0.03) — confirming that
the two-way collision is a deterministic property of the geometry, not seed noise.

| Sub-study | Metric | ICC (seed) | Fixed-effect verdict |
|---|---|---|---|
| Core | success rate | 0.45 | 0/9 algorithm effects significant |
| Multi-vessel | success rate | 0.43 | pairing significant |
| Multi-vessel | vessel collisions/ep | 0.13 | pairing significant |
| Round trip | full-cycle success | 0.19 | agent significant |
| Two-way | vessel collisions/ep | 0.00 | structural; no factor moves it |
| Two-way | head-on events/ep | 0.03 | pairing significant |

> [!INSIGHT]
> **The conclusions survive a stricter, random-seed model.** Treating the seed as a
> random effect — the more conservative choice — leaves the algorithm non-significant
> and the higher stack levels (pairing, control architecture, mission geometry)
> significant, and shows the seed itself explains at most a moderate, never a
> dominant, share of variance.

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
- **To improve the agent, improve its input, not its value-function.** The
  improvement study ([§4.5](#rq1)) shows that appending the port's charted route to
  the observation — a structured navigation prior — is the dominant lever (arm
  partial $\eta^2 = 0.41$), roughly doubling–tripling berthing success, and that it
  combines with policy-invariant reward shaping. The actionable recommendation for
  a Bintulu autonomy stack is therefore a **hybrid** design: feed the Dijkstra /
  charted plan into the learned controller rather than searching for a better DQN
  variant.

These conclusions mirror, on a real chart, what the author's broader Synthetic Port
programme found on a constructed benchmark — strengthening the case that the
pattern is a property of the task class, not of one particular synthetic map.

### 5.1 Synthesis: a design-stack factor hierarchy

Taken together, the four study blocks reframe the design question from *"which DRL
variant wins?"* to *"what **level of the design stack** confers robust, complete,
multi-vessel behaviour?"* The measured partial η² values across all BPAN
experiments arrange the levers into a clear hierarchy (Figure H):

1. **Mission & interaction structure** — the strongest lever (η² ≈ 0.37–0.90):
   the round-trip composite makes completion a different problem (agent η² = 0.90),
   the vessel pairing governs shared-water success and collisions
   (η² ≈ 0.59–0.74), and the two-way channel geometry makes head-on collision
   structural (no measured factor removes it).
2. **Control architecture & navigation prior** — strong (η² ≈ 0.30–0.45): rule-based
   vs learned control decides who completes a round trip and who gives way; the
   charted-route prior is the dominant *within-DQN* lever (arm η² = 0.41, §4.5).
3. **Perception & communications** — metric-dependent (η² ≈ 0.06–0.80): degradation
   dominates *single-vessel* safety/compliance (IALA, static collisions, CTE:
   η² ≈ 0.6–0.9) but is comparatively minor for *inter-vessel* success and
   near-misses.
4. **Learning algorithm (the DQN variant)** — negligible (η² ≈ 0.01–0.05, never
   significant; 0/9 fixed effects under the mixed model).

![The BPAN design-stack factor hierarchy.](results_bintulu/figures/figBPAN_factor_hierarchy.png)
**Figure H.** The design stack ranked by measured variance explained. Higher levels
— mission/interaction framing and control architecture — dominate robust, complete,
multi-vessel behaviour; the perception/communication condition matters chiefly for
single-vessel safety; the choice of value-based algorithm is immaterial. The
actionable ordering for a Bintulu autonomy programme is therefore: get the
**mission framing** and **traffic scheme** right, invest in **control architecture
and the chart prior**, harden **perception/comms** for safety — and only then, if at
all, tune the DRL algorithm.

> [!INSIGHT]
> **The factor hierarchy is the headline.** Across one-way transit, improvement
> arms, shared water, and real operations, robustness is conferred by the *upper*
> levels of the design stack. "Which DQN variant" sits at the bottom and moves
> nothing; mission framing, control architecture, the navigation prior and the
> sensing/comms budget are where autonomy is won or lost.

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
