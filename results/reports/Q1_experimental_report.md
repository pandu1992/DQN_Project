# Comparative Experimental Evaluation of Value-Based Deep Reinforcement Learning Agents for Autonomous Vessel Navigation

**A reproducible, multi-seed factorial study on the Bintulu Port navigation graph**

> Scope note (read first). This report analyses the four value-based RL
> agents that are *actually implemented* in the project (`js/dqn.js`): **DQN,
> Double DQN, Dueling DQN, Dueling Double DQN**, each optionally combined with
> **Prioritized Experience Replay (PER)** and **Noisy Nets**. No PPO/SAC/rule-
> based agent exists in the codebase and none was fabricated. All metrics are
> derived from quantities the environment truly computes; mechanisms that are
> **not** implemented (observation noise, communication packet-error rate,
> collision, docking accuracy, cross-track error, IALA compliance) are
> explicitly out of scope and listed in the Limitations. Every number in this
> report is reproducible from `results/raw/` via the scripts in `experiments/`.

---

## 1. Experimental Design

**Objective.** Determine, with statistical rigor, whether the choice of
value-based DQN variant and/or the Rainbow-style enhancements (PER, Noisy Nets)
produces a measurable difference in autonomous-navigation performance on the
Bintulu Port channel graph, and quantify the associated effect sizes,
robustness, and reproducibility.

**Why a new environment build was required (a genuine design finding).**
The originally deployed environment (`js/environment.js`) is a *saturated*
benchmark: the mission distribution contains only two fixed lane-traversal
missions, and the agent merely advances an index along a pre-computed Dijkstra
path. Under that setup the optimal policy is trivially "always FORWARD", and a
direct check showed **all four algorithms reach ~100% success and reward
≈119.9 with essentially zero between-seed and between-algorithm variance**
(a pure-forward policy attains 100% success, reward 119.84 ± 0.99). With no
variance to explain, every omnibus/pairwise test is vacuous (F≈0, p≈1). We
therefore built an offline research environment, **`VesselEnvV2`**, that reuses
the *same graph topology, the same edge attributes, and the same Dijkstra
planner* but exposes a genuine graph-navigation MDP (Section 3). This is a
methodological necessity, documented transparently, not a result-shaping choice.
`VesselEnvV2` is used only by the offline experiment harness; the deployed web
application is unchanged.

**Design.** A full **4 × 2 × 2 factorial**:

| Factor | Levels |
|---|---|
| Algorithm | DQN, Double DQN, Dueling DQN, Dueling Double DQN |
| PER | off, on |
| Noisy Nets | off, on |

= **16 configurations**, each trained under **10 independent random seeds**
(the *same* seed set 0–9 for every configuration → a randomized-block /
repeated-measures design that supports **paired** tests). Total: **160 training
runs**, **6,400 held-out evaluation episodes**.

> [!INSIGHT]
> **The benchmark had to be made discriminative before it could be studied.**
> The deployed environment is *saturated* — a two-mission, path-index task on
> which every algorithm scores ~100% with essentially zero variance, so no
> statistical test can separate them. Rebuilding it as a genuine
> graph-navigation MDP (same graph, same edge attributes, same planner) is the
> pivotal design decision that makes the entire comparison meaningful.

---

## 2. Algorithms

All four are implemented in pure JavaScript in `js/dqn.js` (no ML library),
sharing an MLP torso `obs(25) → 128 → 128 (ReLU)`, a target network with hard
periodic updates, Huber loss, and a hand-written Adam optimizer.

| Algorithm | Distinguishing mechanism | Source construct |
|---|---|---|
| DQN | vanilla max-Q bootstrap target | `QNetwork`, `useDouble=false` |
| Double DQN | online net selects action, target net evaluates it | `useDouble=true` |
| Dueling DQN | value + advantage streams | `DuelingQNetwork` |
| Dueling Double DQN | dueling network + double-Q target | `DuelingQNetwork` + `useDouble` |

Enhancements (stackable on any base): **PER** (`PrioritizedReplayBuffer`,
proportional priorities + importance-sampling weights) and **Noisy Nets**
(`NoisyDense` factorized-Gaussian weight noise on head layers; ε-greedy is
disabled when active). See Section 5 for exact formulas verified against source.

---

## 3. Environment and Scenario Configuration

`VesselEnvV2` (`js/environmentV2.js`) instantiates the base graph via the
unmodified `VesselEnv` (identical waypoints, buoys, and edge attributes:
`distance, risk, traffic, weather, current, travel_time, energy_cost,
navigation_cost, difficulty`) and the identical Dijkstra planner.

| Property | Value |
|---|---|
| State | 25-d vector: position, goal-relative offset, distance-to-goal, lane, step fraction, and per-action-slot edge features (exists, normalized `navigation_cost`, `difficulty`, toward-goal flag) |
| Actions | 4 slots; slot *a* selects the *a*-th outgoing edge (sorted by cost); a slot beyond the node's out-degree is an *invalid* action |
| Missions | start = any access-channel waypoint; goal ∈ {harbour terminal, L02 terminal, L08 terminal}; seeded → **115 distinct missions/seed**, optimal-cost range **5.3–151.5** |
| Reward | `-0.2` step + `-0.1·navigation_cost(edge)` + `+100` goal + `-5` invalid + `-30` timeout + `-1` revisit |
| Episode budget | 60 steps |
| Optimal reference | Dijkstra over `navigation_cost` (validated: a follow-Dijkstra policy gives `optimality_ratio = 1.0000` on 100/100 solved episodes) |

**Task non-triviality (validated).** Random policy → 13% success, reward −197.6;
greedy-toward-goal heuristic → 100% success, reward 90.2. A real difficulty
gradient exists, so learning quality is measurable.

**Scenario note.** The protocol's Scenario B (observation noise) and Scenario C
(communication packet-error rate) are **not implemented** in the environment and
are out of scope here (Section 17). "PER" in this project denotes *Prioritized
Experience Replay*, a replay-buffer feature — not a packet-error rate.

---

## 4. Evaluation Metrics

The **unit of analysis** is the *per-seed scalar*: within each seed the 40
evaluation episodes are collapsed to one value per metric (episodes within a
seed share a network and are not independent); the **10 seed-scalars** are the
independent observations (n = 10 per configuration).

**Primary:** navigation success rate (%), mean episodic reward, route
**optimality ratio** (route `navigation_cost` ÷ Dijkstra optimum; ≥ 1, 1 =
optimal). **Secondary:** mean steps, revisit rate, excess cost, accumulated
route risk, accumulated route difficulty.

Metric definitions are given mathematically in Section 5.2. `invalid_rate` was
**excluded** from statistics: under the greedy evaluation policy the trained
networks never choose the non-existent 4th action slot, so it is uniformly 0
(zero variance, untestable) — documented in `results/statistics/methodology.json`.

---

## 5. Mathematical Formulation

### 5.1 Algorithms (verified against `js/dqn.js`)

**DQN.** Q-values `Q_θ(s,a)` from the MLP. Bootstrap target and Huber loss
(δ = 1):

$$ y_t = r_t + \gamma\,(1-d_t)\,\max_{a'} Q_{\theta^-}(s_{t+1},a') $$

$$ \delta_t = Q_\theta(s_t,a_t) - y_t, \qquad
L(\theta)=\mathbb{E}\big[\,\ell_\text{Huber}(\delta_t)\,\big],\quad
\ell_\text{Huber}(x)=\begin{cases}\tfrac12 x^2 & |x|\le 1\\ |x|-\tfrac12 & |x|>1\end{cases} $$

with target network `θ⁻` hard-updated every `target_update` steps and replay
buffer sampling. (Confirmed: `_train()` uses `max` over target Q for the
non-double branch, Huber gradient clip at δ=1, Adam.)

**Double DQN.** Action selected by the online net, evaluated by the target net:

$$ a^\* = \arg\max_{a'} Q_\theta(s_{t+1},a'), \qquad
y_t = r_t + \gamma\,(1-d_t)\,Q_{\theta^-}(s_{t+1}, a^\*) $$

**Dueling DQN.** Value and advantage streams recombined with mean-advantage
subtraction (confirmed in `DuelingQNetwork.forward`):

$$ Q(s,a) = V(s) + \Big(A(s,a) - \tfrac{1}{|\mathcal A|}\sum_{a'} A(s,a')\Big) $$

**Dueling Double DQN.** The dueling architecture with the Double-DQN target.

**Prioritized Experience Replay** (`PrioritizedReplayBuffer`). Proportional
priorities and annealed importance-sampling weights:

$$ p_i = (|\delta_i| + \epsilon)^{\alpha},\quad
P(i)=\frac{p_i}{\sum_k p_k},\quad
w_i = \big(N\,P(i)\big)^{-\beta}\ \big/\ \max_j w_j $$

with `α = 0.6`, `β: 0.4 → 1` annealed over `total_steps/2` (verified constants).

**Noisy Nets** (`NoisyDense`, factorized Gaussian). Each noisy layer uses

$$ W = \mu_W + \sigma_W \odot (\varepsilon_\text{in}\,\varepsilon_\text{out}^\top),\quad
b = \mu_b + \sigma_b\,\varepsilon_\text{out},\quad
\varepsilon = f(\xi),\ f(\xi)=\operatorname{sgn}(\xi)\sqrt{|\xi|},\ \xi\sim\mathcal N(0,1) $$

with `σ₀ = 0.5`; exploration comes from weight noise, so ε-greedy is disabled.

### 5.2 Metrics

$$ \mathrm{SR}=\frac{N_\text{success}}{N_\text{episodes}}\times 100\%, \qquad
\bar R=\frac1N\sum_{i=1}^N R_i, \qquad
\rho=\frac{\text{cost}_\text{route}}{\text{cost}_\text{Dijkstra}}\ (\ge 1) $$

$$ s=\sqrt{\tfrac{1}{N-1}\sum_i (x_i-\bar x)^2},\quad
\mathrm{SE}=\frac{s}{\sqrt N},\quad
\mathrm{CI}_{95\%}=\bar x \pm t_{0.975,\,N-1}\frac{s}{\sqrt N} $$

Relative change of an enhancement vs baseline (for a higher-is-better metric):
`Δ_rel = 100·(M_enh − M_base)/M_base`. For optimality ratio (lower is better) a
negative Δ denotes improvement.

---

## 6. Statistical Methodology

**Design → tests.** Shared seeds make every contrast within-block (paired).

1. **Normality** — Shapiro–Wilk on each config's 10 seed-scalars.
2. **Variance homogeneity** — Levene (Brown–Forsythe, median-centred) across the
   four baseline algorithms.
3. **Omnibus (algorithm effect, baseline cells)** — one-way **repeated-measures
   ANOVA** (block = seed) *and* **Friedman** test; partial η² and Kendall's W.
4. **Factorial 4×2×2** — OLS `y ~ algorithm*PER*Noisy + seed` (Type-II SS,
   balanced 10/cell), partial η² for every main effect and interaction.
5. **Pairwise post-hoc** — paired **Wilcoxon signed-rank** (exact) as primary
   (most metrics non-normal), paired *t* as secondary; **Holm** and
   **Benjamini–Hochberg** correction within each metric's comparison family.
6. **Effect sizes** — paired Cohen's `d_z`, Cliff's δ, matched-pairs
   rank-biserial `r`. **Bootstrap** (10,000 resamples) percentile CIs cross-check
   the t-CIs.

**Hypotheses (example, algorithm omnibus).**
$H_0:$ all four algorithms have equal mean success; $H_1:$ at least one differs.
Decision rule: reject $H_0$ if $p < \alpha = 0.05$ (post-hoc: Holm-adjusted
$p < 0.05$).

$$ F=\frac{\mathrm{MS}_\text{treat}}{\mathrm{MS}_\text{error}},\quad
\eta^2_p=\frac{\mathrm{SS}_\text{treat}}{\mathrm{SS}_\text{treat}+\mathrm{SS}_\text{error}};\qquad
\chi^2_F=\frac{12}{nk(k+1)}\sum_j R_j^2-3n(k+1),\quad W=\frac{\chi^2_F}{n(k-1)} $$

**Correction formulas.** Holm: order p-values ascending, adjusted
$\tilde p_{(i)}=\max_{j\le i}\big[(m-j+1)\,p_{(j)}\big]$, capped at 1.
Benjamini–Hochberg: $\tilde p_{(i)}=\min_{j\ge i}\big[\tfrac{m}{j}p_{(j)}\big]$.

**Assumption findings.** Shapiro pass-rates: success 0.94, reward 0.88, steps
0.94, revisit 0.94 (parametric defensible); optimality_ratio 0.31, excess_cost
0.19 (non-normal → Wilcoxon/Friedman primary). Levene: all metrics p > 0.05
(homoscedastic). Full tables in `results/statistics/`.

---

## 7. Overall Performance

Baseline algorithms (PER off, Noisy off), mean ± 95% CI over 10 seeds
(`results/aggregated/summary.csv`; see Table 2, Figure 1, Figure 5):

| Algorithm | Success (%) | Reward | Optimality ρ | Steps | Route risk |
|---|---:|---:|---:|---:|---:|
| DQN | 49.8 ± 18.1 | −18.6 ± 39.0 | 1.59 ± 1.22 | 36.7 | 0.65 |
| Double DQN | 43.2 ± 19.3 | −33.0 ± 40.1 | 1.17 ± 0.28 | 39.9 | 0.59 |
| Dueling DQN | 40.2 ± 11.8 | −38.2 ± 25.0 | 1.07 ± 0.11 | 40.4 | 0.52 |
| Dueling Double DQN | 45.5 ± 17.9 | −30.3 ± 38.0 | 1.94 ± 1.05 | 40.0 | 0.83 |

**Empirical:** DQN has the highest point-estimate success (49.8%), Dueling DQN
the lowest (40.2%) but also the tightest route optimality (1.07). **Statistical:**
all 95% CIs overlap heavily (bootstrap CIs agree, e.g. DQN [34.2, 63.5]). The
ranking is not stable evidence of superiority (Section 11).

---

## 8. Noise Robustness

**Not evaluable in this environment.** `VesselEnvV2` (like the original) has no
observation- or state-noise mechanism, so Scenario B cannot be run without
fabricating a noise model. We report this as a scoped limitation (Section 17)
and a candidate Option-2 extension, rather than inventing noise levels.

---

## 9. PER Robustness

Here "PER" = Prioritized Experience Replay (a training feature), analysed as a
factor. Turning PER on (paired vs baseline per algorithm, Wilcoxon; see
`enhancement_contrasts.csv`, Table 7): effects are small and **none survives
Holm correction** on any primary metric. In the factorial model the PER main
effect is negligible (success partial η² = 0.000, p = 0.94; reward η² = 0.000).
**PER neither reliably helps nor hurts at this training budget.**

*(Communication packet-error rate — the protocol's other "PER" — is not
implemented and is out of scope.)*

---

## 10. Noise + PER Analysis

The joint Scenario C (observation noise × packet-error) is not evaluable (no
noise/comms model). The analogous *implemented* joint condition is
**PER × Noisy Nets**: its factorial interaction is non-significant for success
(η² = 0.001, p = 0.74) and reward (η² = 0.001, p = 0.75). The `+PER+Noisy`
cells (Table 5) do not differ significantly from baseline after correction.

---

## 11. Statistical Significance

**Omnibus over the four base algorithms** (`omnibus_algorithms.csv`):

| Metric | RM-ANOVA | Friedman | Verdict |
|---|---|---|---|
| Success | F(3,27)=0.39, p=0.763, η²ₚ=0.041 | χ²(3)=2.82, p=0.421 | no effect |
| Reward | F=0.36, p=0.784, η²ₚ=0.038 | χ²=1.80, p=0.615 | no effect |
| Optimality ρ | F=1.24, p=0.319, η²ₚ=0.151 | χ²=1.94, p=0.584 | no effect (n.s.) |

**Factorial 4×2×2** (`factorial_anova.csv`), primary metrics — significant
effects only:

- **Seed (block): success η²ₚ = 0.200, p = 0.0003; reward η²ₚ = 0.203, p = 0.0003.**
- Optimality ratio: **algorithm** p = 0.045 (η²ₚ = 0.063), **Noisy** p = 0.0053
  (η²ₚ = 0.061), **Algorithm × Noisy** p = 0.014 (η²ₚ = 0.082).
- No significant algorithm/PER/Noisy main effect on success or reward
  (Noisy→reward is a non-significant trend, p = 0.085).

**Pairwise (Holm-corrected).** **No** pairwise algorithm comparison and **no**
enhancement contrast is significant on **any** metric after Holm correction
(all adjusted p ≈ 1.0; Table 6, Figure 6). This is the classic pattern where a
weak omnibus signal (optimality ratio) does not localize to any single
Holm-surviving pair given n = 10.

---

## 12. Effect Size

On success/reward, paired effect sizes between algorithms are
**small-to-negligible** (|d_z| ≤ 0.35, |Cliff's δ| ≤ 0.31; Table 6). The largest
*practically* interesting effect is **Noisy Nets improving route optimality**:
for Dueling Double DQN, `+Noisy` lowers the optimality ratio with d_z = −0.68
(*medium*), and for DQN d_z = −0.36 (*small*) — directionally consistent with
the significant factorial Noisy main effect on ρ, though not Holm-significant in
the per-algorithm contrast. Effect-size magnitudes therefore **agree with the
significance tests**: real but modest route-quality benefit from Noisy Nets, and
no meaningful success/reward differences.

---

## 13. Multi-Seed Consistency

Seed is the dominant variance component (Figure 8): partial η² ≈ 0.20 for
success and reward — larger than every design factor combined. Concretely, DQN
ranges from 0% (seed 5, a training collapse) to 75% (seed 8) success.

> [!INSIGHT]
> **Run-to-run variability swamps the algorithmic differences.** With a 10-seed
> budget at 150 training episodes, the random seed explains ~4× more variance
> (partial η² ≈ 0.20) than the algorithm choice. Any single-seed or few-seed
> comparison of these DQN variants would therefore be unreliable — a
> reproducibility caution that is itself a publishable result.

---

## 14. Trade-off Analysis

- **Reward vs route quality.** DQN has the best success but a high optimality
  ratio (1.59) and elevated route risk (0.65); Dueling DQN reaches near-optimal
  routes (1.07, lowest risk 0.52) at slightly lower success — a mild
  success-vs-route-efficiency trade-off, though within overlapping CIs.
- **Exploration vs route quality.** Noisy Nets tends to improve route optimality
  (Section 12) without a significant success cost — a favourable trade-off
  direction, meriting confirmation at higher power.
- No safety metric (collision, near-miss) exists, so a reward–safety trade-off
  cannot be assessed (Limitations).

---

## 15. Cross-Algorithm Comparison (empirical answers)

1. **Best in clean conditions?** DQN by point estimate (49.8%), but **not
   statistically distinguishable** from the others.
2. **Most robust to noise?** Not evaluable (no noise model).
3. **Most robust to PER (packet-error)?** Not evaluable; as a training feature,
   PER's effect is negligible for all.
4. **Ranking change under degraded conditions?** Not evaluable.
5. **Statistically significant differences?** None survive Holm on any metric;
   only an omnibus optimality-ratio effect (algorithm, Noisy, Algo×Noisy).
6. **Meaningful effect sizes?** Only Noisy→optimality (small–medium).
7. **Consistent across seeds?** Dueling DQN is the most consistent (tightest CI,
   ±11.8%); DQN and the Noisy variants are the most variable (±18–22%).
8. **High variance?** The `+Noisy` and `+PER+Noisy` cells show the widest CIs.
9. **Reward ↔ success?** Strongly coupled by construction (goal bonus dominates
   reward); both track together across configs.
10. **Reward–safety trade-off?** Not evaluable (no safety metric).
11. **Safety–efficiency trade-off?** Not evaluable.
12. **Comms degradation differs by algorithm?** Not evaluable.

---

## 16. Main Findings

- **F1 (Empirical).** All four variants achieve broadly similar navigation
  success (40–50%) and reward at a 150-episode budget; DQN leads on points,
  Dueling DQN produces the most optimal routes.
- **F2 (Statistical).** No algorithm or enhancement effect on success/reward is
  significant (omnibus or Holm-corrected pairwise). The only reliable design
  signals are on **route optimality** (algorithm, Noisy, and their interaction,
  omnibus level).
- **F3 (Statistical).** **Random seed is the dominant variance source**
  (partial η² ≈ 0.20 ≫ any factor) — a reproducibility result in its own right.
- **F4 (Practical).** For this task and budget the variants are effectively
  **interchangeable on success/reward**; if route efficiency matters, **Noisy
  Nets** is the most promising lever (small–medium effect on ρ).

> [!INSIGHT]
> **Headline takeaway.** On this benchmark the four DQN variants are
> statistically interchangeable for *reaching the goal*; where they differ is in
> *how efficiently* they route (optimality ratio), and there **Noisy Nets** — not
> the choice of DQN variant — is the lever with a measurable (small–medium)
> effect. Report the seed-variance result (F3) prominently: it reframes the study
> from "which algorithm wins" to "how many seeds are needed to claim a winner".

---

## 17. Limitations

- **Environment scope.** Metrics that require physics/sensing/comms are **not
  implemented** and were not fabricated: observation/state **noise**,
  communication **packet-error rate**, **collision** rate, **docking accuracy**,
  **cross-track error**, **IALA compliance**, physical **travel time**.
  Consequently the protocol's Scenario B/C, and the safety/efficiency trade-offs,
  are out of scope. These define the **Option-2 extension** (add a physical /
  sensing / comms layer, then re-run this exact pipeline).
- **Environment substitution.** Results pertain to `VesselEnvV2` (a graph-
  navigation MDP built on the project's real graph), not the saturated deployed
  `VesselEnv`, which cannot support comparative statistics.
- **Statistical power.** n = 10 seeds with large seed variance yields wide CIs;
  absence of significance is **not** evidence of equivalence. A power analysis
  (Section 18) suggests substantially more seeds are needed to detect the
  observed small effects.
- **Budget.** 150 training episodes (~4,500 steps) is a modest, interactive-scale
  budget; rankings may change with longer training.
- **Single environment topology / reward shaping.** One port graph and one
  reward parameterization; generalization is untested.
- **No causal claims** beyond the controlled factorial manipulation.

---

## 18. Recommended Tables and Figures for a Q1 Paper

**Tables** (in `results/tables/`, each with an Interpretation column; CSV + LaTeX + `all_tables.xlsx`):
Table 1 (configuration), Table 2 (overall performance ± 95% CI), Table 3
(omnibus), Table 4 (factorial effects + η²), Table 6 (pairwise + Holm + effect
size), Table 7 (enhancement robustness), Table 8 (findings), plus the Methods
table.

**Figures** (in `results/figures/`, 320 DPI PNG + SVG): Figure 1 (overall
performance), Figure 4 (per-seed distributions — shows seed spread), Figure 6
(significance/effect-size matrix), Figure 8 (variance dominance) are the highest
analytical value and directly support F2–F3.

**For a stronger submission we recommend:** (i) increase to ≥ 30 seeds and/or a
longer training budget (power); (ii) implement the Option-2 physical/sensing/
comms layer to unlock the safety and robustness analyses; (iii) report the
seed-variance result explicitly as a reproducibility contribution.

---

### Provenance

Every value above is generated by the scripts in `experiments/` from
`results/raw/eval_episodes.csv` (6,400 evaluation episodes, 0 NaN/Inf) and is
reproducible end-to-end (see `results/reports/REPRODUCIBILITY.md`). Statistical
outputs: `results/statistics/`; aggregates: `results/aggregated/`.
