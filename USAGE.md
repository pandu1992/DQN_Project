# Usage Guide — Synthetic Port Web DQN Simulation

**Live demo:** https://pandu1992.github.io/DQN_Project/

This guide explains how to use the simulation and how to run a fair,
paper-ready comparison study. Everything runs **100% in your browser** — no
install, no server, no data leaves your machine.

---

## 1. Quick start (30 seconds)

1. Open the [live demo](https://pandu1992.github.io/DQN_Project/) (or open
   `index.html` locally).
2. Click **▶ Train**. Watch the ship learn to navigate the channel — the
   reward chart climbs and epsilon (ε) decays as the policy improves.
3. Click **⏸ Pause** any time, **⟲ Reset** to start fresh, or
   **🎯 Run Greedy Episode** to watch the current policy act greedily
   (no exploration).

---

## 2. The screen at a glance

### Map (left)
- **Green line** — planned path (Dijkstra shortest navigation-cost route).
- **Orange line** — actual path taken by the agent.
- **Blue arrow** — the ship (current waypoint + heading).
- **Green/red triangles** — starboard/port buoys.
- **S / G markers** — start and goal for the current mission.
- **Dashed orange edges** — virtual shortcut branches.

### Reward chart (left, below the map)
- **Orange** = raw reward per episode. **Blue** = 20-episode moving average
  (MA20). Rising MA20 = the agent is learning.

### Side panel (right)
- **Algorithm** dropdown + **PER / Noisy Nets** checkboxes.
- **Controls**: Train / Pause / Reset / Greedy.
- **Speed** slider = environment steps simulated per animation frame.
- **Training Status**: active algorithm, episode, env steps, ε, loss, replay
  buffer size, episode reward, avg reward (100), success rate (100).
- **Current Mission**, **Live Q-values**, **North vs South** table,
  **Representative Episodes** (best/worst).
- **🔥 Heatmap**, **⬇ Export CSV**, **🏆 / 💥 Replay** buttons.

---

## 3. Algorithms & enhancements

### Base algorithm (dropdown — applied on Reset / Train)

| Option | What it does |
|---|---|
| **DQN** | Vanilla Deep Q-Network baseline. |
| **Double DQN** | Reduces Q-value overestimation (online net selects the action, target net evaluates it). |
| **Dueling DQN** | Splits the head into value `V(s)` + advantage `A(s,a)` streams. |
| **Dueling Double DQN** | Combines dueling architecture + double-Q target. |

### Stackable enhancements (checkboxes — apply to any base algorithm)

| Toggle | What it does |
|---|---|
| **Prioritized Replay (PER)** | Samples high-TD-error transitions more often; importance-sampling weights correct the bias. |
| **Noisy Nets** | Learnable parameter noise for exploration; disables ε-greedy (ε shown as 0). |

The active configuration appears in the metrics panel and chart labels, e.g.
`Dueling Double DQN +PER+Noisy`. Changing any of these re-initializes the run.

---

## 4. Analysis features

- **🔥 Heatmap** — overlay coloring each waypoint by visit frequency; reveals
  channel bottlenecks.
- **⬇ Export CSV** (single-run) — per-episode metrics of the current training
  run (lane, start, goal, success, reward, steps, action counts, invalid
  moves, planned distance/cost, epsilon).
- **North (L02) vs South (L08)** — live per-lane success rate, avg reward,
  avg steps.
- **🏆 / 💥 Replay** — replay the best / worst episode's trajectory on the map.

---

## 5. Compare All + paper-ready export

1. (Optional) tick **PER** and/or **Noisy Nets** — they apply to *every*
   algorithm in the comparison.
2. Set **episodes/algo** (default 120; higher = more thorough but slower).
3. Set **seeds** (default 3). With `seeds > 1`, each algorithm is trained
   once per seed (env seed = `42 + seedIndex`); the chart draws the
   **mean curve with a ±std error band**, and the summary table reports
   **mean ± std across seeds**. `seeds = 1` keeps the classic single-run
   behavior (no band). More seeds → more reliable statistics (see §7).
4. Click **⚔ Compare All**. Each of the four algorithms trains in turn on the
   **same seeded environment(s)**; their **MA20 reward learning curves** overlay
   on one chart with a color legend, and a summary table fills in
   (episodes, success rate, avg reward, best MA20, avg steps).
4. When it finishes, the export buttons enable:
   - **⬇ Comparison CSV** — two sections: (a) per-algorithm summary, (b) full
     per-episode learning curves in long format (includes a `seed` column).
     Drop straight into pandas / R / Excel.
   - **🖼 Comparison PNG** — the overlaid learning-curve chart (with the ±std
     band when `seeds > 1`), title, and legend, ready to embed as a figure.

---

## 6. Statistical evaluation (significance tests)

Below the comparison, the **Statistical Evaluation** panel automatically runs
proper significance tests on the multi-seed results — no external tools
needed. Each seed contributes **one sample** per algorithm, so this panel is
only meaningful when you ran **Compare All with `seeds > 1`** (ideally ≥ 5).

### Choose a metric
The **metric** dropdown selects what the tests compare:

| Metric | Definition |
|---|---|
| **Final reward (last 20% eps)** | mean episodic reward over each seed's final 20% of episodes — "converged" performance. |
| **Success rate (last 20% eps)** | goal-reaching rate over the final 20% of episodes. |
| **Avg reward (all eps)** | mean reward across every episode of the run. |
| **Best MA20 reward** | peak of the 20-episode moving-average reward curve. |

### Per-algorithm summary table
For the selected metric, each algorithm shows **n (seeds)**, **mean**, **SD**,
the **95% confidence interval** of the mean (Student-t based), and the
observed **min…max** range across seeds.

### Pairwise significance table
Every algorithm pair is tested (6 pairs for the 4 algorithms), reporting:

- **Δ mean** — difference of means (first minus second).
- **t-test p** — [Welch's two-sample t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)
  (does *not* assume equal variances; robust for unequal seed spread).
- **MW p** — [Mann–Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
  (non-parametric rank-sum; doesn't assume normality — safer for few seeds).
- **Cohen's d** — standardized effect size (pooled SD); how *large* the
  difference is, independent of sample size.
- **Effect** — magnitude label: negligible / small / medium / large.
- **Sig.** — marker from the Welch p-value.

All tests are **two-sided**. `sig` markers:
`***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `ns` not significant.
Effect size: |d|<0.2 negligible · <0.5 small · <0.8 medium · ≥0.8 large.

### Multiple-comparison correction
Testing all pairs inflates the false-positive rate. The warning line shows a
**Bonferroni-adjusted α** (`0.05 / number_of_comparisons`); treat a result as
significant only if its p-value is below that adjusted threshold.

> **⚠ Low-power warning.** With fewer than 5 seeds per algorithm the tests are
> unreliable — the panel flags this explicitly. Bump **seeds** to ≥ 5
> (ideally ≥ 10) before drawing conclusions.

### Export
**⬇ Statistics CSV** dumps, for **all four metrics** at once:
(a) per-algorithm summary (n, mean, SD, SEM, 95% CI half-width, min, max) and
(b) full pairwise test results (Welch t/df/p, Mann–Whitney U/p, Cohen's d,
Hedges g, effect magnitude, significant-at-0.05 flag). Paper-ready.

---

## 7. A fair research workflow

To make claims defensible in a paper, keep everything identical except the
one thing you're studying:

1. **Fix the budget.** Use the same **episodes/algo** for every configuration.
2. **One variable at a time.** Compare, e.g.:
   - *baseline*: all four algorithms, PER off, Noisy off;
   - *+PER*: all four with PER on;
   - *+Noisy*: all four with Noisy on;
   - *+PER+Noisy*: both on.
3. **Export each run's CSV** (rename per configuration, e.g.
   `bintulu_comparison_baseline.csv`, `bintulu_comparison_PER.csv`).
4. **Aggregate & test.** Load the CSVs and compare the metric of interest
   (final MA20 reward, success rate, avg steps). Because a single run is
   noisy, use the **multi-seed** mode (set `seeds ≥ 5`) so you can report
   **mean ± std** and let the built-in **Statistical Evaluation** panel (§6)
   run the significance tests across seeds for you.

### Interpreting the metrics

| Metric | Meaning | Better |
|---|---|---|
| Success rate | fraction of episodes reaching the goal | higher |
| Avg reward | mean episodic return | higher |
| Best MA20 | peak smoothed reward reached | higher |
| Avg steps | mean steps per episode | lower (more direct) |
| Reward std | run-to-run variability | lower (more stable) |

---

## 8. Reproducibility notes

- The environment is **seeded** so the map/mission stream is deterministic;
  the comparison trains every algorithm on the same seed(s).
- Training runs in pure JavaScript on one browser thread, so networks are
  intentionally small. Keep episodes/algo moderate for interactive speed.
- The synthesized channel matches the notebook's **topology** (two access
  channels feeding a shared harbour lane, buoys, virtual branches), not the
  exact nautical-chart pixels — the RL formulation, planner, reward, and
  training loop follow the research notebook.

---

## 9. Troubleshooting

- **Page looks stale after an update** — hard refresh (Ctrl/Cmd + Shift + R).
- **Comparison feels slow** — lower episodes/algo, or reduce the number of
  seeds; dueling + Noisy configurations are the heaviest.
- **Nothing downloads on export** — ensure a comparison has finished (the
  export buttons are disabled until then) and that your browser isn't
  blocking downloads for the page.
