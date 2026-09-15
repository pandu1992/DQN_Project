# Usage Guide — Bintulu Port Web DQN Simulation

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
3. Click **⚔ Compare All**. Each of the four algorithms trains in turn on the
   **same seeded environment**; their **MA20 reward learning curves** overlay
   on one chart with a color legend, and a summary table fills in
   (episodes, success rate, avg reward, best MA20, avg steps).
4. When it finishes, the export buttons enable:
   - **⬇ Comparison CSV** — two sections: (a) per-algorithm summary, (b) full
     per-episode learning curves in long format. Drop straight into
     pandas / R / Excel.
   - **🖼 Comparison PNG** — the overlaid learning-curve chart with title and
     legend, ready to embed as a figure.

---

## 6. A fair research workflow

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
   noisy, prefer the **multi-seed** mode (see below) so you can report
   **mean ± std** and run significance tests across seeds.

### Interpreting the metrics

| Metric | Meaning | Better |
|---|---|---|
| Success rate | fraction of episodes reaching the goal | higher |
| Avg reward | mean episodic return | higher |
| Best MA20 | peak smoothed reward reached | higher |
| Avg steps | mean steps per episode | lower (more direct) |
| Reward std | run-to-run variability | lower (more stable) |

---

## 7. Reproducibility notes

- The environment is **seeded** so the map/mission stream is deterministic;
  the comparison trains every algorithm on the same seed(s).
- Training runs in pure JavaScript on one browser thread, so networks are
  intentionally small. Keep episodes/algo moderate for interactive speed.
- The synthesized channel matches the notebook's **topology** (two access
  channels feeding a shared harbour lane, buoys, virtual branches), not the
  exact nautical-chart pixels — the RL formulation, planner, reward, and
  training loop follow the research notebook.

---

## 8. Troubleshooting

- **Page looks stale after an update** — hard refresh (Ctrl/Cmd + Shift + R).
- **Comparison feels slow** — lower episodes/algo, or reduce the number of
  seeds; dueling + Noisy configurations are the heaviest.
- **Nothing downloads on export** — ensure a comparison has finished (the
  export buttons are disabled until then) and that your browser isn't
  blocking downloads for the page.
