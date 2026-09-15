# Bintulu Port — Web DQN Autonomous Ship Navigation

A **browser-based** simulation of the *Bintulu Port Experiment* research
notebook: a family of value-based RL agents (DQN and its variants) learn to
steer an autonomous vessel along a nautical channel network to its goal. It
runs **100% in the browser** — no Python, no server, no install.

**🌐 Live demo:** https://pandu1992.github.io/DQN_Project/

## Run it

Simply open `index.html` in any modern browser (or use the live demo above).

(Optional local server, only if your browser blocks `file://` scripts:
`python3 -m http.server` then visit `http://localhost:8000`.)

## What you see

- **Navigation map** — synthesized Bintulu-style port with three lanes
  (North `L02`, South `L08`, Harbour `L01`), the waypoint graph, green/red
  buoys, virtual shortcut branches (dashed), the **planned path** (Dijkstra,
  green), the **actual path** taken by the DQN (orange), and the moving ship.
- **Controls** — Train / Pause / Reset, a speed slider (env steps per frame),
  and "Run Greedy Episode" to watch the learned policy act deterministically.
- **Live telemetry** — episode count, env steps, epsilon (ε) decay, training
  loss, replay-buffer size, episode reward, moving-average reward, and
  success rate over the last 100 episodes.
- **Reward chart** — raw reward per episode plus a 20-episode moving average.
- **Q-values** — live bar display of Q(s,·) for WAIT / FORWARD / BACKWARD and
  the chosen action.

### Analysis features (ported from the notebook)

- **🔥 Waypoint visit heatmap** — toggle an overlay coloring each waypoint by
  how often the agent has visited it (YlOrRd ramp), like the notebook's
  `RouteHeatmap`. Reveals channel bottlenecks.
- **⬇ CSV benchmark export** — download per-episode metrics (lane, start,
  goal, success, reward, steps, action counts, invalid moves, planned
  distance/cost, epsilon) as a CSV, mirroring the Sprint 7.3 / 7.4 benchmark
  dumps.
- **North (L02) vs South (L08) comparison** — a live table aggregating
  per-lane episode count, success rate, average reward, and average steps —
  the head-to-head channel analysis from the notebook.
- **🏆 / 💥 Representative episode replay** — the best and worst episodes (by
  reward) are recorded with their trajectories; replay them on the map to
  inspect optimal vs failed navigation, like the notebook's episode animator.

## Algorithm variants & comparison

The **Algorithm** dropdown lets you train the vessel with any of four
value-based RL algorithms (the choice applies on Reset / Train):

| Algorithm | Idea | What changes |
|---|---|---|
| **DQN** (vanilla) | Baseline Deep Q-Network | `Q(s,a) = MLP(s)`; target = `r + γ·maxₐ' Q_target(s',a')` |
| **Double DQN** | Decouple action *selection* from *evaluation* to reduce Q overestimation | online net picks `a' = argmaxₐ' Q_online(s',·)`, target net evaluates `Q_target(s', a')` |
| **Dueling DQN** | Split the head into a state-value `V(s)` and an advantage `A(s,a)` stream | `Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))` |
| **Dueling Double DQN** | Combine both improvements | dueling network **+** double-Q target |

### Stackable enhancements (checkboxes)

Two Rainbow-style enhancements can be toggled on **any** of the four base
algorithms (applied on Reset / Train, and to every algorithm during
Compare All):

| Enhancement | Idea | What changes |
|---|---|---|
| **Prioritized Replay (PER)** | Sample transitions with large TD-error more often to learn faster | proportional priorities `pᵢ = (\|δᵢ\| + ε)^α`; sampling ∝ `pᵢ`; importance-sampling weights `wᵢ = (N·P(i))^{−β}` (β annealed → 1) correct the bias |
| **Noisy Nets** | Learnable parameter noise for exploration (replaces ε-greedy) | head layers become `NoisyDense`: `W = μ_W + σ_W ⊙ εᵂ` with factorized Gaussian noise; ε-greedy is disabled (`ε = 0`) |

The active configuration is shown in the metrics panel and chart labels, e.g.
`Dueling Double DQN +PER+Noisy`.

### ⚔ Compare All + export

Click **Compare All** to automatically train each of the four algorithms for
*N* episodes (configurable) on the same seeded environment — with whatever
PER / Noisy toggles you've enabled applied to all of them — then overlay their
**MA20 reward learning curves** on one chart with a color legend, plus a
summary table of **episodes, success rate, average reward, best MA20 reward,
and average steps** per algorithm.

**Paper-ready export** (buttons enable once a comparison finishes):

- **⬇ Comparison CSV** — a two-section CSV: a per-algorithm summary
  (episodes, success rate, avg reward, reward std, best MA20, avg steps) plus
  the full per-episode learning curves in long format (`algorithm, episode,
  reward, success, steps`) — drop straight into pandas / R / Excel.
- **🖼 Comparison PNG** — the overlaid learning-curve chart rendered with a
  title and color legend, ready to embed in a figure.

> Because the networks are trained in pure JS on one browser thread, keep the
> episodes-per-algorithm modest (the default is 120) for a quick comparison;
> raise it for a more thorough run. Dueling variants and Noisy Nets are a bit
> heavier, so dueling stream heads are intentionally narrower to keep the
> browser responsive.

## How it maps to the research notebook

| Notebook sprint | Web module |
|---|---|
| Sprint 2.x — waypoint graph, lanes, buoys | `js/environment.js` → `buildWaypoints`, `buildGraph`, `buildBuoys` |
| Sprint 6.3 — edge attributes (risk, traffic, weather, current, travel_time, energy_cost, navigation_cost, difficulty) | `buildGraph` edge generation |
| Sprint 6.2 — virtual shortcut branches | `virtualPairs` in `buildGraph` |
| Sprint 6.5C — Dijkstra planner + virtual-edge expansion | `dijkstra`, `expandVirtual`, `physicalBFS` |
| Sprint 3 / 6.8 — Gymnasium `reset`/`step`, WAIT/FORWARD/BACKWARD | `VesselEnv.reset`, `VesselEnv.step` |
| Sprint 6.7B/C — RewardEngine + TerminationEngine | `REWARD_CONFIG` + reward/termination logic in `step` |
| Sprint 4 / 7.2 — DQN (MlpPolicy, target net, replay, ε-greedy) | `js/dqn.js` → `QNetwork`, `DQNAgent`, `ReplayBuffer` |
| Sprint 5 / 7.x — DQN variants (Double, Dueling, Dueling-Double) & benchmark comparison | `js/dqn.js` → `DuelingQNetwork`, `DQNAgent` algorithm flag; `js/main.js` → Compare-All engine |
| Rainbow-style enhancements — Prioritized Replay + Noisy Nets | `js/dqn.js` → `PrioritizedReplayBuffer`, `NoisyDense` (`cfg.per`, `cfg.noisy`) |
| Sprint 7.x — benchmark metrics, reward curve, trajectory viz, CSV/PNG export | `js/main.js` telemetry + canvas rendering + comparison export |

### Agent details (`js/dqn.js`)
- Torso MLP `obsDim(10) → 128 → 128` (ReLU). Standard head → `3` action
  Q-values; dueling head → value `V(s)` + advantage `A(s,a)` streams.
- Selectable algorithm: `DQN`, `DoubleDQN`, `DuelingDQN`, `DuelingDoubleDQN`
  (`cfg.algorithm`). Double-Q changes the target computation; dueling changes
  the network architecture.
- Optional enhancements: **PER** (`cfg.per`, `PrioritizedReplayBuffer` with
  proportional priorities + IS weights) and **Noisy Nets** (`cfg.noisy`,
  `NoisyDense` factorized Gaussian noise on the head layers; disables
  ε-greedy). Both stack on any base algorithm.
- Target network with periodic hard update (`targetUpdate = 1000`).
- Experience replay (`bufferSize = 50000`, `batchSize = 64`).
- ε-greedy, linear decay `1.0 → 0.05` over `epsFraction · totalSteps`
  (skipped when Noisy Nets is on).
- Huber loss, Adam optimizer (hand-written), `γ = 0.99`, `lr = 1e-3`,
  `trainFreq = 4` — mirroring the notebook's `RL_CONFIG`.

### Observation vector (10 dims)
normalized `x`, `y`, heading, lane id, distance-to-goal, remaining steps,
path progress, local avg edge difficulty, local avg navigation cost,
is-terminal flag.

## Notes & simplifications

The raw nautical chart image and OpenCV/HSV semantic-extraction pipeline
(Sprints 1–2.2) cannot run in a browser, so the channel geometry is
**synthesized** to match the notebook's *topology* (two access channels
feeding a shared harbour lane, buoy pairs, virtual branches) rather than
reproducing the exact pixel map. The RL formulation, graph model, planner,
reward/termination semantics, and DQN training loop follow the notebook.

Because everything is pure JS, the neural net is intentionally small so
training is smooth in real time on a single browser thread.
