# Bintulu Port — Web DQN Autonomous Ship Navigation

A **browser-based** simulation of the *Bintulu Port Experiment* research
notebook: a Deep Q-Network (DQN) learns to steer an autonomous vessel along
a nautical channel network to its goal. It runs **100% in the browser** — no
Python, no server, no install. Just open `index.html`.

## Run it

Simply open `index.html` in any modern browser. That's it.

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
| Sprint 7.x — benchmark metrics, reward curve, trajectory viz | `js/main.js` telemetry + canvas rendering |

### DQN details (`js/dqn.js`)
- MLP `obsDim(10) → 128 → 128 → 3` actions, ReLU hidden, linear output.
- Target network with periodic hard update (`targetUpdate = 1000`).
- Experience replay (`bufferSize = 50000`, `batchSize = 64`).
- ε-greedy, linear decay `1.0 → 0.05` over `epsFraction · totalSteps`.
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
