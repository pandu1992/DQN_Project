# Finding (A): The Deployed Benchmark Is Saturated and Non-Discriminative

This short note documents *why* a comparative statistical study could not be run
on the originally deployed environment (`js/environment.js`) and why an offline
research build (`js/environmentV2.js`) was required. It is intended as the
"experimental design justification" narrative for the paper.

## Observation

`js/environment.js` (`VesselEnv`) exposes an MDP in which:

- the **mission distribution has only two fixed missions** — a full traversal of
  the North lane (`L02_WP001 → L02_WP022`) or the South lane
  (`L08_WP001 → L08_WP020`); across 200 resets on any seed exactly **2 distinct
  missions** occur;
- the agent's actions (`WAIT`/`FORWARD`/`BACKWARD`) merely **advance an index
  along a pre-computed Dijkstra path**, so the solution is handed to the agent;
- the optimal policy is therefore trivially "always FORWARD".

## Evidence

A hardcoded pure-`FORWARD` policy attains **100% success, mean reward
119.84 ± 0.99** (range 119–121) over 200 missions. Every trained algorithm
reaches this ceiling within ~40 episodes, and different seeds produce
**identical** evaluation trajectories.

## Consequence

With near-zero between-algorithm and between-seed variance:

- any omnibus test has F ≈ 0, p ≈ 1 (nothing to explain);
- confidence intervals collapse to a point;
- effect-size denominators → 0 (undefined).

A comparative analysis on this environment would yield "119.9 ± 0.0 vs
119.9 ± 0.0, p ≈ 1.0" for every pair — technically real but scientifically
empty, and unsuitable for a Q1 manuscript.

## Resolution

We built `VesselEnvV2`, which reuses the **same graph topology, the same edge
attributes, and the same Dijkstra planner**, but (i) samples **115 distinct
missions per seed** with optimal-cost range 5.3–151.5, and (ii) requires the
agent to **choose among the current node's real outgoing edges** (genuine
routing) rather than advancing a pre-solved index. Task non-triviality was
validated (random policy 13% success; near-optimal heuristic 100%), and the
optimality metric was validated (follow-Dijkstra → ratio 1.0000 on 100/100
episodes). All reported metrics remain derived from the environment's real edge
attributes; nothing is fabricated. The deployed web application is unchanged —
`VesselEnvV2` is used only by the offline experiment harness.
