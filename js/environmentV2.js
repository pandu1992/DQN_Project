/* ============================================================
 * BINTULU PORT — Environment V2 (research/experiment build)
 * environmentV2.js
 *
 * PURPOSE
 *   The original environment.js is a *saturated* benchmark: only two
 *   fixed lane-traversal missions, and the agent merely advances an
 *   index along a pre-computed Dijkstra path — so "always FORWARD" is
 *   optimal and every algorithm trivially scores ~100% / reward ~119.9.
 *   That leaves ZERO between-algorithm / between-seed variance, which
 *   makes any comparative statistical analysis vacuous.
 *
 *   V2 turns the SAME graph + SAME edge attributes into a genuine
 *   graph-navigation MDP so that policy quality actually varies:
 *     - the agent chooses among the current node's REAL outgoing edges
 *       (routing decisions at junctions + virtual shortcuts),
 *     - missions are varied (many distinct start->goal pairs, incl.
 *       cross-lane routing to the harbour),
 *     - reward is driven by the edges' REAL navigation_cost,
 *     - all metrics derive from EXISTING edge attributes
 *       (navigation_cost, risk, difficulty) — nothing fabricated.
 *
 *   The topology, edge-attribute generation, buoys, and Dijkstra
 *   planner are reused UNCHANGED from environment.js (imported), so the
 *   underlying "physics" is identical; only the MDP interface changes.
 *
 * NOTE: This file is for the offline experiment harness (Node). It does
 *       NOT alter the deployed web app (which still uses environment.js).
 * ============================================================ */

(function (root) {
  "use strict";

  // Reuse the base building blocks from environment.js. In Node the
  // harness loads environment.js first into the same sandbox, exposing
  // window.BintuluEnv; in the browser the same holds.
  const base = root.BintuluEnv;
  if (!base) throw new Error("environmentV2 requires environment.js loaded first");
  const { dijkstra, MAP_W, MAP_H } = base;

  // We need the internal builders. environment.js does not export them,
  // so V2 rebuilds an equivalent graph via the exported VesselEnv: we
  // instantiate a base env to borrow its states/order/graph/buoys
  // (identical construction, same seed => same graph).
  const BaseVesselEnv = base.VesselEnv;

  // ---------- seeded RNG (mulberry32), same as environment.js ----------
  function makeRNG(seed) {
    let s = seed >>> 0;
    return function () {
      s |= 0;
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  // ---------- reward configuration (V2) ----------
  // Reward is shaped by the REAL navigation_cost of each traversed edge:
  //   step reward  = -kappa * navigation_cost(edge)      (cheaper edge => less penalty)
  //   goal bonus   = + GOAL
  //   invalid move = + INVALID (chose a non-existent action slot)
  //   timeout      = + TIMEOUT (failed to reach goal in budget)
  // This makes route quality matter without inventing any new quantity.
  const REWARD_CONFIG_V2 = {
    goal: 100.0,
    invalid: -5.0,
    timeout: -30.0,
    costScale: 1.0,     // kappa: weight on real navigation_cost per edge
    stepPenalty: -0.2,  // small per-step penalty to discourage dithering
    revisitPenalty: -1.0, // discourage looping over already-visited nodes
  };

  const MAX_EPISODE_STEPS_V2 = 60;

  // Fixed number of action slots = max out-degree observed in the graph.
  // Action a in [0, A) selects the a-th outgoing edge (sorted deterministically);
  // if a >= outdegree(current), it is an INVALID action (no move).
  // This gives the agent genuine routing choices at junctions.
  const N_ACTION_SLOTS = 4;

  class VesselEnvV2 {
    constructor(seed = 42) {
      this.seed = seed;
      // borrow identical graph/states/buoys from the base env (same seed => same graph)
      const b = new BaseVesselEnv(seed);
      this.states = b.states;
      this.order = b.order;
      this.graph = b.graph;
      this.buoys = b.buoys;

      this.MAX_X = MAP_W;
      this.MAX_Y = MAP_H;
      this.maxSteps = MAX_EPISODE_STEPS_V2;
      this.missionRng = makeRNG(seed + 777);

      // deterministic sorted adjacency (stable action indexing)
      this.adj = {};
      for (const id of Object.keys(this.graph)) {
        const edges = this.graph[id].slice().sort((e1, e2) => {
          if (e1.navigation_cost !== e2.navigation_cost) return e1.navigation_cost - e2.navigation_cost;
          return e1.neighbor < e2.neighbor ? -1 : 1;
        });
        this.adj[id] = edges;
      }

      // maximum out-degree (for action space sizing / normalization)
      this.maxOutDeg = Math.max(...Object.values(this.adj).map((e) => e.length));

      this.nActions = N_ACTION_SLOTS;
      // obs: [x, y, gx, gy, dx_norm, dy_norm, distGoal, lane, stepFrac] (9)
      //   + per-slot [exists, cost_norm, difficulty, towardGoal] * N_ACTION_SLOTS
      this.obsDim = 9 + 4 * N_ACTION_SLOTS;

      // precompute all-pairs optimal (Dijkstra) cost lazily via cache
      this._planCache = {};
      this.reset();
    }

    laneEncoding(lane) {
      return { L02: 0, L08: 1, L01: 2 }[lane] ?? 0;
    }

    _plan(start, goal) {
      const key = start + "|" + goal;
      if (!(key in this._planCache)) {
        this._planCache[key] = dijkstra(this.graph, start, goal);
      }
      return this._planCache[key];
    }

    // Varied mission distribution:
    //   start: any spawn on an access channel (L02/L08 head)
    //          OR a random interior waypoint (harder/varied path lengths)
    //   goal : harbour terminal (L01 end), the opposite-lane terminal,
    //          or the same-lane terminal — chosen at random.
    // All are reachable via the physical graph (harbour junction connects lanes).
    _randomMission() {
      const ids = Object.keys(this.states);
      // candidate starts: spawns + a sampling of interior access-lane nodes
      const startPool = ids.filter((id) => {
        const s = this.states[id];
        return s.lane !== "L01"; // start on an access channel
      });
      let start, goal, plan, tries = 0;
      const harbourEnd = this.order.L01[this.order.L01.length - 1];
      const l02End = this.order.L02[this.order.L02.length - 1];
      const l08End = this.order.L08[this.order.L08.length - 1];
      const goalPool = [harbourEnd, l02End, l08End];
      do {
        start = startPool[Math.floor(this.missionRng() * startPool.length)];
        goal = goalPool[Math.floor(this.missionRng() * goalPool.length)];
        tries++;
        if (start === goal) { plan = { found: false }; continue; }
        plan = this._plan(start, goal);
      } while ((!plan.found || plan.totalCost <= 0) && tries < 50);
      if (!plan.found) { start = this.order.L02[0]; goal = harbourEnd; plan = this._plan(start, goal); }
      return { start, goal, plan };
    }

    reset(mission = null) {
      this.stepCount = 0;
      this.totalReward = 0;
      this.done = false;
      this.truncated = false;

      const m = mission || this._randomMission();
      this.startWp = m.start;
      this.goalWp = m.goal;
      const plan = m.plan || this._plan(this.startWp, this.goalWp);
      this.plannedPath = plan.found ? plan.nodes.slice() : [this.startWp, this.goalWp];
      this.plannedDistance = plan.totalDistance || 0;
      this.optimalCost = plan.totalCost || 0; // Dijkstra optimal navigation_cost

      this.currentWp = this.startWp;
      this.actualPath = [this.currentWp];
      this.visited = { [this.currentWp]: 1 };

      // accumulated REAL metrics along the agent's actual route
      this.routeCost = 0;       // sum of traversed edges' navigation_cost
      this.routeRisk = 0;       // sum of traversed edges' risk
      this.routeDifficulty = 0; // sum of traversed edges' difficulty
      this.routeDistance = 0;   // sum of traversed edges' geometric distance
      this.invalidCount = 0;
      this.revisitCount = 0;

      return this._obs();
    }

    _towardGoal(fromId, toId) {
      // 1 if the edge reduces straight-line distance to goal, else 0
      const g = this.states[this.goalWp];
      const a = this.states[fromId], b = this.states[toId];
      return dist(b, g) < dist(a, g) ? 1 : 0;
    }

    _obs() {
      const s = this.states[this.currentWp];
      const g = this.states[this.goalWp];
      const dgoal = dist(s, g);
      const edges = this.adj[this.currentWp] || [];
      const obs = [
        s.x / this.MAX_X,
        s.y / this.MAX_Y,
        g.x / this.MAX_X,
        g.y / this.MAX_Y,
        (g.x - s.x) / this.MAX_X,      // dx toward goal
        (g.y - s.y) / this.MAX_Y,      // dy toward goal
        dgoal / 900,
        this.laneEncoding(s.lane) / 2,
        this.stepCount / this.maxSteps,
      ];
      for (let a = 0; a < N_ACTION_SLOTS; a++) {
        if (a < edges.length) {
          const e = edges[a];
          obs.push(1); // exists
          obs.push(Math.min(1, e.navigation_cost / 30));
          obs.push(e.difficulty);
          obs.push(this._towardGoal(this.currentWp, e.neighbor));
        } else {
          obs.push(0, 0, 0, 0); // non-existent slot
        }
      }
      return obs;
    }

    step(action) {
      this.stepCount++;
      const edges = this.adj[this.currentWp] || [];
      let invalid = false;
      let reward = REWARD_CONFIG_V2.stepPenalty;

      if (action < 0 || action >= edges.length) {
        // chose a non-existent edge slot => invalid, stay put
        invalid = true;
        this.invalidCount++;
        reward += REWARD_CONFIG_V2.invalid;
      } else {
        const e = edges[action];
        // pay the REAL navigation cost of the chosen edge
        reward += REWARD_CONFIG_V2.costScale * (-e.navigation_cost) * 0.1;
        this.routeCost += e.navigation_cost;
        this.routeRisk += e.risk;
        this.routeDifficulty += e.difficulty;
        this.routeDistance += e.distance;
        this.currentWp = e.neighbor;
        this.actualPath.push(this.currentWp);
        if (this.visited[this.currentWp]) {
          this.revisitCount++;
          reward += REWARD_CONFIG_V2.revisitPenalty;
        }
        this.visited[this.currentWp] = (this.visited[this.currentWp] || 0) + 1;
      }

      const reachedGoal = this.currentWp === this.goalWp;
      let timeout = this.stepCount >= this.maxSteps;

      if (reachedGoal) reward += REWARD_CONFIG_V2.goal;
      else if (timeout) reward += REWARD_CONFIG_V2.timeout;

      this.totalReward += reward;

      let terminated = reachedGoal;
      if (timeout && !reachedGoal) this.truncated = true;
      this.done = terminated || this.truncated;

      // path optimality ratio (>=1; 1.0 == matched Dijkstra optimal cost)
      const optimalityRatio =
        this.optimalCost > 0 && reachedGoal ? this.routeCost / this.optimalCost : null;

      const info = {
        currentWp: this.currentWp,
        goalWp: this.goalWp,
        reachedGoal,
        invalid,
        timeout,
        routeCost: this.routeCost,
        optimalCost: this.optimalCost,
        excessCost: reachedGoal ? this.routeCost - this.optimalCost : null,
        optimalityRatio,
        routeRisk: this.routeRisk,
        routeDifficulty: this.routeDifficulty,
        routeDistance: this.routeDistance,
        invalidCount: this.invalidCount,
        revisitCount: this.revisitCount,
      };
      return { obs: this._obs(), reward, terminated, truncated: this.truncated, info };
    }
  }

  root.BintuluEnvV2 = {
    VesselEnvV2,
    REWARD_CONFIG_V2,
    MAX_EPISODE_STEPS_V2,
    N_ACTION_SLOTS,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.BintuluEnvV2;
  }
})(typeof window !== "undefined" ? window : globalThis);
