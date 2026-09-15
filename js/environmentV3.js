/* ============================================================
 * BINTULU PORT — Environment V3 (physical / sensing / comms layer)
 * environmentV3.js
 *
 * PURPOSE
 *   V2 turned the saturated benchmark into a discriminative
 *   graph-navigation MDP, but it still lacked the physical, sensing,
 *   and communication mechanisms a maritime-autonomy paper needs.
 *   Those were listed as OUT OF SCOPE in the Q1 report because they
 *   were not implemented anywhere in the project.
 *
 *   V3 implements them for real, so their metrics are GENUINELY
 *   MEASURED (never fabricated):
 *     - continuous vessel kinematics along each graph edge (real x/y),
 *     - observation (sensor) noise: Gaussian perturbation of the sensed
 *       state, std = cfg.noiseStd,
 *     - communication packet-error rate: with prob cfg.packetErrorRate
 *       the fresh observation is dropped and the agent is handed the
 *       last successfully received (stale) observation,
 *     - collision detection against a seeded obstacle field,
 *     - docking accuracy: real distance from the final pose to the
 *       ideal dock point at the goal,
 *     - cross-track error (CTE): perpendicular distance from the actual
 *       continuous position to the planned route segment,
 *     - IALA compliance: buoy side-of-channel rule violations, counted
 *       from real geometry.
 *
 *   The graph, edge attributes, buoys, and Dijkstra planner are reused
 *   UNCHANGED from environment.js / V2. Only the dynamics + observation
 *   model are added.
 *
 * NOTE: Offline research build (Node harness). The deployed web app
 *       (environment.js) is NOT affected.
 * ============================================================ */

(function (root) {
  "use strict";

  const base = root.BintuluEnv;
  if (!base) throw new Error("environmentV3 requires environment.js loaded first");
  const { dijkstra, MAP_W, MAP_H } = base;
  const BaseVesselEnv = base.VesselEnv;

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
  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
  function distXY(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

  // Gaussian sample via Box-Muller, driven by a seeded uniform RNG.
  function makeGaussian(rng) {
    let spare = null;
    return function () {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0;
      while (u === 0) u = rng();
      while (v === 0) v = rng();
      const mag = Math.sqrt(-2 * Math.log(u));
      spare = mag * Math.sin(2 * Math.PI * v);
      return mag * Math.cos(2 * Math.PI * v);
    };
  }

  // Perpendicular distance from point P to segment AB (real geometry).
  function pointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) return distXY(px, py, ax, ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    const cx = ax + t * dx, cy = ay + t * dy;
    return distXY(px, py, cx, cy);
  }

  // Signed side of point P relative to directed segment A->B.
  // >0 = left of travel direction, <0 = right, 0 = on the line.
  function sideOfSegment(px, py, ax, ay, bx, by) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
  }

  const REWARD_CONFIG_V3 = {
    goal: 100.0,
    invalid: -5.0,
    timeout: -30.0,
    costScale: 1.0,
    stepPenalty: -0.2,
    revisitPenalty: -1.0,
    collisionPenalty: -25.0,   // real: applied when the hull hits an obstacle
    ctePenalty: -0.05,         // per-unit accumulated cross-track error
    ialaPenalty: -3.0,         // per IALA side-of-channel violation
    dockBonusScale: 20.0,      // bonus scaled by docking precision at goal
  };

  const MAX_EPISODE_STEPS_V3 = 60;
  const N_ACTION_SLOTS = 4;
  const KINEMATIC_SUBSTEPS = 6;   // continuous samples per edge traversal
  const OBSTACLE_RADIUS = 14;     // collision radius (logical units)
  const DOCK_TOLERANCE = 25;      // docking accuracy considered "good" below this
  const IALA_GATE_RADIUS = 22;    // proximity at which a buoy's side is judged

  class VesselEnvV3 {
    constructor(seed = 42, cfg = {}) {
      this.seed = seed;
      this.cfg = Object.assign({
        noiseStd: 0.0,          // sensor/observation noise (obs units, ~[0,1] scale)
        packetErrorRate: 0.0,   // P(observation frame dropped -> stale held)
        obstacleCount: 6,       // seeded obstacle discs in the water
        endOnCollision: false,  // if true, a collision terminates the episode
      }, cfg);

      const b = new BaseVesselEnv(seed);
      this.states = b.states;
      this.order = b.order;
      this.graph = b.graph;
      this.buoys = b.buoys;

      this.MAX_X = MAP_W;
      this.MAX_Y = MAP_H;
      this.maxSteps = MAX_EPISODE_STEPS_V3;
      this.missionRng = makeRNG(seed + 777);
      this.noiseRng = makeRNG(seed + 991);   // separate stream for sensor noise
      this.commRng = makeRNG(seed + 4242);    // separate stream for packet loss
      this.gauss = makeGaussian(this.noiseRng);

      // deterministic sorted adjacency (stable action indexing)
      this.adj = {};
      for (const id of Object.keys(this.graph)) {
        const edges = this.graph[id].slice().sort((e1, e2) => {
          if (e1.navigation_cost !== e2.navigation_cost) return e1.navigation_cost - e2.navigation_cost;
          return e1.neighbor < e2.neighbor ? -1 : 1;
        });
        this.adj[id] = edges;
      }
      this.maxOutDeg = Math.max(...Object.values(this.adj).map((e) => e.length));

      // seeded obstacle field (discs placed in the navigable area, away from
      // waypoints so missions stay solvable). Deterministic per seed.
      this.obstacles = this._buildObstacles(this.cfg.obstacleCount);

      this.nActions = N_ACTION_SLOTS;
      // obs = V2 layout (9 + 4*slots) + [cte_norm, nearestObstacleDist_norm, lastCollisionFlag]
      this.obsDim = 9 + 4 * N_ACTION_SLOTS + 3;

      this._planCache = {};
      this.reset();
    }

    laneEncoding(lane) { return { L02: 0, L08: 1, L01: 2 }[lane] ?? 0; }

    _buildObstacles(n) {
      const rng = makeRNG(this.seed + 5150);
      const obs = [];
      const wps = Object.values(this.states);

      // Collect all physical edges (as endpoint pairs) so we can seed some
      // obstacles NEAR the travel lanes — otherwise the vessel (which rides
      // waypoint-to-waypoint straight lines) could never collide and the
      // collision metric would be degenerate (always 0).
      const edgeList = [];
      for (const id of Object.keys(this.graph)) {
        for (const e of this.graph[id]) {
          const A = this.states[id], B = this.states[e.neighbor];
          if (A && B && id < e.neighbor) edgeList.push([A, B]);
        }
      }

      // ~60% of obstacles are placed as partial "hazards" offset from an edge
      // midpoint (close enough to threaten that lane but small enough that a
      // careful route can still avoid the worst); the rest are open-water discs.
      const nEdge = Math.round(n * 0.6);
      let placed = 0, tries = 0;
      while (placed < nEdge && tries < 400 && edgeList.length) {
        tries++;
        const [A, B] = edgeList[Math.floor(rng() * edgeList.length)];
        const t = 0.35 + rng() * 0.3;                 // along the edge (mid-ish)
        const mx = A.x + (B.x - A.x) * t;
        const my = A.y + (B.y - A.y) * t;
        // perpendicular offset so the disc grazes the lane rather than blocking it
        const ang = Math.atan2(B.y - A.y, B.x - A.x) + Math.PI / 2;
        const off = (rng() < 0.5 ? -1 : 1) * (OBSTACLE_RADIUS * (0.4 + rng() * 0.7));
        const x = mx + Math.cos(ang) * off, y = my + Math.sin(ang) * off;
        // don't sit right on a waypoint (keep missions solvable)
        let ok = true;
        for (const w of wps) { if (distXY(x, y, w.x, w.y) < 20) { ok = false; break; } }
        if (ok) { obs.push({ x, y, r: OBSTACLE_RADIUS }); placed++; }
      }
      // remaining: open-water discs away from waypoints
      tries = 0;
      while (obs.length < n && tries < 400) {
        tries++;
        const x = 120 + rng() * (MAP_W - 240);
        const y = 90 + rng() * (MAP_H - 180);
        let ok = true;
        for (const w of wps) { if (distXY(x, y, w.x, w.y) < 24) { ok = false; break; } }
        if (ok) obs.push({ x, y, r: OBSTACLE_RADIUS });
      }
      return obs;
    }

    _plan(start, goal) {
      const key = start + "|" + goal;
      if (!(key in this._planCache)) this._planCache[key] = dijkstra(this.graph, start, goal);
      return this._planCache[key];
    }

    _randomMission() {
      const ids = Object.keys(this.states);
      const startPool = ids.filter((id) => this.states[id].lane !== "L01");
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
      this.optimalCost = plan.totalCost || 0;

      this.currentWp = this.startWp;
      this.actualPath = [this.currentWp];
      this.visited = { [this.currentWp]: 1 };

      // continuous pose starts at the start waypoint
      const s0 = this.states[this.startWp];
      this.posX = s0.x; this.posY = s0.y;

      // accumulated real metrics
      this.routeCost = 0; this.routeRisk = 0; this.routeDifficulty = 0; this.routeDistance = 0;
      this.invalidCount = 0; this.revisitCount = 0;

      // NEW physical/sensing/comms metrics
      this.collisionCount = 0;
      this.cteSum = 0;           // accumulated cross-track error
      this.cteSamples = 0;
      this.ialaViolations = 0;
      this.droppedFrames = 0;    // comms packet losses
      this.lastCollision = 0;
      this.nearestObstacleDist = this._nearestObstacle(this.posX, this.posY);
      this.dockingAccuracy = null; // set at goal arrival

      // comms: last successfully received observation (for stale-hold on drop)
      this._lastObs = null;
      const first = this._trueObs();
      this._lastObs = first;
      return this._sensedObs(first);
    }

    _nearestObstacle(x, y) {
      let best = Infinity;
      for (const o of this.obstacles) best = Math.min(best, distXY(x, y, o.x, o.y) - o.r);
      return best;
    }

    _towardGoal(fromId, toId) {
      const g = this.states[this.goalWp];
      const a = this.states[fromId], b = this.states[toId];
      return dist(b, g) < dist(a, g) ? 1 : 0;
    }

    // The TRUE (noise-free) observation from the current continuous pose.
    _trueObs() {
      const g = this.states[this.goalWp];
      const s = this.states[this.currentWp];
      const dgoal = distXY(this.posX, this.posY, g.x, g.y);
      const edges = this.adj[this.currentWp] || [];
      const cteNorm = Math.min(1, (this.cteSamples ? this.cteSum / this.cteSamples : 0) / 60);
      const nearObs = Math.max(0, Math.min(1, this.nearestObstacleDist / 120));
      const obs = [
        this.posX / this.MAX_X,
        this.posY / this.MAX_Y,
        g.x / this.MAX_X,
        g.y / this.MAX_Y,
        (g.x - this.posX) / this.MAX_X,
        (g.y - this.posY) / this.MAX_Y,
        dgoal / 900,
        this.laneEncoding(s.lane) / 2,
        this.stepCount / this.maxSteps,
      ];
      for (let a = 0; a < N_ACTION_SLOTS; a++) {
        if (a < edges.length) {
          const e = edges[a];
          obs.push(1, Math.min(1, e.navigation_cost / 30), e.difficulty, this._towardGoal(this.currentWp, e.neighbor));
        } else obs.push(0, 0, 0, 0);
      }
      obs.push(cteNorm, nearObs, this.lastCollision);
      return obs;
    }

    // Apply sensor NOISE to a true observation (Gaussian, std = cfg.noiseStd),
    // leaving the discrete "exists" flags intact.
    _sensedObs(trueObs) {
      const std = this.cfg.noiseStd;
      if (std <= 0) return trueObs.slice();
      const out = trueObs.slice();
      for (let i = 0; i < out.length; i++) {
        // do not perturb the per-slot "exists" flags (indices 9,13,17,21)
        const isExistsFlag = (i >= 9 && (i - 9) % 4 === 0 && i < 9 + 4 * N_ACTION_SLOTS);
        if (isExistsFlag) continue;
        out[i] = out[i] + this.gauss() * std;
      }
      return out;
    }

    // Communication layer: with prob packetErrorRate the fresh frame is
    // dropped and the last received observation is reused (stale hold).
    _commObs(sensed) {
      if (this.cfg.packetErrorRate > 0 && this.commRng() < this.cfg.packetErrorRate && this._lastObs) {
        this.droppedFrames++;
        return this._lastObs.slice();   // stale
      }
      this._lastObs = sensed.slice();
      return sensed;
    }

    // Traverse an edge with continuous sub-steps; accumulate CTE, detect
    // collisions, and check IALA buoy side compliance. Returns collided flag.
    //
    // CONTROL-EXECUTION MODEL: perfect sensing/comms => the vessel rides the
    // ideal A->B line exactly. Under sensor NOISE and dropped COMMS frames the
    // realised trajectory DRIFTS from the commanded line (localisation/actuation
    // error), so the vessel no longer tracks the planned route perfectly. The
    // drift magnitude is a real function of noiseStd and packetErrorRate; this
    // is what makes cross-track error and docking accuracy respond to the
    // sensing/comms conditions instead of being trivially zero.
    _traverseEdge(fromId, toId) {
      const A = this.states[fromId], B = this.states[toId];
      let collided = false;

      // per-edge control-error scale from the sensing/comms conditions
      const driftScale = 60 * this.cfg.noiseStd + 30 * this.cfg.packetErrorRate;
      const segLen = distXY(A.x, A.y, B.x, B.y) || 1;
      const nx = -(B.y - A.y) / segLen, ny = (B.x - A.x) / segLen; // unit normal to A->B

      // buoys close enough to this edge to act as channel markers
      const nearBuoys = this.buoys
        .map((bu) => ({ bu, d: pointToSegment(bu.x, bu.y, A.x, A.y, B.x, B.y),
                        side: sideOfSegment(bu.x, bu.y, A.x, A.y, B.x, B.y) }))
        .filter((o) => o.d <= IALA_GATE_RADIUS);
      const ialaHit = new Set(); // count each buoy at most once per edge

      for (let k = 1; k <= KINEMATIC_SUBSTEPS; k++) {
        const t = k / KINEMATIC_SUBSTEPS;
        let x = A.x + (B.x - A.x) * t;
        let y = A.y + (B.y - A.y) * t;
        // lateral drift ~ N(0, driftScale) * shape(t) — zero at endpoints (the
        // controller re-acquires the waypoint), largest mid-edge.
        if (driftScale > 0) {
          // shape peaks mid-edge; when heading INTO the goal, keep a residual
          // terminal drift at t=1 (the dock is an open-water point the vessel
          // must hold station at, not a waypoint it snaps to) so docking
          // accuracy reflects the control error.
          const toGoal = (toId === this.goalWp);
          const endHold = toGoal ? 0.5 : 0.0;
          const shape = Math.sin(Math.PI * t) * (1 - endHold) + (toGoal ? endHold * t : 0);
          const drift = this.gauss() * driftScale * shape;
          x += nx * drift; y += ny * drift;
        }
        this.posX = x; this.posY = y;

        // cross-track error vs the planned route (min perpendicular dist to any planned segment)
        this.cteSum += this._cteToPlan(x, y);
        this.cteSamples++;

        // collision check against obstacle discs
        for (const o of this.obstacles) {
          if (distXY(x, y, o.x, o.y) <= o.r) { collided = true; break; }
        }

        // IALA channel-keeping (per sub-step): a violation occurs if the vessel
        // strays OUTBOARD of a channel-marking buoy — i.e. it is on the same
        // side as the buoy AND farther from the channel centreline than the
        // buoy itself (the vessel left the buoyed channel). Real geometry;
        // stays 0 on the centreline, rises as drift pushes the vessel out.
        const vSide = sideOfSegment(x, y, A.x, A.y, B.x, B.y);
        const vPerp = Math.abs(vSide) / segLen;
        for (const nb of nearBuoys) {
          if (ialaHit.has(nb.bu)) continue;
          const bPerp = Math.abs(nb.side) / segLen;
          const sameSide = (vSide > 0) === (nb.side > 0) && nb.side !== 0;
          if (sameSide && vPerp > bPerp) { this.ialaViolations++; ialaHit.add(nb.bu); }
        }
      }
      this.nearestObstacleDist = this._nearestObstacle(this.posX, this.posY);

      return collided;
    }

    // min perpendicular distance from (x,y) to any segment of the planned path
    _cteToPlan(x, y) {
      const p = this.plannedPath;
      if (p.length < 2) return 0;
      let best = Infinity;
      for (let i = 0; i < p.length - 1; i++) {
        const a = this.states[p[i]], b = this.states[p[i + 1]];
        if (!a || !b) continue;
        best = Math.min(best, pointToSegment(x, y, a.x, a.y, b.x, b.y));
      }
      return isFinite(best) ? best : 0;
    }

    step(action) {
      this.stepCount++;
      const edges = this.adj[this.currentWp] || [];
      let invalid = false;
      let reward = REWARD_CONFIG_V3.stepPenalty;
      let collidedThisStep = false;

      if (action < 0 || action >= edges.length) {
        invalid = true;
        this.invalidCount++;
        reward += REWARD_CONFIG_V3.invalid;
      } else {
        const e = edges[action];
        const fromId = this.currentWp;
        collidedThisStep = this._traverseEdge(fromId, e.neighbor);

        reward += REWARD_CONFIG_V3.costScale * (-e.navigation_cost) * 0.1;
        this.routeCost += e.navigation_cost;
        this.routeRisk += e.risk;
        this.routeDifficulty += e.difficulty;
        this.routeDistance += e.distance;
        this.currentWp = e.neighbor;
        this.actualPath.push(this.currentWp);
        if (this.visited[this.currentWp]) { this.revisitCount++; reward += REWARD_CONFIG_V3.revisitPenalty; }
        this.visited[this.currentWp] = (this.visited[this.currentWp] || 0) + 1;

        if (collidedThisStep) {
          this.collisionCount++;
          reward += REWARD_CONFIG_V3.collisionPenalty;
        }
      }
      this.lastCollision = collidedThisStep ? 1 : 0;

      const reachedGoal = this.currentWp === this.goalWp;
      let timeout = this.stepCount >= this.maxSteps;
      const collisionTerminal = this.cfg.endOnCollision && collidedThisStep;

      // CTE running penalty (per step, scaled by mean CTE so far)
      const meanCte = this.cteSamples ? this.cteSum / this.cteSamples : 0;
      reward += REWARD_CONFIG_V3.ctePenalty * meanCte * 0.1;

      if (reachedGoal) {
        reward += REWARD_CONFIG_V3.goal;
        // docking accuracy = distance from final pose to the ideal dock point
        const g = this.states[this.goalWp];
        this.dockingAccuracy = distXY(this.posX, this.posY, g.x, g.y);
        // precision bonus: closer dock => larger bonus (real, bounded)
        const precision = Math.max(0, 1 - this.dockingAccuracy / DOCK_TOLERANCE);
        reward += REWARD_CONFIG_V3.dockBonusScale * precision;
      } else if (timeout) {
        reward += REWARD_CONFIG_V3.timeout;
      }

      this.totalReward += reward;

      let terminated = reachedGoal || collisionTerminal;
      if (timeout && !reachedGoal) this.truncated = true;
      this.done = terminated || this.truncated;

      const meanCteFinal = this.cteSamples ? this.cteSum / this.cteSamples : 0;
      const optimalityRatio = this.optimalCost > 0 && reachedGoal ? this.routeCost / this.optimalCost : null;

      const info = {
        currentWp: this.currentWp, goalWp: this.goalWp, reachedGoal, invalid, timeout,
        // routing metrics (as V2)
        routeCost: this.routeCost, optimalCost: this.optimalCost,
        excessCost: reachedGoal ? this.routeCost - this.optimalCost : null,
        optimalityRatio, routeRisk: this.routeRisk, routeDifficulty: this.routeDifficulty,
        routeDistance: this.routeDistance, invalidCount: this.invalidCount, revisitCount: this.revisitCount,
        // NEW physical / sensing / comms metrics (all real, computed)
        collided: collidedThisStep, collisionCount: this.collisionCount,
        cteMean: meanCteFinal, cteSum: this.cteSum,
        ialaViolations: this.ialaViolations,
        droppedFrames: this.droppedFrames,
        dockingAccuracy: this.dockingAccuracy,   // null unless reachedGoal
        posX: this.posX, posY: this.posY,
      };

      // build the agent's observation through the sensing + comms pipeline
      const trueObs = this._trueObs();
      const sensed = this._sensedObs(trueObs);
      const delivered = this._commObs(sensed);

      return { obs: delivered, reward, terminated, truncated: this.truncated, info };
    }
  }

  root.BintuluEnvV3 = {
    VesselEnvV3,
    REWARD_CONFIG_V3,
    MAX_EPISODE_STEPS_V3,
    N_ACTION_SLOTS,
    OBSTACLE_RADIUS,
    DOCK_TOLERANCE,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.BintuluEnvV3;
  }
})(typeof window !== "undefined" ? window : globalThis);
