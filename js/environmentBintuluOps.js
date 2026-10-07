/* ============================================================
 * BPAN — environmentBintuluOps.js  (extension e: operational realism)
 *   TwoPhaseBintulu : single vessel, inbound -> dock -> outbound round trip on
 *                     the Bintulu channels (success = full cycle).
 *   TwoWayBintulu   : an inbound and an outbound vessel share one access channel
 *                     in opposing directions -> head-on encounters (reuses the
 *                     multi-vessel inter-vessel geometry).
 * Mirrors the Synthetic Port TwoPhaseEnvV5 / TwoWayEnvV5 on the real chart.
 * ============================================================ */
(function (root) {
  "use strict";
  const base = root.BintuluPortEnv;
  if (!base) throw new Error("environmentBintuluOps requires environmentBintulu.js loaded first");
  const { BintuluEnv, dijkstra } = base;
  const N_SLOTS = 4, SUBSTEPS = 6;
  const VESSEL_COLLISION_RADIUS = 26, VESSEL_NEARMISS_RADIUS = 60, CROSSING_BEARING = Math.PI / 8;
  const REWARD = { phaseBonus: 40.0, vesselCollision: -30.0, nearMiss: -3.0, giveWay: -5.0 };
  function makeRNG(seed) { let s = seed >>> 0; return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function distXY(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }
  function pointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
    if (len2 === 0) return distXY(px, py, ax, ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2; t = Math.max(0, Math.min(1, t));
    return distXY(px, py, ax + t * dx, ay + t * dy);
  }

  /* Operational-realism invariant: a working port ALWAYS keeps a navigable
   * approach to each berth. The core single-leg study averages over many
   * missions, so a seed that happens to seal one berth's final approach edge is
   * masked by aggregation; but a round-trip/opposed-traffic mission that MUST
   * reach a specific berth would then be impossible. We therefore enforce a
   * minimum clearance along each terminal's final approach segment on the
   * operational envs only (the committed core env is left byte-identical).
   * Obstacles that intrude on a berth-approach corridor are nudged clear. */
  const LANE_CLEARANCE = 8;   // px of navigable water the vessel body needs past the obstacle radius
  // Collect every sequential centreline segment of the access/harbour lanes.
  function laneSegments(env) {
    const segs = [];
    for (const lane of ["NORTH", "SOUTH", "HARBOUR"]) {
      const ids = env.order[lane]; if (!ids || ids.length < 2) continue;
      for (let i = 0; i < ids.length - 1; i++) segs.push([env.states[ids[i]], env.states[ids[i + 1]]]);
    }
    return segs;
  }
  // A surveyed channel is kept navigable: nudge any obstacle that would seal a
  // lane centreline segment just clear of the corridor (perpendicular to that
  // segment), preserving its side. Obstacles in open water are untouched, so
  // drift under noise/comms degradation can still carry a vessel into them.
  // Applied to the OPERATIONAL envs only; the committed core env is unchanged.
  function clearLaneCorridors(env) {
    const segs = laneSegments(env);
    for (const o of env.obstacles) {
      // iterate a few passes so a nudge away from one segment that pushes it onto
      // another still converges to a globally clear position.
      for (let pass = 0; pass < 4; pass++) {
        let moved = false;
        for (const [A, B] of segs) {
          const d = pointToSegment(o.x, o.y, A.x, A.y, B.x, B.y);
          const need = o.r + LANE_CLEARANCE;
          if (d < need) {
            const dx = B.x - A.x, dy = B.y - A.y, len = Math.hypot(dx, dy) || 1;
            let nx = -dy / len, ny = dx / len;
            const side = (B.x - A.x) * (o.y - A.y) - (B.y - A.y) * (o.x - A.x);
            if (side < 0) { nx = -nx; ny = -ny; }
            const shift = (need - d) + 2;
            o.x += nx * shift; o.y += ny * shift; moved = true;
          }
        }
        if (!moved) break;
      }
    }
    return env;
  }
  const clearBerthApproaches = clearLaneCorridors;   // alias (name used at call sites)

  // ---- Scenario A: two-phase round trip (single vessel) ----
  class TwoPhaseBintulu {
    constructor(seed = 42, cfg = {}) {
      this.seed = seed; this.cfg = Object.assign({ noiseStd: 0, packetErrorRate: 0, obstacleCount: 6 }, cfg);
      this.v = new BintuluEnv(seed, this.cfg);
      clearBerthApproaches(this.v);
      this.states = this.v.states; this.order = this.v.order; this.graph = this.v.graph;
      this.MAX_X = this.v.MAX_X; this.MAX_Y = this.v.MAX_Y; this.nActions = N_SLOTS;
      this.v.maxSteps = 120; this.maxSteps = 120;
      this.obsDim = this.v.obsDim + 2;              // + phase flag, cycle progress
      this.missionRng = makeRNG(seed + 51000);
      this.reset();
    }
    _roundTrip() {
      // Round trip within ONE access channel (realistic: a vessel transits its
      // approach channel to the inner berth and departs the same way). Berth =
      // the harbour-end terminal of the chosen lane; no cross-channel shortcut.
      const lane = this.missionRng() < 0.5 ? "NORTH" : "SOUTH";
      const ids = this.order[lane];
      const entrance = ids[0];
      const berth = ids[ids.length - 1];          // inner berth of this lane
      const exit = ids[0];                          // depart back out the same channel
      const inbound = dijkstra(this.graph, entrance, berth);
      const outbound = dijkstra(this.graph, berth, exit);
      if (!inbound.found || !outbound.found) {
        const e0 = this.order.NORTH[0], b0 = this.order.NORTH[this.order.NORTH.length - 1];
        return { entrance: e0, berth: b0, exit: e0, inbound: dijkstra(this.graph, e0, b0), outbound: dijkstra(this.graph, b0, e0) };
      }
      return { entrance, berth, exit, inbound, outbound };
    }
    reset(mission = null) {
      this.m = mission || this._roundTrip();
      this.phase = 1; this.cycleComplete = false; this.phase1Steps = null; this.dockAccuracyLeg1 = null;
      this.v.reset({ start: this.m.entrance, goal: this.m.berth, plan: this.m.inbound });
      this.stepCount = 0; this.totalReward = 0; this.done = false; this.truncated = false; this._bonus = 0;
      return this._compose(this.v._sensedObs(this.v._trueObs()));
    }
    _compose(o) { return o.concat([this.phase === 2 ? 1 : 0, this.phase === 1 ? 0.0 : 0.5]); }
    _retarget() {
      const v = this.v, plan = this.m.outbound.found ? this.m.outbound : dijkstra(this.graph, v.currentWp, this.m.exit);
      v.goalWp = this.m.exit; v.plannedPath = plan.found ? plan.nodes.slice() : [v.currentWp, this.m.exit];
      v.plannedDistance = plan.totalDistance || 0; v.optimalCost = plan.totalCost || 0;
      v.routeCost = 0; v.visited = { [v.currentWp]: 1 }; v.dockingAccuracy = null;
    }
    step(action) {
      this.stepCount++;
      const r = this.v.step(action); let extra = 0;
      if (r.info.reachedGoal) {
        if (this.phase === 1) { this.phase1Steps = this.stepCount; this.dockAccuracyLeg1 = this.v.dockingAccuracy; extra += REWARD.phaseBonus; this.phase = 2; this._retarget(); }
        else this.cycleComplete = true;
      }
      const timeout = this.stepCount >= this.maxSteps;
      let terminated = this.cycleComplete;
      if (timeout && !this.cycleComplete) this.truncated = true;
      this.done = terminated || this.truncated;
      this._bonus += extra; this.totalReward = this.v.totalReward + this._bonus;
      const info = Object.assign({}, r.info, { phase: this.phase, cycleComplete: this.cycleComplete, phase1Steps: this.phase1Steps, dockAccuracyLeg1: this.dockAccuracyLeg1, fullSuccess: this.cycleComplete ? 1 : 0 });
      return { obs: this._compose(r.obs), reward: r.reward + extra, terminated: this.done && !this.truncated, truncated: this.truncated, info };
    }
    get collisionCount() { return this.v.collisionCount; } get cteSum() { return this.v.cteSum; } get cteSamples() { return this.v.cteSamples; }
    get ialaViolations() { return this.v.ialaViolations; } get droppedFrames() { return this.v.droppedFrames; }
    get currentWp() { return this.v.currentWp; } get goalWp() { return this.v.goalWp; } get posX() { return this.v.posX; } get posY() { return this.v.posY; }
    get startWp() { return this.m.entrance; } get plannedPath() { return this.v.plannedPath; }
  }

  // ---- Scenario B: two-way opposing traffic ----
  class TwoWayBintulu {
    constructor(seed = 42, cfg = {}) {
      this.seed = seed; this.cfg = Object.assign({ noiseStd: 0, packetErrorRate: 0, obstacleCount: 6 }, cfg);
      this.IN = new BintuluEnv(seed, this.cfg); this.OUT = new BintuluEnv(seed, this.cfg);
      clearBerthApproaches(this.IN); clearBerthApproaches(this.OUT);
      this.states = this.IN.states; this.order = this.IN.order; this.graph = this.IN.graph;
      this.MAX_X = this.IN.MAX_X; this.MAX_Y = this.IN.MAX_Y; this.nActions = N_SLOTS;
      this.obsDim = this.IN.obsDim + 3;
      this.laneRng = makeRNG(seed + 52000);
      this.pnIn = makeRNG(seed + 53001); this.pnOut = makeRNG(seed + 53002);
      this.reset();
    }
    _gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    _opposing() {
      // Opposing traffic in ONE access channel: an INBOUND vessel runs from the
      // sea entrance toward an inner berth-approach node, while an OUTBOUND
      // vessel runs the SAME channel from that inner node back toward the sea
      // entrance -> a genuine head-on on a shared centreline. Goals stay on the
      // lane (no harbour-connector diversion, no sealed far berth) so the task
      // is a clean COLREGs encounter; difficulty comes from degradation + the
      // opposing vessel, not from an impassable channel.
      const lane = this.laneRng() < 0.5 ? "NORTH" : "SOUTH";
      const ids = this.order[lane];
      const entrance = ids[0];
      // inner turn-point: a node near (but not at) the inner end of the lane
      const innerIdx = ids.length - 2;         // second-to-last lane node (berth approach)
      const inner = ids[innerIdx];
      return { lane,
               inbound: { start: entrance, goal: inner, plan: dijkstra(this.graph, entrance, inner) },
               outbound: { start: inner, goal: entrance, plan: dijkstra(this.graph, inner, entrance) } };
    }
    _partner(obsv, other, rng) {
      const dx = (other.posX - obsv.posX) / this.MAX_X, dy = (other.posY - obsv.posY) / this.MAX_Y;
      const range = Math.min(1, distXY(obsv.posX, obsv.posY, other.posX, other.posY) / 400);
      const std = this.cfg.noiseStd;
      if (std <= 0) return [dx, dy, range];
      return [dx + this._gauss(rng) * std, dy + this._gauss(rng) * std, range + this._gauss(rng) * std];
    }
    _compose(o, obsv, other, rng) { return o.concat(this._partner(obsv, other, rng)); }
    reset() {
      this.stepCount = 0; this.done = false;
      const mm = this._opposing(); this.lane = mm.lane;
      this.IN.reset(mm.inbound); this.OUT.reset(mm.outbound);
      this.vesselCollisions = 0; this.nearMisses = 0; this.collisionContactSteps = 0;
      this.minCPA = Infinity; this.encounterSteps = 0; this.giveWayEvents = 0; this.headOnEvents = 0;
      this._inColl = false; this._inNear = false; this.IN._h = null; this.OUT._h = null;
      return [this._compose(this.IN._sensedObs(this.IN._trueObs()), this.IN, this.OUT, this.pnIn),
              this._compose(this.OUT._sensedObs(this.OUT._trueObs()), this.OUT, this.IN, this.pnOut)];
    }
    _role(self, other) {
      const sh = self._h, oh = other._h; if (sh == null || oh == null) return "none";
      const bearing = Math.atan2(other.posY - self.posY, other.posX - self.posX);
      let rel = bearing - sh; while (rel > Math.PI) rel -= 2 * Math.PI; while (rel < -Math.PI) rel += 2 * Math.PI;
      let dh = sh - oh; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
      if (Math.abs(Math.abs(dh) - Math.PI) < CROSSING_BEARING && Math.abs(rel) < CROSSING_BEARING) return "head_on";
      if (rel > 0 && rel < Math.PI / 2) return "give_way";
      if (rel < 0 && rel > -Math.PI / 2) return "stand_on";
      return "none";
    }
    _hold(env) { return { obs: env._trueObs(), reward: 0, terminated: env.done && !env.truncated, truncated: env.truncated, info: { reachedGoal: env.currentWp === env.goalWp, held: true } }; }
    stepBoth(aIn, aOut) {
      this.stepCount++;
      const A = this.IN, B = this.OUT;
      const preA = { x: A.posX, y: A.posY }, preB = { x: B.posX, y: B.posY };
      const rA = A.done ? this._hold(A) : A.step(aIn);
      const rB = B.done ? this._hold(B) : B.step(aOut);
      const postA = { x: A.posX, y: A.posY }, postB = { x: B.posX, y: B.posY };
      A._h = (postA.x !== preA.x || postA.y !== preA.y) ? Math.atan2(postA.y - preA.y, postA.x - preA.x) : A._h;
      B._h = (postB.x !== preB.x || postB.y !== preB.y) ? Math.atan2(postB.y - preB.y, postB.x - preB.x) : B._h;
      let collided = false, nearMiss = false, stepMinCPA = Infinity;
      for (let k = 1; k <= SUBSTEPS; k++) {
        const t = k / SUBSTEPS;
        const ax = preA.x + (postA.x - preA.x) * t, ay = preA.y + (postA.y - preA.y) * t;
        const bx = preB.x + (postB.x - preB.x) * t, by = preB.y + (postB.y - preB.y) * t;
        const sep = distXY(ax, ay, bx, by);
        if (sep < stepMinCPA) stepMinCPA = sep;
        if (sep <= VESSEL_COLLISION_RADIUS) collided = true; else if (sep <= VESSEL_NEARMISS_RADIUS) nearMiss = true;
      }
      if (stepMinCPA < this.minCPA) this.minCPA = stepMinCPA;
      let roleA = "none", roleB = "none";
      if (stepMinCPA <= VESSEL_NEARMISS_RADIUS * 1.5) {
        this.encounterSteps++; roleA = this._role(A, B); roleB = this._role(B, A);
        if (roleA === "head_on" || roleB === "head_on") this.headOnEvents++;
        if (["give_way", "head_on"].includes(roleA) || ["give_way", "head_on"].includes(roleB)) this.giveWayEvents++;
      }
      let eA = 0, eB = 0;
      if (collided) { if (!this._inColl) this.vesselCollisions++; this._inColl = true; this._inNear = true; this.collisionContactSteps++; eA += REWARD.vesselCollision; eB += REWARD.vesselCollision; }
      else this._inColl = false;
      if (nearMiss && !collided) { if (!this._inNear) this.nearMisses++; this._inNear = true; eA += REWARD.nearMiss; eB += REWARD.nearMiss; } else if (!collided) this._inNear = false;
      if (nearMiss || collided) { if (["give_way", "head_on"].includes(roleA)) eA += REWARD.giveWay; if (["give_way", "head_on"].includes(roleB)) eB += REWARD.giveWay; }
      rA.reward += eA; rB.reward += eB; A.totalReward += eA; B.totalReward += eB;
      rA.obs = this._compose(rA.obs, A, B, this.pnIn); rB.obs = this._compose(rB.obs, B, A, this.pnOut);
      this.done = A.done && B.done;
      return { IN: rA, OUT: rB, shared: { vesselCollisions: this.vesselCollisions, nearMisses: this.nearMisses, minCPA: this.minCPA, headOnEvents: this.headOnEvents, done: this.done, lane: this.lane } };
    }
  }
  root.BintuluOps = { TwoPhaseBintulu, TwoWayBintulu, VESSEL_COLLISION_RADIUS, VESSEL_NEARMISS_RADIUS };
  if (typeof module !== "undefined" && module.exports) module.exports = root.BintuluOps;
})(typeof window !== "undefined" ? window : globalThis);
