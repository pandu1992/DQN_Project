/* ============================================================
 * BPAN — environmentBintuluMulti.js  (extension d: two independent vessels)
 * MultiVesselBintulu: two INDEPENDENT BintuluEnv vessels sharing the Bintulu
 * approach channels, each with its own policy + mission, that must avoid each
 * other. Mirrors the Synthetic Port MultiVesselEnvV4 design on the real chart.
 * Measures real inter-vessel collision / near-miss / closest-point-of-approach.
 * ============================================================ */
(function (root) {
  "use strict";
  const base = root.BintuluPortEnv;
  if (!base) throw new Error("environmentBintuluMulti requires environmentBintulu.js loaded first");
  const { BintuluEnv } = base;
  const N_SLOTS = 4;
  const VESSEL_COLLISION_RADIUS = 26;   // chart px; hull-hull separation below this = collision
  const VESSEL_NEARMISS_RADIUS = 60;
  const SUBSTEPS = 6;
  const CROSSING_BEARING = Math.PI / 8;
  const REWARD = { vesselCollision: -30.0, nearMiss: -3.0, giveWay: -5.0 };

  function makeRNG(seed) { let s = seed >>> 0; return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function distXY(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }
  function pointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
    if (len2 === 0) return distXY(px, py, ax, ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2; t = Math.max(0, Math.min(1, t));
    return distXY(px, py, ax + t * dx, ay + t * dy);
  }
  // Operational invariant shared with environmentBintuluOps: a surveyed port
  // channel is kept navigable, so no obstacle may fully seal a lane centreline
  // segment. Obstacles that would block a lane are nudged just clear of it,
  // perpendicular to that segment (preserving side); obstacles in open water are
  // untouched so drift under degradation can still carry a vessel into them.
  // Applied to the MULTI-VESSEL env only; the committed single-leg core env is
  // left byte-identical.
  const LANE_CLEARANCE = 8;
  function clearLaneCorridors(env) {
    const segs = [];
    for (const lane of ["NORTH", "SOUTH", "HARBOUR"]) {
      const ids = env.order[lane]; if (!ids || ids.length < 2) continue;
      for (let i = 0; i < ids.length - 1; i++) segs.push([env.states[ids[i]], env.states[ids[i + 1]]]);
    }
    for (const o of env.obstacles) {
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

  class MultiVesselBintulu {
    constructor(seed = 42, cfg = {}) {
      this.seed = seed;
      this.cfg = Object.assign({ noiseStd: 0, packetErrorRate: 0, obstacleCount: 6 }, cfg);
      this.A = new BintuluEnv(seed, this.cfg);
      this.B = new BintuluEnv(seed, this.cfg);
      clearLaneCorridors(this.A); clearLaneCorridors(this.B);
      // decorrelate vessel B's mission stream so the two get different missions
      this.B.missionRng = makeRNG(seed + 20240);
      this.MAX_X = this.A.MAX_X; this.MAX_Y = this.A.MAX_Y;
      this.nActions = N_SLOTS;
      this.perVesselExtra = 3;                 // sensed partner dx, dy, range
      this.obsDim = this.A.obsDim + this.perVesselExtra;
      this.partnerNoiseRngA = makeRNG(seed + 33001);
      this.partnerNoiseRngB = makeRNG(seed + 33002);
      this.reset();
    }
    _gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    _partnerChannel(obsv, other, rng) {
      const dx = (other.posX - obsv.posX) / this.MAX_X, dy = (other.posY - obsv.posY) / this.MAX_Y;
      const range = Math.min(1, distXY(obsv.posX, obsv.posY, other.posX, other.posY) / 400);
      const std = this.cfg.noiseStd;
      if (std <= 0) return [dx, dy, range];
      return [dx + this._gauss(rng) * std, dy + this._gauss(rng) * std, range + this._gauss(rng) * std];
    }
    _compose(baseObs, obsv, other, rng) { return baseObs.concat(this._partnerChannel(obsv, other, rng)); }

    reset() {
      this.stepCount = 0; this.done = false;
      const a0 = this.A.reset();
      let b0 = this.B.reset();
      let tries = 0;
      while (this.B.goalWp === this.A.goalWp && tries < 25) { b0 = this.B.reset(); tries++; }
      this.vesselCollisions = 0; this.nearMisses = 0; this.collisionContactSteps = 0;
      this.minCPA = Infinity; this.encounterSteps = 0; this.giveWayEvents = 0;
      this._inCollision = false; this._inNearMiss = false; this.lastVesselCollision = 0;
      this.A._v4h = null; this.B._v4h = null;
      return [this._compose(a0, this.A, this.B, this.partnerNoiseRngA),
              this._compose(b0, this.B, this.A, this.partnerNoiseRngB)];
    }
    _role(self, other) {
      const sh = self._v4h, oh = other._v4h;
      if (sh == null || oh == null) return "none";
      const bearing = Math.atan2(other.posY - self.posY, other.posX - self.posX);
      let rel = bearing - sh; while (rel > Math.PI) rel -= 2 * Math.PI; while (rel < -Math.PI) rel += 2 * Math.PI;
      let dh = sh - oh; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
      if (Math.abs(Math.abs(dh) - Math.PI) < CROSSING_BEARING && Math.abs(rel) < CROSSING_BEARING) return "head_on";
      if (rel > 0 && rel < Math.PI / 2) return "give_way";
      if (rel < 0 && rel > -Math.PI / 2) return "stand_on";
      return "none";
    }
    _hold(env) {
      const o = env._trueObs();
      return { obs: o, reward: 0, terminated: env.done && !env.truncated, truncated: env.truncated,
               info: { reachedGoal: env.currentWp === env.goalWp, invalid: false, held: true } };
    }
    stepBoth(aA, aB) {
      this.stepCount++;
      const A = this.A, B = this.B;
      const preA = { x: A.posX, y: A.posY }, preB = { x: B.posX, y: B.posY };
      const rA = A.done ? this._hold(A) : A.step(aA);
      const rB = B.done ? this._hold(B) : B.step(aB);
      const postA = { x: A.posX, y: A.posY }, postB = { x: B.posX, y: B.posY };
      A._v4h = (postA.x !== preA.x || postA.y !== preA.y) ? Math.atan2(postA.y - preA.y, postA.x - preA.x) : A._v4h;
      B._v4h = (postB.x !== preB.x || postB.y !== preB.y) ? Math.atan2(postB.y - preB.y, postB.x - preB.x) : B._v4h;
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
        if (["give_way", "head_on"].includes(roleA) || ["give_way", "head_on"].includes(roleB)) this.giveWayEvents++;
      }
      let eA = 0, eB = 0;
      if (collided) {
        if (!this._inCollision) this.vesselCollisions++;
        this._inCollision = true; this._inNearMiss = true; this.collisionContactSteps++;
        eA += REWARD.vesselCollision; eB += REWARD.vesselCollision; this.lastVesselCollision = 1;
      } else { this._inCollision = false; this.lastVesselCollision = 0; }
      if (nearMiss && !collided) {
        if (!this._inNearMiss) this.nearMisses++;
        this._inNearMiss = true; eA += REWARD.nearMiss; eB += REWARD.nearMiss;
      } else if (!collided) this._inNearMiss = false;
      if (nearMiss || collided) {
        if (["give_way", "head_on"].includes(roleA)) eA += REWARD.giveWay;
        if (["give_way", "head_on"].includes(roleB)) eB += REWARD.giveWay;
      }
      rA.reward += eA; rB.reward += eB; A.totalReward += eA; B.totalReward += eB;
      rA.obs = this._compose(rA.obs, A, B, this.partnerNoiseRngA);
      rB.obs = this._compose(rB.obs, B, A, this.partnerNoiseRngB);
      this.done = A.done && B.done;
      return { A: rA, B: rB, shared: { vesselCollisions: this.vesselCollisions, nearMisses: this.nearMisses, minCPA: this.minCPA, giveWayEvents: this.giveWayEvents, done: this.done } };
    }
  }
  root.BintuluMulti = { MultiVesselBintulu, VESSEL_COLLISION_RADIUS, VESSEL_NEARMISS_RADIUS };
  if (typeof module !== "undefined" && module.exports) module.exports = root.BintuluMulti;
})(typeof window !== "undefined" ? window : globalThis);
