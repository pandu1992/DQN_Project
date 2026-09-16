/* ============================================================
 * SYNTHETIC PORT — Environment V5 (mission-cycle + two-way traffic)
 * environmentV5.js
 *
 * STUDY 5 — Two complementary realism extensions of the single-vessel
 * physical/sensing/comms testbed (VesselEnvV3), each combined with the
 * Study-2 degradation grid:
 *
 *   SCENARIO A — TWO-PHASE ROUND-TRIP MISSION (single vessel).
 *     A real port call is not one-way: a vessel comes IN, DOCKS, then goes
 *     back OUT. TwoPhaseEnvV5 wraps a VesselEnvV3 and requires the agent to
 *     complete BOTH legs in one episode:
 *        phase 1 (INBOUND):  entrance  -> harbour berth (dock)
 *        phase 2 (OUTBOUND): harbour berth -> a channel exit
 *     When the inbound goal is reached the environment AUTOMATICALLY re-targets
 *     the vessel to the outbound goal (same continuous pose, no teleport) and
 *     flips a phase flag in the observation. "Success" means the FULL cycle
 *     completed within the (larger) step budget. All Study-2 metrics accumulate
 *     across both legs; docking accuracy is captured at the berth (end of leg 1).
 *
 *   SCENARIO B — TWO-WAY TRAFFIC (two vessels, opposite directions).
 *     Two vessels share the channel travelling in OPPOSITE directions — one
 *     INBOUND (entrance -> harbour), one OUTBOUND (harbour -> entrance) — so
 *     they must pass each other head-on / crossing in the same water.
 *     TwoWayEnvV5 composes two VesselEnvV3 vessels (like Study 4's multi-vessel
 *     env) but FORCES opposing missions on the SAME access lane, then measures
 *     the same real inter-vessel geometry (collision r=18, near-miss=40, CPA,
 *     COLREGs give-way role) as Study 4. The difference from Study 4 is the
 *     traffic PATTERN: guaranteed opposing flow on a shared lane rather than
 *     two independent random missions.
 *
 * The base graph/edges/Dijkstra/buoys are reused UNCHANGED. Only the mission
 * structure (round-trip re-targeting) and the traffic pattern (opposing flow)
 * are added on top of the existing physical/sensing/comms machinery. The
 * deployed web app (environment.js) is NOT affected — offline research build.
 * ============================================================ */

(function (root) {
  "use strict";

  const baseV3 = root.BintuluEnvV3;
  const baseEnv = root.BintuluEnv;
  if (!baseV3) throw new Error("environmentV5 requires environmentV3.js loaded first");
  if (!baseEnv) throw new Error("environmentV5 requires environment.js loaded first");
  const { VesselEnvV3, N_ACTION_SLOTS, DOCK_TOLERANCE } = baseV3;
  const { MAP_W, MAP_H, dijkstra } = baseEnv;

  function makeRNG(seed) {
    let s = seed >>> 0;
    return function () {
      s |= 0; s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function distXY(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

  const VESSEL_COLLISION_RADIUS = 18;
  const VESSEL_NEARMISS_RADIUS = 40;
  const KINEMATIC_SUBSTEPS = 6;
  const CROSSING_BEARING = Math.PI / 8;

  const REWARD_CONFIG_V5 = {
    phaseBonus: 40.0,             // reward for completing the inbound leg (docking)
    vesselCollisionPenalty: -30.0,
    nearMissPenalty: -3.0,
    giveWayViolationPenalty: -5.0,
  };

  // ------------------------------------------------------------------
  // SCENARIO A — TwoPhaseEnvV5 (single vessel, inbound -> dock -> outbound)
  // ------------------------------------------------------------------
  class TwoPhaseEnvV5 {
    constructor(seed = 42, cfg = {}) {
      this.seed = seed;
      this.cfg = Object.assign({
        noiseStd: 0.0, packetErrorRate: 0.0, obstacleCount: 6,
      }, cfg);

      // underlying physical/sensing/comms vessel (its own metrics are reused)
      this.v = new VesselEnvV3(seed, this.cfg);
      this.states = this.v.states; this.order = this.v.order; this.graph = this.v.graph;
      this.buoys = this.v.buoys; this.adj = this.v.adj; this.obstacles = this.v.obstacles;
      this.MAX_X = MAP_W; this.MAX_Y = MAP_H;

      // the round trip needs a bigger budget than a single leg
      this.v.maxSteps = 110;
      this.maxSteps = this.v.maxSteps;
      this.nActions = N_ACTION_SLOTS;
      // obs = VesselEnvV3 obs (28) + [phase flag, cycle progress]
      this.obsDim = this.v.obsDim + 2;

      this.missionRng = makeRNG(seed + 51000);
      this.reset();
    }

    laneEncoding(l) { return this.v.laneEncoding(l); }

    // pick an entrance waypoint on an access lane + the harbour berth + an exit.
    _roundTripMission() {
      const harbourEnd = this.order.L01[this.order.L01.length - 1];
      // choose which access lane the vessel enters/leaves through
      const lane = this.missionRng() < 0.5 ? "L02" : "L08";
      const laneIds = this.order[lane];
      const entrance = laneIds[0];
      // outbound exit: the far entrance of the SAME lane (a full transit) — or
      // the other lane's entrance, chosen at random for variety.
      const exitLane = this.missionRng() < 0.5 ? lane : (lane === "L02" ? "L08" : "L02");
      const exit = this.order[exitLane][0];
      const inbound = dijkstra(this.graph, entrance, harbourEnd);
      const outbound = dijkstra(this.graph, harbourEnd, exit);
      if (!inbound.found || !outbound.found) {
        // fallback: L02 in, L02 out
        const e0 = this.order.L02[0];
        return { entrance: e0, berth: harbourEnd, exit: e0,
                 inbound: dijkstra(this.graph, e0, harbourEnd),
                 outbound: dijkstra(this.graph, harbourEnd, e0) };
      }
      return { entrance, berth: harbourEnd, exit, inbound, outbound };
    }

    reset(mission = null) {
      this.m = mission || this._roundTripMission();
      this.phase = 1;                       // 1 = inbound, 2 = outbound
      this.cycleComplete = false;
      this.phase1Steps = null;
      this.dockAccuracyLeg1 = null;
      // start the underlying vessel on the inbound leg
      this.v.reset({ start: this.m.entrance, goal: this.m.berth, plan: this.m.inbound });
      this.stepCount = 0;
      this.totalReward = 0;
      this.done = false; this.truncated = false;
      return this._compose(this.v._sensedObs(this.v._trueObs()));
    }

    _compose(baseObs) {
      const cycleProgress = this.phase === 1 ? 0.0 : 0.5;  // coarse phase progress
      return baseObs.concat([this.phase === 2 ? 1 : 0, cycleProgress]);
    }

    step(action) {
      this.stepCount++;
      const r = this.v.step(action);
      let reward = r.reward;

      // Did we just reach the current leg's goal?
      if (r.info.reachedGoal) {
        if (this.phase === 1) {
          // completed the INBOUND leg (docked). Capture docking accuracy, award
          // the phase bonus, and AUTOMATICALLY re-target to the OUTBOUND leg
          // WITHOUT moving the vessel (continuous pose is preserved).
          this.phase1Steps = this.stepCount;
          this.dockAccuracyLeg1 = this.v.dockingAccuracy;
          reward += REWARD_CONFIG_V5.phaseBonus;
          this.phase = 2;
          this._retargetOutbound();
        } else {
          // completed the OUTBOUND leg -> full cycle done
          this.cycleComplete = true;
        }
      }

      const timeout = this.stepCount >= this.maxSteps;
      // episode terminates on full cycle OR timeout
      let terminated = this.cycleComplete;
      if (timeout && !this.cycleComplete) this.truncated = true;
      this.done = terminated || this.truncated;
      this.totalReward += (reward - r.reward); // add only our extra shaping; v.totalReward already has r.reward
      // keep a single authoritative total: v.totalReward + accumulated bonuses
      // (recompute cleanly below)

      const info = Object.assign({}, r.info, {
        phase: this.phase, cycleComplete: this.cycleComplete,
        phase1Steps: this.phase1Steps, dockAccuracyLeg1: this.dockAccuracyLeg1,
        fullSuccess: this.cycleComplete ? 1 : 0,
      });

      const delivered = this._compose(r.obs);
      // authoritative total reward = underlying vessel total + our phase bonuses
      this._bonusTotal = (this._bonusTotal || 0) + (reward - r.reward);
      this.totalReward = this.v.totalReward + this._bonusTotal;

      return { obs: delivered, reward, terminated: this.done && !this.truncated, truncated: this.truncated, info };
    }

    // Re-target the underlying vessel to the outbound goal, preserving its
    // current continuous pose and current waypoint. We rebuild the plan/goal
    // fields the vessel uses for observation + CTE, but do NOT reset the pose.
    _retargetOutbound() {
      const v = this.v;
      const plan = this.m.outbound.found ? this.m.outbound : dijkstra(this.graph, v.currentWp, this.m.exit);
      v.goalWp = this.m.exit;
      v.plannedPath = plan.found ? plan.nodes.slice() : [v.currentWp, this.m.exit];
      v.plannedDistance = plan.totalDistance || 0;
      v.optimalCost = plan.totalCost || 0;
      // reset per-leg routing accumulators so leg-2 optimality is meaningful,
      // but keep safety counters (collisions, IALA, dropped frames) cumulative.
      v.routeCost = 0; v.routeRisk = 0; v.routeDifficulty = 0; v.routeDistance = 0;
      v.visited = { [v.currentWp]: 1 };
      v.dockingAccuracy = null;
    }

    // expose vessel metrics for the harness
    get collisionCount() { return this.v.collisionCount; }
    get cteSum() { return this.v.cteSum; }
    get cteSamples() { return this.v.cteSamples; }
    get ialaViolations() { return this.v.ialaViolations; }
    get droppedFrames() { return this.v.droppedFrames; }
    get currentWp() { return this.v.currentWp; }
    get goalWp() { return this.v.goalWp; }
    get plannedPath() { return this.v.plannedPath; }
    get posX() { return this.v.posX; }
    get posY() { return this.v.posY; }
    get optimalCost() { return this.v.optimalCost; }
    get routeCost() { return this.v.routeCost; }
    get startWp() { return this.m.entrance; }
  }

  // ------------------------------------------------------------------
  // SCENARIO B — TwoWayEnvV5 (two vessels, opposing flow on a shared lane)
  // ------------------------------------------------------------------
  class TwoWayEnvV5 {
    constructor(seed = 42, cfg = {}) {
      this.seed = seed;
      this.cfg = Object.assign({
        noiseStd: 0.0, packetErrorRate: 0.0, obstacleCount: 6,
      }, cfg);

      this.IN = new VesselEnvV3(seed, this.cfg);   // inbound vessel
      this.OUT = new VesselEnvV3(seed, this.cfg);  // outbound vessel (opposite direction)

      this.states = this.IN.states; this.order = this.IN.order; this.graph = this.IN.graph;
      this.MAX_X = MAP_W; this.MAX_Y = MAP_H;
      this.nActions = N_ACTION_SLOTS;
      this.perVesselExtra = 3;                     // sensed partner dx,dy,range
      this.obsDim = this.IN.obsDim + this.perVesselExtra;

      this.laneRng = makeRNG(seed + 52000);
      this.partnerNoiseRngIn = makeRNG(seed + 53001);
      this.partnerNoiseRngOut = makeRNG(seed + 53002);
      this.reset();
    }

    _gauss(rng) {
      let u = 0, v = 0;
      while (u === 0) u = rng();
      while (v === 0) v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    // Force OPPOSING missions on the SAME access lane so the two vessels meet
    // head-on in a SHARED mid-lane segment. Both transits are kept to a
    // tractable length (a full end-to-end lane transit is ~24 steps, too long
    // to learn in budget and would make the encounter timing brittle): the
    // inbound vessel runs from a mid-lane point INTO the harbour, and the
    // outbound vessel runs FROM the harbour to a mid-lane exit point on the
    // SAME lane. Their paths overlap on the lane's upper half, so they still
    // pass each other head-on, but neither has to cross the entire lane.
    _opposingMissions() {
      const lane = this.laneRng() < 0.5 ? "L02" : "L08";
      const ids = this.order[lane];
      const harbourEnd = this.order.L01[this.order.L01.length - 1];
      // inbound starts partway up the lane (index ~ 45-60% of the way in) so it
      // still has to transit the shared upper segment + junction into harbour.
      const inIdx = Math.floor(ids.length * (0.45 + 0.15 * this.laneRng()));
      const inStart = ids[Math.min(inIdx, ids.length - 2)];
      // outbound exits to a mid-lane point on the same lane (index ~ 55-70%),
      // so it travels harbour -> down the shared segment (meeting the inbound).
      const outIdx = Math.floor(ids.length * (0.55 + 0.15 * this.laneRng()));
      const outExit = ids[Math.min(outIdx, ids.length - 2)];
      return {
        lane,
        inbound: { start: inStart, goal: harbourEnd, plan: dijkstra(this.graph, inStart, harbourEnd) },
        outbound: { start: harbourEnd, goal: outExit, plan: dijkstra(this.graph, harbourEnd, outExit) },
      };
    }

    _partnerChannel(observer, other, rng) {
      const dx = (other.posX - observer.posX) / this.MAX_X;
      const dy = (other.posY - observer.posY) / this.MAX_Y;
      const range = Math.min(1, distXY(observer.posX, observer.posY, other.posX, other.posY) / 300);
      const std = this.cfg.noiseStd;
      if (std <= 0) return [dx, dy, range];
      return [dx + this._gauss(rng) * std, dy + this._gauss(rng) * std, range + this._gauss(rng) * std];
    }

    _compose(baseObs, observer, other, rng) {
      return baseObs.concat(this._partnerChannel(observer, other, rng));
    }

    reset() {
      this.stepCount = 0; this.done = false;
      const mm = this._opposingMissions();
      this.lane = mm.lane;
      this.IN.reset(mm.inbound);
      this.OUT.reset(mm.outbound);
      // inter-vessel metrics
      this.vesselCollisions = 0; this.nearMisses = 0; this.collisionContactSteps = 0;
      this.minCPA = Infinity; this.encounterSteps = 0; this.giveWayEvents = 0;
      this.headOnEvents = 0;
      this._inCollision = false; this._inNearMiss = false;
      this.IN._v5heading = null; this.OUT._v5heading = null;
      const oIn = this._compose(this.IN._sensedObs(this.IN._trueObs()), this.IN, this.OUT, this.partnerNoiseRngIn);
      const oOut = this._compose(this.OUT._sensedObs(this.OUT._trueObs()), this.OUT, this.IN, this.partnerNoiseRngOut);
      return [oIn, oOut];
    }

    _giveWayRole(self, other) {
      const sh = self._v5heading, oh = other._v5heading;
      if (sh == null || oh == null) return "none";
      const bearing = Math.atan2(other.posY - self.posY, other.posX - self.posX);
      let rel = bearing - sh;
      while (rel > Math.PI) rel -= 2 * Math.PI;
      while (rel < -Math.PI) rel += 2 * Math.PI;
      let dh = sh - oh;
      while (dh > Math.PI) dh -= 2 * Math.PI;
      while (dh < -Math.PI) dh += 2 * Math.PI;
      const reciprocal = Math.abs(Math.abs(dh) - Math.PI) < CROSSING_BEARING;
      if (reciprocal && Math.abs(rel) < CROSSING_BEARING) return "head_on";
      if (rel > 0 && rel < Math.PI / 2) return "give_way";
      if (rel < 0 && rel > -Math.PI / 2) return "stand_on";
      return "none";
    }

    _holdResult(env) {
      const trueObs = env._trueObs();
      return {
        obs: trueObs, reward: 0, terminated: env.done && !env.truncated, truncated: env.truncated,
        info: { currentWp: env.currentWp, goalWp: env.goalWp, reachedGoal: env.currentWp === env.goalWp,
          invalid: false, timeout: env.truncated, collided: false, held: true },
      };
    }

    stepBoth(actIn, actOut) {
      this.stepCount++;
      const A = this.IN, B = this.OUT;
      const preA = { x: A.posX, y: A.posY }, preB = { x: B.posX, y: B.posY };
      const rA = A.done ? this._holdResult(A) : A.step(actIn);
      const rB = B.done ? this._holdResult(B) : B.step(actOut);
      const postA = { x: A.posX, y: A.posY }, postB = { x: B.posX, y: B.posY };

      A._v5heading = (postA.x !== preA.x || postA.y !== preA.y) ? Math.atan2(postA.y - preA.y, postA.x - preA.x) : A._v5heading;
      B._v5heading = (postB.x !== preB.x || postB.y !== preB.y) ? Math.atan2(postB.y - preB.y, postB.x - preB.x) : B._v5heading;

      let collided = false, nearMiss = false, stepMinCPA = Infinity;
      for (let k = 1; k <= KINEMATIC_SUBSTEPS; k++) {
        const t = k / KINEMATIC_SUBSTEPS;
        const ax = preA.x + (postA.x - preA.x) * t, ay = preA.y + (postA.y - preA.y) * t;
        const bx = preB.x + (postB.x - preB.x) * t, by = preB.y + (postB.y - preB.y) * t;
        const sep = distXY(ax, ay, bx, by);
        if (sep < stepMinCPA) stepMinCPA = sep;
        if (sep <= VESSEL_COLLISION_RADIUS) collided = true;
        else if (sep <= VESSEL_NEARMISS_RADIUS) nearMiss = true;
      }
      if (stepMinCPA < this.minCPA) this.minCPA = stepMinCPA;

      let roleA = "none", roleB = "none";
      if (stepMinCPA <= VESSEL_NEARMISS_RADIUS * 1.5) {
        this.encounterSteps++;
        roleA = this._giveWayRole(A, B); roleB = this._giveWayRole(B, A);
        if (roleA === "head_on" || roleB === "head_on") this.headOnEvents++;
        if (roleA === "give_way" || roleA === "head_on" || roleB === "give_way" || roleB === "head_on") this.giveWayEvents++;
      }

      let extraA = 0, extraB = 0;
      if (collided) {
        if (!this._inCollision) this.vesselCollisions++;
        this._inCollision = true; this._inNearMiss = true;
        extraA += REWARD_CONFIG_V5.vesselCollisionPenalty; extraB += REWARD_CONFIG_V5.vesselCollisionPenalty;
        this.collisionContactSteps++;
      } else {
        this._inCollision = false;
      }
      if (nearMiss && !collided) {
        if (!this._inNearMiss) this.nearMisses++;
        this._inNearMiss = true;
        extraA += REWARD_CONFIG_V5.nearMissPenalty; extraB += REWARD_CONFIG_V5.nearMissPenalty;
      } else if (!collided) {
        this._inNearMiss = false;
      }
      if (nearMiss || collided) {
        if (roleA === "give_way" || roleA === "head_on") extraA += REWARD_CONFIG_V5.giveWayViolationPenalty;
        if (roleB === "give_way" || roleB === "head_on") extraB += REWARD_CONFIG_V5.giveWayViolationPenalty;
      }
      rA.reward += extraA; rB.reward += extraB;
      A.totalReward += extraA; B.totalReward += extraB;

      rA.info = Object.assign({}, rA.info, { vesselCollision: collided ? 1 : 0, nearMiss: nearMiss ? 1 : 0, cpa: stepMinCPA, giveWayRole: roleA });
      rB.info = Object.assign({}, rB.info, { vesselCollision: collided ? 1 : 0, nearMiss: nearMiss ? 1 : 0, cpa: stepMinCPA, giveWayRole: roleB });

      rA.obs = this._compose(rA.obs, A, B, this.partnerNoiseRngIn);
      rB.obs = this._compose(rB.obs, B, A, this.partnerNoiseRngOut);

      this.done = A.done && B.done;
      const shared = {
        vesselCollisions: this.vesselCollisions, nearMisses: this.nearMisses,
        collisionContactSteps: this.collisionContactSteps, minCPA: this.minCPA,
        encounterSteps: this.encounterSteps, giveWayEvents: this.giveWayEvents, headOnEvents: this.headOnEvents,
        stepCount: this.stepCount, done: this.done, lane: this.lane,
      };
      return { IN: rA, OUT: rB, shared };
    }
  }

  root.BintuluEnvV5 = {
    TwoPhaseEnvV5,
    TwoWayEnvV5,
    REWARD_CONFIG_V5,
    VESSEL_COLLISION_RADIUS,
    VESSEL_NEARMISS_RADIUS,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.BintuluEnvV5;
  }
})(typeof window !== "undefined" ? window : globalThis);
