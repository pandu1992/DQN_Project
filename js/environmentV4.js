/* ============================================================
 * SYNTHETIC PORT — Environment V4 (multi-vessel encounter layer)
 * environmentV4.js
 *
 * STUDY 4 — Two INDEPENDENT autonomous vessels share the same port
 * channel network and must AVOID EACH OTHER. Each vessel is driven by
 * its own policy (a separate agent instance); neither is a central
 * controller. This is the multi-agent extension of the single-vessel
 * physical/sensing/comms testbed (VesselEnvV3, Study 2/3).
 *
 * WHY IT MATTERS
 *   A single agent on an empty channel never has to negotiate a moving
 *   partner. With two independent vessels, the safety-critical event is
 *   no longer "hull hits a static obstacle" but "two hulls converge" —
 *   i.e. a close-quarters situation / collision between vessels. An agent
 *   that ignores the other vessel WILL cause inter-vessel collisions;
 *   an agent that yields (COLREGs give-way spirit) avoids them.
 *
 * DESIGN (honest, everything measured — nothing fabricated)
 *   - Composes TWO VesselEnvV3 instances ("A" and "B") over the SAME
 *     synthesized graph, obstacle field and buoys (shared world). Each
 *     sub-env keeps ALL its real Study-2 metrics (collisions vs static
 *     obstacles, cross-track error, IALA violations, docking accuracy,
 *     dropped comms frames) computed exactly as before.
 *   - The two vessels step in LOCKSTEP. On each macro-step both choose an
 *     outgoing-edge action; we then advance BOTH along their chosen edges
 *     with synchronized continuous sub-steps and, at every sub-step,
 *     measure the real Euclidean distance between the two hulls.
 *       * closest-point-of-approach (CPA) is tracked (real minimum distance);
 *       * if the hull separation drops below VESSEL_COLLISION_RADIUS an
 *         INTER-VESSEL COLLISION is recorded for BOTH vessels (a genuine
 *         geometric event);
 *       * separations below VESSEL_NEARMISS_RADIUS (but above collision)
 *         are counted as near-miss / close-quarters situations.
 *   - Each vessel's observation is EXTENDED with the other vessel's SENSED
 *     relative position + range (3 extra values), passed through the SAME
 *     sensor-noise + packet-loss pipeline as its own state. So an agent can
 *     perceive and react to the partner, but only through degraded sensing
 *     under Study-2 conditions.
 *   - A COLREGs-style GIVE-WAY role is computed from real geometry at each
 *     encounter (relative bearing / crossing situation) to attribute
 *     stand-on vs give-way. This is recorded for analysis; it does not
 *     override either agent's own action (the agents remain independent).
 *
 * The base graph/edges/Dijkstra/buoys are reused UNCHANGED. Only the
 * two-vessel coupling + inter-vessel geometry is added. The deployed web
 * app (environment.js) is NOT affected — this is an offline research build.
 * ============================================================ */

(function (root) {
  "use strict";

  const baseV3 = root.BintuluEnvV3;
  const baseEnv = root.BintuluEnv;
  if (!baseV3) throw new Error("environmentV4 requires environmentV3.js loaded first");
  if (!baseEnv) throw new Error("environmentV4 requires environment.js loaded first");
  const { VesselEnvV3, N_ACTION_SLOTS } = baseV3;
  const { MAP_W, MAP_H } = baseEnv;

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

  // Inter-vessel geometry constants (logical units, same scale as OBSTACLE_RADIUS=14).
  const VESSEL_COLLISION_RADIUS = 18;   // hull-hull separation below this = collision
  const VESSEL_NEARMISS_RADIUS = 40;    // separation below this (but > collision) = close-quarters
  const KINEMATIC_SUBSTEPS = 6;         // must match VesselEnvV3 sub-step count
  const CROSSING_BEARING = Math.PI / 8; // head-on window (+/-22.5 deg) for role assignment

  // Reward shaping added ON TOP of each vessel's own VesselEnvV3 reward, for
  // the multi-vessel interaction only. Kept modest so single-vessel behaviour
  // is preserved; the dominant new signal is the collision penalty.
  const REWARD_CONFIG_V4 = {
    vesselCollisionPenalty: -30.0,  // applied to BOTH vessels on an inter-vessel collision
    nearMissPenalty: -3.0,          // close-quarters (per macro-step, once)
    giveWayViolationPenalty: -5.0,  // give-way vessel that closed the range into a near-miss/collision
  };

  /**
   * MultiVesselEnvV4 — two independent VesselEnvV3 vessels on one shared world.
   *
   * Usage (mirrors the single-env loop, but for two agents in lockstep):
   *   const menv = new MultiVesselEnvV4(seed, cfg);
   *   let [obsA, obsB] = menv.reset();          // one observation per vessel
   *   const rA = menv.stepBoth(actionA, actionB);
   *   // rA = { A:{obs,reward,terminated,truncated,info}, B:{...}, shared:{...} }
   */
  class MultiVesselEnvV4 {
    constructor(seed = 42, cfg = {}) {
      this.seed = seed;
      this.cfg = Object.assign({
        noiseStd: 0.0,
        packetErrorRate: 0.0,
        obstacleCount: 6,
      }, cfg);

      // Two vessels over the SAME world. Same env seed => identical graph,
      // obstacle field and buoys for both (they truly share the water). We
      // offset the MISSION rng per vessel so they don't get identical missions.
      this.A = new VesselEnvV3(seed, this.cfg);
      this.B = new VesselEnvV3(seed, this.cfg);
      // Re-seed vessel B's mission stream so the two vessels get DIFFERENT
      // start/goal pairs (otherwise they would ride the same line and the
      // encounter would be degenerate). Sensing/comms streams stay per-vessel.
      this.B.missionRng = makeRNG(seed + 20240);

      this.MAX_X = MAP_W; this.MAX_Y = MAP_H;
      this.nActions = N_ACTION_SLOTS;
      // each vessel observes its own VesselEnvV3 obs (obsDim) + 3 partner values
      this.perVesselExtra = 3;
      this.obsDim = this.A.obsDim + this.perVesselExtra;

      // separate noise streams for the partner-channel sensing (so partner
      // perception is degraded independently of own-state perception)
      this.partnerNoiseRngA = makeRNG(seed + 33001);
      this.partnerNoiseRngB = makeRNG(seed + 33002);

      this.reset();
    }

    _gauss(rng) {
      // Box-Muller with a cached spare per call site is overkill here; use a
      // simple 2-uniform draw (sufficient; deterministic per seed).
      let u = 0, v = 0;
      while (u === 0) u = rng();
      while (v === 0) v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    // Build the partner-relative channel [dx_norm, dy_norm, range_norm] for the
    // observer vessel, passed through sensor noise (own noiseStd) so it degrades
    // under Study-2 conditions exactly like the rest of the observation.
    _partnerChannel(observer, other, noiseRng) {
      const dx = (other.posX - observer.posX) / this.MAX_X;
      const dy = (other.posY - observer.posY) / this.MAX_Y;
      const range = Math.min(1, distXY(observer.posX, observer.posY, other.posX, other.posY) / 300);
      const std = this.cfg.noiseStd;
      if (std <= 0) return [dx, dy, range];
      return [
        dx + this._gauss(noiseRng) * std,
        dy + this._gauss(noiseRng) * std,
        range + this._gauss(noiseRng) * std,
      ];
    }

    _composeObs(baseObs, observer, other, noiseRng) {
      return baseObs.concat(this._partnerChannel(observer, other, noiseRng));
    }

    reset() {
      this.stepCount = 0;
      this.done = false;
      // per-vessel obs from the underlying V3 env
      const a0 = this.A.reset();
      let b0 = this.B.reset();
      // Two vessels must NOT share the same berth/goal: a single dock point
      // cannot host both, and a docked-and-holding vessel sitting on the shared
      // goal would generate spurious repeated "collisions" with the other
      // vessel still trying to reach it. Re-roll B's mission until its GOAL
      // differs from A's (bounded; deterministic per seed). This keeps the two
      // vessels on genuinely different missions that cross in the channel.
      let tries = 0;
      while (this.B.goalWp === this.A.goalWp && tries < 25) {
        b0 = this.B.reset();
        tries++;
      }
      // inter-vessel metrics (real, measured)
      this.vesselCollisions = 0;    // DISTINCT inter-vessel collision events (shared count)
      this.nearMisses = 0;          // DISTINCT close-quarters events (shared count)
      this.collisionContactSteps = 0; // macro-steps spent hull-to-hull (secondary saturation-free)
      this.minCPA = Infinity;       // closest point of approach over the whole episode
      this.encounterSteps = 0;      // macro-steps where vessels were within near-miss range
      this.giveWayEvents = 0;       // macro-steps flagged as a crossing/head-on encounter
      // engagement state for distinct-event counting
      this._inCollision = false;
      this._inNearMiss = false;
      // last known partner collision flag per vessel (for obs / info)
      this.lastVesselCollision = 0;
      const obsA = this._composeObs(a0, this.A, this.B, this.partnerNoiseRngA);
      const obsB = this._composeObs(b0, this.B, this.A, this.partnerNoiseRngB);
      return [obsA, obsB];
    }

    // Determine COLREGs-style role for vessel `self` relative to `other`, from
    // real geometry using each vessel's current heading (direction of its last
    // motion) and relative bearing. Returns "give_way" | "stand_on" | "none".
    // (Recorded for analysis; does not force either agent's action.)
    _giveWayRole(self, other) {
      const shead = self._v4heading, ohead = other._v4heading;
      if (shead == null || ohead == null) return "none";
      // relative bearing of `other` from `self`'s heading
      const bearing = Math.atan2(other.posY - self.posY, other.posX - self.posX);
      let rel = bearing - shead;
      while (rel > Math.PI) rel -= 2 * Math.PI;
      while (rel < -Math.PI) rel += 2 * Math.PI;
      // head-on: nearly reciprocal courses AND other roughly ahead
      let dh = shead - ohead;
      while (dh > Math.PI) dh -= 2 * Math.PI;
      while (dh < -Math.PI) dh += 2 * Math.PI;
      const reciprocal = Math.abs(Math.abs(dh) - Math.PI) < CROSSING_BEARING;
      if (reciprocal && Math.abs(rel) < CROSSING_BEARING) return "head_on"; // both give way (starboard)
      // crossing: other on own starboard side (rel in (0, pi/2)) => self gives way
      if (rel > 0 && rel < Math.PI / 2) return "give_way";
      if (rel < 0 && rel > -Math.PI / 2) return "stand_on";
      return "none";
    }

    /**
     * Advance BOTH vessels one macro-step. Each vessel resolves its own edge
     * action through VesselEnvV3.step() (which runs its own continuous
     * sub-steps, static-obstacle collisions, CTE, IALA, docking). We then
     * REPLAY the two synchronized traversals sub-step-by-sub-step purely to
     * measure inter-vessel geometry (CPA, collision, near-miss) — this does
     * not alter either vessel's own trajectory or its own Study-2 metrics.
     */
    stepBoth(actionA, actionB) {
      this.stepCount++;

      // record pre-move poses so we can replay the two straight-line traversals
      const A = this.A, B = this.B;
      const preA = { x: A.posX, y: A.posY, wp: A.currentWp, done: A.done };
      const preB = { x: B.posX, y: B.posY, wp: B.currentWp, done: B.done };

      // resolve each vessel's own step (advances its real pose + own metrics).
      // A vessel that is already done holds station (no further stepping).
      const rA = A.done ? this._holdResult(A) : A.step(actionA);
      const rB = B.done ? this._holdResult(B) : B.step(actionB);

      const postA = { x: A.posX, y: A.posY };
      const postB = { x: B.posX, y: B.posY };

      // headings from actual motion this step (for COLREGs role attribution)
      A._v4heading = (postA.x !== preA.x || postA.y !== preA.y)
        ? Math.atan2(postA.y - preA.y, postA.x - preA.x) : A._v4heading;
      B._v4heading = (postB.x !== preB.x || postB.y !== preB.y)
        ? Math.atan2(postB.y - preB.y, postB.x - preB.x) : B._v4heading;

      // --- synchronized inter-vessel geometry over the macro-step ---
      // Linearly interpolate both vessels between pre and post pose across the
      // same KINEMATIC_SUBSTEPS. (VesselEnvV3 already advanced the true pose;
      // this replay uses the endpoints, which is the honest first-order model
      // of where each hull was during the shared interval.)
      let collidedThisStep = false, nearMissThisStep = false, stepMinCPA = Infinity;
      for (let k = 1; k <= KINEMATIC_SUBSTEPS; k++) {
        const t = k / KINEMATIC_SUBSTEPS;
        const ax = preA.x + (postA.x - preA.x) * t, ay = preA.y + (postA.y - preA.y) * t;
        const bx = preB.x + (postB.x - preB.x) * t, by = preB.y + (postB.y - preB.y) * t;
        const sep = distXY(ax, ay, bx, by);
        if (sep < stepMinCPA) stepMinCPA = sep;
        if (sep <= VESSEL_COLLISION_RADIUS) collidedThisStep = true;
        else if (sep <= VESSEL_NEARMISS_RADIUS) nearMissThisStep = true;
      }
      if (stepMinCPA < this.minCPA) this.minCPA = stepMinCPA;

      // COLREGs role for this encounter (only meaningful when close-ish)
      let roleA = "none", roleB = "none";
      const inEncounter = stepMinCPA <= VESSEL_NEARMISS_RADIUS * 1.5;
      if (inEncounter) {
        this.encounterSteps++;
        roleA = this._giveWayRole(A, B);
        roleB = this._giveWayRole(B, A);
        if (roleA === "give_way" || roleA === "head_on" || roleB === "give_way" || roleB === "head_on") {
          this.giveWayEvents++;
        }
      }

      // apply inter-vessel reward shaping + book-keeping.
      //
      // EVENT vs STEP accounting (important for honest metrics): two vessels
      // locked together on the same channel line would otherwise be counted as
      // a fresh "collision" on every macro-step, saturating the count (40+ per
      // episode) and obscuring how MANY distinct close-quarters situations
      // occurred. So the reported counts (vesselCollisions / nearMisses) are
      // DISTINCT EVENTS: incremented only on the transition INTO the state,
      // not while the pair remains engaged. The reward penalty, by contrast,
      // is applied EVERY step the hulls are in contact/close — a persistently
      // unsafe situation should keep costing the agents (correct incentive).
      let extraA = 0, extraB = 0;
      if (collidedThisStep) {
        if (!this._inCollision) this.vesselCollisions++;   // new distinct event
        this._inCollision = true; this._inNearMiss = true;
        extraA += REWARD_CONFIG_V4.vesselCollisionPenalty;
        extraB += REWARD_CONFIG_V4.vesselCollisionPenalty;
        this.lastVesselCollision = 1;
        this.collisionContactSteps++;                      // steps in contact (secondary)
      } else {
        this._inCollision = false;
        this.lastVesselCollision = 0;
      }
      if (nearMissThisStep && !collidedThisStep) {
        if (!this._inNearMiss) this.nearMisses++;           // new distinct event
        this._inNearMiss = true;
        extraA += REWARD_CONFIG_V4.nearMissPenalty;
        extraB += REWARD_CONFIG_V4.nearMissPenalty;
      } else if (!collidedThisStep) {
        this._inNearMiss = false;
      }
      // give-way accountability: the vessel that was obliged to yield but
      // still allowed a near-miss/collision takes an extra penalty (teaches
      // give-way behaviour without dictating the action).
      if ((nearMissThisStep || collidedThisStep)) {
        if (roleA === "give_way" || roleA === "head_on") extraA += REWARD_CONFIG_V4.giveWayViolationPenalty;
        if (roleB === "give_way" || roleB === "head_on") extraB += REWARD_CONFIG_V4.giveWayViolationPenalty;
      }

      rA.reward += extraA; rB.reward += extraB;
      A.totalReward += extraA; B.totalReward += extraB;

      // extend infos with inter-vessel data
      rA.info = Object.assign({}, rA.info, {
        vesselCollision: collidedThisStep ? 1 : 0, nearMiss: nearMissThisStep ? 1 : 0,
        cpa: stepMinCPA, giveWayRole: roleA,
      });
      rB.info = Object.assign({}, rB.info, {
        vesselCollision: collidedThisStep ? 1 : 0, nearMiss: nearMissThisStep ? 1 : 0,
        cpa: stepMinCPA, giveWayRole: roleB,
      });

      // compose observations (own obs + sensed partner channel)
      rA.obs = this._composeObs(rA.obs, A, B, this.partnerNoiseRngA);
      rB.obs = this._composeObs(rB.obs, B, A, this.partnerNoiseRngB);

      // episode ends when BOTH vessels are done (reached goal / truncated)
      this.done = A.done && B.done;

      const shared = {
        vesselCollisions: this.vesselCollisions, nearMisses: this.nearMisses,
        collisionContactSteps: this.collisionContactSteps,
        minCPA: this.minCPA, encounterSteps: this.encounterSteps, giveWayEvents: this.giveWayEvents,
        stepCount: this.stepCount, done: this.done,
      };
      return { A: rA, B: rB, shared };
    }

    // a "hold station" result for a vessel that has already finished, so the
    // other vessel can keep running until it also finishes.
    _holdResult(env) {
      const trueObs = env._trueObs();
      return {
        obs: trueObs, reward: 0, terminated: env.done && !env.truncated, truncated: env.truncated,
        info: {
          currentWp: env.currentWp, goalWp: env.goalWp, reachedGoal: env.currentWp === env.goalWp,
          invalid: false, timeout: env.truncated, collided: false, collisionCount: env.collisionCount,
          cteMean: env.cteSamples ? env.cteSum / env.cteSamples : 0, ialaViolations: env.ialaViolations,
          droppedFrames: env.droppedFrames, dockingAccuracy: env.dockingAccuracy,
          posX: env.posX, posY: env.posY, held: true,
        },
      };
    }
  }

  root.BintuluEnvV4 = {
    MultiVesselEnvV4,
    REWARD_CONFIG_V4,
    VESSEL_COLLISION_RADIUS,
    VESSEL_NEARMISS_RADIUS,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.BintuluEnvV4;
  }
})(typeof window !== "undefined" ? window : globalThis);
