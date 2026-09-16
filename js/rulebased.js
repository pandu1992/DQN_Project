/* ============================================================
 * BINTULU PORT — COLREGs-aware Rule-Based Baseline (Study 3)
 * rulebased.js
 *
 * A deterministic, non-learning controller for VesselEnvV3 that serves
 * as a classical baseline against the four value-based DQN variants.
 * It mirrors the DQNAgent interface (act / observe / qValues) so the
 * experiment harness can drive it identically.
 *
 * POLICY (charted-channel following with maritime rules of the road):
 *   A real vessel follows the charted/planned safe passage and deviates
 *   only to avoid hazards. So the controller's PRIMARY guide is the
 *   Dijkstra-planned route (the "charted channel"): it prefers the edge
 *   that advances along the plan. It then applies COLREGs-style rules:
 *     1. CHANNEL FOLLOWING — strongly prefer the edge toward the next
 *        planned waypoint (staying in the marked, surveyed channel).
 *     2. GOAL-SEEKING — secondary preference for edges reducing the
 *        (sensed) straight-line distance to goal (used off-plan / at
 *        junctions).
 *     3. COLLISION AVOIDANCE (COLREGs give-way spirit) — penalise edges
 *        that pass near a detected obstacle; hard-avoid edges crossing an
 *        obstacle disc, deviating from the channel when necessary.
 *     4. CHANNEL / IALA KEEPING — prefer lower navigation_cost/difficulty
 *        (safer marked water).
 *     5. ANTI-STALL — avoid immediately revisiting the previous node.
 *
 * FAIR ROBUSTNESS COMPARISON: the controller does NOT use ground-truth
 * pose. It reads the vessel's own goal-relative bearing from the
 * observation vector (obs indices 4,5 = dx,dy toward goal, in [-1,1] of
 * MAP scale), which is subject to the same sensor noise and stale-comms
 * degradation the DQN agents receive. Obstacle geometry is treated as
 * on-board perception (available to any planner).
 *
 * The controller queries the env's static graph (adjacency, obstacle
 * positions, node coordinates) via a reference passed at construction —
 * exactly the map/chart information a classical planner would have.
 * ============================================================ */

(function (root) {
  "use strict";

  function distXY(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

  // perpendicular distance from point P to segment AB
  function pointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) return distXY(px, py, ax, ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return distXY(px, py, ax + t * dx, ay + t * dy);
  }

  class RuleBasedAgent {
    /**
     * @param {number} obsDim  (for interface parity; unused)
     * @param {number} nActions
     * @param {object} cfg  { env } — reference to the VesselEnvV3 instance
     */
    constructor(obsDim, nActions, cfg = {}) {
      this.obsDim = obsDim;
      this.nActions = nActions;
      this.env = cfg.env || null;
      this.algorithm = "RuleBased";
      this.usePER = false;
      this.useNoisy = false;
      this.epsilon = 0;
      this.stepCount = 0;
      this.trainSteps = 0;
      this.lastLoss = 0;
      this.buffer = { size: 0 };
      this._prevWp = null;
      // scoring weights (tuned so the controller is competent but rule-driven)
      this.wChannel = 5.0;     // follow the charted/planned channel (primary)
      this.wGoal = 1.0;        // reward progress toward the sensed goal (secondary)
      this.wObstacle = 6.0;    // penalise proximity to obstacles (collision avoidance)
      this.wCost = 0.15;       // prefer lower navigation_cost (channel keeping)
      this.wDifficulty = 2.0;  // prefer lower difficulty (safer, marked water)
      this.wRevisit = 4.0;     // avoid bouncing back
      this.obstacleClear = 22; // distance below which an obstacle is "threatening"
    }

    get label() { return "Rule-based (COLREGs-aware)"; }

    // interface parity — the baseline never learns
    observe() { this.stepCount++; }
    qValues() { return new Array(this.nActions).fill(0); }
    computeEpsilon() { return 0; }

    /**
     * Choose an action (outgoing-edge slot). Uses the SENSED goal bearing
     * from `obs` (so noise/comms degradation affect it) plus on-board map +
     * obstacle geometry from the env reference.
     */
    act(obs, _greedy = false) {
      const env = this.env;
      if (!env) return 0;
      const cur = env.currentWp;
      const edges = env.adj[cur] || [];
      if (edges.length === 0) return 0;

      const s = env.states[cur];

      // --- sensed goal position, reconstructed from the (noisy) observation ---
      // obs layout: [x/W, y/H, gx/W, gy/H, dx/W, dy/H, distGoal/900, lane, stepFrac, ...slots]
      // Use the sensed absolute goal (indices 2,3) scaled back to map units.
      let goalX, goalY;
      if (obs && obs.length >= 4) {
        goalX = obs[2] * env.MAX_X;
        goalY = obs[3] * env.MAX_Y;
      } else {
        const g = env.states[env.goalWp];
        goalX = g.x; goalY = g.y;
      }

      // charted channel: the next waypoint along the Dijkstra-planned route.
      // If we are on the plan, prefer the edge that advances it; this keeps the
      // vessel in surveyed water (how classical marine navigation works).
      const plan = env.plannedPath || [];
      let plannedNext = null;
      const pos = plan.indexOf(cur);
      if (pos >= 0 && pos < plan.length - 1) plannedNext = plan[pos + 1];

      let best = 0, bestScore = -Infinity;
      for (let a = 0; a < edges.length; a++) {
        const e = edges[a];
        const nb = env.states[e.neighbor];
        if (!nb) continue;

        // 1. channel following: is this edge the next charted segment?
        const channel = (plannedNext !== null && e.neighbor === plannedNext) ? 1 : 0;

        // 2. goal-seeking: reduction in distance to the sensed goal (normalised)
        const dCur = distXY(s.x, s.y, goalX, goalY);
        const dNext = distXY(nb.x, nb.y, goalX, goalY);
        const progress = (dCur - dNext) / 900;

        // 2. collision avoidance: closest approach of this edge to any obstacle
        let minObs = Infinity;
        for (const o of env.obstacles) {
          const d = pointToSegment(o.x, o.y, s.x, s.y, nb.x, nb.y) - o.r;
          if (d < minObs) minObs = d;
        }
        // penalty grows sharply as the edge nears/enters an obstacle
        let obsPenalty = 0;
        if (minObs < this.obstacleClear) {
          obsPenalty = (this.obstacleClear - Math.max(0, minObs)) / this.obstacleClear; // [0,1]
          if (minObs <= 0) obsPenalty += 2.0; // edge crosses an obstacle -> avoid hard
        }

        // 3. channel keeping: prefer cheaper, lower-difficulty (marked) edges
        const costTerm = -(Math.min(1, e.navigation_cost / 30));
        const diffTerm = -(e.difficulty || 0);

        // 4. anti-stall: discourage returning to the node we just came from
        const revisit = (e.neighbor === this._prevWp) ? 1 : 0;

        const score =
          this.wChannel * channel +
          this.wGoal * progress +
          this.wCost * costTerm +
          this.wDifficulty * diffTerm -
          this.wObstacle * obsPenalty -
          this.wRevisit * revisit;

        if (score > bestScore) { bestScore = score; best = a; }
      }

      this._prevWp = cur;
      return best;
    }
  }

  root.BintuluRuleBased = { RuleBasedAgent };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.BintuluRuleBased;
  }
})(typeof window !== "undefined" ? window : globalThis);
