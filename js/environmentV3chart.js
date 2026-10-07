/* ============================================================
 * SYNTHETIC PORT — Environment V3-Chart (baseline-fairness study)
 * environmentV3chart.js
 *
 * Q1 AUDIT FIX (Task 3) — baseline information asymmetry.
 *
 * The COLREGs rule-based baseline (js/rulebased.js) follows the DIJKSTRA-CHARTED
 * channel: its dominant scoring term (weight 5.0) is "is this outgoing edge the
 * next charted segment?", read directly from env.plannedPath. The value-based
 * DQN, by contrast, receives only the 28-dim VesselEnvV3 observation, whose
 * per-edge slots carry [exists, navCost, difficulty, towardGoal] — a GREEDY
 * geometric "reduces straight-line distance to goal" flag, NOT the planner's
 * "advances the charted route" flag. So the baseline has a structured navigation
 * prior the learner does not.
 *
 * A reviewer will (correctly) ask: is the rule-based advantage because classical
 * control is inherently more robust, or simply because it is handed chart
 * information the learner is denied?
 *
 * This env makes chart access an EXPLICIT, controllable factor. With
 * cfg.chartAware = true it APPENDS one extra feature per action slot — an
 * `onPlan` flag (1 if that outgoing edge is the next segment of the Dijkstra
 * plan, else 0) — to the observation, giving the DQN the SAME charted-route
 * prior the rule-based controller uses. With cfg.chartAware = false the
 * observation is byte-identical to VesselEnvV3 (the "DRL-no-chart" condition).
 *
 * Everything else — graph, kinematics, sensing/comms degradation, obstacles,
 * CTE/IALA/docking metrics — is inherited UNCHANGED from VesselEnvV3, so the
 * ONLY manipulated variable is the presence of the chart prior. The `onPlan`
 * feature is passed through the SAME sensor-noise / packet-loss pipeline as the
 * rest of the observation (it is not a privileged noise-free channel), except
 * that, like the other per-slot "exists" flags, it is left un-perturbed by the
 * additive Gaussian noise (it is a discrete routing indicator, not a continuous
 * measurement).
 *
 * Offline research build; the deployed web app is unaffected.
 * ============================================================ */

(function (root) {
  "use strict";

  const baseV3 = root.BintuluEnvV3;
  if (!baseV3) throw new Error("environmentV3chart requires environmentV3.js loaded first");
  const { VesselEnvV3, N_ACTION_SLOTS } = baseV3;

  class VesselEnvV3Chart extends VesselEnvV3 {
    constructor(seed = 42, cfg = {}) {
      super(seed, cfg);
      this.chartAware = !!cfg.chartAware;
      // extend obsDim by one onPlan feature per action slot when chart-aware
      if (this.chartAware) this.obsDim = this.obsDim + N_ACTION_SLOTS;
    }

    // Which outgoing edge (by sorted-adjacency slot index) is the next charted
    // segment along the Dijkstra plan from the current node? Returns a slot
    // index or -1. Uses the SAME planned path the rule-based baseline uses.
    _plannedSlot() {
      const plan = this.plannedPath || [];
      const cur = this.currentWp;
      const pos = plan.indexOf(cur);
      if (pos < 0 || pos >= plan.length - 1) return -1;
      const plannedNext = plan[pos + 1];
      const edges = this.adj[cur] || [];
      for (let a = 0; a < edges.length; a++) {
        if (edges[a].neighbor === plannedNext) return a;
      }
      return -1;
    }

    // append per-slot onPlan flags to a base VesselEnvV3 observation
    _appendChart(baseObs) {
      const slot = this._plannedSlot();
      const out = baseObs.slice();
      for (let a = 0; a < N_ACTION_SLOTS; a++) out.push(a === slot ? 1 : 0);
      return out;
    }

    _trueObs() {
      const base = super._trueObs();
      return this.chartAware ? this._appendChart(base) : base;
    }

    // Sensor noise: reuse the parent's perturbation on the V3 portion, and
    // leave the appended onPlan flags un-perturbed (discrete routing indicator,
    // like the per-slot "exists" flags the parent already skips).
    _sensedObs(trueObs) {
      if (!this.chartAware) return super._sensedObs(trueObs);
      const v3len = trueObs.length - N_ACTION_SLOTS;
      const v3part = super._sensedObs(trueObs.slice(0, v3len));
      return v3part.concat(trueObs.slice(v3len)); // onPlan flags pass through
    }
  }

  root.BintuluEnvV3Chart = { VesselEnvV3Chart };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.BintuluEnvV3Chart;
  }
})(typeof window !== "undefined" ? window : globalThis);
