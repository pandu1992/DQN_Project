/* ============================================================
 * BPAN — environmentBintuluPlus.js
 * DQN-improvement variant of the Bintulu environment. Adds, as toggle-able
 * options, the concrete improvements evaluated in the BPAN improvement study:
 *   cfg.chartAware : append per-slot onPlan flags (the Dijkstra charted-route
 *                    prior) to the observation (obsDim 28 -> 32). This is the
 *                    structured-navigation-prior lever shown to help on the
 *                    Synthetic Port chart.
 *   cfg.shaping    : add potential-based reward shaping toward the goal
 *                    (F = gamma*phi(s') - phi(s), phi = -dist_to_goal), which is
 *                    policy-invariant (Ng et al. 1999) so it cannot change the
 *                    optimal policy, only speed/stabilise learning.
 * Everything else is inherited UNCHANGED from BintuluEnv, so the ONLY variables
 * are the two improvements. Byte-equivalent to BintuluEnv when both are off.
 * ============================================================ */
(function (root) {
  "use strict";
  const base = root.BintuluPortEnv;
  if (!base) throw new Error("environmentBintuluPlus requires environmentBintulu.js loaded first");
  const { BintuluEnv } = base;
  const N_SLOTS = 4;

  function distXY(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

  class BintuluEnvPlus extends BintuluEnv {
    constructor(seed = 42, cfg = {}) {
      super(seed, cfg);
      this.chartAware = !!cfg.chartAware;
      this.shaping = !!cfg.shaping;
      this.gammaShape = cfg.gammaShape != null ? cfg.gammaShape : 0.99;
      if (this.chartAware) this.obsDim = this.obsDim + N_SLOTS;
      // re-run reset so observation width matches the (possibly extended) obsDim
      this.reset();
    }

    // slot index (in sorted adjacency) of the next Dijkstra-plan segment, or -1
    _plannedSlot() {
      const plan = this.plannedPath || [], cur = this.currentWp;
      const pos = plan.indexOf(cur);
      if (pos < 0 || pos >= plan.length - 1) return -1;
      const nxt = plan[pos + 1], edges = this.adj[cur] || [];
      for (let a = 0; a < edges.length; a++) if (edges[a].neighbor === nxt) return a;
      return -1;
    }
    _appendChart(o) {
      const slot = this._plannedSlot(), out = o.slice();
      for (let a = 0; a < N_SLOTS; a++) out.push(a === slot ? 1 : 0);
      return out;
    }
    _trueObs() { const b = super._trueObs(); return this.chartAware ? this._appendChart(b) : b; }
    _sensedObs(trueObs) {
      if (!this.chartAware) return super._sensedObs(trueObs);
      const v = trueObs.length - N_SLOTS;
      return super._sensedObs(trueObs.slice(0, v)).concat(trueObs.slice(v)); // onPlan flags pass through
    }

    // potential toward goal, normalised by map diagonal
    _phi() {
      const g = this.states[this.goalWp];
      const diag = Math.hypot(this.MAX_X, this.MAX_Y);
      return -distXY(this.posX, this.posY, g.x, g.y) / diag;
    }

    step(action) {
      const phiBefore = this.shaping ? this._phi() : 0;
      const r = super.step(action);
      if (this.shaping) {
        const phiAfter = this._phi();
        const shapeScale = 20.0; // modest; policy-invariant potential shaping
        const F = (this.gammaShape * phiAfter - phiBefore) * shapeScale;
        r.reward += F; this.totalReward += F;
      }
      return r;
    }
  }

  root.BintuluPortEnvPlus = { BintuluEnvPlus };
  if (typeof module !== "undefined" && module.exports) module.exports = root.BintuluPortEnvPlus;
})(typeof window !== "undefined" ? window : globalThis);
