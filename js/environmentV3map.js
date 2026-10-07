/* ============================================================
 * SYNTHETIC PORT — Environment V3-Chart-Map (external-validity study)
 * environmentV3map.js
 *
 * Q1 AUDIT FIX (Task 4). Combines the chart-aware observation option
 * (VesselEnvV3Chart) with a swappable, parameterized map (mapgen.js), so the
 * chart-fairness experiment can be replicated across MULTIPLE synthetic port
 * geometries — removing the single-graph external-validity bottleneck.
 *
 * cfg.mapId selects the geometry (0 = canonical anchor, matching all prior
 * studies; 1..N = distinct layouts). Everything else — kinematics, sensing/
 * comms degradation, obstacles, CTE/IALA/docking metrics, and the optional
 * chart prior — is inherited UNCHANGED. The map's graph edge attributes are
 * seeded by (seed, mapId) so they are deterministic and reproducible.
 *
 * Offline research build; the deployed web app is unaffected.
 * ============================================================ */

(function (root) {
  "use strict";

  const chartMod = root.BintuluEnvV3Chart;
  const mapMod = root.BintuluMapGen;
  const baseV3 = root.BintuluEnvV3;
  if (!chartMod) throw new Error("environmentV3map requires environmentV3chart.js loaded first");
  if (!mapMod) throw new Error("environmentV3map requires mapgen.js loaded first");
  const { VesselEnvV3Chart } = chartMod;
  const { N_ACTION_SLOTS } = baseV3;

  function makeRNG(seed) {
    let s = seed >>> 0;
    return function () {
      s |= 0; s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class VesselEnvV3ChartMap extends VesselEnvV3Chart {
    constructor(seed = 42, cfg = {}) {
      // build with the canonical map first (super sets up everything), then, if
      // a non-canonical map is requested, swap the geometry/graph and rebuild
      // the derived structures (sorted adjacency, obstacle field) before reset.
      super(seed, cfg);
      this.mapId = cfg.mapId || 0;
      if (this.mapId !== 0) {
        const m = mapMod.buildMap(this.mapId, seed);
        this.states = m.states;
        this.order = m.order;
        this.graph = m.graph;
        this.buoys = m.buoys;
        // rebuild deterministic sorted adjacency (stable action indexing)
        this.adj = {};
        for (const id of Object.keys(this.graph)) {
          const edges = this.graph[id].slice().sort((e1, e2) => {
            if (e1.navigation_cost !== e2.navigation_cost) return e1.navigation_cost - e2.navigation_cost;
            return e1.neighbor < e2.neighbor ? -1 : 1;
          });
          this.adj[id] = edges;
        }
        this.maxOutDeg = Math.max(...Object.values(this.adj).map((e) => e.length));
        // rebuild the plan cache + obstacle field for the new map, then reset
        this._planCache = {};
        this.obstacles = this._buildObstacles(this.cfg.obstacleCount);
        this.reset();
      }
    }
  }

  root.BintuluEnvV3ChartMap = { VesselEnvV3ChartMap };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.BintuluEnvV3ChartMap;
  }
})(typeof window !== "undefined" ? window : globalThis);
