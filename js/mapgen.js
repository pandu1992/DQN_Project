/* ============================================================
 * SYNTHETIC PORT — Parameterized map generator (external-validity study)
 * mapgen.js
 *
 * Q1 AUDIT FIX (Task 4) — external validity beyond a single graph.
 *
 * All of Studies 1-5 reuse ONE hard-coded channel geometry, which a reviewer
 * can (correctly) call a single-graph external-validity bottleneck. This module
 * generates a FAMILY of distinct but structurally-comparable synthetic port
 * maps, parameterized by a map id: it varies the number of waypoints per lane,
 * the channel curvature/amplitude, the vertical separation of the two access
 * channels, the harbour berth position, and the virtual-shortcut placement.
 *
 * MAP 0 reproduces the CANONICAL geometry used by environment.js
 * (nNorth=22, nSouth=20, nHarb=6, the same coordinate formulas), so it is an
 * exact validity anchor; maps 1..N are genuinely different layouts. The graph
 * edge-attribute and buoy generation logic mirrors environment.js exactly, so
 * the ONLY thing that changes across maps is the geometry/topology.
 *
 * Returns {states, order, graph, buoys} with the same object shape the
 * VesselEnv family expects, so it can be swapped into VesselEnvV3 / V3Chart.
 * Deployed web app unaffected (offline research module).
 * ============================================================ */

(function (root) {
  "use strict";

  const base = root.BintuluEnv;
  if (!base) throw new Error("mapgen requires environment.js loaded first");
  const { MAP_W, MAP_H } = base;

  function makeRNG(seed) {
    let s = seed >>> 0;
    return function () {
      s |= 0; s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

  // Per-map geometry parameters. Map 0 == canonical (matches environment.js).
  const MAP_PARAMS = [
    // id 0: CANONICAL anchor (identical to environment.js buildWaypoints)
    { nNorth: 22, nSouth: 20, nHarb: 6, x0: 70, xspan: 640,
      nY: 150, nAmp: -40, nSlope: 30, sY: 410, sAmp: 40, sSlope: -20,
      hX: 760, hXspan: 70, hY: 300, hYslope: -20,
      virtual: [["L02", 3, 9], ["L02", 8, 14], ["L08", 3, 10]] },
    // id 1: longer, straighter channels, wider separation, berth higher
    { nNorth: 26, nSouth: 24, nHarb: 7, x0: 60, xspan: 700,
      nY: 110, nAmp: -20, nSlope: 20, sY: 450, sAmp: 25, sSlope: -10,
      hX: 790, hXspan: 60, hY: 250, hYslope: 15,
      virtual: [["L02", 5, 12], ["L08", 4, 13]] },
    // id 2: shorter, strongly curved channels, tight separation, low berth
    { nNorth: 18, nSouth: 18, nHarb: 5, x0: 90, xspan: 600,
      nY: 190, nAmp: -70, nSlope: 45, sY: 380, sAmp: 70, sSlope: -35,
      hX: 720, hXspan: 90, hY: 330, hYslope: -30,
      virtual: [["L02", 2, 8], ["L02", 7, 13], ["L08", 3, 9], ["L08", 6, 12]] },
    // id 3: asymmetric (long north, short south), S-curve north, mid berth
    { nNorth: 28, nSouth: 16, nHarb: 6, x0: 65, xspan: 660,
      nY: 130, nAmp: -55, nSlope: 60, sY: 430, sAmp: 30, sSlope: 5,
      hX: 770, hXspan: 75, hY: 285, hYslope: -10,
      virtual: [["L02", 4, 11], ["L02", 12, 20], ["L08", 2, 8]] },
    // id 4: compact map, gentle curves, high berth, few shortcuts
    { nNorth: 20, nSouth: 22, nHarb: 6, x0: 80, xspan: 620,
      nY: 160, nAmp: -30, nSlope: 10, sY: 400, sAmp: 50, sSlope: -25,
      hX: 740, hXspan: 80, hY: 270, hYslope: 20,
      virtual: [["L02", 5, 13], ["L08", 5, 14]] },
  ];

  function nMaps() { return MAP_PARAMS.length; }

  function buildWaypoints(p) {
    const states = {};
    const order = { L02: [], L08: [], L01: [] };
    for (let i = 0; i < p.nNorth; i++) {
      const t = i / (p.nNorth - 1);
      const x = p.x0 + t * p.xspan;
      const y = p.nY + Math.sin(t * Math.PI) * p.nAmp + t * p.nSlope;
      const id = `L02_WP${String(i + 1).padStart(3, "0")}`;
      states[id] = { id, x, y, lane: "L02", isSpawn: i === 0, isTerminal: i === p.nNorth - 1 };
      order.L02.push(id);
    }
    for (let i = 0; i < p.nSouth; i++) {
      const t = i / (p.nSouth - 1);
      const x = p.x0 + t * p.xspan;
      const y = p.sY + Math.sin(t * Math.PI) * p.sAmp + t * p.sSlope;
      const id = `L08_WP${String(i + 1).padStart(3, "0")}`;
      states[id] = { id, x, y, lane: "L08", isSpawn: i === 0, isTerminal: i === p.nSouth - 1 };
      order.L08.push(id);
    }
    for (let i = 0; i < p.nHarb; i++) {
      const t = i / (p.nHarb - 1);
      const x = p.hX + t * p.hXspan;
      const y = p.hY + t * p.hYslope;
      const id = `L01_WP${String(i + 1).padStart(3, "0")}`;
      states[id] = { id, x, y, lane: "L01", isSpawn: i === 0, isTerminal: i === p.nHarb - 1 };
      order.L01.push(id);
    }
    for (const lane of Object.keys(order)) {
      const ids = order[lane];
      for (let i = 0; i < ids.length; i++) {
        const cur = states[ids[i]];
        const nxt = states[ids[Math.min(i + 1, ids.length - 1)]];
        cur.heading = Math.atan2(nxt.y - cur.y, nxt.x - cur.x) * (180 / Math.PI);
      }
    }
    return { states, order };
  }

  // Edge-attribute generation — identical formulas to environment.js buildGraph.
  function buildGraph(states, order, rng, virtualPairs) {
    const graph = {};
    for (const id of Object.keys(states)) graph[id] = [];
    const BASE_SPEED = 8.0;
    function attach(u, v, edgeType, direction) {
      const d = dist(states[u], states[v]);
      const risk = edgeType === "virtual" ? 0.25 + rng() * 0.2 : 0.02 + rng() * 0.06;
      const traffic = rng() * 0.5;
      const weather = rng() * 0.3;
      const current = -0.2 + rng() * 0.4;
      let speed = Math.max(2.0, BASE_SPEED * (1 - current));
      const travelTime = d / speed;
      const energyCost = d * 0.02;
      const navigationCost = travelTime + energyCost + 2.0 * traffic + 3.0 * risk;
      let difficulty = Math.min(1.0, 0.35 * risk + 0.25 * traffic + 0.2 * weather + 0.2 * Math.abs(current));
      graph[u].push({
        neighbor: v, distance: d, direction, edge_type: edgeType,
        risk: +risk.toFixed(3), traffic: +traffic.toFixed(3), weather: +weather.toFixed(3),
        current: +current.toFixed(3), speed: +speed.toFixed(2), travel_time: +travelTime.toFixed(2),
        energy_cost: +energyCost.toFixed(2), navigation_cost: +navigationCost.toFixed(2),
        difficulty: +difficulty.toFixed(3), edge_id: `${u}->${v}`,
      });
    }
    for (const lane of Object.keys(order)) {
      const ids = order[lane];
      for (let i = 0; i < ids.length - 1; i++) {
        attach(ids[i], ids[i + 1], "normal", "forward");
        attach(ids[i + 1], ids[i], "normal", "backward");
      }
    }
    const northEnd = order.L02[order.L02.length - 1];
    const southEnd = order.L08[order.L08.length - 1];
    const harbStart = order.L01[0];
    attach(northEnd, harbStart, "normal", "forward");
    attach(harbStart, northEnd, "normal", "backward");
    attach(southEnd, harbStart, "normal", "forward");
    attach(harbStart, southEnd, "normal", "backward");
    // virtual shortcuts: [lane, i, j] -> connect order[lane][i] and order[lane][j]
    for (const [lane, i, j] of (virtualPairs || [])) {
      const a = order[lane][i], b = order[lane][j];
      if (a && b) { attach(a, b, "virtual", "virtual"); attach(b, a, "virtual", "virtual"); }
    }
    return graph;
  }

  function buildBuoys(states, order) {
    const buoys = [];
    let g = 1, r = 1;
    for (const lane of ["L02", "L08"]) {
      const ids = order[lane];
      for (let i = 1; i < ids.length - 1; i += 2) {
        const cur = states[ids[i]];
        const nxt = states[ids[Math.min(i + 1, ids.length - 1)]];
        const ang = Math.atan2(nxt.y - cur.y, nxt.x - cur.x);
        const nx = -Math.sin(ang), ny = Math.cos(ang), off = 16;
        buoys.push({ id: `G${String(g++).padStart(3, "0")}`, color: "GREEN", x: cur.x + nx * off, y: cur.y + ny * off });
        buoys.push({ id: `R${String(r++).padStart(3, "0")}`, color: "RED", x: cur.x - nx * off, y: cur.y - ny * off });
      }
    }
    return buoys;
  }

  // Build the full {states, order, graph, buoys} for a given map id, using a
  // graph RNG seeded by (envSeed, mapId) so edge attributes are deterministic.
  function buildMap(mapId, seed) {
    const p = MAP_PARAMS[mapId % MAP_PARAMS.length];
    const { states, order } = buildWaypoints(p);
    const rng = makeRNG((seed >>> 0) ^ (0x9e3779b1 * (mapId + 1)));
    const graph = buildGraph(states, order, rng, p.virtual);
    const buoys = buildBuoys(states, order);
    return { states, order, graph, buoys, mapId };
  }

  root.BintuluMapGen = { buildMap, nMaps, MAP_W, MAP_H };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.BintuluMapGen;
  }
})(typeof window !== "undefined" ? window : globalThis);
