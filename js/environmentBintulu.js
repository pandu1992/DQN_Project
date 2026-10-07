/* ============================================================
 * BINTULU PORT AUTONOMOUS NAVIGATION (BPAN)
 * environmentBintulu.js
 *
 * A SEPARATE, chart-grounded study — distinct from the "Synthetic Port"
 * Studies 1-5. The waypoint graph, buoys and berths are placed to match the
 * REAL Bintulu Port approach chart (assets/bintulu/bintulu_chart.png), which is
 * rendered as the live-simulation background.
 *
 * COORDINATE SYSTEM
 *   The map uses the chart image's native pixel frame (1536 x 1024) so every
 *   waypoint/buoy lands on the corresponding feature of the background chart.
 *   The renderer scales this frame to the canvas.
 *
 * TOPOLOGY (faithful to the chart, with the navigable path EXTENDED to the
 * berths so missions are complete — the chart's magenta channel lines stop in
 * open water; a vessel must still reach a jetty/harbour):
 *   - NORTH ACCESS CHANNEL  : sea entrance -> ... -> Southern Jetty /
 *                             Container Terminal / Inner Harbour 2 (berths).
 *   - SOUTH ACCESS CHANNEL  : sea entrance -> ... -> Inner Harbour 1 (berth).
 *   Both channels are marked by the chart's lateral buoys (red "NO/R" to one
 *   side, green "N/G" to the other).
 *
 * This module reuses the SAME physics/sensing/comms + metric machinery contract
 * as the project's other environments (reset()/step(), continuous kinematics,
 * Gaussian obs-noise, packet-loss, obstacle collisions, docking accuracy, CTE,
 * IALA channel-keeping) so results are commensurable with the rest of the repo,
 * but the geometry is Bintulu-specific. All metrics are genuinely computed.
 * ============================================================ */

(function (root) {
  "use strict";

  // Native chart frame (background image is 1536x1024).
  const MAP_W = 1536;
  const MAP_H = 1024;

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
  function distXY(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

  // ---------- Dijkstra over navigation_cost (self-contained) ----------
  function dijkstra(graph, start, goal, weight) {
    weight = weight || "navigation_cost";
    const cost = { [start]: 0 }, parent = {}, visited = new Set();
    const pq = [[0, start]];
    while (pq.length) {
      pq.sort((a, b) => a[0] - b[0]);
      const [c, node] = pq.shift();
      if (visited.has(node)) continue;
      visited.add(node);
      if (node === goal) break;
      for (const e of graph[node] || []) {
        const nc = c + e[weight];
        if (nc < (cost[e.neighbor] ?? Infinity)) {
          cost[e.neighbor] = nc; parent[e.neighbor] = [node, e]; pq.push([nc, e.neighbor]);
        }
      }
    }
    if (!(goal in cost)) return { nodes: [], found: false, totalDistance: 0, totalCost: 0 };
    let nodes = [goal], n = goal;
    while (n !== start) { const [prev] = parent[n]; nodes.push(prev); n = prev; }
    nodes.reverse();
    let totalDistance = 0, totalCost = 0;
    for (let i = 0; i < nodes.length - 1; i++) {
      for (const e of graph[nodes[i]] || []) {
        if (e.neighbor === nodes[i + 1]) { totalDistance += e.distance; totalCost += e.navigation_cost; break; }
      }
    }
    return { nodes, found: true, totalDistance, totalCost };
  }

  /* ------------------------------------------------------------
   * Bintulu lane geometry — pixel coordinates read off the chart.
   * NORTH access channel (upper) and SOUTH access channel (lower), each a
   * centreline polyline running from the sea entrance (left) toward the port,
   * then EXTENDED into berth nodes on the harbour (right).
   * ------------------------------------------------------------ */
  function buildWaypoints() {
    const states = {};
    const order = { NORTH: [], SOUTH: [], HARBOUR: [] };

    // NORTH ACCESS CHANNEL centreline (between the red NO* and green N* buoys),
    // sea entrance -> approach -> Southern Jetty / Container Terminal berths.
    const north = [
      [70, 405], [250, 440], [430, 480], [610, 515], [775, 545],   // along the buoyed channel
      [900, 560], [1010, 565],                                       // approach to OUTER BCH / jetty mouth
      [1095, 560], [1180, 540], [1255, 525],                         // Southern Jetty / Container Terminal berth
    ];
    north.forEach((p, i) => {
      const id = `N_WP${String(i + 1).padStart(3, "0")}`;
      states[id] = { id, x: p[0], y: p[1], lane: "NORTH", isSpawn: i === 0, isTerminal: i === north.length - 1 };
      order.NORTH.push(id);
    });

    // SOUTH ACCESS CHANNEL centreline (between red R* and green G* buoys),
    // sea entrance -> approach -> Inner Harbour 1 berth.
    const south = [
      [55, 615], [230, 655], [410, 700], [585, 735], [730, 755],    // along the buoyed channel
      [860, 775], [945, 790],                                        // approach to Inner Harbour 1 mouth
      [1030, 820], [1120, 850], [1190, 870],                         // Inner Harbour 1 berth
    ];
    south.forEach((p, i) => {
      const id = `S_WP${String(i + 1).padStart(3, "0")}`;
      states[id] = { id, x: p[0], y: p[1], lane: "SOUTH", isSpawn: i === 0, isTerminal: i === south.length - 1 };
      order.SOUTH.push(id);
    });

    // HARBOUR connector: a short shared lane linking the two channel mouths
    // through the Southern Jetty / Inner Harbour 2 area (so a vessel can also
    // cross between the north approach and the inner harbours).
    const harb = [
      [1050, 620], [1120, 650], [1190, 690],                        // Southern Jetty -> Inner Harbour 2 berth
    ];
    harb.forEach((p, i) => {
      const id = `H_WP${String(i + 1).padStart(3, "0")}`;
      states[id] = { id, x: p[0], y: p[1], lane: "HARBOUR", isSpawn: false, isTerminal: i === harb.length - 1 };
      order.HARBOUR.push(id);
    });

    // headings along each lane
    for (const lane of Object.keys(order)) {
      const ids = order[lane];
      for (let i = 0; i < ids.length; i++) {
        const cur = states[ids[i]], nxt = states[ids[Math.min(i + 1, ids.length - 1)]];
        cur.heading = Math.atan2(nxt.y - cur.y, nxt.x - cur.x) * (180 / Math.PI);
      }
    }
    return { states, order };
  }

  /* ------------------------------------------------------------
   * Lateral buoys read off the chart (pixel coords). color GREEN/RED.
   * Used for IALA channel-keeping checks + rendering over the chart.
   * ------------------------------------------------------------ */
  function buildBuoys() {
    return [
      // NORTH channel — red "NO" (port side) and green "N" (starboard side)
      { id: "NO9", color: "RED", x: 60, y: 370 }, { id: "N1", color: "GREEN", x: 70, y: 455 },
      { id: "NO7", color: "RED", x: 240, y: 405 }, { id: "N3", color: "GREEN", x: 225, y: 490 },
      { id: "NO5", color: "RED", x: 420, y: 445 }, { id: "N5", color: "GREEN", x: 415, y: 530 },
      { id: "NO3", color: "RED", x: 605, y: 475 }, { id: "N7", color: "GREEN", x: 600, y: 565 },
      { id: "NO1", color: "RED", x: 775, y: 500 }, { id: "N9", color: "GREEN", x: 775, y: 605 },
      // SOUTH channel — red "R" (port side) and green "G" (starboard side)
      { id: "R2", color: "RED", x: 60, y: 585 }, { id: "G2", color: "GREEN", x: 45, y: 665 },
      { id: "R4", color: "RED", x: 210, y: 620 }, { id: "G4", color: "GREEN", x: 215, y: 700 },
      { id: "R6", color: "RED", x: 385, y: 660 }, { id: "G6", color: "GREEN", x: 390, y: 745 },
      { id: "R8", color: "RED", x: 560, y: 695 }, { id: "G8", color: "GREEN", x: 560, y: 780 },
      { id: "R10", color: "RED", x: 715, y: 715 }, { id: "G10", color: "GREEN", x: 700, y: 800 },
    ];
  }

  function buildGraph(states, order, rng) {
    const graph = {};
    for (const id of Object.keys(states)) graph[id] = [];
    const BASE_SPEED = 8.0;
    function attach(u, v, edgeType, direction) {
      const d = dist(states[u], states[v]);
      const risk = edgeType === "virtual" ? 0.25 + rng() * 0.2 : 0.02 + rng() * 0.06;
      const traffic = rng() * 0.5, weather = rng() * 0.3, current = -0.2 + rng() * 0.4;
      const speed = Math.max(2.0, BASE_SPEED * (1 - current));
      const travelTime = d / speed, energyCost = d * 0.01;
      const navigationCost = travelTime + energyCost + 2.0 * traffic + 3.0 * risk;
      const difficulty = Math.min(1.0, 0.35 * risk + 0.25 * traffic + 0.2 * weather + 0.2 * Math.abs(current));
      graph[u].push({
        neighbor: v, distance: d, direction, edge_type: edgeType,
        risk: +risk.toFixed(3), traffic: +traffic.toFixed(3), weather: +weather.toFixed(3),
        current: +current.toFixed(3), speed: +speed.toFixed(2), travel_time: +travelTime.toFixed(2),
        energy_cost: +energyCost.toFixed(2), navigation_cost: +navigationCost.toFixed(2),
        difficulty: +difficulty.toFixed(3), edge_id: `${u}->${v}`,
      });
    }
    // sequential edges within each lane (both directions)
    for (const lane of Object.keys(order)) {
      const ids = order[lane];
      for (let i = 0; i < ids.length - 1; i++) {
        attach(ids[i], ids[i + 1], "normal", "forward");
        attach(ids[i + 1], ids[i], "normal", "backward");
      }
    }
    // junctions: NORTH end <-> HARBOUR start; HARBOUR end <-> SOUTH approach;
    // link the two channels through the harbour connector so the port is one graph.
    const nEnd = order.NORTH[order.NORTH.length - 1];
    const nApproach = order.NORTH[order.NORTH.length - 4]; // mouth near jetty
    const hStart = order.HARBOUR[0], hEnd = order.HARBOUR[order.HARBOUR.length - 1];
    const sApproach = order.SOUTH[order.SOUTH.length - 4];
    attach(nApproach, hStart, "normal", "forward"); attach(hStart, nApproach, "normal", "backward");
    attach(hEnd, sApproach, "normal", "forward"); attach(sApproach, hEnd, "normal", "backward");
    // a virtual cross-channel shortcut (mid-approach) as in the other envs
    attach(order.NORTH[5], order.SOUTH[5], "virtual", "virtual");
    attach(order.SOUTH[5], order.NORTH[5], "virtual", "virtual");
    return graph;
  }

  // ---------- Gaussian via Box-Muller ----------
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
  function pointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
    if (len2 === 0) return distXY(px, py, ax, ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2; t = Math.max(0, Math.min(1, t));
    return distXY(px, py, ax + t * dx, ay + t * dy);
  }
  function sideOfSegment(px, py, ax, ay, bx, by) { return (bx - ax) * (py - ay) - (by - ay) * (px - ax); }

  const REWARD = {
    goal: 100.0, invalid: -5.0, timeout: -30.0, stepPenalty: -0.2, revisitPenalty: -1.0,
    collisionPenalty: -25.0, ctePenalty: -0.05, dockBonusScale: 20.0, costScale: 1.0,
  };
  const MAX_STEPS = 60, N_SLOTS = 4, SUBSTEPS = 6, OBSTACLE_R = 22, DOCK_TOL = 40, IALA_GATE = 60;

  class BintuluEnv {
    constructor(seed = 42, cfg = {}) {
      this.seed = seed;
      this.cfg = Object.assign({ noiseStd: 0, packetErrorRate: 0, obstacleCount: 6 }, cfg);
      const rng = makeRNG(seed);
      const { states, order } = buildWaypoints();
      this.states = states; this.order = order;
      this.buoys = buildBuoys();
      this.graph = buildGraph(states, order, rng);
      this.MAX_X = MAP_W; this.MAX_Y = MAP_H;
      this.maxSteps = MAX_STEPS; this.nActions = N_SLOTS;
      this.missionRng = makeRNG(seed + 777);
      this.noiseRng = makeRNG(seed + 991);
      this.commRng = makeRNG(seed + 4242);
      this.gauss = makeGaussian(this.noiseRng);
      this.adj = {};
      for (const id of Object.keys(this.graph)) {
        this.adj[id] = this.graph[id].slice().sort((e1, e2) =>
          e1.navigation_cost !== e2.navigation_cost ? e1.navigation_cost - e2.navigation_cost
            : (e1.neighbor < e2.neighbor ? -1 : 1));
      }
      this.obstacles = this._buildObstacles(this.cfg.obstacleCount);
      this.obsDim = 9 + 4 * N_SLOTS + 3;
      this._planCache = {};
      this.reset();
    }

    laneEncoding(l) { return { NORTH: 0, SOUTH: 1, HARBOUR: 2 }[l] ?? 0; }

    _buildObstacles(n) {
      const rng = makeRNG(this.seed + 5150), obs = [], wps = Object.values(this.states);
      const edges = [];
      for (const id of Object.keys(this.graph))
        for (const e of this.graph[id]) {
          const A = this.states[id], B = this.states[e.neighbor];
          if (A && B && id < e.neighbor && e.edge_type !== "virtual") edges.push([A, B]);
        }
      let placed = 0, tries = 0; const nEdge = Math.round(n * 0.6);
      while (placed < nEdge && tries < 400 && edges.length) {
        tries++;
        const [A, B] = edges[Math.floor(rng() * edges.length)];
        const t = 0.35 + rng() * 0.3, mx = A.x + (B.x - A.x) * t, my = A.y + (B.y - A.y) * t;
        const ang = Math.atan2(B.y - A.y, B.x - A.x) + Math.PI / 2;
        const off = (rng() < 0.5 ? -1 : 1) * (OBSTACLE_R * (0.4 + rng() * 0.7));
        const x = mx + Math.cos(ang) * off, y = my + Math.sin(ang) * off;
        let ok = true;
        for (const w of wps) if (distXY(x, y, w.x, w.y) < 34) { ok = false; break; }
        if (ok) { obs.push({ x, y, r: OBSTACLE_R }); placed++; }
      }
      tries = 0;
      while (obs.length < n && tries < 400) {
        tries++;
        const x = 120 + rng() * (MAP_W - 500), y = 300 + rng() * (MAP_H - 450);
        let ok = true;
        for (const w of wps) if (distXY(x, y, w.x, w.y) < 40) { ok = false; break; }
        if (ok) obs.push({ x, y, r: OBSTACLE_R });
      }
      return obs;
    }

    _plan(start, goal) {
      const key = start + "|" + goal;
      if (!(key in this._planCache)) this._planCache[key] = dijkstra(this.graph, start, goal);
      return this._planCache[key];
    }

    _randomMission() {
      // start at a sea entrance of either channel; goal = a berth terminal.
      const starts = [this.order.NORTH[0], this.order.SOUTH[0]];
      const goals = [
        this.order.NORTH[this.order.NORTH.length - 1],   // Southern Jetty / Container Terminal
        this.order.SOUTH[this.order.SOUTH.length - 1],   // Inner Harbour 1
        this.order.HARBOUR[this.order.HARBOUR.length - 1],// Inner Harbour 2
      ];
      let start, goal, plan, tries = 0;
      do {
        start = starts[Math.floor(this.missionRng() * starts.length)];
        goal = goals[Math.floor(this.missionRng() * goals.length)];
        tries++;
        if (start === goal) { plan = { found: false }; continue; }
        plan = this._plan(start, goal);
      } while ((!plan.found || plan.totalCost <= 0) && tries < 50);
      if (!plan.found) { start = this.order.NORTH[0]; goal = this.order.NORTH[this.order.NORTH.length - 1]; plan = this._plan(start, goal); }
      return { start, goal, plan };
    }

    reset(mission = null) {
      this.stepCount = 0; this.totalReward = 0; this.done = false; this.truncated = false;
      const m = mission || this._randomMission();
      this.startWp = m.start; this.goalWp = m.goal;
      const plan = m.plan || this._plan(this.startWp, this.goalWp);
      this.plannedPath = plan.found ? plan.nodes.slice() : [this.startWp, this.goalWp];
      this.plannedDistance = plan.totalDistance || 0; this.optimalCost = plan.totalCost || 0;
      this.currentWp = this.startWp; this.actualPath = [this.currentWp]; this.visited = { [this.currentWp]: 1 };
      const s0 = this.states[this.startWp]; this.posX = s0.x; this.posY = s0.y;
      this.routeCost = 0; this.routeRisk = 0; this.routeDifficulty = 0; this.routeDistance = 0;
      this.invalidCount = 0; this.revisitCount = 0;
      this.collisionCount = 0; this.cteSum = 0; this.cteSamples = 0; this.ialaViolations = 0;
      this.droppedFrames = 0; this.lastCollision = 0; this.dockingAccuracy = null;
      this.nearestObstacleDist = this._nearestObstacle(this.posX, this.posY);
      this._lastObs = null;
      const first = this._trueObs(); this._lastObs = first;
      return this._commObs(this._sensedObs(first));
    }

    _nearestObstacle(x, y) { let b = Infinity; for (const o of this.obstacles) b = Math.min(b, distXY(x, y, o.x, o.y) - o.r); return b; }
    _towardGoal(fromId, toId) {
      const g = this.states[this.goalWp], a = this.states[fromId], b = this.states[toId];
      return dist(b, g) < dist(a, g) ? 1 : 0;
    }

    _trueObs() {
      const g = this.states[this.goalWp], s = this.states[this.currentWp];
      const dgoal = distXY(this.posX, this.posY, g.x, g.y);
      const edges = this.adj[this.currentWp] || [];
      const cteNorm = Math.min(1, (this.cteSamples ? this.cteSum / this.cteSamples : 0) / 80);
      const nearObs = Math.max(0, Math.min(1, this.nearestObstacleDist / 160));
      const obs = [
        this.posX / this.MAX_X, this.posY / this.MAX_Y, g.x / this.MAX_X, g.y / this.MAX_Y,
        (g.x - this.posX) / this.MAX_X, (g.y - this.posY) / this.MAX_Y, dgoal / 1500,
        this.laneEncoding(s.lane) / 2, this.stepCount / this.maxSteps,
      ];
      for (let a = 0; a < N_SLOTS; a++) {
        if (a < edges.length) { const e = edges[a]; obs.push(1, Math.min(1, e.navigation_cost / 40), e.difficulty, this._towardGoal(this.currentWp, e.neighbor)); }
        else obs.push(0, 0, 0, 0);
      }
      obs.push(cteNorm, nearObs, this.lastCollision);
      return obs;
    }

    _sensedObs(trueObs) {
      const std = this.cfg.noiseStd;
      if (std <= 0) return trueObs.slice();
      const out = trueObs.slice();
      for (let i = 0; i < out.length; i++) {
        const isExists = (i >= 9 && (i - 9) % 4 === 0 && i < 9 + 4 * N_SLOTS);
        if (isExists) continue;
        out[i] += this.gauss() * std;
      }
      return out;
    }
    _commObs(sensed) {
      if (this.cfg.packetErrorRate > 0 && this.commRng() < this.cfg.packetErrorRate && this._lastObs) {
        this.droppedFrames++; return this._lastObs.slice();
      }
      this._lastObs = sensed.slice(); return sensed;
    }

    _cteToPlan(x, y) {
      const p = this.plannedPath; if (p.length < 2) return 0;
      let best = Infinity;
      for (let i = 0; i < p.length - 1; i++) {
        const a = this.states[p[i]], b = this.states[p[i + 1]];
        if (a && b) best = Math.min(best, pointToSegment(x, y, a.x, a.y, b.x, b.y));
      }
      return isFinite(best) ? best : 0;
    }

    _traverseEdge(fromId, toId) {
      const A = this.states[fromId], B = this.states[toId];
      let collided = false;
      const driftScale = 90 * this.cfg.noiseStd + 45 * this.cfg.packetErrorRate;
      const segLen = distXY(A.x, A.y, B.x, B.y) || 1;
      const nx = -(B.y - A.y) / segLen, ny = (B.x - A.x) / segLen;
      const nearBuoys = this.buoys
        .map((bu) => ({ bu, d: pointToSegment(bu.x, bu.y, A.x, A.y, B.x, B.y), side: sideOfSegment(bu.x, bu.y, A.x, A.y, B.x, B.y) }))
        .filter((o) => o.d <= IALA_GATE);
      const ialaHit = new Set();
      for (let k = 1; k <= SUBSTEPS; k++) {
        const t = k / SUBSTEPS;
        let x = A.x + (B.x - A.x) * t, y = A.y + (B.y - A.y) * t;
        if (driftScale > 0) {
          const toGoal = (toId === this.goalWp), endHold = toGoal ? 0.5 : 0.0;
          const shape = Math.sin(Math.PI * t) * (1 - endHold) + (toGoal ? endHold * t : 0);
          const drift = this.gauss() * driftScale * shape; x += nx * drift; y += ny * drift;
        }
        this.posX = x; this.posY = y;
        this.cteSum += this._cteToPlan(x, y); this.cteSamples++;
        for (const o of this.obstacles) if (distXY(x, y, o.x, o.y) <= o.r) { collided = true; break; }
        const vSide = sideOfSegment(x, y, A.x, A.y, B.x, B.y), vPerp = Math.abs(vSide) / segLen;
        for (const nb of nearBuoys) {
          if (ialaHit.has(nb.bu)) continue;
          const bPerp = Math.abs(nb.side) / segLen, sameSide = (vSide > 0) === (nb.side > 0) && nb.side !== 0;
          if (sameSide && vPerp > bPerp) { this.ialaViolations++; ialaHit.add(nb.bu); }
        }
      }
      this.nearestObstacleDist = this._nearestObstacle(this.posX, this.posY);
      return collided;
    }

    step(action) {
      this.stepCount++;
      const edges = this.adj[this.currentWp] || [];
      let invalid = false, reward = REWARD.stepPenalty, collidedThisStep = false;
      if (action < 0 || action >= edges.length) { invalid = true; this.invalidCount++; reward += REWARD.invalid; }
      else {
        const e = edges[action], fromId = this.currentWp;
        collidedThisStep = this._traverseEdge(fromId, e.neighbor);
        reward += REWARD.costScale * (-e.navigation_cost) * 0.1;
        this.routeCost += e.navigation_cost; this.routeRisk += e.risk; this.routeDifficulty += e.difficulty; this.routeDistance += e.distance;
        this.currentWp = e.neighbor; this.actualPath.push(this.currentWp);
        if (this.visited[this.currentWp]) { this.revisitCount++; reward += REWARD.revisitPenalty; }
        this.visited[this.currentWp] = (this.visited[this.currentWp] || 0) + 1;
        if (collidedThisStep) { this.collisionCount++; reward += REWARD.collisionPenalty; }
      }
      this.lastCollision = collidedThisStep ? 1 : 0;
      const reachedGoal = this.currentWp === this.goalWp;
      const timeout = this.stepCount >= this.maxSteps;
      const meanCte = this.cteSamples ? this.cteSum / this.cteSamples : 0;
      reward += REWARD.ctePenalty * meanCte * 0.1;
      if (reachedGoal) {
        reward += REWARD.goal;
        const g = this.states[this.goalWp];
        this.dockingAccuracy = distXY(this.posX, this.posY, g.x, g.y);
        reward += REWARD.dockBonusScale * Math.max(0, 1 - this.dockingAccuracy / DOCK_TOL);
      } else if (timeout) reward += REWARD.timeout;
      this.totalReward += reward;
      let terminated = reachedGoal;
      if (timeout && !reachedGoal) this.truncated = true;
      this.done = terminated || this.truncated;
      const meanCteFinal = this.cteSamples ? this.cteSum / this.cteSamples : 0;
      const info = {
        currentWp: this.currentWp, goalWp: this.goalWp, reachedGoal, invalid, timeout,
        routeCost: this.routeCost, optimalCost: this.optimalCost,
        optimalityRatio: this.optimalCost > 0 && reachedGoal ? this.routeCost / this.optimalCost : null,
        collided: collidedThisStep, collisionCount: this.collisionCount, cteMean: meanCteFinal,
        ialaViolations: this.ialaViolations, droppedFrames: this.droppedFrames,
        dockingAccuracy: this.dockingAccuracy, posX: this.posX, posY: this.posY,
      };
      const delivered = this._commObs(this._sensedObs(this._trueObs()));
      return { obs: delivered, reward, terminated, truncated: this.truncated, info };
    }
  }

  root.BintuluPortEnv = { BintuluEnv, dijkstra, MAP_W, MAP_H, REWARD, MAX_STEPS, DOCK_TOL };
  if (typeof module !== "undefined" && module.exports) module.exports = root.BintuluPortEnv;
})(typeof window !== "undefined" ? window : globalThis);
