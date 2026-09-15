/* ============================================================
 * BINTULU PORT — Web DQN Simulation
 * environment.js
 *
 * Ports the essential structure of the research notebook:
 *   - Waypoint graph (North L02, South L08, Harbour L01)
 *   - Green / red buoys along the channels
 *   - Edge attributes: distance, risk, traffic, weather, current,
 *     travel_time, energy_cost, navigation_cost, difficulty
 *   - Virtual "shortcut" branches (edge_type = "virtual")
 *   - Gymnasium-style reset()/step() with Stay/Forward/Backward
 *   - RewardEngine + TerminationEngine semantics
 *   - Dijkstra planner (navigation_cost weighted) with virtual-edge
 *     expansion into physical waypoints
 *
 * Everything is deterministic given a seed so results are
 * reproducible, matching the notebook's RANDOM_SEED = 42 intent.
 * ============================================================ */

// ---------- Seeded RNG (mulberry32) ----------
function makeRNG(seed) {
  let s = seed >>> 0;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- Action space ----------
const ACTIONS = { WAIT: 0, FORWARD: 1, BACKWARD: 2 };
const ACTION_NAMES = ["WAIT", "FORWARD", "BACKWARD"];

// ---------- Reward configuration (from notebook RewardConfig) ----------
const REWARD_CONFIG = {
  progress: 1.0,
  goal: 100.0,
  wait: -1.0,
  backward: -2.0,
  invalid: -10.0,
  timeout: -50.0,
};

const MAX_EPISODE_STEPS = 200;

// ---------- Map canvas dimensions (logical coordinates) ----------
const MAP_W = 900;
const MAP_H = 560;

/* ------------------------------------------------------------
 * Build the lane geometry.
 * We synthesize Bintulu-style channels since the raw chart image
 * is not available in the browser. The topology mirrors the
 * notebook: two long access channels (North/South) that both feed
 * into a shared harbour lane, plus virtual shortcuts.
 * ------------------------------------------------------------ */
function buildWaypoints() {
  const states = {}; // id -> {id, x, y, lane, isSpawn, isTerminal, heading}
  const order = { L02: [], L08: [], L01: [] };

  // North access channel (L02): sweeps left -> right, gentle curve up
  const nNorth = 22;
  for (let i = 0; i < nNorth; i++) {
    const t = i / (nNorth - 1);
    const x = 70 + t * 640;
    const y = 150 + Math.sin(t * Math.PI) * -40 + t * 30;
    const id = `L02_WP${String(i + 1).padStart(3, "0")}`;
    states[id] = { id, x, y, lane: "L02", isSpawn: i === 0, isTerminal: i === nNorth - 1 };
    order.L02.push(id);
  }

  // South access channel (L08): sweeps left -> right, curve down
  const nSouth = 20;
  for (let i = 0; i < nSouth; i++) {
    const t = i / (nSouth - 1);
    const x = 70 + t * 640;
    const y = 410 + Math.sin(t * Math.PI) * 40 - t * 20;
    const id = `L08_WP${String(i + 1).padStart(3, "0")}`;
    states[id] = { id, x, y, lane: "L08", isSpawn: i === 0, isTerminal: i === nSouth - 1 };
    order.L08.push(id);
  }

  // Harbour lane (L01): short vertical berth on the right
  const nHarb = 6;
  for (let i = 0; i < nHarb; i++) {
    const t = i / (nHarb - 1);
    const x = 760 + t * 70;
    const y = 300 - t * 20;
    const id = `L01_WP${String(i + 1).padStart(3, "0")}`;
    states[id] = { id, x, y, lane: "L01", isSpawn: i === 0, isTerminal: i === nHarb - 1 };
    order.L01.push(id);
  }

  // heading per waypoint (angle to next in lane)
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

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/* ------------------------------------------------------------
 * Build the undirected navigation graph with edge attributes.
 * Mirrors Sprint 6.3 edge property generation.
 * ------------------------------------------------------------ */
function buildGraph(states, order, rng) {
  const graph = {}; // id -> [ {neighbor, distance, direction, edge_type, ...attrs} ]
  for (const id of Object.keys(states)) graph[id] = [];

  const BASE_SPEED = 8.0;

  function attach(u, v, edgeType, direction) {
    const d = dist(states[u], states[v]);
    const risk = edgeType === "virtual" ? 0.25 + rng() * 0.2 : 0.02 + rng() * 0.06;
    const traffic = rng() * 0.5;
    const weather = rng() * 0.3;
    const current = -0.2 + rng() * 0.4;
    let speed = BASE_SPEED * (1 - current);
    speed = Math.max(2.0, speed);
    const travelTime = d / speed;
    const energyCost = d * 0.02;
    const navigationCost = travelTime + energyCost + 2.0 * traffic + 3.0 * risk;
    let difficulty = 0.35 * risk + 0.25 * traffic + 0.2 * weather + 0.2 * Math.abs(current);
    difficulty = Math.min(1.0, difficulty);
    graph[u].push({
      neighbor: v,
      distance: d,
      direction,
      edge_type: edgeType,
      risk: +risk.toFixed(3),
      traffic: +traffic.toFixed(3),
      weather: +weather.toFixed(3),
      current: +current.toFixed(3),
      speed: +speed.toFixed(2),
      travel_time: +travelTime.toFixed(2),
      energy_cost: +energyCost.toFixed(2),
      navigation_cost: +navigationCost.toFixed(2),
      difficulty: +difficulty.toFixed(3),
      edge_id: `${u}->${v}`,
    });
  }

  // sequential (physical) edges within each lane, both directions
  for (const lane of Object.keys(order)) {
    const ids = order[lane];
    for (let i = 0; i < ids.length - 1; i++) {
      attach(ids[i], ids[i + 1], "normal", "forward");
      attach(ids[i + 1], ids[i], "normal", "backward");
    }
  }

  // connect harbour to the end of both access channels (junctions)
  const northEnd = order.L02[order.L02.length - 1];
  const southEnd = order.L08[order.L08.length - 1];
  const harbStart = order.L01[0];
  attach(northEnd, harbStart, "normal", "forward");
  attach(harbStart, northEnd, "normal", "backward");
  attach(southEnd, harbStart, "normal", "forward");
  attach(harbStart, southEnd, "normal", "backward");

  // a few virtual shortcut branches within North lane (Sprint 6.2)
  const virtualPairs = [
    ["L02_WP004", "L02_WP010"],
    ["L02_WP009", "L02_WP015"],
    ["L08_WP004", "L08_WP011"],
  ];
  for (const [a, b] of virtualPairs) {
    if (states[a] && states[b]) {
      attach(a, b, "virtual", "virtual");
      attach(b, a, "virtual", "virtual");
    }
  }

  return graph;
}

/* ------------------------------------------------------------
 * Build buoys: green (starboard) and red (port) markers spaced
 * along both access channels — offset perpendicular to the lane.
 * ------------------------------------------------------------ */
function buildBuoys(states, order) {
  const buoys = [];
  let g = 1;
  let r = 1;
  for (const lane of ["L02", "L08"]) {
    const ids = order[lane];
    for (let i = 1; i < ids.length - 1; i += 2) {
      const cur = states[ids[i]];
      const nxt = states[ids[Math.min(i + 1, ids.length - 1)]];
      const ang = Math.atan2(nxt.y - cur.y, nxt.x - cur.x);
      const nx = -Math.sin(ang);
      const ny = Math.cos(ang);
      const off = 16;
      buoys.push({ id: `G${String(g++).padStart(3, "0")}`, color: "GREEN", x: cur.x + nx * off, y: cur.y + ny * off });
      buoys.push({ id: `R${String(r++).padStart(3, "0")}`, color: "RED", x: cur.x - nx * off, y: cur.y - ny * off });
    }
  }
  return buoys;
}

/* ------------------------------------------------------------
 * Dijkstra planner over navigation_cost, with virtual-edge
 * expansion into physical waypoints (Sprint 6.5C semantics).
 * Returns { nodes: [...], found, totalDistance, totalCost }.
 * ------------------------------------------------------------ */
function dijkstra(graph, start, goal, weight = "navigation_cost") {
  const cost = { [start]: 0 };
  const parent = {};
  const visited = new Set();
  // simple priority via array (graph is small)
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
        cost[e.neighbor] = nc;
        parent[e.neighbor] = [node, e];
        pq.push([nc, e.neighbor]);
      }
    }
  }
  if (!(goal in cost)) return { nodes: [], found: false, totalDistance: 0, totalCost: 0 };

  // reconstruct
  let nodes = [goal];
  let n = goal;
  while (n !== start) {
    const [prev] = parent[n];
    nodes.push(prev);
    n = prev;
  }
  nodes.reverse();

  // expand virtual edges into physical shortest paths (BFS on physical edges)
  const expanded = expandVirtual(graph, nodes);

  let totalDistance = 0;
  let totalCost = 0;
  for (let i = 0; i < expanded.length - 1; i++) {
    const e = findEdge(graph, expanded[i], expanded[i + 1]);
    if (e) {
      totalDistance += e.distance;
      totalCost += e.navigation_cost;
    }
  }
  return { nodes: expanded, found: true, totalDistance, totalCost };
}

function findEdge(graph, u, v) {
  for (const e of graph[u] || []) if (e.neighbor === v) return e;
  return null;
}

function isPhysicalNeighbor(graph, u, v) {
  for (const e of graph[u] || []) if (e.neighbor === v && e.edge_type !== "virtual") return true;
  return false;
}

function physicalBFS(graph, start, goal) {
  const parent = { [start]: null };
  const q = [start];
  while (q.length) {
    const cur = q.shift();
    if (cur === goal) break;
    for (const e of graph[cur] || []) {
      if (e.edge_type === "virtual") continue;
      if (e.neighbor in parent) continue;
      parent[e.neighbor] = cur;
      q.push(e.neighbor);
    }
  }
  if (!(goal in parent)) return [start, goal];
  const path = [];
  let n = goal;
  while (n !== null) {
    path.push(n);
    n = parent[n];
  }
  path.reverse();
  return path;
}

function expandVirtual(graph, nodes) {
  if (nodes.length <= 1) return nodes;
  const out = [nodes[0]];
  for (let i = 0; i < nodes.length - 1; i++) {
    const u = nodes[i];
    const v = nodes[i + 1];
    if (isPhysicalNeighbor(graph, u, v)) {
      out.push(v);
    } else {
      const phys = physicalBFS(graph, u, v);
      for (let k = 1; k < phys.length; k++) out.push(phys[k]);
    }
  }
  return out;
}

/* ============================================================
 * VesselNavigationEnv — Gymnasium-style environment
 * ============================================================ */
class VesselEnv {
  constructor(seed = 42) {
    this.seed = seed;
    this.rng = makeRNG(seed);
    const { states, order } = buildWaypoints();
    this.states = states;
    this.order = order;
    this.graph = buildGraph(states, order, this.rng);
    this.buoys = buildBuoys(states, order);
    this.maxSteps = MAX_EPISODE_STEPS;

    // normalization constants
    this.MAX_X = MAP_W;
    this.MAX_Y = MAP_H;

    // episode RNG for mission selection (separate stream)
    this.missionRng = makeRNG(seed + 777);

    this.obsDim = 10;
    this.nActions = 3;
    this.reset();
  }

  laneEncoding(lane) {
    return { L02: 0, L08: 1, L01: 2 }[lane] ?? 0;
  }

  _randomMission() {
    // pick a spawn on an access channel, goal = harbour terminal or lane terminal
    const spawnCandidates = Object.values(this.states).filter((s) => s.isSpawn && s.lane !== "L01");
    const start = spawnCandidates[Math.floor(this.missionRng() * spawnCandidates.length)];
    // goal: end of the same access lane (kept solvable & interpretable)
    const laneIds = this.order[start.lane];
    const goal = this.states[laneIds[laneIds.length - 1]];
    return { start: start.id, goal: goal.id };
  }

  reset(mission = null) {
    this.stepCount = 0;
    this.totalReward = 0;
    this.done = false;
    this.truncated = false;

    const m = mission || this._randomMission();
    this.startWp = m.start;
    this.goalWp = m.goal;

    const plan = dijkstra(this.graph, this.startWp, this.goalWp);
    this.plannedPath = plan.found ? plan.nodes.slice() : [this.startWp, this.goalWp];
    this.plannedDistance = plan.totalDistance;
    this.plannedCost = plan.totalCost;

    // agent follows the planned path index (Sprint 6.8C path-index model)
    this.pathNodes = this.plannedPath.slice();
    this.pathIndex = 0;
    this.currentWp = this.pathNodes[0];
    this.actualPath = [this.currentWp];

    return this._obs();
  }

  _obs() {
    const s = this.states[this.currentWp];
    const g = this.states[this.goalWp];
    const remainingSteps = this.pathNodes.length - 1 - this.pathIndex;
    const distGoal = dist(s, g);
    // local edge stats
    const edges = this.graph[this.currentWp] || [];
    let avgDiff = 0;
    let avgCost = 0;
    for (const e of edges) {
      avgDiff += e.difficulty;
      avgCost += e.navigation_cost;
    }
    if (edges.length) {
      avgDiff /= edges.length;
      avgCost /= edges.length;
    }
    return [
      s.x / this.MAX_X,
      s.y / this.MAX_Y,
      (s.heading + 180) / 360,
      this.laneEncoding(s.lane) / 2,
      distGoal / 900,
      remainingSteps / this.pathNodes.length,
      this.pathIndex / this.pathNodes.length,
      avgDiff,
      Math.min(1, avgCost / 30),
      s.isTerminal ? 1 : 0,
    ];
  }

  step(action) {
    this.stepCount++;
    let invalid = false;

    if (action === ACTIONS.FORWARD) {
      if (this.pathIndex < this.pathNodes.length - 1) this.pathIndex++;
      else invalid = true;
    } else if (action === ACTIONS.BACKWARD) {
      if (this.pathIndex > 0) this.pathIndex--;
      else invalid = true;
    }
    // WAIT: no movement

    this.currentWp = this.pathNodes[this.pathIndex];
    this.actualPath.push(this.currentWp);

    const reachedGoal = this.currentWp === this.goalWp;
    let timeout = false;
    if (this.stepCount >= this.maxSteps) timeout = true;

    // ---- reward (RewardEngine) ----
    let reward = 0;
    if (invalid) {
      reward = REWARD_CONFIG.invalid;
    } else if (timeout && !reachedGoal) {
      reward = REWARD_CONFIG.timeout;
    } else {
      if (action === ACTIONS.WAIT) reward += REWARD_CONFIG.wait;
      else if (action === ACTIONS.FORWARD) reward += REWARD_CONFIG.progress;
      else if (action === ACTIONS.BACKWARD) reward += REWARD_CONFIG.backward;
      if (reachedGoal) reward += REWARD_CONFIG.goal;
    }
    this.totalReward += reward;

    // ---- termination ----
    let terminated = false;
    if (reachedGoal) terminated = true;
    if (timeout) this.truncated = true;
    this.done = terminated || this.truncated;

    const info = {
      currentWp: this.currentWp,
      goalWp: this.goalWp,
      pathIndex: this.pathIndex,
      remainingSteps: this.pathNodes.length - 1 - this.pathIndex,
      reachedGoal,
      invalid,
      timeout,
    };
    return { obs: this._obs(), reward, terminated, truncated: this.truncated, info };
  }
}

// expose to window
window.BintuluEnv = {
  VesselEnv,
  ACTIONS,
  ACTION_NAMES,
  REWARD_CONFIG,
  MAP_W,
  MAP_H,
  dijkstra,
};
