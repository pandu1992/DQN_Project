/* ============================================================
 * Record two-vessel greedy trajectories for GIF rendering, on the Bintulu chart:
 *   multi_interaction : two independent vessels (MultiVesselBintulu) that must
 *                       avoid each other (extension d).
 *   twoway_headon     : an inbound + an outbound vessel share one access channel
 *                       in opposing directions (extension e, TwoWayBintulu).
 * Trains DQN where needed, plays greedy, logs both vessels' per-step pose plus
 * the shared static scene, planned paths, goals, and inter-vessel events.
 * Output: results_bintulu/trajectories/<name>.json
 * ============================================================ */
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
function load() {
  const jsDir = path.join(__dirname, "..", "js"), sb = {};
  sb.window = sb; sb.globalThis = sb; sb.Math = Math; sb.Float64Array = Float64Array; sb.Array = Array; sb.console = console; sb.module = undefined;
  vm.createContext(sb);
  for (const f of ["dqn.js", "rulebased.js", "environmentBintulu.js", "environmentBintuluMulti.js", "environmentBintuluOps.js"])
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  return { Multi: sb.window.BintuluMulti, Ops: sb.window.BintuluOps, DQN: sb.window.BintuluDQN, RB: sb.window.BintuluRuleBased };
}
const { Multi, Ops, DQN, RB } = load();
const { MultiVesselBintulu, VESSEL_COLLISION_RADIUS } = Multi;
const { TwoWayBintulu } = Ops;
const { DQNAgent } = DQN; const { RuleBasedAgent } = RB;
function mulberry32(seed) { let s = seed >>> 0; return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function sceneOf(env) {
  return {
    mapW: env.MAX_X, mapH: env.MAX_Y,
    buoys: env.buoys ? env.buoys.map((b) => ({ x: b.x, y: b.y, color: b.color })) : [],
    obstacles: env.obstacles.map((o) => ({ x: o.x, y: o.y, r: o.r })),
    states: Object.fromEntries(Object.entries(env.states).map(([id, s]) => [id, { x: s.x, y: s.y, lane: s.lane }])),
  };
}
function pathOf(env) { return env.plannedPath.map((id) => ({ id, x: env.states[id].x, y: env.states[id].y })); }

// ---------- multi-vessel ----------
function trainMulti(menv, episodes, budget) {
  const aA = new DQNAgent(menv.obsDim, menv.nActions, { algorithm: "DQN", per: false, noisy: false, totalSteps: budget });
  const aB = new DQNAgent(menv.obsDim, menv.nActions, { algorithm: "DQN", per: false, noisy: false, totalSteps: budget });
  for (let ep = 0; ep < episodes; ep++) {
    let [oA, oB] = menv.reset();
    const MX = Math.max(menv.A.maxSteps, menv.B.maxSteps);
    while (true) {
      const a = aA.act(oA, false), b = aB.act(oB, false);
      const r = menv.stepBoth(a, b);
      aA.observe(oA, a, r.A.reward, r.A.obs, r.A.terminated);
      aB.observe(oB, b, r.B.reward, r.B.obs, r.B.terminated);
      oA = r.A.obs; oB = r.B.obs;
      if (menv.done || menv.stepCount >= MX) break;
    }
  }
  return [aA, aB];
}
function playMulti(menv, aA, aB) {
  let [oA, oB] = menv.reset();
  const A = menv.A, B = menv.B, MX = Math.max(A.maxSteps, B.maxSteps);
  const fA = [{ x: A.posX, y: A.posY }], fB = [{ x: B.posX, y: B.posY }];
  const planA = pathOf(A), planB = pathOf(B), goalA = A.goalWp, goalB = B.goalWp;
  let minCPAstep = [];
  while (true) {
    const a = aA.act(oA, true), b = aB.act(oB, true);
    const r = menv.stepBoth(a, b);
    fA.push({ x: A.posX, y: A.posY }); fB.push({ x: B.posX, y: B.posY });
    minCPAstep.push(+menv.minCPA.toFixed(1));
    oA = r.A.obs; oB = r.B.obs;
    if (menv.done || menv.stepCount >= MX) break;
  }
  return { fA, fB, planA, planB,
    goalA: { x: menv.A.states[goalA].x, y: menv.A.states[goalA].y },
    goalB: { x: menv.B.states[goalB].x, y: menv.B.states[goalB].y },
    successA: A.currentWp === goalA ? 1 : 0, successB: B.currentWp === goalB ? 1 : 0,
    vesselCollisions: menv.vesselCollisions, nearMisses: menv.nearMisses, minCPA: +menv.minCPA.toFixed(1),
    steps: fA.length - 1 };
}

// ---------- two-way ----------
function trainWay(menv, episodes, budget) {
  const aIn = new DQNAgent(menv.obsDim, menv.IN.nActions, { algorithm: "DQN", per: false, noisy: false, totalSteps: budget });
  const aOut = new RuleBasedAgent(menv.OUT.obsDim, menv.OUT.nActions, { env: menv.OUT });
  for (let ep = 0; ep < episodes; ep++) {
    let [oi, oo] = menv.reset();
    const MX = Math.max(menv.IN.maxSteps, menv.OUT.maxSteps);
    while (true) {
      const ai = aIn.act(oi, false), ao = aOut.act(oo, false);
      const r = menv.stepBoth(ai, ao);
      aIn.observe(oi, ai, r.IN.reward, r.IN.obs, r.IN.terminated);
      oi = r.IN.obs; oo = r.OUT.obs;
      if (menv.done || menv.stepCount >= MX) break;
    }
  }
  return [aIn, aOut];
}
function playWay(menv, aIn, aOut) {
  let [oi, oo] = menv.reset();
  const A = menv.IN, B = menv.OUT, MX = Math.max(A.maxSteps, B.maxSteps);
  const fA = [{ x: A.posX, y: A.posY }], fB = [{ x: B.posX, y: B.posY }];
  const planA = pathOf(A), planB = pathOf(B), goalA = A.goalWp, goalB = B.goalWp;
  while (true) {
    const ai = aIn.act(oi, true), ao = aOut.act(oo, true);
    const r = menv.stepBoth(ai, ao);
    fA.push({ x: A.posX, y: A.posY }); fB.push({ x: B.posX, y: B.posY });
    oi = r.IN.obs; oo = r.OUT.obs;
    if (menv.done || menv.stepCount >= MX) break;
  }
  return { fA, fB, planA, planB,
    goalA: { x: A.states[goalA].x, y: A.states[goalA].y }, goalB: { x: B.states[goalB].x, y: B.states[goalB].y },
    successA: A.currentWp === goalA ? 1 : 0, successB: B.currentWp === goalB ? 1 : 0,
    vesselCollisions: menv.vesselCollisions, nearMisses: menv.nearMisses, headOnEvents: menv.headOnEvents,
    minCPA: +(isFinite(menv.minCPA) ? menv.minCPA : 0).toFixed(1), lane: menv.lane, steps: fA.length - 1 };
}

function main() {
  const OUT = path.join(__dirname, "..", "results_bintulu", "trajectories");
  fs.mkdirSync(OUT, { recursive: true });

  // 1) multi-vessel interaction (two learned vessels, mid condition for a lively encounter)
  {
    const seed = 3; Math.random = mulberry32(seed * 1000003 + 12345);
    const menv = new MultiVesselBintulu(42 + seed, { noiseStd: 0.1, packetErrorRate: 0.2, obstacleCount: 6 });
    const [aA, aB] = trainMulti(menv, 150, 150 * 60);
    // play a few, prefer an episode where the two actually encounter (minCPA small)
    let best = null;
    for (let k = 0; k < 25; k++) { const ep = playMulti(menv, aA, aB); if (!best || ep.minCPA < best.minCPA) best = ep; if (ep.minCPA < 80) { best = ep; break; } }
    const data = { name: "multi_interaction", scene: sceneOf(menv.A), episode: best, radius: VESSEL_COLLISION_RADIUS, kind: "multi" };
    fs.writeFileSync(path.join(OUT, "multi_interaction.json"), JSON.stringify(data));
    console.log(`multi_interaction: steps=${best.steps} succA=${best.successA} succB=${best.successB} vColl=${best.vesselCollisions} nearMiss=${best.nearMisses} minCPA=${best.minCPA}`);
  }

  // 2) two-way head-on (learned inbound vs rule-based outbound in one channel)
  {
    const seed = 0; Math.random = mulberry32(seed * 1000003 + 12345);
    const menv = new TwoWayBintulu(42 + seed, { noiseStd: 0.1, packetErrorRate: 0.2, obstacleCount: 6 });
    const [aIn, aOut] = trainWay(menv, 200, 200 * 60);
    let best = null;
    for (let k = 0; k < 25; k++) { const ep = playWay(menv, aIn, aOut); if (!best || ep.headOnEvents > best.headOnEvents) best = ep; if (ep.headOnEvents >= 1) { best = ep; break; } }
    const data = { name: "twoway_headon", scene: sceneOf(menv.IN), episode: best, radius: VESSEL_COLLISION_RADIUS, kind: "twoway" };
    fs.writeFileSync(path.join(OUT, "twoway_headon.json"), JSON.stringify(data));
    console.log(`twoway_headon: lane=${best.lane} steps=${best.steps} succIn=${best.successA} succOut=${best.successB} vColl=${best.vesselCollisions} headOn=${best.headOnEvents} minCPA=${best.minCPA}`);
  }
  console.log("-> results_bintulu/trajectories/{multi_interaction,twoway_headon}.json");
}
main();
