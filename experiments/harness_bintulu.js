/* ============================================================
 * BPAN HARNESS — DQN variants on the Bintulu Port environment.
 * Mirrors the project's single-vessel protocol (train then greedy-eval),
 * logging the same real metrics. Self-contained; nothing fabricated.
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function loadModules() {
  const jsDir = path.join(__dirname, "..", "js");
  const sb = {};
  sb.window = sb; sb.globalThis = sb; sb.Math = Math; sb.Float64Array = Float64Array;
  sb.Array = Array; sb.console = console; sb.module = undefined;
  vm.createContext(sb);
  for (const f of ["dqn.js", "environmentBintulu.js"]) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  }
  return { BintuluPortEnv: sb.window.BintuluPortEnv, BintuluDQN: sb.window.BintuluDQN };
}

const { BintuluPortEnv, BintuluDQN } = loadModules();
const { BintuluEnv } = BintuluPortEnv;
const { DQNAgent } = BintuluDQN;
const ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"];

function mulberry32(seed) {
  let s = seed >>> 0;
  return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const realRandom = Math.random;
const setSeeded = (n) => { Math.random = mulberry32(n); };
const restore = () => { Math.random = realRandom; };

function runEpisode(env, agent, greedy) {
  let obs = env.reset();
  const lane = env.states[env.startWp].lane, goalLane = env.states[env.goalWp].lane;
  let steps = 0, invalid = 0, reachedGoal = false, terminated = false, truncated = false;
  while (true) {
    const a = agent.act(obs, greedy);
    const r = env.step(a);
    if (r.info.invalid) invalid++;
    steps++;
    if (!greedy && agent.observe) agent.observe(obs, a, r.reward, r.obs, r.terminated);
    obs = r.obs;
    if (r.terminated || r.truncated) { reachedGoal = r.info.reachedGoal; terminated = r.terminated; truncated = r.truncated; break; }
  }
  const solved = reachedGoal;
  return {
    reward: +env.totalReward.toFixed(4), success: solved ? 1 : 0, steps, invalid,
    route_cost: +env.routeCost.toFixed(4), optimal_cost: +env.optimalCost.toFixed(4),
    optimality_ratio: solved && env.optimalCost > 0 ? +(env.routeCost / env.optimalCost).toFixed(4) : "",
    collisions: env.collisionCount, collision_flag: env.collisionCount > 0 ? 1 : 0,
    cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(4),
    iala_violations: env.ialaViolations, dropped_frames: env.droppedFrames,
    docking_accuracy: solved && env.dockingAccuracy != null ? +env.dockingAccuracy.toFixed(4) : "",
    lane, goal_lane: goalLane, start: env.startWp, goal: env.goalWp,
    terminated: terminated ? 1 : 0, truncated: truncated ? 1 : 0,
  };
}

function runCell(cfg) {
  const { algorithm, seed, envSeed, trainEpisodes, evalEpisodes, totalStepsBudget,
          noiseStd = 0, packetErrorRate = 0, obstacleCount = 6 } = cfg;
  setSeeded(seed * 1000003 + 12345);
  const env = new BintuluEnv(envSeed, { noiseStd, packetErrorRate, obstacleCount });
  const agent = new DQNAgent(env.obsDim, env.nActions, { algorithm, per: false, noisy: false, totalSteps: totalStepsBudget });
  for (let ep = 0; ep < trainEpisodes; ep++) runEpisode(env, agent, false);
  const evalRows = [];
  for (let ep = 0; ep < evalEpisodes; ep++) evalRows.push(Object.assign({ phase: "eval", episode: ep + 1 }, runEpisode(env, agent, true)));
  restore();
  return { evalRows };
}

module.exports = { runCell, ALGOS };

if (require.main === module) {
  const t0 = Date.now();
  console.log("BPAN harness self-test (DQN, clean & harsh, seed 0):");
  for (const [c, ns, per] of [["clean", 0, 0], ["harsh", 0.25, 0.4]]) {
    const { evalRows } = runCell({ algorithm: "DQN", seed: 0, envSeed: 42, trainEpisodes: 150, evalEpisodes: 30, totalStepsBudget: 4500, noiseStd: ns, packetErrorRate: per });
    const succ = 100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
    const solved = evalRows.filter((r) => r.success === 1);
    const cteS = solved.length ? solved.reduce((a, r) => a + r.cte_mean, 0) / solved.length : 0;
    const coll = evalRows.reduce((a, r) => a + r.collisions, 0) / evalRows.length;
    console.log(`  ${c.padEnd(6)} succ=${succ.toFixed(0)}% cte|succ=${cteS.toFixed(1)} coll/ep=${coll.toFixed(2)} iala/ep=${(evalRows.reduce((a,r)=>a+r.iala_violations,0)/evalRows.length).toFixed(1)}`);
  }
  console.log(`  elapsed ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
