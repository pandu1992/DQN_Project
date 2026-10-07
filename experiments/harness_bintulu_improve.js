/* ============================================================
 * BPAN DQN-IMPROVEMENT HARNESS
 * Compares concrete DQN improvements on the Bintulu env:
 *   baseline        : BintuluEnv (no improvement)
 *   chart           : + Dijkstra charted-route prior (onPlan features)
 *   shaping         : + potential-based reward shaping toward goal
 *   chart_shaping    : both
 *   chart_long       : chart prior + a larger training budget
 * Logs the same real metrics; CTE reported on successful episodes downstream.
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function load() {
  const jsDir = path.join(__dirname, "..", "js");
  const sb = {};
  sb.window = sb; sb.globalThis = sb; sb.Math = Math; sb.Float64Array = Float64Array; sb.Array = Array; sb.console = console; sb.module = undefined;
  vm.createContext(sb);
  for (const f of ["dqn.js", "environmentBintulu.js", "environmentBintuluPlus.js"]) vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  return { Plus: sb.window.BintuluPortEnvPlus, DQN: sb.window.BintuluDQN };
}
const { Plus, DQN } = load();
const { BintuluEnvPlus } = Plus;
const { DQNAgent } = DQN;

function mulberry32(seed) { let s = seed >>> 0; return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const realRandom = Math.random;
const setSeeded = (n) => { Math.random = mulberry32(n); };
const restore = () => { Math.random = realRandom; };

const ARMS = {
  baseline:      { chartAware: false, shaping: false, train: 150, budget: 4500 },
  chart:         { chartAware: true,  shaping: false, train: 150, budget: 4500 },
  shaping:       { chartAware: false, shaping: true,  train: 150, budget: 4500 },
  chart_shaping: { chartAware: true,  shaping: true,  train: 150, budget: 4500 },
  chart_long:    { chartAware: true,  shaping: false, train: 300, budget: 9000 },
};

function runEpisode(env, agent, greedy) {
  let obs = env.reset();
  let steps = 0, reachedGoal = false;
  while (true) {
    const a = agent.act(obs, greedy);
    const r = env.step(a);
    steps++;
    if (!greedy && agent.observe) agent.observe(obs, a, r.reward, r.obs, r.terminated);
    obs = r.obs;
    if (r.terminated || r.truncated) { reachedGoal = r.info.reachedGoal; break; }
  }
  const solved = reachedGoal;
  return {
    reward: +env.totalReward.toFixed(4), success: solved ? 1 : 0, steps,
    optimality_ratio: solved && env.optimalCost > 0 ? +(env.routeCost / env.optimalCost).toFixed(4) : "",
    collisions: env.collisionCount, collision_flag: env.collisionCount > 0 ? 1 : 0,
    cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(4),
    iala_violations: env.ialaViolations, dropped_frames: env.droppedFrames,
    docking_accuracy: solved && env.dockingAccuracy != null ? +env.dockingAccuracy.toFixed(4) : "",
    start: env.startWp, goal: env.goalWp,
  };
}

function runCell(cfg) {
  const { arm, seed, envSeed, evalEpisodes, noiseStd = 0, packetErrorRate = 0, obstacleCount = 6 } = cfg;
  const spec = ARMS[arm];
  setSeeded(seed * 1000003 + 12345);
  const env = new BintuluEnvPlus(envSeed, { noiseStd, packetErrorRate, obstacleCount, chartAware: spec.chartAware, shaping: spec.shaping });
  const agent = new DQNAgent(env.obsDim, env.nActions, { algorithm: "DQN", per: false, noisy: false, totalSteps: spec.budget });
  for (let ep = 0; ep < spec.train; ep++) runEpisode(env, agent, false);
  const evalRows = [];
  for (let ep = 0; ep < evalEpisodes; ep++) evalRows.push(Object.assign({ phase: "eval", episode: ep + 1 }, runEpisode(env, agent, true)));
  restore();
  return { evalRows };
}

module.exports = { runCell, ARMS: Object.keys(ARMS) };

if (require.main === module) {
  const t0 = Date.now();
  console.log("BPAN improvement pilot (seed 0, clean & harsh):");
  for (const arm of Object.keys(ARMS)) {
    const line = [];
    for (const [c, ns, per] of [["clean", 0, 0], ["harsh", 0.25, 0.4]]) {
      const { evalRows } = runCell({ arm, seed: 0, envSeed: 42, evalEpisodes: 30, noiseStd: ns, packetErrorRate: per });
      const s = 100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
      line.push(`${c}=${s.toFixed(0)}%`);
    }
    console.log(`  ${arm.padEnd(14)} ${line.join("  ")}`);
  }
  console.log(`  elapsed ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
