/* ============================================================
 * BASELINE-FAIRNESS HARNESS (Q1 audit Task 3)
 *
 * Disentangles "classical control is more robust" from "the classical
 * controller simply has chart information the learner lacks", by making chart
 * access an EXPLICIT experimental arm on VesselEnvV3Chart:
 *
 *   arm "DRL_no_chart"   : DQN, cfg.chartAware=false  (28-dim obs; current setup)
 *   arm "DRL_with_chart" : DQN, cfg.chartAware=true   (28 + 4 onPlan features)
 *   arm "RuleBased"      : COLREGs rule-based (already reads env.plannedPath)
 *
 * All three are run on the SAME env/seed/degradation grid as Study 3, and log
 * the same real metrics. If DRL_with_chart closes the gap to RuleBased, the
 * bottleneck is access to structured navigation priors (an architectural
 * insight), not learning-vs-classical control per se.
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function loadProjectModules() {
  const jsDir = path.join(__dirname, "..", "js");
  const sb = {};
  sb.window = sb; sb.globalThis = sb; sb.Math = Math; sb.Float64Array = Float64Array;
  sb.Array = Array; sb.console = console; sb.module = undefined;
  vm.createContext(sb);
  for (const f of ["environment.js", "dqn.js", "environmentV2.js", "environmentV3.js",
                   "environmentV3chart.js", "rulebased.js"]) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  }
  return {
    BintuluEnvV3Chart: sb.window.BintuluEnvV3Chart, BintuluDQN: sb.window.BintuluDQN,
    BintuluRuleBased: sb.window.BintuluRuleBased,
  };
}

const { BintuluEnvV3Chart, BintuluDQN, BintuluRuleBased } = loadProjectModules();
const { VesselEnvV3Chart } = BintuluEnvV3Chart;
const { DQNAgent } = BintuluDQN;
const { RuleBasedAgent } = BintuluRuleBased;

function mulberry32(seed) {
  let s = seed >>> 0;
  return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const realMathRandom = Math.random;
const setSeededRandom = (n) => { Math.random = mulberry32(n); };
const restoreRandom = () => { Math.random = realMathRandom; };

function runEpisode(env, agent, greedy) {
  let obs = env.reset();
  const lane = env.states[env.startWp].lane;
  const goalLane = env.states[env.goalWp].lane;
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

// arm in {DRL_no_chart, DRL_with_chart, RuleBased}
function runCell(cfg) {
  const { arm, seed, envSeed, trainEpisodes, evalEpisodes, totalStepsBudget,
          noiseStd = 0, packetErrorRate = 0, obstacleCount = 6 } = cfg;
  setSeededRandom(seed * 1000003 + 12345);
  const chartAware = arm === "DRL_with_chart";
  const env = new VesselEnvV3Chart(envSeed, { noiseStd, packetErrorRate, obstacleCount, chartAware });

  let agent;
  if (arm === "RuleBased") {
    agent = new RuleBasedAgent(env.obsDim, env.nActions, { env });
  } else {
    agent = new DQNAgent(env.obsDim, env.nActions, { algorithm: "DQN", per: false, noisy: false, totalSteps: totalStepsBudget });
    for (let ep = 0; ep < trainEpisodes; ep++) runEpisode(env, agent, false);
  }
  const evalRows = [];
  for (let ep = 0; ep < evalEpisodes; ep++) evalRows.push(Object.assign({ phase: "eval", episode: ep + 1 }, runEpisode(env, agent, true)));
  restoreRandom();
  return { evalRows };
}

module.exports = { runCell };

if (require.main === module) {
  const t0 = Date.now();
  console.log("Baseline-fairness harness self-test (clean & harsh, seed 0):");
  for (const arm of ["RuleBased", "DRL_no_chart", "DRL_with_chart"]) {
    for (const [c, ns, per] of [["clean", 0, 0], ["harsh", 0.25, 0.4]]) {
      const { evalRows } = runCell({ arm, seed: 0, envSeed: 42, trainEpisodes: arm === "RuleBased" ? 0 : 130, evalEpisodes: 30, totalStepsBudget: 3900, noiseStd: ns, packetErrorRate: per });
      const succ = 100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
      const solved = evalRows.filter((r) => r.success === 1);
      const cteS = solved.length ? solved.reduce((a, r) => a + r.cte_mean, 0) / solved.length : 0;
      console.log(`  ${arm.padEnd(15)} ${c.padEnd(6)} succ=${succ.toFixed(0)}% cte|success=${cteS.toFixed(1)} obsDim=${arm === "DRL_with_chart" ? 32 : 28}`);
    }
  }
  console.log(`  elapsed ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
