/* ============================================================
 * STUDY 3 HARNESS — rule-based baseline vs DQN on VesselEnvV3
 *
 * Handles TWO agent types identically through the same episode loop:
 *   - RL variants (DQN/DoubleDQN/DuelingDQN/DuelingDoubleDQN): train
 *     `trainEpisodes` then greedy-evaluate `evalEpisodes`.
 *   - "RuleBased": no training; evaluated directly (deterministic).
 *
 * Logs the same real physical/sensing/comms metrics as the Study 2
 * (V3) harness. harness_v3.js is left untouched.
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function loadProjectModules() {
  const jsDir = path.join(__dirname, "..", "js");
  const sb = {};
  sb.window = sb; sb.globalThis = sb; sb.Math = Math; sb.Float64Array = Float64Array; sb.Array = Array; sb.console = console; sb.module = undefined;
  vm.createContext(sb);
  for (const f of ["environment.js", "dqn.js", "environmentV2.js", "environmentV3.js", "rulebased.js"]) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  }
  return { BintuluEnvV3: sb.window.BintuluEnvV3, BintuluDQN: sb.window.BintuluDQN, BintuluRuleBased: sb.window.BintuluRuleBased };
}

const { BintuluEnvV3, BintuluDQN, BintuluRuleBased } = loadProjectModules();
const { VesselEnvV3 } = BintuluEnvV3;
const { DQNAgent } = BintuluDQN;
const { RuleBasedAgent } = BintuluRuleBased;
const RL_ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"];

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
  let steps = 0, invalid = 0, terminated = false, truncated = false, reachedGoal = false;
  while (true) {
    const a = agent.act(obs, greedy);
    const r = env.step(a);
    if (r.info.invalid) invalid++;
    steps++;
    if (!greedy && agent.observe) agent.observe(obs, a, r.reward, r.obs, r.terminated);
    obs = r.obs;
    if (r.terminated || r.truncated) { terminated = r.terminated; truncated = r.truncated; reachedGoal = r.info.reachedGoal; break; }
  }
  return {
    reward: +env.totalReward.toFixed(4), success: reachedGoal ? 1 : 0, steps, invalid,
    route_cost: +env.routeCost.toFixed(4), optimal_cost: +env.optimalCost.toFixed(4),
    excess_cost: reachedGoal ? +(env.routeCost - env.optimalCost).toFixed(4) : "",
    optimality_ratio: reachedGoal && env.optimalCost > 0 ? +(env.routeCost / env.optimalCost).toFixed(4) : "",
    route_risk: +env.routeRisk.toFixed(4), route_difficulty: +env.routeDifficulty.toFixed(4),
    route_distance: +env.routeDistance.toFixed(2), revisit: env.revisitCount,
    collisions: env.collisionCount, collision_flag: env.collisionCount > 0 ? 1 : 0,
    cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(4),
    iala_violations: env.ialaViolations, dropped_frames: env.droppedFrames,
    docking_accuracy: reachedGoal && env.dockingAccuracy != null ? +env.dockingAccuracy.toFixed(4) : "",
    lane, goal_lane: goalLane, start: env.startWp, goal: env.goalWp,
    terminated: terminated ? 1 : 0, truncated: truncated ? 1 : 0,
  };
}

function runCell(cfg) {
  const { agent: agentName, seed, envSeed, trainEpisodes, evalEpisodes, totalStepsBudget,
          noiseStd = 0, packetErrorRate = 0, obstacleCount = 6 } = cfg;
  setSeededRandom(seed * 1000003 + 12345);
  const env = new VesselEnvV3(envSeed, { noiseStd, packetErrorRate, obstacleCount });

  let agent, trainRows = [];
  if (agentName === "RuleBased") {
    agent = new RuleBasedAgent(env.obsDim, env.nActions, { env });
    // no training phase
  } else {
    agent = new DQNAgent(env.obsDim, env.nActions, { algorithm: agentName, per: false, noisy: false, totalSteps: totalStepsBudget });
    for (let ep = 0; ep < trainEpisodes; ep++) {
      trainRows.push(Object.assign({ phase: "train", episode: ep + 1 }, runEpisode(env, agent, false)));
    }
  }
  const evalRows = [];
  for (let ep = 0; ep < evalEpisodes; ep++) {
    evalRows.push(Object.assign({ phase: "eval", episode: ep + 1 }, runEpisode(env, agent, true)));
  }
  restoreRandom();
  return { trainRows, evalRows };
}

module.exports = { runCell, RL_ALGOS };

if (require.main === module) {
  const t0 = Date.now();
  console.log("Study 3 harness self-test (clean condition, seed 0):");
  for (const ag of ["RuleBased", "DQN"]) {
    const { evalRows } = runCell({ agent: ag, seed: 0, envSeed: 42, trainEpisodes: 100, evalEpisodes: 30, totalStepsBudget: 3000, noiseStd: 0, packetErrorRate: 0 });
    const mean = (k) => { const v = evalRows.map(r => r[k]).filter(x => x !== "" && isFinite(x)); return v.reduce((a, b) => a + b, 0) / (v.length || 1); };
    const succ = 100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
    console.log(`  ${ag.padEnd(10)} succ=${succ.toFixed(0)}% reward=${mean("reward").toFixed(1)} coll/ep=${mean("collisions").toFixed(2)} cte=${mean("cte_mean").toFixed(1)}`);
  }
  console.log(`  elapsed ${Date.now() - t0}ms`);
}
