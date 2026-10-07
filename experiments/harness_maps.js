/* ============================================================
 * EXTERNAL-VALIDITY HARNESS (Q1 audit Task 4)
 *
 * Replicates the baseline-fairness arms (DRL_no_chart / DRL_with_chart /
 * RuleBased) across MULTIPLE synthetic maps (VesselEnvV3ChartMap, cfg.mapId),
 * to test whether the chart-prior / factor-hierarchy findings hold beyond the
 * single canonical graph. Same metrics as harness_chart.js, plus map_id.
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
                   "environmentV3chart.js", "mapgen.js", "environmentV3map.js", "rulebased.js"]) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  }
  return {
    BintuluEnvV3ChartMap: sb.window.BintuluEnvV3ChartMap, BintuluDQN: sb.window.BintuluDQN,
    BintuluRuleBased: sb.window.BintuluRuleBased, BintuluMapGen: sb.window.BintuluMapGen,
  };
}

const { BintuluEnvV3ChartMap, BintuluDQN, BintuluRuleBased, BintuluMapGen } = loadProjectModules();
const { VesselEnvV3ChartMap } = BintuluEnvV3ChartMap;
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
  };
}

function runCell(cfg) {
  const { arm, mapId, seed, envSeed, trainEpisodes, evalEpisodes, totalStepsBudget,
          noiseStd = 0, packetErrorRate = 0, obstacleCount = 6 } = cfg;
  setSeededRandom(seed * 1000003 + 12345);
  const chartAware = arm === "DRL_with_chart";
  const env = new VesselEnvV3ChartMap(envSeed, { noiseStd, packetErrorRate, obstacleCount, chartAware, mapId });

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

module.exports = { runCell, nMaps: BintuluMapGen.nMaps };

if (require.main === module) {
  const t0 = Date.now();
  console.log("External-validity harness self-test (maps 0,2,3; clean & harsh; seed 0):");
  for (const mapId of [0, 2, 3]) {
    for (const arm of ["DRL_no_chart", "DRL_with_chart", "RuleBased"]) {
      const parts = [];
      for (const [c, ns, per] of [["clean", 0, 0], ["harsh", 0.25, 0.4]]) {
        const { evalRows } = runCell({ arm, mapId, seed: 0, envSeed: 42, trainEpisodes: arm === "RuleBased" ? 0 : 130, evalEpisodes: 30, totalStepsBudget: 3900, noiseStd: ns, packetErrorRate: per });
        const succ = 100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
        parts.push(`${c}=${succ.toFixed(0)}%`);
      }
      console.log(`  map${mapId} ${arm.padEnd(15)} ${parts.join(" ")}`);
    }
  }
  console.log(`  elapsed ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
