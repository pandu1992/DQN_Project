/* ============================================================
 * V3 EXPERIMENT HARNESS — physical / sensing / comms layer
 *
 * Same design as harness.js (Option 1) but runs VesselEnvV3, which
 * adds genuinely-implemented mechanisms so the following metrics are
 * REAL (measured, never fabricated):
 *   collision_rate, docking_accuracy, cte_mean, iala_violations,
 *   dropped_frames  — in addition to success / reward / steps / route.
 *
 * New scenario factors: noiseStd (sensor/observation noise) and
 * packetErrorRate (communication packet loss). The V2 harness
 * (harness.js) is left completely untouched.
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function loadProjectModules() {
  const jsDir = path.join(__dirname, "..", "js");
  const sandbox = {};
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.Math = Math;
  sandbox.Float64Array = Float64Array;
  sandbox.Array = Array;
  sandbox.console = console;
  sandbox.module = undefined;
  vm.createContext(sandbox);
  for (const f of ["environment.js", "dqn.js", "environmentV2.js", "environmentV3.js"]) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sandbox, { filename: f });
  }
  return {
    BintuluEnvV3: sandbox.window.BintuluEnvV3,
    BintuluDQN: sandbox.window.BintuluDQN,
  };
}

const { BintuluEnvV3, BintuluDQN } = loadProjectModules();
const { VesselEnvV3 } = BintuluEnvV3;
const { DQNAgent, ALGORITHMS, ALGO_LABELS } = BintuluDQN;

function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const realMathRandom = Math.random;
function setSeededRandom(seedInt) { Math.random = mulberry32(seedInt); }
function restoreRandom() { Math.random = realMathRandom; }

function runEpisode(env, agent, greedy) {
  let obs = env.reset();
  const startWp = env.startWp;
  const lane = env.states[startWp].lane;
  const goalLane = env.states[env.goalWp].lane;
  let steps = 0, invalid = 0;
  let terminated = false, truncated = false, reachedGoal = false, info = null;

  while (true) {
    const a = agent.act(obs, greedy);
    const r = env.step(a);
    info = r.info;
    if (r.info.invalid) invalid++;
    steps++;
    if (!greedy) agent.observe(obs, a, r.reward, r.obs, r.terminated);
    obs = r.obs;
    if (r.terminated || r.truncated) {
      terminated = r.terminated; truncated = r.truncated; reachedGoal = r.info.reachedGoal;
      break;
    }
  }
  return {
    reward: +env.totalReward.toFixed(4),
    success: reachedGoal ? 1 : 0,
    steps,
    invalid,
    // routing metrics (as V2)
    route_cost: +env.routeCost.toFixed(4),
    optimal_cost: +env.optimalCost.toFixed(4),
    excess_cost: reachedGoal ? +(env.routeCost - env.optimalCost).toFixed(4) : "",
    optimality_ratio: reachedGoal && env.optimalCost > 0 ? +(env.routeCost / env.optimalCost).toFixed(4) : "",
    route_risk: +env.routeRisk.toFixed(4),
    route_difficulty: +env.routeDifficulty.toFixed(4),
    route_distance: +env.routeDistance.toFixed(2),
    revisit: env.revisitCount,
    // NEW physical / sensing / comms metrics (all real)
    collisions: env.collisionCount,
    collision_flag: env.collisionCount > 0 ? 1 : 0,
    cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(4),
    iala_violations: env.ialaViolations,
    dropped_frames: env.droppedFrames,
    docking_accuracy: reachedGoal && env.dockingAccuracy != null ? +env.dockingAccuracy.toFixed(4) : "",
    lane, goal_lane: goalLane, start: startWp, goal: env.goalWp,
    terminated: terminated ? 1 : 0, truncated: truncated ? 1 : 0,
  };
}

/* Run one (algorithm, per, noisy, seed, noiseStd, packetErrorRate) cell. */
function runCell(cfg) {
  const {
    algorithm, per, noisy, seed, envSeed,
    trainEpisodes, evalEpisodes, totalStepsBudget,
    noiseStd = 0, packetErrorRate = 0, obstacleCount = 6,
  } = cfg;

  setSeededRandom(seed * 1000003 + 12345);

  const envCfg = { noiseStd, packetErrorRate, obstacleCount };
  const env = new VesselEnvV3(envSeed, envCfg);
  const agent = new DQNAgent(env.obsDim, env.nActions, {
    algorithm, per, noisy, totalSteps: totalStepsBudget,
  });

  const trainRows = [];
  for (let ep = 0; ep < trainEpisodes; ep++) {
    trainRows.push(Object.assign({ phase: "train", episode: ep + 1 }, runEpisode(env, agent, false)));
  }
  const evalRows = [];
  for (let ep = 0; ep < evalEpisodes; ep++) {
    evalRows.push(Object.assign({ phase: "eval", episode: ep + 1 }, runEpisode(env, agent, true)));
  }

  restoreRandom();
  return { trainRows, evalRows };
}

module.exports = { runCell, ALGORITHMS, ALGO_LABELS };

// ---- self-test ----
if (require.main === module) {
  const t0 = Date.now();
  console.log("V3 harness. Algorithms:", ALGORITHMS.join(", "));
  for (const [ns, per] of [[0, 0], [0.15, 0.2]]) {
    const { evalRows } = runCell({
      algorithm: "DQN", per: false, noisy: false, seed: 0, envSeed: 42,
      trainEpisodes: 120, evalEpisodes: 40, totalStepsBudget: 120 * 30,
      noiseStd: ns, packetErrorRate: per,
    });
    const mean = (k) => {
      const v = evalRows.map((r) => r[k]).filter((x) => x !== "" && isFinite(x));
      return v.reduce((a, b) => a + b, 0) / (v.length || 1);
    };
    const succ = 100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
    console.log(`  noise=${ns} PER=${per}: succ=${succ.toFixed(1)}% reward=${mean("reward").toFixed(1)} ` +
      `coll/ep=${mean("collisions").toFixed(2)} cte=${mean("cte_mean").toFixed(2)} ` +
      `iala/ep=${mean("iala_violations").toFixed(2)} dock=${mean("docking_accuracy").toFixed(2)} ` +
      `drop/ep=${mean("dropped_frames").toFixed(1)}`);
  }
  console.log(`  elapsed ${Date.now() - t0}ms`);
}
