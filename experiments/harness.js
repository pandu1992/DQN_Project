/* ============================================================
 * Q1 EXPERIMENT HARNESS  (Option 1 — real metrics only)
 *
 * Loads the ACTUAL browser modules environment.js + dqn.js UNCHANGED
 * by providing a minimal `window` shim, then runs, per configuration
 * and per seed:
 *
 *   1. a TRAINING phase (agent explores; ε-greedy or Noisy weights)
 *   2. a held-out GREEDY EVALUATION phase (no exploration) that
 *      measures the *converged* policy — this is what we analyze.
 *
 * Only metrics the real environment produces are logged:
 *   reward, success (reached goal within step budget), steps,
 *   invalid-action count, WAIT/FORWARD/BACKWARD counts, lane, mission.
 *
 * Nothing about noise / packet-error / collision / docking / CTE is
 * fabricated — the environment does not implement them.
 * ============================================================ */

"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

// ---- Load the real browser JS into a shared sandbox with a window shim ----
function loadProjectModules() {
  const jsDir = path.join(__dirname, "..", "js");
  const sandbox = {};
  sandbox.window = sandbox;              // window === global (browser-like)
  sandbox.globalThis = sandbox;
  sandbox.Math = Math;
  sandbox.Float64Array = Float64Array;
  sandbox.Array = Array;
  sandbox.console = console;
  sandbox.module = undefined;            // ensure stats.js browser branch is fine either way
  vm.createContext(sandbox);

  for (const f of ["environment.js", "dqn.js", "environmentV2.js"]) {
    const code = fs.readFileSync(path.join(jsDir, f), "utf8");
    vm.runInContext(code, sandbox, { filename: f });
  }
  return {
    BintuluEnv: sandbox.window.BintuluEnv,
    BintuluEnvV2: sandbox.window.BintuluEnvV2,
    BintuluDQN: sandbox.window.BintuluDQN,
    sandbox,
  };
}

const { BintuluEnv, BintuluEnvV2, BintuluDQN } = loadProjectModules();
const { VesselEnv } = BintuluEnv;
const { VesselEnvV2 } = BintuluEnvV2;
const { DQNAgent, ALGORITHMS, ALGO_LABELS } = BintuluDQN;

/* ------------------------------------------------------------
 * IMPORTANT reproducibility note about RNG:
 *   - VesselEnv uses a SEEDED mulberry32 stream for graph + mission
 *     selection (fully controlled by the integer seed).
 *   - DQNAgent / NoisyDense / ReplayBuffer use the GLOBAL Math.random
 *     (not seeded in the source). To make the *agent* side reproducible
 *     too, we override Math.random inside the shared sandbox with a
 *     seeded generator for the duration of each run. Because the modules
 *     were compiled against `sandbox.Math`, replacing sandbox.Math.random
 *     controls every Math.random() call inside environment.js + dqn.js.
 * ------------------------------------------------------------ */
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
function setSeededRandom(seedInt) {
  const gen = mulberry32(seedInt);
  Math.random = gen; // sandbox.Math === global Math here, so this affects the modules
}
function restoreRandom() {
  Math.random = realMathRandom;
}

/* ------------------------------------------------------------
 * Run one episode. If `agent` is provided and greedy=false, the
 * agent learns (observe). If greedy=true, it acts greedily and does
 * NOT learn — used for evaluation of the converged policy.
 * Returns per-episode real metrics.
 * ------------------------------------------------------------ */
function runEpisode(env, agent, greedy) {
  let obs = env.reset();
  const startWp = env.startWp;
  const lane = env.states[startWp].lane;
  const goalLane = env.states[env.goalWp].lane;
  let steps = 0;
  let invalid = 0;
  let lastInfo = null;
  let terminated = false, truncated = false, reachedGoal = false;

  while (true) {
    const a = agent.act(obs, greedy);
    const r = env.step(a);
    lastInfo = r.info;
    if (r.info.invalid) invalid++;
    steps++;
    if (!greedy) agent.observe(obs, a, r.reward, r.obs, r.terminated);
    obs = r.obs;
    if (r.terminated || r.truncated) {
      terminated = r.terminated;
      truncated = r.truncated;
      reachedGoal = r.info.reachedGoal;
      break;
    }
  }
  return {
    reward: env.totalReward,
    success: reachedGoal ? 1 : 0,
    steps,
    invalid,
    // real route-quality metrics (from existing edge attributes)
    route_cost: +env.routeCost.toFixed(4),
    optimal_cost: +env.optimalCost.toFixed(4),
    excess_cost: reachedGoal ? +(env.routeCost - env.optimalCost).toFixed(4) : "",
    optimality_ratio: reachedGoal && env.optimalCost > 0 ? +(env.routeCost / env.optimalCost).toFixed(4) : "",
    route_risk: +env.routeRisk.toFixed(4),
    route_difficulty: +env.routeDifficulty.toFixed(4),
    route_distance: +env.routeDistance.toFixed(2),
    revisit: env.revisitCount,
    lane,
    goal_lane: goalLane,
    start: startWp,
    goal: env.goalWp,
    terminated: terminated ? 1 : 0,
    truncated: truncated ? 1 : 0,
  };
}

/* ------------------------------------------------------------
 * Run a full (algorithm, per, noisy, seed) cell:
 *   - build env with env seed
 *   - build agent with matching cfg
 *   - TRAIN for trainEpisodes (logging training curve)
 *   - EVALUATE greedily for evalEpisodes on a fresh held-out env
 *     (same seed family, continued mission stream) — analyzed data
 * Returns { trainRows, evalRows }.
 * ------------------------------------------------------------ */
function runCell(cfg) {
  const {
    algorithm, per, noisy, seed,
    envSeed, trainEpisodes, evalEpisodes, totalStepsBudget,
  } = cfg;

  setSeededRandom(seed * 1000003 + 12345); // deterministic agent-side RNG per seed

  const env = new VesselEnvV2(envSeed);
  const agent = new DQNAgent(env.obsDim, env.nActions, {
    algorithm, per, noisy,
    totalSteps: totalStepsBudget,
  });

  const trainRows = [];
  for (let ep = 0; ep < trainEpisodes; ep++) {
    const m = runEpisode(env, agent, false);
    trainRows.push(Object.assign({ phase: "train", episode: ep + 1 }, m));
  }

  // Evaluation phase: greedy policy, continue the SAME seeded mission
  // stream (missionRng keeps producing missions), no learning.
  const evalRows = [];
  for (let ep = 0; ep < evalEpisodes; ep++) {
    const m = runEpisode(env, agent, true);
    evalRows.push(Object.assign({ phase: "eval", episode: ep + 1 }, m));
  }

  restoreRandom();
  return { trainRows, evalRows, finalEpsilon: agent.epsilon, trainSteps: agent.trainSteps, envSteps: agent.stepCount };
}

module.exports = { runCell, ALGORITHMS, ALGO_LABELS, VesselEnv, DQNAgent };

// ---- self-test when run directly ----
if (require.main === module) {
  console.log("Modules loaded. Algorithms:", ALGORITHMS.join(", "));
  const t0 = Date.now();
  const { trainRows, evalRows } = runCell({
    algorithm: "DQN", per: false, noisy: false, seed: 0,
    envSeed: 42, trainEpisodes: 150, evalEpisodes: 40, totalStepsBudget: 150 * 25,
  });
  const evalSucc = evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
  const evalRew = evalRows.reduce((a, r) => a + r.reward, 0) / evalRows.length;
  const lastTrain = trainRows.slice(-20);
  const trainSucc = lastTrain.reduce((a, r) => a + r.success, 0) / lastTrain.length;
  console.log(`self-test DQN seed0: trainEps=${trainRows.length} evalEps=${evalRows.length}`);
  console.log(`  last-20 train success=${(trainSucc * 100).toFixed(1)}%  eval success=${(evalSucc * 100).toFixed(1)}%  eval meanReward=${evalRew.toFixed(2)}`);
  console.log(`  elapsed ${(Date.now() - t0)}ms`);
}
