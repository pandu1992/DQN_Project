/* ============================================================
 * STUDY 4 HARNESS — two INDEPENDENT vessels that must avoid each other
 * on MultiVesselEnvV4, layered on the Study-2 sensing/comms degradation.
 *
 * Two independent agents (vessel A, vessel B) step in LOCKSTEP through the
 * shared world. Each agent is its own instance with its own policy; there is
 * no central controller. Agents may be RL variants (trained) or the COLREGs
 * rule-based baseline (deterministic). We support MIXED pairings, e.g.
 * DQN vs RuleBased, to ask "does a learned policy avoid a moving partner as
 * well as a classical give-way controller?".
 *
 * Training for an RL agent is done AGAINST THE LIVE PARTNER (self-play style):
 * both vessels act every macro-step, and the RL agent observes its own
 * transitions including the inter-vessel reward shaping. A RuleBased partner
 * is deterministic and never learns. Evaluation is greedy/deterministic.
 *
 * Every logged number is genuinely computed by MultiVesselEnvV4 /
 * VesselEnvV3 — inter-vessel collisions, near-misses, closest-point-of-
 * approach, plus each vessel's own static-obstacle / CTE / IALA / docking /
 * dropped-frame metrics. Nothing is fabricated.
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
  for (const f of ["environment.js", "dqn.js", "environmentV2.js", "environmentV3.js", "environmentV4.js", "rulebased.js"]) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  }
  return {
    BintuluEnvV4: sb.window.BintuluEnvV4, BintuluDQN: sb.window.BintuluDQN,
    BintuluRuleBased: sb.window.BintuluRuleBased,
  };
}

const { BintuluEnvV4, BintuluDQN, BintuluRuleBased } = loadProjectModules();
const { MultiVesselEnvV4 } = BintuluEnvV4;
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

// Build an agent for a given sub-env. RuleBased needs the sub-env reference;
// RL agents are sized to the COMPOSED obsDim (own obs + partner channel).
function makeAgent(name, subEnv, composedObsDim, totalStepsBudget) {
  if (name === "RuleBased") {
    return new RuleBasedAgent(subEnv.obsDim, subEnv.nActions, { env: subEnv });
  }
  return new DQNAgent(composedObsDim, subEnv.nActions, {
    algorithm: name, per: false, noisy: false, totalSteps: totalStepsBudget,
  });
}

// One two-vessel episode in lockstep. Returns per-vessel summary rows +
// shared inter-vessel metrics. `greedy` disables exploration & learning.
function runEpisode(menv, agentA, agentB, greedy) {
  let [obsA, obsB] = menv.reset();
  const startA = menv.A.startWp, goalA = menv.A.goalWp;
  const startB = menv.B.startWp, goalB = menv.B.goalWp;
  const laneA = menv.A.states[startA].lane, goalLaneA = menv.A.states[goalA].lane;
  const laneB = menv.B.states[startB].lane, goalLaneB = menv.B.states[goalB].lane;
  let stepsA = 0, stepsB = 0, invalidA = 0, invalidB = 0;
  const MAX = Math.max(menv.A.maxSteps, menv.B.maxSteps);

  while (true) {
    const aA = agentA.act(obsA, greedy);
    const aB = agentB.act(obsB, greedy);
    const r = menv.stepBoth(aA, aB);
    if (!menv.A.done) { stepsA++; if (r.A.info.invalid) invalidA++; }
    if (!menv.B.done) { stepsB++; if (r.B.info.invalid) invalidB++; }
    if (!greedy) {
      if (agentA.observe) agentA.observe(obsA, aA, r.A.reward, r.A.obs, r.A.terminated);
      if (agentB.observe) agentB.observe(obsB, aB, r.B.reward, r.B.obs, r.B.terminated);
    }
    obsA = r.A.obs; obsB = r.B.obs;
    if (menv.done || menv.stepCount >= MAX) break;
  }

  const A = menv.A, B = menv.B;
  const rowFor = (env, agentName, steps, invalid, lane, goalLane, start, goal) => ({
    agent: agentName,
    reward: +env.totalReward.toFixed(4),
    success: env.currentWp === env.goalWp ? 1 : 0,
    steps, invalid,
    route_cost: +env.routeCost.toFixed(4), optimal_cost: +env.optimalCost.toFixed(4),
    optimality_ratio: (env.currentWp === env.goalWp && env.optimalCost > 0) ? +(env.routeCost / env.optimalCost).toFixed(4) : "",
    static_collisions: env.collisionCount,
    cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(4),
    iala_violations: env.ialaViolations, dropped_frames: env.droppedFrames,
    docking_accuracy: (env.currentWp === env.goalWp && env.dockingAccuracy != null) ? +env.dockingAccuracy.toFixed(4) : "",
    lane, goal_lane: goalLane, start, goal,
    // shared inter-vessel metrics duplicated onto each vessel row for convenience
    vessel_collisions: menv.vesselCollisions,
    vessel_collision_flag: menv.vesselCollisions > 0 ? 1 : 0,
    collision_contact_steps: menv.collisionContactSteps,
    near_misses: menv.nearMisses,
    min_cpa: +(isFinite(menv.minCPA) ? menv.minCPA : 0).toFixed(4),
    encounter_steps: menv.encounterSteps,
    give_way_events: menv.giveWayEvents,
  });

  return {
    rowA: rowFor(A, agentA.algorithm || "RuleBased", stepsA, invalidA, laneA, goalLaneA, startA, goalA),
    rowB: rowFor(B, agentB.algorithm || "RuleBased", stepsB, invalidB, laneB, goalLaneB, startB, goalB),
  };
}

// Run one (pairing, condition, seed) cell. pairing = { A: name, B: name }.
function runCell(cfg) {
  const { pairA, pairB, seed, envSeed, trainEpisodes, evalEpisodes, totalStepsBudget,
          noiseStd = 0, packetErrorRate = 0, obstacleCount = 6 } = cfg;
  setSeededRandom(seed * 1000003 + 12345);
  const menv = new MultiVesselEnvV4(envSeed, { noiseStd, packetErrorRate, obstacleCount });

  const agentA = makeAgent(pairA, menv.A, menv.obsDim, totalStepsBudget);
  const agentB = makeAgent(pairB, menv.B, menv.obsDim, totalStepsBudget);

  // training: both agents act live; RL agents learn against the live partner.
  const anyRL = pairA !== "RuleBased" || pairB !== "RuleBased";
  if (anyRL) {
    for (let ep = 0; ep < trainEpisodes; ep++) runEpisode(menv, agentA, agentB, false);
  }

  const evalRowsA = [], evalRowsB = [];
  for (let ep = 0; ep < evalEpisodes; ep++) {
    const { rowA, rowB } = runEpisode(menv, agentA, agentB, true);
    evalRowsA.push(Object.assign({ phase: "eval", episode: ep + 1, vessel: "A" }, rowA));
    evalRowsB.push(Object.assign({ phase: "eval", episode: ep + 1, vessel: "B" }, rowB));
  }
  restoreRandom();
  return { evalRowsA, evalRowsB };
}

module.exports = { runCell, RL_ALGOS };

if (require.main === module) {
  const t0 = Date.now();
  console.log("Study 4 harness self-test (two vessels, clean vs harsh, seed 0):");
  const pairings = [["DQN", "DQN"], ["DQN", "RuleBased"], ["RuleBased", "RuleBased"]];
  for (const [pa, pb] of pairings) {
    for (const [cname, ns, per] of [["clean", 0, 0], ["harsh", 0.25, 0.4]]) {
      const { evalRowsA, evalRowsB } = runCell({
        pairA: pa, pairB: pb, seed: 0, envSeed: 42, trainEpisodes: 60, evalEpisodes: 20,
        totalStepsBudget: 2400, noiseStd: ns, packetErrorRate: per,
      });
      const all = evalRowsA.concat(evalRowsB);
      const mean = (rows, k) => { const v = rows.map(r => r[k]).filter(x => x !== "" && isFinite(x)); return v.reduce((a, b) => a + b, 0) / (v.length || 1); };
      const succA = 100 * evalRowsA.reduce((a, r) => a + r.success, 0) / evalRowsA.length;
      const succB = 100 * evalRowsB.reduce((a, r) => a + r.success, 0) / evalRowsB.length;
      const vcoll = mean(evalRowsA, "vessel_collisions"); // shared, same on A & B
      const nmiss = mean(evalRowsA, "near_misses");
      const cpa = mean(evalRowsA, "min_cpa");
      console.log(`  ${(pa + " vs " + pb).padEnd(22)} ${cname.padEnd(6)} succA=${succA.toFixed(0)}% succB=${succB.toFixed(0)}% vColl/ep=${vcoll.toFixed(2)} nearMiss/ep=${nmiss.toFixed(2)} minCPA=${cpa.toFixed(1)}`);
    }
  }
  console.log(`  elapsed ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
