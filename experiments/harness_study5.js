/* ============================================================
 * STUDY 5 HARNESS — mission-cycle (two-phase) + two-way traffic,
 * on the VesselEnvV3 physical/sensing/comms base, under Study-2 degradation.
 *
 * Two scenarios share this harness:
 *   scenario "twophase": SINGLE vessel, inbound -> dock -> outbound round trip
 *                        (TwoPhaseEnvV5). Success = FULL cycle completed.
 *   scenario "twoway":   TWO vessels in OPPOSING flow on a shared lane
 *                        (TwoWayEnvV5), inbound vs outbound, head-on encounters.
 *
 * Agents: the four RL variants (trained) or the COLREGs rule-based baseline.
 * For twoway we support agent pairings identical on both vessels (both learned
 * or both classical) plus the mixed learned-vs-classical pairing.
 *
 * Every logged number is genuinely computed by the environment; nothing is
 * fabricated. RL agents train `trainEpisodes` then greedy-evaluate.
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
  for (const f of ["environment.js", "dqn.js", "environmentV2.js", "environmentV3.js", "environmentV5.js", "rulebased.js"]) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  }
  return { BintuluEnvV5: sb.window.BintuluEnvV5, BintuluDQN: sb.window.BintuluDQN, BintuluRuleBased: sb.window.BintuluRuleBased };
}

const { BintuluEnvV5, BintuluDQN, BintuluRuleBased } = loadProjectModules();
const { TwoPhaseEnvV5, TwoWayEnvV5 } = BintuluEnvV5;
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

// ---- SCENARIO A: two-phase round trip (single vessel) ----
function makeAgentPhase(name, env, budget) {
  if (name === "RuleBased") return new RuleBasedAgent(env.v.obsDim, env.nActions, { env: env.v });
  return new DQNAgent(env.obsDim, env.nActions, { algorithm: name, per: false, noisy: false, totalSteps: budget });
}

function runPhaseEpisode(env, agent, greedy) {
  let obs = env.reset();
  let steps = 0, invalid = 0;
  while (true) {
    const a = agent.act(obs, greedy);
    const r = env.step(a);
    steps++; if (r.info.invalid) invalid++;
    if (!greedy && agent.observe) agent.observe(obs, a, r.reward, r.obs, r.terminated);
    obs = r.obs;
    if (r.terminated || r.truncated) break;
  }
  return {
    reward: +env.totalReward.toFixed(4),
    full_success: env.cycleComplete ? 1 : 0,
    dock_success: env.phase1Steps != null ? 1 : 0,
    final_phase: env.phase, steps, invalid,
    phase1_steps: env.phase1Steps != null ? env.phase1Steps : "",
    dock_accuracy: env.dockAccuracyLeg1 != null ? +env.dockAccuracyLeg1.toFixed(4) : "",
    collisions: env.collisionCount, collision_flag: env.collisionCount > 0 ? 1 : 0,
    cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(4),
    iala_violations: env.ialaViolations, dropped_frames: env.droppedFrames,
    lane: env.states[env.startWp].lane, start: env.startWp,
  };
}

// ---- SCENARIO B: two-way opposing traffic (two vessels) ----
function makeAgentTwoWay(name, subEnv, composedObsDim, budget) {
  if (name === "RuleBased") return new RuleBasedAgent(subEnv.obsDim, subEnv.nActions, { env: subEnv });
  return new DQNAgent(composedObsDim, subEnv.nActions, { algorithm: name, per: false, noisy: false, totalSteps: budget });
}

function runTwoWayEpisode(menv, agIn, agOut, greedy) {
  let [oIn, oOut] = menv.reset();
  const MX = Math.max(menv.IN.maxSteps, menv.OUT.maxSteps);
  let stIn = 0, stOut = 0;
  while (true) {
    const aI = agIn.act(oIn, greedy), aO = agOut.act(oOut, greedy);
    const r = menv.stepBoth(aI, aO);
    if (!menv.IN.done) stIn++;
    if (!menv.OUT.done) stOut++;
    if (!greedy) {
      if (agIn.observe) agIn.observe(oIn, aI, r.IN.reward, r.IN.obs, r.IN.terminated);
      if (agOut.observe) agOut.observe(oOut, aO, r.OUT.reward, r.OUT.obs, r.OUT.terminated);
    }
    oIn = r.IN.obs; oOut = r.OUT.obs;
    if (menv.done || menv.stepCount >= MX) break;
  }
  const rowFor = (env, agentName, dir, steps) => ({
    direction: dir, agent: agentName,
    reward: +env.totalReward.toFixed(4),
    success: env.currentWp === env.goalWp ? 1 : 0, steps,
    collisions: env.collisionCount,
    cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(4),
    iala_violations: env.ialaViolations, dropped_frames: env.droppedFrames,
    docking_accuracy: (env.currentWp === env.goalWp && env.dockingAccuracy != null) ? +env.dockingAccuracy.toFixed(4) : "",
    lane: menv.lane,
    vessel_collisions: menv.vesselCollisions, vessel_collision_flag: menv.vesselCollisions > 0 ? 1 : 0,
    collision_contact_steps: menv.collisionContactSteps, near_misses: menv.nearMisses,
    min_cpa: +(isFinite(menv.minCPA) ? menv.minCPA : 0).toFixed(4),
    encounter_steps: menv.encounterSteps, give_way_events: menv.giveWayEvents, head_on_events: menv.headOnEvents,
  });
  return { rowIn: rowFor(menv.IN, agIn.algorithm || "RuleBased", "inbound", stIn),
           rowOut: rowFor(menv.OUT, agOut.algorithm || "RuleBased", "outbound", stOut) };
}

function runCell(cfg) {
  const { scenario, agent, pairIn, pairOut, seed, envSeed, trainEpisodes, evalEpisodes,
          totalStepsBudget, noiseStd = 0, packetErrorRate = 0, obstacleCount = 6 } = cfg;
  setSeededRandom(seed * 1000003 + 12345);

  if (scenario === "twophase") {
    const env = new TwoPhaseEnvV5(envSeed, { noiseStd, packetErrorRate, obstacleCount });
    const ag = makeAgentPhase(agent, env, totalStepsBudget);
    if (agent !== "RuleBased") for (let ep = 0; ep < trainEpisodes; ep++) runPhaseEpisode(env, ag, false);
    const evalRows = [];
    for (let ep = 0; ep < evalEpisodes; ep++) evalRows.push(Object.assign({ phase: "eval", episode: ep + 1 }, runPhaseEpisode(env, ag, true)));
    restoreRandom();
    return { evalRows };
  }

  // twoway
  const menv = new TwoWayEnvV5(envSeed, { noiseStd, packetErrorRate, obstacleCount });
  const agIn = makeAgentTwoWay(pairIn, menv.IN, menv.obsDim, totalStepsBudget);
  const agOut = makeAgentTwoWay(pairOut, menv.OUT, menv.obsDim, totalStepsBudget);
  const anyRL = pairIn !== "RuleBased" || pairOut !== "RuleBased";
  if (anyRL) for (let ep = 0; ep < trainEpisodes; ep++) runTwoWayEpisode(menv, agIn, agOut, false);
  const evalRowsIn = [], evalRowsOut = [];
  for (let ep = 0; ep < evalEpisodes; ep++) {
    const { rowIn, rowOut } = runTwoWayEpisode(menv, agIn, agOut, true);
    evalRowsIn.push(Object.assign({ phase: "eval", episode: ep + 1 }, rowIn));
    evalRowsOut.push(Object.assign({ phase: "eval", episode: ep + 1 }, rowOut));
  }
  restoreRandom();
  return { evalRowsIn, evalRowsOut };
}

module.exports = { runCell, RL_ALGOS };

if (require.main === module) {
  const t0 = Date.now();
  console.log("Study 5 harness self-test:");
  console.log(" [A] two-phase round trip (RuleBased, DQN), clean:");
  for (const ag of ["RuleBased", "DQN"]) {
    const { evalRows } = runCell({ scenario: "twophase", agent: ag, seed: 0, envSeed: 42, trainEpisodes: ag === "RuleBased" ? 0 : 120, evalEpisodes: 20, totalStepsBudget: 6600, noiseStd: 0, packetErrorRate: 0 });
    const full = 100 * evalRows.reduce((a, r) => a + r.full_success, 0) / evalRows.length;
    const dock = 100 * evalRows.reduce((a, r) => a + r.dock_success, 0) / evalRows.length;
    console.log(`   ${ag.padEnd(10)} dock=${dock.toFixed(0)}% fullCycle=${full.toFixed(0)}%`);
  }
  console.log(" [B] two-way traffic (RuleBased x2, DQN x2), clean vs harsh:");
  for (const [pi, po] of [["RuleBased", "RuleBased"], ["DQN", "DQN"]]) {
    for (const [c, ns, per] of [["clean", 0, 0], ["harsh", 0.25, 0.4]]) {
      const { evalRowsIn, evalRowsOut } = runCell({ scenario: "twoway", pairIn: pi, pairOut: po, seed: 0, envSeed: 42, trainEpisodes: pi === "RuleBased" && po === "RuleBased" ? 0 : 80, evalEpisodes: 20, totalStepsBudget: 2400, noiseStd: ns, packetErrorRate: per });
      const vc = evalRowsIn.reduce((a, r) => a + r.vessel_collisions, 0) / evalRowsIn.length;
      const ho = evalRowsIn.reduce((a, r) => a + r.head_on_events, 0) / evalRowsIn.length;
      const cpa = evalRowsIn.reduce((a, r) => a + r.min_cpa, 0) / evalRowsIn.length;
      const sI = 100 * evalRowsIn.reduce((a, r) => a + r.success, 0) / evalRowsIn.length;
      const sO = 100 * evalRowsOut.reduce((a, r) => a + r.success, 0) / evalRowsOut.length;
      console.log(`   ${(pi + "/" + po).padEnd(18)} ${c.padEnd(6)} vColl/ep=${vc.toFixed(2)} headOn/ep=${ho.toFixed(1)} minCPA=${cpa.toFixed(0)} succIn=${sI.toFixed(0)}% succOut=${sO.toFixed(0)}%`);
    }
  }
  console.log(`  elapsed ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
