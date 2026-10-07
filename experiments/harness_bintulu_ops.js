/* BPAN extension (e) — operational realism on the Bintulu chart.
 * twophase: single vessel inbound->dock->outbound (success=full cycle).
 * twoway:   inbound + outbound vessels sharing a lane (head-on). */
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
function load() {
  const jsDir = path.join(__dirname, "..", "js"), sb = {};
  sb.window = sb; sb.globalThis = sb; sb.Math = Math; sb.Float64Array = Float64Array; sb.Array = Array; sb.console = console; sb.module = undefined;
  vm.createContext(sb);
  for (const f of ["dqn.js", "rulebased.js", "environmentBintulu.js", "environmentBintuluOps.js"]) vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  return { Ops: sb.window.BintuluOps, DQN: sb.window.BintuluDQN, RB: sb.window.BintuluRuleBased };
}
const { Ops, DQN, RB } = load();
const { TwoPhaseBintulu, TwoWayBintulu } = Ops;
const { DQNAgent } = DQN; const { RuleBasedAgent } = RB;
function mulberry32(seed) { let s = seed >>> 0; return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const realR = Math.random, setSeeded = (n) => { Math.random = mulberry32(n); }, restore = () => { Math.random = realR; };

// ---- twophase ----
function makeAgentPhase(name, env, budget) {
  if (name === "RuleBased") return new RuleBasedAgent(env.v.obsDim, env.nActions, { env: env.v });
  return new DQNAgent(env.obsDim, env.nActions, { algorithm: name, per: false, noisy: false, totalSteps: budget });
}
function runPhaseEp(env, agent, greedy) {
  let obs = env.reset();
  while (true) { const a = agent.act(obs, greedy); const r = env.step(a); if (!greedy && agent.observe) agent.observe(obs, a, r.reward, r.obs, r.terminated); obs = r.obs; if (r.terminated || r.truncated) break; }
  return { full_success: env.cycleComplete ? 1 : 0, dock_success: env.phase1Steps != null ? 1 : 0, reward: +env.totalReward.toFixed(2),
    dock_accuracy: env.dockAccuracyLeg1 != null ? +env.dockAccuracyLeg1.toFixed(2) : "", collisions: env.collisionCount,
    cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(3), iala_violations: env.ialaViolations };
}
// ---- twoway ----
function makeAgentWay(name, sub, obsDim, budget) {
  if (name === "RuleBased") return new RuleBasedAgent(sub.obsDim, sub.nActions, { env: sub });
  return new DQNAgent(obsDim, sub.nActions, { algorithm: name, per: false, noisy: false, totalSteps: budget });
}
function runWayEp(menv, agIn, agOut, greedy) {
  let [oi, oo] = menv.reset();
  const MX = Math.max(menv.IN.maxSteps, menv.OUT.maxSteps);
  while (true) { const ai = agIn.act(oi, greedy), ao = agOut.act(oo, greedy); const r = menv.stepBoth(ai, ao);
    if (!greedy) { if (agIn.observe) agIn.observe(oi, ai, r.IN.reward, r.IN.obs, r.IN.terminated); if (agOut.observe) agOut.observe(oo, ao, r.OUT.reward, r.OUT.obs, r.OUT.terminated); }
    oi = r.IN.obs; oo = r.OUT.obs; if (menv.done || menv.stepCount >= MX) break; }
  const row = (env, nm, dir) => ({ direction: dir, agent: nm, success: env.currentWp === env.goalWp ? 1 : 0,
    vessel_collisions: menv.vesselCollisions, vessel_collision_flag: menv.vesselCollisions > 0 ? 1 : 0, near_misses: menv.nearMisses,
    min_cpa: +(isFinite(menv.minCPA) ? menv.minCPA : 0).toFixed(2), head_on_events: menv.headOnEvents, iala_violations: env.ialaViolations });
  return { rowIn: row(menv.IN, agIn.algorithm || "RuleBased", "inbound"), rowOut: row(menv.OUT, agOut.algorithm || "RuleBased", "outbound") };
}
function runCell(cfg) {
  const { scenario, agent, pairIn, pairOut, seed, envSeed, trainEpisodes, evalEpisodes, budget, noiseStd = 0, packetErrorRate = 0 } = cfg;
  setSeeded(seed * 1000003 + 12345);
  if (scenario === "twophase") {
    const env = new TwoPhaseBintulu(envSeed, { noiseStd, packetErrorRate, obstacleCount: 6 });
    const ag = makeAgentPhase(agent, env, budget);
    if (agent !== "RuleBased") for (let ep = 0; ep < trainEpisodes; ep++) runPhaseEp(env, ag, false);
    const rows = []; for (let ep = 0; ep < evalEpisodes; ep++) rows.push(Object.assign({ phase: "eval", episode: ep + 1 }, runPhaseEp(env, ag, true)));
    restore(); return { rows };
  }
  const menv = new TwoWayBintulu(envSeed, { noiseStd, packetErrorRate, obstacleCount: 6 });
  const agIn = makeAgentWay(pairIn, menv.IN, menv.obsDim, budget), agOut = makeAgentWay(pairOut, menv.OUT, menv.obsDim, budget);
  const anyRL = pairIn !== "RuleBased" || pairOut !== "RuleBased";
  if (anyRL) for (let ep = 0; ep < trainEpisodes; ep++) runWayEp(menv, agIn, agOut, false);
  const rIn = [], rOut = [];
  for (let ep = 0; ep < evalEpisodes; ep++) { const { rowIn, rowOut } = runWayEp(menv, agIn, agOut, true); rIn.push(Object.assign({ phase: "eval", episode: ep + 1 }, rowIn)); rOut.push(Object.assign({ phase: "eval", episode: ep + 1 }, rowOut)); }
  restore(); return { rIn, rOut };
}
module.exports = { runCell };
if (require.main === module) {
  console.log("ops pilot:");
  for (const ag of ["RuleBased", "DQN"]) { const { rows } = runCell({ scenario: "twophase", agent: ag, seed: 0, envSeed: 42, trainEpisodes: ag === "RuleBased" ? 0 : 120, evalEpisodes: 20, budget: 6600 }); console.log(`  twophase ${ag}: dock=${(100 * rows.reduce((a, r) => a + r.dock_success, 0) / rows.length).toFixed(0)}% full=${(100 * rows.reduce((a, r) => a + r.full_success, 0) / rows.length).toFixed(0)}%`); }
  for (const [pi, po] of [["RuleBased", "RuleBased"], ["DQN", "RuleBased"]]) { const { rIn, rOut } = runCell({ scenario: "twoway", pairIn: pi, pairOut: po, seed: 0, envSeed: 42, trainEpisodes: pi === "RuleBased" && po === "RuleBased" ? 0 : 100, evalEpisodes: 20, budget: 3000, noiseStd: 0.25, packetErrorRate: 0.4 }); console.log(`  twoway ${pi}/${po} harsh: headOn/ep=${(rIn.reduce((a, r) => a + r.head_on_events, 0) / rIn.length).toFixed(1)} vColl/ep=${(rIn.reduce((a, r) => a + r.vessel_collisions, 0) / rIn.length).toFixed(2)} succIn=${(100 * rIn.reduce((a, r) => a + r.success, 0) / rIn.length).toFixed(0)}%`); }
}
