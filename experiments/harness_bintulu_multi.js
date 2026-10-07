/* BPAN extension (d) — two independent vessels on the Bintulu chart.
 * Pairings: DQN_vs_DQN, DQN_vs_RuleBased, RuleBased_vs_RuleBased.
 * Lockstep two-agent loop; logs inter-vessel + per-vessel metrics. */
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
function load() {
  const jsDir = path.join(__dirname, "..", "js"), sb = {};
  sb.window = sb; sb.globalThis = sb; sb.Math = Math; sb.Float64Array = Float64Array; sb.Array = Array; sb.console = console; sb.module = undefined;
  vm.createContext(sb);
  for (const f of ["dqn.js", "rulebased.js", "environmentBintulu.js", "environmentBintuluMulti.js"]) vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  return { Multi: sb.window.BintuluMulti, DQN: sb.window.BintuluDQN, RB: sb.window.BintuluRuleBased };
}
const { Multi, DQN, RB } = load();
const { MultiVesselBintulu } = Multi;
const { DQNAgent } = DQN;
const { RuleBasedAgent } = RB;
function mulberry32(seed) { let s = seed >>> 0; return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const realR = Math.random, setSeeded = (n) => { Math.random = mulberry32(n); }, restore = () => { Math.random = realR; };

function makeAgent(name, sub, obsDim, budget) {
  if (name === "RuleBased") return new RuleBasedAgent(sub.obsDim, sub.nActions, { env: sub });
  return new DQNAgent(obsDim, sub.nActions, { algorithm: name, per: false, noisy: false, totalSteps: budget });
}
function runEpisode(menv, agA, agB, greedy) {
  let [oA, oB] = menv.reset();
  const MAX = Math.max(menv.A.maxSteps, menv.B.maxSteps);
  while (true) {
    const aA = agA.act(oA, greedy), aB = agB.act(oB, greedy);
    const r = menv.stepBoth(aA, aB);
    if (!greedy) { if (agA.observe) agA.observe(oA, aA, r.A.reward, r.A.obs, r.A.terminated); if (agB.observe) agB.observe(oB, aB, r.B.reward, r.B.obs, r.B.terminated); }
    oA = r.A.obs; oB = r.B.obs;
    if (menv.done || menv.stepCount >= MAX) break;
  }
  const A = menv.A, B = menv.B;
  const row = (env, nm) => ({ agent: nm, success: env.currentWp === env.goalWp ? 1 : 0, reward: +env.totalReward.toFixed(2),
    static_collisions: env.collisionCount, cte_mean: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(3), iala_violations: env.ialaViolations,
    vessel_collisions: menv.vesselCollisions, vessel_collision_flag: menv.vesselCollisions > 0 ? 1 : 0, near_misses: menv.nearMisses,
    min_cpa: +(isFinite(menv.minCPA) ? menv.minCPA : 0).toFixed(2), give_way_events: menv.giveWayEvents });
  return { rowA: row(A, agA.algorithm || "RuleBased"), rowB: row(B, agB.algorithm || "RuleBased") };
}
function runCell(cfg) {
  const { pairA, pairB, seed, envSeed, trainEpisodes, evalEpisodes, budget, noiseStd = 0, packetErrorRate = 0 } = cfg;
  setSeeded(seed * 1000003 + 12345);
  const menv = new MultiVesselBintulu(envSeed, { noiseStd, packetErrorRate, obstacleCount: 6 });
  const agA = makeAgent(pairA, menv.A, menv.obsDim, budget), agB = makeAgent(pairB, menv.B, menv.obsDim, budget);
  const anyRL = pairA !== "RuleBased" || pairB !== "RuleBased";
  if (anyRL) for (let ep = 0; ep < trainEpisodes; ep++) runEpisode(menv, agA, agB, false);
  const rowsA = [], rowsB = [];
  for (let ep = 0; ep < evalEpisodes; ep++) { const { rowA, rowB } = runEpisode(menv, agA, agB, true); rowsA.push(Object.assign({ phase: "eval", episode: ep + 1, vessel: "A" }, rowA)); rowsB.push(Object.assign({ phase: "eval", episode: ep + 1, vessel: "B" }, rowB)); }
  restore();
  return { rowsA, rowsB };
}
module.exports = { runCell };
if (require.main === module) {
  for (const [pa, pb] of [["DQN", "DQN"], ["DQN", "RuleBased"], ["RuleBased", "RuleBased"]]) {
    const line = [];
    for (const [c, ns, per] of [["clean", 0, 0], ["harsh", 0.25, 0.4]]) {
      const { rowsA } = runCell({ pairA: pa, pairB: pb, seed: 0, envSeed: 42, trainEpisodes: pa === "RuleBased" && pb === "RuleBased" ? 0 : 120, evalEpisodes: 20, budget: 3600, noiseStd: ns, packetErrorRate: per });
      const vc = rowsA.reduce((a, r) => a + r.vessel_collisions, 0) / rowsA.length, cpa = rowsA.reduce((a, r) => a + r.min_cpa, 0) / rowsA.length;
      line.push(`${c} vColl/ep=${vc.toFixed(2)} CPA=${cpa.toFixed(0)}`);
    }
    console.log(`${(pa + "/" + pb).padEnd(20)} ${line.join("  ")}`);
  }
}
