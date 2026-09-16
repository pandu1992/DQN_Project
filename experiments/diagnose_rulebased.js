"use strict";
const path = require("path"), fs = require("fs"), vm = require("vm");
function load() {
  const jsDir = path.join(__dirname, "..", "js");
  const s = {}; s.window = s; s.globalThis = s; s.Math = Math; s.Float64Array = Float64Array; s.Array = Array; s.console = console;
  vm.createContext(s);
  for (const f of ["environment.js", "dqn.js", "environmentV2.js", "environmentV3.js", "rulebased.js"])
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), s, { filename: f });
  return s.window;
}
const W = load();
const { VesselEnvV3 } = W.BintuluEnvV3;
const { RuleBasedAgent } = W.BintuluRuleBased;

function rollAgent(env, agent, N) {
  let succ = 0, rew = 0, coll = 0, cte = 0, iala = 0, dockSum = 0, dockN = 0, steps = 0;
  for (let i = 0; i < N; i++) {
    let obs = env.reset(); let done = false, r, g = 0;
    while (!done && g++ < 80) { const a = agent.act(obs, true); r = env.step(a); obs = r.obs; done = r.terminated || r.truncated; }
    succ += r.info.reachedGoal ? 1 : 0; rew += env.totalReward; coll += r.info.collisionCount;
    cte += r.info.cteMean; iala += r.info.ialaViolations; steps += env.stepCount;
    if (r.info.dockingAccuracy != null) { dockSum += r.info.dockingAccuracy; dockN++; }
  }
  return { succ: 100 * succ / N, rew: rew / N, coll: coll / N, cte: cte / N, iala: iala / N,
           dock: dockN ? dockSum / dockN : NaN, steps: steps / N };
}

console.log("=== Rule-based COLREGs baseline on VesselEnvV3 ===\n");
console.log("condition        success  reward   coll/ep  cte    iala/ep  dock   steps");
for (const [ns, per] of [[0, 0], [0.1, 0.2], [0.25, 0.4]]) {
  const env = new VesselEnvV3(42, { noiseStd: ns, packetErrorRate: per });
  const agent = new RuleBasedAgent(env.obsDim, env.nActions, { env });
  const m = rollAgent(env, agent, 100);
  console.log(`n${ns} p${per}         ${m.succ.toFixed(0).padStart(4)}%   ${m.rew.toFixed(1).padStart(6)}   ${m.coll.toFixed(2)}    ${m.cte.toFixed(1).padStart(5)}  ${m.iala.toFixed(1).padStart(5)}   ${isNaN(m.dock) ? "n/a" : m.dock.toFixed(1)}   ${m.steps.toFixed(1)}`);
}

// vs random baseline (clean)
function randAgent(nA) { return { act: () => Math.floor(Math.random() * nA), observe() {}, qValues() { return []; } }; }
const envR = new VesselEnvV3(42, {});
const rnd = rollAgent(envR, randAgent(envR.nActions), 100);
console.log(`\nRandom (clean):    ${rnd.succ.toFixed(0)}%   reward ${rnd.rew.toFixed(1)}  coll/ep ${rnd.coll.toFixed(2)}`);

// multi-seed stability of the baseline (clean)
let s = 0;
for (let seed = 0; seed < 6; seed++) {
  const env = new VesselEnvV3(42 + seed, {});
  const agent = new RuleBasedAgent(env.obsDim, env.nActions, { env });
  s += rollAgent(env, agent, 40).succ;
}
console.log(`\nRule-based clean success across 6 seeds: mean=${(s / 6).toFixed(1)}%  (should be stable & > random)`);
