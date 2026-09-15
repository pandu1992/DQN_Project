"use strict";
const path = require("path"), fs = require("fs"), vm = require("vm");
function load() {
  const jsDir = path.join(__dirname, "..", "js");
  const s = {}; s.window = s; s.globalThis = s; s.Math = Math; s.Float64Array = Float64Array; s.Array = Array; s.console = console;
  vm.createContext(s);
  for (const f of ["environment.js", "dqn.js", "environmentV2.js"]) vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), s, { filename: f });
  return s.window;
}
const W = load();
const { VesselEnvV2 } = W.BintuluEnvV2;

// mission diversity + optimal-cost spread
for (const seed of [42, 43]) {
  const env = new VesselEnvV2(seed);
  const missions = new Set(); const costs = [];
  for (let i = 0; i < 300; i++) { env.reset(); missions.add(env.startWp + "->" + env.goalWp); costs.push(env.optimalCost); }
  const mn = Math.min(...costs), mx = Math.max(...costs);
  console.log(`envSeed=${seed}: distinct missions=${missions.size}, optimalCost min/max=${mn.toFixed(1)}/${mx.toFixed(1)}, maxOutDeg=${env.maxOutDeg}, obsDim=${env.obsDim}, nActions=${env.nActions}`);
}

// Random policy baseline vs greedy-toward-goal heuristic — is the task non-trivial?
function rollPolicy(env, N, policyFn) {
  let succ = 0, rew = 0, ratios = [];
  for (let i = 0; i < N; i++) {
    let obs = env.reset(); let done = false, r;
    while (!done) { const a = policyFn(env, obs); r = env.step(a); obs = r.obs; done = r.terminated || r.truncated; }
    succ += r.info.reachedGoal ? 1 : 0; rew += env.totalReward;
    if (r.info.optimalityRatio) ratios.push(r.info.optimalityRatio);
  }
  const mr = ratios.length ? ratios.reduce((a,b)=>a+b,0)/ratios.length : NaN;
  return { succ: 100*succ/N, rew: rew/N, optRatio: mr, nOpt: ratios.length };
}
const env = new VesselEnvV2(42);
const rand = rollPolicy(env, 200, (e) => Math.floor(Math.random() * e.nActions));
// greedy: pick the existing slot whose towardGoal flag is 1 and lowest cost; else slot 0
function greedyToward(e) {
  const edges = e.adj[e.currentWp] || [];
  let best = -1, bestScore = Infinity;
  const g = e.states[e.goalWp];
  for (let a = 0; a < edges.length; a++) {
    const nb = e.states[edges[a].neighbor];
    const d = Math.hypot(nb.x - g.x, nb.y - g.y);
    if (d < bestScore) { bestScore = d; best = a; }
  }
  return best < 0 ? 0 : best;
}
const greedy = rollPolicy(new VesselEnvV2(42), 200, (e) => greedyToward(e));
console.log(`RANDOM  policy: success=${rand.succ.toFixed(0)}% meanReward=${rand.rew.toFixed(1)} optRatio=${isNaN(rand.optRatio)?"n/a":rand.optRatio.toFixed(2)} (n=${rand.nOpt})`);
console.log(`GREEDY-goal heuristic: success=${greedy.succ.toFixed(0)}% meanReward=${greedy.rew.toFixed(1)} optRatio=${isNaN(greedy.optRatio)?"n/a":greedy.optRatio.toFixed(2)} (n=${greedy.nOpt})`);
