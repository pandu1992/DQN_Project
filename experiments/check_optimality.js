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
const { dijkstra } = W.BintuluEnv;

// Force the agent to follow the EXACT Dijkstra planned path by selecting,
// at each node, the action-slot edge whose neighbor equals the next planned node.
// The resulting routeCost should equal optimalCost => ratio exactly 1.0.
const env = new VesselEnvV2(42);
let below1 = 0, exactly1 = 0, above1 = 0, n = 0;
const ratios = [];
for (let i = 0; i < 100; i++) {
  const obs0 = env.reset();
  const planned = env.plannedPath.slice(); // optimal node sequence
  let idx = 0, done = false, r;
  while (!done) {
    const cur = env.currentWp;
    const edges = env.adj[cur] || [];
    // find next planned node after current
    let nextNode = null;
    const pos = planned.indexOf(cur);
    if (pos >= 0 && pos < planned.length - 1) nextNode = planned[pos + 1];
    let a = edges.findIndex((e) => e.neighbor === nextNode);
    if (a < 0) a = 0; // fallback
    r = env.step(a);
    done = r.terminated || r.truncated;
    if (env.stepCount > 70) break;
  }
  if (r.info.reachedGoal && r.info.optimalityRatio) {
    const rr = r.info.optimalityRatio; ratios.push(rr); n++;
    if (rr < 0.999) below1++; else if (rr <= 1.001) exactly1++; else above1++;
  }
}
console.log(`Follow-Dijkstra policy over ${n} solved eps: ratio <1: ${below1}, ~1: ${exactly1}, >1: ${above1}`);
console.log(`  ratio mean=${(ratios.reduce((a,b)=>a+b,0)/ratios.length).toFixed(4)} min=${Math.min(...ratios).toFixed(4)} max=${Math.max(...ratios).toFixed(4)}`);
console.log(ratios.every(r => r >= 0.999) ? "OK: no ratio below 1.0 (planner is the true optimum)" : "WARN: ratios below 1.0 exist -> metric/planner mismatch");
