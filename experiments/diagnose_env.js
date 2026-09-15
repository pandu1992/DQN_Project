"use strict";
const path = require("path");
const fs = require("fs");
const vm = require("vm");
function load() {
  const jsDir = path.join(__dirname, "..", "js");
  const s = {}; s.window = s; s.globalThis = s; s.Math = Math; s.Float64Array = Float64Array; s.Array = Array; s.console = console;
  vm.createContext(s);
  for (const f of ["environment.js", "dqn.js"]) vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), s, { filename: f });
  return s.window;
}
const W = load();
const { VesselEnv } = W.BintuluEnv;

// How many distinct missions does the seeded mission stream yield?
for (const seed of [42, 43, 44]) {
  const env = new VesselEnv(seed);
  const missions = new Set();
  const pathLens = [];
  for (let i = 0; i < 200; i++) {
    env.reset();
    missions.add(env.startWp + "->" + env.goalWp);
    pathLens.push(env.pathNodes.length - 1);
  }
  console.log(`envSeed=${seed}: distinct missions in 200 resets = ${missions.size}, plannedLen min/max = ${Math.min(...pathLens)}/${Math.max(...pathLens)}`);
  console.log("   sample missions:", [...missions].slice(0, 6).join(", "));
}

// Is the optimum trivially "always FORWARD"? Roll a pure-forward policy.
const env = new VesselEnv(42);
let succ = 0, rew = 0, N = 50;
for (let i = 0; i < N; i++) {
  env.reset();
  let done = false, R = 0, r2;
  while (!done) { r2 = env.step(1); R = env.totalReward; done = r2.terminated || r2.truncated; }
  succ += r2.info.reachedGoal ? 1 : 0; rew += R;
}
console.log(`Pure-FORWARD policy over ${N} eps: success=${(100*succ/N).toFixed(0)}%  meanReward=${(rew/N).toFixed(2)}`);

// Reward spread of the optimal (forward) policy across missions
env.reset();
const rewards = [];
for (let i = 0; i < 200; i++) {
  env.reset(); let done=false, r2;
  while(!done){ r2 = env.step(1); done = r2.terminated || r2.truncated; }
  rewards.push(env.totalReward);
}
const mean = rewards.reduce((a,b)=>a+b,0)/rewards.length;
const sd = Math.sqrt(rewards.reduce((a,b)=>a+(b-mean)**2,0)/rewards.length);
console.log(`Forward-policy reward across 200 missions: mean=${mean.toFixed(2)} sd=${sd.toFixed(2)} min=${Math.min(...rewards)} max=${Math.max(...rewards)}`);
