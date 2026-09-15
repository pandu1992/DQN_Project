"use strict";
const path = require("path"), fs = require("fs"), vm = require("vm");
function load() {
  const jsDir = path.join(__dirname, "..", "js");
  const s = {}; s.window = s; s.globalThis = s; s.Math = Math; s.Float64Array = Float64Array; s.Array = Array; s.console = console;
  vm.createContext(s);
  for (const f of ["environment.js", "dqn.js", "environmentV2.js", "environmentV3.js"])
    vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), s, { filename: f });
  return s.window;
}
const W = load();
const { VesselEnvV3 } = W.BintuluEnvV3;

function rollPlan(env, N) {
  // follow the planned Dijkstra path exactly (best-case route)
  let succ = 0, coll = 0, cteSum = 0, iala = 0, drop = 0, dockSum = 0, dockN = 0, rew = 0;
  for (let i = 0; i < N; i++) {
    env.reset();
    let done = false, r, guard = 0;
    while (!done && guard++ < 80) {
      const planned = env.plannedPath;
      const pos = planned.indexOf(env.currentWp);
      let next = (pos >= 0 && pos < planned.length - 1) ? planned[pos + 1] : null;
      const edges = env.adj[env.currentWp] || [];
      let a = edges.findIndex(e => e.neighbor === next);
      if (a < 0) a = 0;
      r = env.step(a); done = r.terminated || r.truncated;
    }
    succ += r.info.reachedGoal ? 1 : 0;
    coll += r.info.collisionCount;
    cteSum += r.info.cteMean;
    iala += r.info.ialaViolations;
    drop += r.info.droppedFrames;
    rew += env.totalReward;
    if (r.info.dockingAccuracy != null) { dockSum += r.info.dockingAccuracy; dockN++; }
  }
  return {
    successPct: 100 * succ / N, meanColl: coll / N, meanCte: cteSum / N,
    meanIala: iala / N, meanDrop: drop / N, meanReward: rew / N,
    meanDock: dockN ? dockSum / dockN : NaN,
  };
}

console.log("obstacles per env (seed42):", new VesselEnvV3(42).obstacles.length, "obsDim:", new VesselEnvV3(42).obsDim, "nActions:", new VesselEnvV3(42).nActions);

// 1) Observation noise: same pose, higher noiseStd => larger obs variance
function obsNoiseStd(std) {
  const env = new VesselEnvV3(42, { noiseStd: std, packetErrorRate: 0 });
  const trueO = env._trueObs();
  let sq = 0, n = 0;
  for (let k = 0; k < 400; k++) {
    const o = env._sensedObs(trueO);
    for (let i = 0; i < o.length; i++) { const d = o[i] - trueO[i]; sq += d * d; n++; }
  }
  return Math.sqrt(sq / n);
}
console.log("\n[Observation noise] empirical RMS deviation of sensed obs:");
for (const s of [0.0, 0.05, 0.15, 0.30]) console.log(`  noiseStd=${s.toFixed(2)} -> rms=${obsNoiseStd(s).toFixed(4)}`);

// 2) Packet-error rate: dropped-frame fraction should track packetErrorRate
console.log("\n[Comms packet loss] dropped-frame fraction over rollouts (per-episode counted, summed):");
for (const per of [0.0, 0.1, 0.3, 0.6]) {
  let steps = 0, dropped = 0;
  for (let ep = 0; ep < 40; ep++) {
    const env = new VesselEnvV3(42 + ep, { packetErrorRate: per });
    env.reset(); let done = false, g = 0;
    while (!done && g++ < 60) { const r = env.step(Math.floor(Math.random() * env.nActions)); steps++; done = r.terminated || r.truncated; }
    dropped += env.droppedFrames;
  }
  console.log(`  packetErrorRate=${per.toFixed(2)} -> observed drop fraction=${(dropped / steps).toFixed(3)} (steps=${steps}, dropped=${dropped})`);
}

// 3) Follow-plan policy across noise/PER: metrics should stay sane; collisions/CTE/IALA are REAL
console.log("\n[Follow-Dijkstra policy] real physical metrics (200 eps):");
for (const [ns, per] of [[0, 0], [0.15, 0], [0, 0.3], [0.15, 0.3]]) {
  const env = new VesselEnvV3(42, { noiseStd: ns, packetErrorRate: per });
  const m = rollPlan(env, 200);
  console.log(`  noise=${ns} PER=${per}: succ=${m.successPct.toFixed(0)}% coll/ep=${m.meanColl.toFixed(2)} cte=${m.meanCte.toFixed(1)} iala/ep=${m.meanIala.toFixed(1)} drop/ep=${m.meanDrop.toFixed(1)} dockAcc=${isNaN(m.meanDock)?"n/a":m.meanDock.toFixed(1)} reward=${m.meanReward.toFixed(1)}`);
}

// 4) Collisions actually happen for a random policy (wanders into obstacles)
console.log("\n[Random policy] collisions occur (100 eps):");
const envR = new VesselEnvV3(42);
let colEps = 0, colTot = 0;
for (let i = 0; i < 100; i++) { envR.reset(); let done = false, g = 0, r; while (!done && g++ < 60) { r = envR.step(Math.floor(Math.random() * envR.nActions)); done = r.terminated || r.truncated; } if (envR.collisionCount > 0) colEps++; colTot += envR.collisionCount; }
console.log(`  episodes with >=1 collision: ${colEps}/100, total collisions: ${colTot}`);

// 5) Docking accuracy is a real distance (>=0) and CTE is 0 when perfectly on-plan
console.log("\n[Sanity] follow-plan CTE should be ~0 (agent rides the planned segments):");
const envP = new VesselEnvV3(42);
const mp = rollPlan(envP, 50);
console.log(`  follow-plan mean CTE=${mp.meanCte.toFixed(3)} (expect near 0), dockAcc=${mp.meanDock.toFixed(2)} (>=0 real distance)`);
