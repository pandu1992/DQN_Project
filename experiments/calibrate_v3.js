"use strict";
const { runCell } = require("./harness_v3.js");

function meanOf(rows, k) {
  const v = rows.map(r => r[k]).filter(x => x !== "" && x != null && isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
}
function evalStats(evalRows) {
  const succ = 100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
  return {
    succ,
    reward: meanOf(evalRows, "reward"),
    coll: meanOf(evalRows, "collisions"),
    cte: meanOf(evalRows, "cte_mean"),
    iala: meanOf(evalRows, "iala_violations"),
    dock: meanOf(evalRows, "docking_accuracy"),
  };
}

// timing for one representative cell
const t0 = Date.now();
const warm = runCell({ algorithm: "DQN", per: false, noisy: false, seed: 0, envSeed: 42,
  trainEpisodes: 120, evalEpisodes: 40, totalStepsBudget: 120 * 30, noiseStd: 0.1, packetErrorRate: 0.1 });
console.log(`one cell (120 train ep) elapsed: ${Date.now() - t0}ms`);

// discriminative check: DQN across a noise x PER grid, 2 seeds each
const noiseLevels = [0.0, 0.1, 0.25];
const perLevels = [0.0, 0.2, 0.4];
console.log("\nDQN across noise x PER (2 seeds, 100 train ep):");
console.log("noise  PER   succ   reward  coll   cte    iala   dock");
for (const ns of noiseLevels) {
  for (const per of perLevels) {
    const acc = { succ: 0, reward: 0, coll: 0, cte: 0, iala: 0, dock: 0, dn: 0 };
    for (let seed = 0; seed < 2; seed++) {
      const { evalRows } = runCell({ algorithm: "DQN", per: false, noisy: false, seed, envSeed: 42 + seed,
        trainEpisodes: 100, evalEpisodes: 30, totalStepsBudget: 100 * 30, noiseStd: ns, packetErrorRate: per });
      const s = evalStats(evalRows);
      acc.succ += s.succ; acc.reward += s.reward; acc.coll += s.coll; acc.cte += s.cte; acc.iala += s.iala;
      if (isFinite(s.dock)) { acc.dock += s.dock; acc.dn++; }
    }
    const d = (x) => (x / 2);
    console.log(`${ns.toFixed(2)}  ${per.toFixed(2)}  ${d(acc.succ).toFixed(0).padStart(4)}%  ${d(acc.reward).toFixed(1).padStart(6)}  ${d(acc.coll).toFixed(2)}  ${d(acc.cte).toFixed(1).padStart(5)}  ${d(acc.iala).toFixed(1).padStart(5)}  ${acc.dn ? (acc.dock / acc.dn).toFixed(1) : "n/a"}`);
  }
}
