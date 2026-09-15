"use strict";
const { runCell } = require("./harness.js");

// 1) Determinism: same seed -> identical eval metrics
function digest(rows) {
  return rows.map(r => `${r.success}:${r.reward.toFixed(4)}:${r.steps}`).join("|");
}
const cfgA = { algorithm: "DQN", per: false, noisy: false, seed: 3, envSeed: 45, trainEpisodes: 60, evalEpisodes: 20, totalStepsBudget: 60 * 25 };
const r1 = runCell(cfgA);
const r2 = runCell(cfgA);
const det = digest(r1.evalRows) === digest(r2.evalRows) && digest(r1.trainRows) === digest(r2.trainRows);
console.log("Determinism (same seed => identical):", det ? "PASS" : "FAIL");

// 2) Different seeds -> different trajectories (RNG actually varies)
const r3 = runCell(Object.assign({}, cfgA, { seed: 7 }));
console.log("Seed sensitivity (diff seed => diff data):",
  digest(r1.evalRows) !== digest(r3.evalRows) ? "PASS" : "FAIL (identical!)");

// 3) Spread across algorithms at a SHORT budget (to avoid ceiling=100% everywhere)
const budgets = [40, 80, 150];
for (const te of budgets) {
  const line = [];
  for (const algo of ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]) {
    let succ = 0, rew = 0, n = 0;
    for (let s = 0; s < 3; s++) {
      const rr = runCell({ algorithm: algo, per: false, noisy: false, seed: s, envSeed: 42 + s, trainEpisodes: te, evalEpisodes: 30, totalStepsBudget: te * 25 });
      const es = rr.evalRows.reduce((a, x) => a + x.success, 0) / rr.evalRows.length;
      const er = rr.evalRows.reduce((a, x) => a + x.reward, 0) / rr.evalRows.length;
      succ += es; rew += er; n++;
    }
    line.push(`${algo}: succ=${(100 * succ / n).toFixed(0)}% rew=${(rew / n).toFixed(1)}`);
  }
  console.log(`trainEpisodes=${te} (3 seeds each):`);
  console.log("   " + line.join(" | "));
}
