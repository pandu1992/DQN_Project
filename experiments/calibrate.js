"use strict";
const { runCell } = require("./harness.js");

function agg(rows, key) {
  const v = rows.map(r => r[key]).filter(x => x !== "" && x != null && isFinite(x));
  const m = v.reduce((a, b) => a + b, 0) / (v.length || 1);
  return m;
}
function evalStats(evalRows) {
  const succ = evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length;
  const rew = agg(evalRows, "reward");
  const solved = evalRows.filter(r => r.success === 1);
  const optR = solved.length ? agg(solved, "optimality_ratio") : NaN;
  return { succ: 100 * succ, rew, optR, nSolved: solved.length };
}

const budgets = [150, 300];
const algos = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"];
for (const te of budgets) {
  console.log(`\n=== trainEpisodes=${te}, evalEpisodes=40, 3 seeds ===`);
  for (const algo of algos) {
    const s = [];
    for (let seed = 0; seed < 3; seed++) {
      const rr = runCell({ algorithm: algo, per: false, noisy: false, seed, envSeed: 42 + seed, trainEpisodes: te, evalEpisodes: 40, totalStepsBudget: te * 30 });
      s.push(evalStats(rr.evalRows));
    }
    const ms = (k) => (s.reduce((a, x) => a + x[k], 0) / s.length);
    const sd = (k) => { const m = ms(k); return Math.sqrt(s.reduce((a, x) => a + (x[k] - m) ** 2, 0) / s.length); };
    console.log(`  ${algo.padEnd(17)} success=${ms("succ").toFixed(1)}±${sd("succ").toFixed(1)}%  reward=${ms("rew").toFixed(1)}±${sd("rew").toFixed(1)}  optRatio=${ms("optR").toFixed(3)}`);
  }
}
