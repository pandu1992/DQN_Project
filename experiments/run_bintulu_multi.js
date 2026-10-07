"use strict";
const fs = require("fs"), path = require("path");
const { runCell } = require("./harness_bintulu_multi.js");
const ROOT = path.join(__dirname, "..");
const RAW = path.join(ROOT, "results_bintulu/raw_multi"); fs.mkdirSync(RAW, { recursive: true });
const CSV = path.join(RAW, "eval_episodes.csv"), MAN = path.join(RAW, "run_manifest.json");
const PAIRINGS = [["DQN", "DQN"], ["DQN", "RuleBased"], ["RuleBased", "RuleBased"]];
const PNAME = ["DQN_vs_DQN", "DQN_vs_RuleBased", "RuleBased_vs_RuleBased"];
const CONDS = [["clean", 0, 0], ["mid", 0.1, 0.2], ["harsh", 0.25, 0.4]];
const SEEDS = [0, 1, 2, 3, 4, 5];
const COLS = ["config_id", "pairing", "condition", "noise_std", "packet_error_rate", "seed", "env_seed", "phase", "episode", "vessel", "agent", "success", "reward", "static_collisions", "cte_mean", "iala_violations", "vessel_collisions", "vessel_collision_flag", "near_misses", "min_cpa", "give_way_events"];
const line = (o) => COLS.map((c) => (o[c] === "" || o[c] == null ? "" : o[c])).join(",");
function main() {
  const t0 = Date.now();
  const argCell = (process.argv.find((a) => a.startsWith("--cell=")) || "").split("=")[1] || null;
  const doAppend = process.argv.includes("--append");
  let pis = PAIRINGS.map((_, i) => i), cis = CONDS.map((_, i) => i);
  if (argCell) { const [p, c] = argCell.split(":"); pis = [+p]; cis = [+c]; }
  if (!doAppend || !fs.existsSync(CSV)) fs.writeFileSync(CSV, COLS.join(",") + "\n");
  let man = { experiment_id: "bpan_multi_vessel", cells: [] };
  if (doAppend && fs.existsSync(MAN)) { try { man = JSON.parse(fs.readFileSync(MAN, "utf8")); } catch (e) {} man.cells = man.cells || []; }
  let idx = 0, total = pis.length * cis.length * SEEDS.length;
  for (const pi of pis) for (const ci of cis) {
    const [pa, pb] = PAIRINGS[pi], [cn, ns, per] = CONDS[ci], config_id = `${PNAME[pi]}_${cn}`;
    for (const seed of SEEDS) {
      const env_seed = 42 + seed, c0 = Date.now();
      const { rowsA, rowsB } = runCell({ pairA: pa, pairB: pb, seed, envSeed: env_seed, trainEpisodes: (pa === "RuleBased" && pb === "RuleBased") ? 0 : 150, evalEpisodes: 30, budget: 4500, noiseStd: ns, packetErrorRate: per });
      const base = { config_id, pairing: PNAME[pi], condition: cn, noise_std: ns, packet_error_rate: per, seed, env_seed };
      fs.appendFileSync(CSV, rowsA.concat(rowsB).map((r) => line(Object.assign({}, base, r))).join("\n") + "\n");
      idx++; man.cells.push({ config_id, seed, ms: Date.now() - c0 });
      const vc = (rowsA.reduce((a, r) => a + r.vessel_collisions, 0) / rowsA.length).toFixed(2);
      console.log(`[${argCell || ""}][${idx}/${total}] ${config_id} seed=${seed} vColl/ep=${vc} (${Date.now() - c0}ms)`);
    }
  }
  man.total_cells = man.cells.length; fs.writeFileSync(MAN, JSON.stringify(man, null, 2));
  console.log(`\nDONE +${idx} in ${((Date.now() - t0) / 1000).toFixed(1)}s. Total ${man.cells.length}.`);
}
main();
