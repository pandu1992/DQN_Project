/* ============================================================
 * STUDY 4 RUNNER — two independent vessels, 4 pairings x 3 conditions x 8 seeds
 * Sharded + resumable. Shard filter: --cell=pairingIdx:condIdx
 * (pairingIdx 0..3, condIdx 0=clean,1=mid,2=harsh). --append accumulates.
 *
 * Emits ONE row per vessel per eval episode (vessel A and B), with the shared
 * inter-vessel metrics duplicated onto both rows. Deterministic per seed.
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const { runCell } = require("./harness_study4.js");

const ROOT = path.join(__dirname, "..");
const CFG = JSON.parse(fs.readFileSync(path.join(ROOT, "results_study4/configs/experiment_config_study4.json"), "utf8"));
const RAW = path.join(ROOT, "results_study4/raw");
fs.mkdirSync(RAW, { recursive: true });
const EVAL_CSV = path.join(RAW, "eval_episodes.csv");
const MANIFEST = path.join(RAW, "run_manifest.json");

const COLS = [
  "config_id", "pairing", "condition", "noise_std", "packet_error_rate", "seed", "env_seed",
  "phase", "episode", "vessel", "agent",
  "reward", "success", "steps", "invalid",
  "route_cost", "optimal_cost", "optimality_ratio",
  "static_collisions", "cte_mean", "iala_violations", "dropped_frames", "docking_accuracy",
  "vessel_collisions", "vessel_collision_flag", "collision_contact_steps", "near_misses",
  "min_cpa", "encounter_steps", "give_way_events",
  "lane", "goal_lane", "start", "goal",
];
const line = (o) => COLS.map((c) => (o[c] === "" || o[c] == null ? "" : o[c])).join(",");

function main() {
  const t0 = Date.now();
  const argCell = (process.argv.find((a) => a.startsWith("--cell=")) || "").split("=")[1] || null;
  const doAppend = process.argv.includes("--append");

  let pairIdxs = CFG.factors.pairing.map((_, i) => i);
  let condIdxs = CFG.factors.condition.map((_, i) => i);
  if (argCell) { const [pi, ci] = argCell.split(":"); pairIdxs = [parseInt(pi, 10)]; condIdxs = [parseInt(ci, 10)]; }

  if (!doAppend || !fs.existsSync(EVAL_CSV)) fs.writeFileSync(EVAL_CSV, COLS.join(",") + "\n");
  let manifest = { experiment_id: CFG.experiment_id, cells: [], nan_inf_flags: [] };
  if (doAppend && fs.existsSync(MANIFEST)) {
    try { manifest = JSON.parse(fs.readFileSync(MANIFEST, "utf8")); } catch (e) {}
    manifest.cells = manifest.cells || []; manifest.nan_inf_flags = manifest.nan_inf_flags || [];
  }

  const argSeeds = (process.argv.find((a) => a.startsWith("--seeds=")) || "").split("=")[1] || null;
  const seeds = argSeeds ? argSeeds.split(",").map((s) => parseInt(s, 10)) : CFG.seeds;
  const te = CFG.training.train_episodes, ee = CFG.evaluation.eval_episodes;
  const budget = CFG.training.total_steps_budget, obstacleCount = CFG.environment.obstacle_count;
  let idx = 0;
  const total = pairIdxs.length * condIdxs.length * seeds.length;
  const tag = argCell ? `[${argCell}] ` : "";

  for (const pi of pairIdxs) {
    const pairing = CFG.factors.pairing[pi];
    for (const ci of condIdxs) {
      const cond = CFG.factors.condition[ci];
      const config_id = `${pairing.name}_${cond.name}`;
      for (const seed of seeds) {
        const env_seed = 42 + seed;
        const cellT0 = Date.now();
        const { evalRowsA, evalRowsB } = runCell({
          pairA: pairing.A, pairB: pairing.B, seed, envSeed: env_seed,
          trainEpisodes: te, evalEpisodes: ee, totalStepsBudget: budget,
          noiseStd: cond.noise_std, packetErrorRate: cond.packet_error_rate, obstacleCount,
        });
        const base = {
          config_id, pairing: pairing.name, condition: cond.name,
          noise_std: cond.noise_std, packet_error_rate: cond.packet_error_rate, seed, env_seed,
        };
        let nan = false;
        const rows = evalRowsA.concat(evalRowsB);
        const lines = rows.map((r) => {
          if (!isFinite(r.reward) || !isFinite(r.min_cpa)) nan = true;
          return line(Object.assign({}, base, r));
        });
        fs.appendFileSync(EVAL_CSV, lines.join("\n") + "\n");
        idx++;
        const ms = Date.now() - cellT0;
        manifest.cells.push({ config_id, seed, ms, evalRows: rows.length });
        if (nan) manifest.nan_inf_flags.push({ config_id, seed });
        const sA = (100 * evalRowsA.reduce((a, r) => a + r.success, 0) / evalRowsA.length).toFixed(0);
        const sB = (100 * evalRowsB.reduce((a, r) => a + r.success, 0) / evalRowsB.length).toFixed(0);
        const vc = (evalRowsA.reduce((a, r) => a + r.vessel_collisions, 0) / evalRowsA.length).toFixed(2);
        console.log(`${tag}[${idx}/${total}] ${config_id} seed=${seed} succA=${sA}% succB=${sB}% vColl/ep=${vc} (${ms}ms)`);
      }
    }
  }
  manifest.total_cells = manifest.cells.length;
  manifest.eval_rows = manifest.cells.reduce((a, c) => a + c.evalRows, 0);
  fs.writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2));
  console.log(`\n${tag}DONE +${idx} cells in ${((Date.now() - t0) / 1000).toFixed(1)}s. Total ${manifest.cells.length}. NaN: ${manifest.nan_inf_flags.length}`);
}
main();
