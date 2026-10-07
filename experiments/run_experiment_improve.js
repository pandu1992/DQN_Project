/* ============================================================
 * BPAN improvement runner — 4 arms x 3 conditions x 6 seeds. Sharded + resumable.
 * --cell=armIdx:condIdx  --seeds=CSV  --append
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const { runCell } = require("./harness_bintulu_improve.js");

const ROOT = path.join(__dirname, "..");
const CFG = JSON.parse(fs.readFileSync(path.join(ROOT, "results_bintulu/configs/experiment_config_improve.json"), "utf8"));
const RAW = path.join(ROOT, "results_bintulu/raw_improve");
fs.mkdirSync(RAW, { recursive: true });
const EVAL_CSV = path.join(RAW, "eval_episodes.csv");
const MANIFEST = path.join(RAW, "run_manifest.json");

const COLS = ["config_id", "arm", "condition", "noise_std", "packet_error_rate", "seed", "env_seed",
  "phase", "episode", "reward", "success", "steps", "optimality_ratio",
  "collisions", "collision_flag", "cte_mean", "iala_violations", "dropped_frames", "docking_accuracy", "start", "goal"];
const line = (o) => COLS.map((c) => (o[c] === "" || o[c] == null ? "" : o[c])).join(",");

function main() {
  const t0 = Date.now();
  const argCell = (process.argv.find((a) => a.startsWith("--cell=")) || "").split("=")[1] || null;
  const argSeedsRaw = (process.argv.find((a) => a.startsWith("--seeds=")) || "").split("=")[1] || null;
  const argSeeds = argSeedsRaw ? argSeedsRaw.split(",").map((s) => parseInt(s, 10)) : null;
  const doAppend = process.argv.includes("--append");

  let armIdxs = CFG.factors.arm.map((_, i) => i);
  let condIdxs = CFG.factors.condition.map((_, i) => i);
  if (argCell) { const [ai, ci] = argCell.split(":"); armIdxs = [parseInt(ai, 10)]; condIdxs = [parseInt(ci, 10)]; }
  const seeds = argSeeds || CFG.seeds;

  if (!doAppend || !fs.existsSync(EVAL_CSV)) fs.writeFileSync(EVAL_CSV, COLS.join(",") + "\n");
  let manifest = { experiment_id: CFG.experiment_id, cells: [] };
  if (doAppend && fs.existsSync(MANIFEST)) { try { manifest = JSON.parse(fs.readFileSync(MANIFEST, "utf8")); } catch (e) {} manifest.cells = manifest.cells || []; }

  const ee = CFG.evaluation.eval_episodes, obstacleCount = 6;
  let idx = 0; const total = armIdxs.length * condIdxs.length * seeds.length;
  const tag = argCell ? `[${argCell}] ` : "";
  for (const ai of armIdxs) {
    const arm = CFG.factors.arm[ai];
    for (const ci of condIdxs) {
      const cond = CFG.factors.condition[ci];
      const config_id = `${arm}_${cond.name}`;
      for (const seed of seeds) {
        const env_seed = 42 + seed, cellT0 = Date.now();
        const { evalRows } = runCell({ arm, seed, envSeed: env_seed, evalEpisodes: ee, noiseStd: cond.noise_std, packetErrorRate: cond.packet_error_rate, obstacleCount });
        const base = { config_id, arm, condition: cond.name, noise_std: cond.noise_std, packet_error_rate: cond.packet_error_rate, seed, env_seed };
        fs.appendFileSync(EVAL_CSV, evalRows.map((r) => line(Object.assign({}, base, r))).join("\n") + "\n");
        idx++; const ms = Date.now() - cellT0;
        manifest.cells.push({ config_id, seed, ms, evalEps: evalRows.length });
        const succ = (100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length).toFixed(0);
        console.log(`${tag}[${idx}/${total}] ${config_id} seed=${seed} succ=${succ}% (${ms}ms)`);
      }
    }
  }
  manifest.total_cells = manifest.cells.length; manifest.eval_rows = manifest.cells.reduce((a, c) => a + c.evalEps, 0);
  fs.writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2));
  console.log(`\n${tag}DONE +${idx} in ${((Date.now() - t0) / 1000).toFixed(1)}s. Total ${manifest.cells.length}.`);
}
main();
