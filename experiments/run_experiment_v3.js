/* ============================================================
 * V3 ROBUSTNESS EXPERIMENT RUNNER
 * Reads results_v3/configs/experiment_config_v3.json, runs every
 * (algorithm x noise x packet_error x seed) cell via harness_v3.js,
 * and streams INDIVIDUAL per-episode evaluation rows (all real
 * metrics) to results_v3/raw/. Sharded + resumable like the V2 runner.
 *
 * Shard filters:
 *   --cell=Algorithm:noiseIdx:perIdx   run one (algo,noise,per) across all seeds
 *   --algo=Algorithm                   run one algorithm (all noise/per/seed)
 *   --append                           keep existing CSV (accumulate shards)
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const { runCell } = require("./harness_v3.js");

const ROOT = path.join(__dirname, "..");
const CFG = JSON.parse(fs.readFileSync(path.join(ROOT, "results_v3/configs/experiment_config_v3.json"), "utf8"));
const RAW = path.join(ROOT, "results_v3/raw");
fs.mkdirSync(RAW, { recursive: true });

const EVAL_CSV = path.join(RAW, "eval_episodes.csv");
const TRAIN_CSV = path.join(RAW, "train_episodes.csv");
const MANIFEST = path.join(RAW, "run_manifest.json");

const EVAL_COLS = [
  "config_id", "algorithm", "noise_std", "packet_error_rate", "seed", "env_seed",
  "phase", "episode", "reward", "success", "steps", "invalid", "revisit",
  "route_cost", "optimal_cost", "excess_cost", "optimality_ratio",
  "route_risk", "route_difficulty", "route_distance",
  "collisions", "collision_flag", "cte_mean", "iala_violations", "dropped_frames", "docking_accuracy",
  "lane", "goal_lane", "start", "goal", "terminated", "truncated",
];
const TRAIN_COLS = EVAL_COLS.slice();

function line(cols, obj) {
  return cols.map((c) => {
    const v = obj[c];
    if (v === "" || v === null || v === undefined) return "";
    return v;
  }).join(",");
}

function main() {
  const t0 = Date.now();
  const argAlgo = (process.argv.find((a) => a.startsWith("--algo=")) || "").split("=")[1] || null;
  const argCell = (process.argv.find((a) => a.startsWith("--cell=")) || "").split("=")[1] || null;
  const doAppend = process.argv.includes("--append");

  let algorithms = CFG.factors.algorithm;
  let noiseIdxs = CFG.factors.noise_std.map((_, i) => i);
  let perIdxs = CFG.factors.packet_error_rate.map((_, i) => i);
  if (argAlgo) algorithms = [argAlgo];
  if (argCell) {
    const [ca, ni, pi] = argCell.split(":");
    algorithms = [ca]; noiseIdxs = [parseInt(ni, 10)]; perIdxs = [parseInt(pi, 10)];
  }

  if (!doAppend || !fs.existsSync(EVAL_CSV)) {
    fs.writeFileSync(EVAL_CSV, EVAL_COLS.join(",") + "\n");
    fs.writeFileSync(TRAIN_CSV, TRAIN_COLS.join(",") + "\n");
  }
  let manifest = { experiment_id: CFG.experiment_id, started: new Date().toISOString(), cells: [], nan_inf_flags: [] };
  if (doAppend && fs.existsSync(MANIFEST)) {
    try { manifest = JSON.parse(fs.readFileSync(MANIFEST, "utf8")); } catch (e) {}
    manifest.cells = manifest.cells || []; manifest.nan_inf_flags = manifest.nan_inf_flags || [];
  }

  const seeds = CFG.seeds;
  const trainEpisodes = CFG.training.train_episodes;
  const evalEpisodes = CFG.evaluation.eval_episodes;
  const totalStepsBudget = CFG.training.total_steps_budget;
  const obstacleCount = CFG.environment.obstacle_count;

  let cellIdx = 0;
  const totalCells = algorithms.length * noiseIdxs.length * perIdxs.length * seeds.length;
  const shardTag = argCell ? `[${argCell}] ` : argAlgo ? `[${argAlgo}] ` : "";

  for (const algorithm of algorithms) {
    for (const ni of noiseIdxs) {
      const noiseStd = CFG.factors.noise_std[ni];
      for (const pi of perIdxs) {
        const packetErrorRate = CFG.factors.packet_error_rate[pi];
        const config_id = `${algorithm}_n${noiseStd}_p${packetErrorRate}`;
        for (const seed of seeds) {
          const env_seed = 42 + seed;
          const cellT0 = Date.now();
          const { trainRows, evalRows } = runCell({
            algorithm, per: false, noisy: false, seed, envSeed: env_seed,
            trainEpisodes, evalEpisodes, totalStepsBudget, noiseStd, packetErrorRate, obstacleCount,
          });
          const base = { config_id, algorithm, noise_std: noiseStd, packet_error_rate: packetErrorRate, seed, env_seed };
          let nanFlag = false;
          const check = (r) => { for (const k of ["reward", "steps", "cte_mean"]) if (!isFinite(r[k])) nanFlag = true; };
          const evalLines = evalRows.map((r) => { check(r); return line(EVAL_COLS, Object.assign({}, base, r)); });
          const trainLines = trainRows.map((r) => line(TRAIN_COLS, Object.assign({}, base, r)));
          fs.appendFileSync(EVAL_CSV, evalLines.join("\n") + "\n");
          fs.appendFileSync(TRAIN_CSV, trainLines.join("\n") + "\n");

          cellIdx++;
          const ms = Date.now() - cellT0;
          manifest.cells.push({ config_id, seed, ms, evalEps: evalRows.length });
          if (nanFlag) manifest.nan_inf_flags.push({ config_id, seed });
          const succ = (100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length).toFixed(0);
          console.log(`${shardTag}[${cellIdx}/${totalCells}] ${config_id} seed=${seed} succ=${succ}% (${ms}ms)`);
        }
      }
    }
  }

  manifest.finished = new Date().toISOString();
  manifest.total_cells = manifest.cells.length;
  manifest.eval_rows = manifest.cells.reduce((a, c) => a + c.evalEps, 0);
  fs.writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2));
  console.log(`\n${shardTag}DONE +${cellIdx} cells in ${((Date.now() - t0) / 1000).toFixed(1)}s. Total ${manifest.cells.length}. NaN flags: ${manifest.nan_inf_flags.length}`);
}
main();
