/* ============================================================
 * FULL EXPERIMENT RUNNER
 * Reads results/configs/experiment_config.json, runs every
 * (algorithm x per x noisy x seed) cell, and streams INDIVIDUAL
 * per-episode observations to results/raw/ (no aggregation here).
 *
 * Outputs:
 *   results/raw/eval_episodes.csv   — one row per evaluation episode
 *   results/raw/train_episodes.csv  — one row per training episode
 *   results/raw/run_manifest.json   — completion manifest + timing
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const { runCell } = require("./harness.js");

const ROOT = path.join(__dirname, "..");
const CFG = JSON.parse(fs.readFileSync(path.join(ROOT, "results/configs/experiment_config.json"), "utf8"));
const RAW = path.join(ROOT, "results/raw");
fs.mkdirSync(RAW, { recursive: true });

const EVAL_CSV = path.join(RAW, "eval_episodes.csv");
const TRAIN_CSV = path.join(RAW, "train_episodes.csv");
const MANIFEST = path.join(RAW, "run_manifest.json");

const EVAL_COLS = [
  "config_id", "algorithm", "per", "noisy", "seed", "env_seed", "phase", "episode",
  "reward", "success", "steps", "invalid", "revisit",
  "route_cost", "optimal_cost", "excess_cost", "optimality_ratio",
  "route_risk", "route_difficulty", "route_distance",
  "lane", "goal_lane", "start", "goal", "terminated", "truncated",
];
const TRAIN_COLS = EVAL_COLS.slice();

function rowToLine(cols, obj) {
  return cols.map((c) => {
    const v = obj[c];
    if (v === "" || v === null || v === undefined) return "";
    if (typeof v === "string") return v;
    return v;
  }).join(",");
}

function main() {
  const t0 = Date.now();
  // optional sharding: --algo=DQN runs only that algorithm; --append keeps existing CSVs
  const argAlgo = (process.argv.find((a) => a.startsWith("--algo=")) || "").split("=")[1] || null;
  // --cell=Algorithm:per:noisy  (e.g. --cell=DoubleDQN:1:0) runs exactly one (algo,per,noisy) across all seeds
  const argCell = (process.argv.find((a) => a.startsWith("--cell=")) || "").split("=")[1] || null;
  const doAppend = process.argv.includes("--append");
  let algorithms = argAlgo ? [argAlgo] : CFG.factors.algorithm;
  let perLevels = CFG.factors.per;
  let noisyLevels = CFG.factors.noisy;
  if (argCell) {
    const [ca, cp, cn] = argCell.split(":");
    algorithms = [ca];
    perLevels = [cp === "1"];
    noisyLevels = [cn === "1"];
  }
  const seeds = CFG.seeds;
  const trainEpisodes = CFG.training.train_episodes;
  const evalEpisodes = CFG.evaluation.eval_episodes;
  const totalStepsBudget = CFG.training.total_steps_budget;

  // fresh files with headers (unless appending to an in-progress sharded run)
  if (!doAppend || !fs.existsSync(EVAL_CSV)) {
    fs.writeFileSync(EVAL_CSV, EVAL_COLS.join(",") + "\n");
    fs.writeFileSync(TRAIN_CSV, TRAIN_COLS.join(",") + "\n");
  }

  const manifestPath = MANIFEST;
  let manifest = { experiment_id: CFG.experiment_id, started: new Date().toISOString(), cells: [], nan_inf_flags: [] };
  if (doAppend && fs.existsSync(manifestPath)) {
    try { manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")); } catch (e) {}
    if (!manifest.cells) manifest.cells = [];
    if (!manifest.nan_inf_flags) manifest.nan_inf_flags = [];
  }
  let cellIdx = 0;
  const totalCells = algorithms.length * perLevels.length * noisyLevels.length * seeds.length;
  const shardTag = argAlgo ? `[shard ${argAlgo}] ` : "";

  for (const algorithm of algorithms) {
    for (const per of perLevels) {
      for (const noisy of noisyLevels) {
        const config_id = `${algorithm}${per ? "+PER" : ""}${noisy ? "+Noisy" : ""}`;
        for (const seed of seeds) {
          const env_seed = 42 + seed;
          const cellT0 = Date.now();
          const { trainRows, evalRows } = runCell({
            algorithm, per, noisy, seed, envSeed: env_seed,
            trainEpisodes, evalEpisodes, totalStepsBudget,
          });

          const base = { config_id, algorithm, per: per ? 1 : 0, noisy: noisy ? 1 : 0, seed, env_seed };

          // stream train + eval rows, checking for NaN/Inf
          let nanFlag = false;
          const checkNum = (r) => {
            for (const k of ["reward", "steps", "route_cost"]) {
              if (!isFinite(r[k])) { nanFlag = true; }
            }
          };
          const evalLines = [];
          for (const r of evalRows) { checkNum(r); evalLines.push(rowToLine(EVAL_COLS, Object.assign({}, base, r))); }
          const trainLines = [];
          for (const r of trainRows) { checkNum(r); trainLines.push(rowToLine(TRAIN_COLS, Object.assign({}, base, r))); }
          fs.appendFileSync(EVAL_CSV, evalLines.join("\n") + "\n");
          fs.appendFileSync(TRAIN_CSV, trainLines.join("\n") + "\n");

          cellIdx++;
          const cellMs = Date.now() - cellT0;
          manifest.cells.push({ config_id, seed, env_seed, trainEps: trainRows.length, evalEps: evalRows.length, ms: cellMs });
          if (nanFlag) manifest.nan_inf_flags.push({ config_id, seed });

          // progress line to stdout (captured to a log by the caller)
          const evalSucc = (100 * evalRows.reduce((a, x) => a + x.success, 0) / evalRows.length).toFixed(1);
          console.log(`${shardTag}[${cellIdx}/${totalCells}] ${config_id} seed=${seed} evalSucc=${evalSucc}% (${cellMs}ms)`);
        }
      }
    }
  }

  manifest.finished = new Date().toISOString();
  manifest.total_cells = manifest.cells.length;
  manifest.eval_rows = manifest.cells.reduce((a, c) => a + c.evalEps, 0);
  manifest.train_rows = manifest.cells.reduce((a, c) => a + c.trainEps, 0);
  fs.writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2));
  console.log(`\n${shardTag}DONE: +${cellIdx} cells this shard in ${((Date.now() - t0) / 1000).toFixed(1)}s. Total cells: ${manifest.cells.length}. NaN/Inf flags: ${manifest.nan_inf_flags.length}`);
}

main();
