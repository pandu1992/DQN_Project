/* ============================================================
 * STUDY 3 RUNNER — rule-based vs DQN, 3 conditions x 15 seeds
 * Sharded + resumable. Shard filter: --cell=Agent:condIdx
 * (condIdx 0=clean,1=mid,2=harsh). --append accumulates.
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const { runCell } = require("./harness_study3.js");

const ROOT = path.join(__dirname, "..");
const CFG = JSON.parse(fs.readFileSync(path.join(ROOT, "results_study3/configs/experiment_config_study3.json"), "utf8"));
const RAW = path.join(ROOT, "results_study3/raw");
fs.mkdirSync(RAW, { recursive: true });
const EVAL_CSV = path.join(RAW, "eval_episodes.csv");
const MANIFEST = path.join(RAW, "run_manifest.json");

const COLS = [
  "config_id", "agent", "condition", "noise_std", "packet_error_rate", "seed", "env_seed",
  "phase", "episode", "reward", "success", "steps", "invalid", "revisit",
  "route_cost", "optimal_cost", "excess_cost", "optimality_ratio", "route_risk", "route_difficulty", "route_distance",
  "collisions", "collision_flag", "cte_mean", "iala_violations", "dropped_frames", "docking_accuracy",
  "lane", "goal_lane", "start", "goal", "terminated", "truncated",
];
const line = (o) => COLS.map((c) => (o[c] === "" || o[c] == null ? "" : o[c])).join(",");

function main() {
  const t0 = Date.now();
  const argCell = (process.argv.find((a) => a.startsWith("--cell=")) || "").split("=")[1] || null;
  const doAppend = process.argv.includes("--append");

  let agents = CFG.factors.agent;
  let condIdxs = CFG.factors.condition.map((_, i) => i);
  if (argCell) { const [ag, ci] = argCell.split(":"); agents = [ag]; condIdxs = [parseInt(ci, 10)]; }

  if (!doAppend || !fs.existsSync(EVAL_CSV)) fs.writeFileSync(EVAL_CSV, COLS.join(",") + "\n");
  let manifest = { experiment_id: CFG.experiment_id, cells: [], nan_inf_flags: [] };
  if (doAppend && fs.existsSync(MANIFEST)) { try { manifest = JSON.parse(fs.readFileSync(MANIFEST, "utf8")); } catch (e) {} manifest.cells = manifest.cells || []; manifest.nan_inf_flags = manifest.nan_inf_flags || []; }

  const seeds = CFG.seeds, te = CFG.training.train_episodes, ee = CFG.evaluation.eval_episodes, budget = CFG.training.total_steps_budget, obstacleCount = CFG.environment.obstacle_count;
  let idx = 0;
  const total = agents.length * condIdxs.length * seeds.length;
  const tag = argCell ? `[${argCell}] ` : "";

  for (const agent of agents) {
    for (const ci of condIdxs) {
      const cond = CFG.factors.condition[ci];
      const config_id = `${agent}_${cond.name}`;
      for (const seed of seeds) {
        const env_seed = 42 + seed;
        const cellT0 = Date.now();
        const { evalRows } = runCell({
          agent, seed, envSeed: env_seed, trainEpisodes: te, evalEpisodes: ee,
          totalStepsBudget: budget, noiseStd: cond.noise_std, packetErrorRate: cond.packet_error_rate, obstacleCount,
        });
        const base = { config_id, agent, condition: cond.name, noise_std: cond.noise_std, packet_error_rate: cond.packet_error_rate, seed, env_seed };
        let nan = false;
        const lines = evalRows.map((r) => { if (!isFinite(r.reward) || !isFinite(r.cte_mean)) nan = true; return line(Object.assign({}, base, r)); });
        fs.appendFileSync(EVAL_CSV, lines.join("\n") + "\n");
        idx++;
        const ms = Date.now() - cellT0;
        manifest.cells.push({ config_id, seed, ms, evalEps: evalRows.length });
        if (nan) manifest.nan_inf_flags.push({ config_id, seed });
        const succ = (100 * evalRows.reduce((a, r) => a + r.success, 0) / evalRows.length).toFixed(0);
        console.log(`${tag}[${idx}/${total}] ${config_id} seed=${seed} succ=${succ}% (${ms}ms)`);
      }
    }
  }
  manifest.total_cells = manifest.cells.length;
  manifest.eval_rows = manifest.cells.reduce((a, c) => a + c.evalEps, 0);
  fs.writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2));
  console.log(`\n${tag}DONE +${idx} cells in ${((Date.now() - t0) / 1000).toFixed(1)}s. Total ${manifest.cells.length}. NaN: ${manifest.nan_inf_flags.length}`);
}
main();
