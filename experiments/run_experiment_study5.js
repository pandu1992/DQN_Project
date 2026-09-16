/* ============================================================
 * STUDY 5 RUNNER — two-phase round trip + two-way opposing traffic.
 * Sharded + resumable.
 *   Scenario A (twophase): --scn=twophase --cell=agentIdx:condIdx
 *   Scenario B (twoway):   --scn=twoway   --cell=pairIdx:condIdx
 * --seeds=CSV limits seeds. --append accumulates.
 * Writes results_study5/raw/{twophase,twoway}_episodes.csv + run_manifest.json.
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const { runCell } = require("./harness_study5.js");

const ROOT = path.join(__dirname, "..");
const CFG = JSON.parse(fs.readFileSync(path.join(ROOT, "results_study5/configs/experiment_config_study5.json"), "utf8"));
const RAW = path.join(ROOT, "results_study5/raw");
fs.mkdirSync(RAW, { recursive: true });
const MANIFEST = path.join(RAW, "run_manifest.json");

const PHASE_CSV = path.join(RAW, "twophase_episodes.csv");
const WAY_CSV = path.join(RAW, "twoway_episodes.csv");

const PHASE_COLS = [
  "config_id", "agent", "condition", "noise_std", "packet_error_rate", "seed", "env_seed",
  "phase", "episode", "reward", "full_success", "dock_success", "final_phase", "steps", "invalid",
  "phase1_steps", "dock_accuracy", "collisions", "collision_flag", "cte_mean", "iala_violations", "dropped_frames",
  "lane", "start",
];
const WAY_COLS = [
  "config_id", "pairing", "condition", "noise_std", "packet_error_rate", "seed", "env_seed",
  "phase", "episode", "direction", "agent", "reward", "success", "steps",
  "collisions", "cte_mean", "iala_violations", "dropped_frames", "docking_accuracy", "lane",
  "vessel_collisions", "vessel_collision_flag", "collision_contact_steps", "near_misses",
  "min_cpa", "encounter_steps", "give_way_events", "head_on_events",
];
const line = (cols, o) => cols.map((c) => (o[c] === "" || o[c] == null ? "" : o[c])).join(",");

function loadManifest(doAppend) {
  let m = { experiment_id: CFG.experiment_id, cells: [], nan_inf_flags: [] };
  if (doAppend && fs.existsSync(MANIFEST)) { try { m = JSON.parse(fs.readFileSync(MANIFEST, "utf8")); } catch (e) {} m.cells = m.cells || []; m.nan_inf_flags = m.nan_inf_flags || []; }
  return m;
}
function saveManifest(m) {
  m.total_cells = m.cells.length;
  fs.writeFileSync(MANIFEST, JSON.stringify(m, null, 2));
}

function runTwoPhase(argCell, argSeeds, doAppend) {
  const agents = CFG.scenarios.twophase.agents;
  let agentIdxs = agents.map((_, i) => i);
  let condIdxs = CFG.factors.condition.map((_, i) => i);
  if (argCell) { const [ai, ci] = argCell.split(":"); agentIdxs = [parseInt(ai, 10)]; condIdxs = [parseInt(ci, 10)]; }
  const seeds = argSeeds || CFG.seeds;
  const te = CFG.training.twophase.train_episodes, ee = CFG.evaluation.eval_episodes, budget = CFG.training.twophase.total_steps_budget;

  if (!doAppend || !fs.existsSync(PHASE_CSV)) fs.writeFileSync(PHASE_CSV, PHASE_COLS.join(",") + "\n");
  const manifest = loadManifest(doAppend);
  let idx = 0; const total = agentIdxs.length * condIdxs.length * seeds.length;
  for (const ai of agentIdxs) {
    const agent = agents[ai];
    for (const ci of condIdxs) {
      const cond = CFG.factors.condition[ci];
      const config_id = `twophase_${agent}_${cond.name}`;
      for (const seed of seeds) {
        const env_seed = 42 + seed, t0 = Date.now();
        const { evalRows } = runCell({ scenario: "twophase", agent, seed, envSeed: env_seed,
          trainEpisodes: agent === "RuleBased" ? 0 : te, evalEpisodes: ee, totalStepsBudget: budget,
          noiseStd: cond.noise_std, packetErrorRate: cond.packet_error_rate, obstacleCount: 6 });
        const base = { config_id, agent, condition: cond.name, noise_std: cond.noise_std, packet_error_rate: cond.packet_error_rate, seed, env_seed };
        let nan = false;
        const lines = evalRows.map((r) => { if (!isFinite(r.reward)) nan = true; return line(PHASE_COLS, Object.assign({}, base, r)); });
        fs.appendFileSync(PHASE_CSV, lines.join("\n") + "\n");
        idx++; const ms = Date.now() - t0;
        manifest.cells.push({ scenario: "twophase", config_id, seed, ms, evalRows: evalRows.length });
        if (nan) manifest.nan_inf_flags.push({ config_id, seed });
        const full = (100 * evalRows.reduce((a, r) => a + r.full_success, 0) / evalRows.length).toFixed(0);
        const dock = (100 * evalRows.reduce((a, r) => a + r.dock_success, 0) / evalRows.length).toFixed(0);
        console.log(`[twophase ${argCell || ""}][${idx}/${total}] ${config_id} seed=${seed} dock=${dock}% full=${full}% (${ms}ms)`);
      }
    }
  }
  saveManifest(manifest);
}

function runTwoWay(argCell, argSeeds, doAppend) {
  const pairings = CFG.scenarios.twoway.pairings;
  let pairIdxs = pairings.map((_, i) => i);
  let condIdxs = CFG.factors.condition.map((_, i) => i);
  if (argCell) { const [pi, ci] = argCell.split(":"); pairIdxs = [parseInt(pi, 10)]; condIdxs = [parseInt(ci, 10)]; }
  const seeds = argSeeds || CFG.seeds;
  const te = CFG.training.twoway.train_episodes, ee = CFG.evaluation.eval_episodes, budget = CFG.training.twoway.total_steps_budget;

  if (!doAppend || !fs.existsSync(WAY_CSV)) fs.writeFileSync(WAY_CSV, WAY_COLS.join(",") + "\n");
  const manifest = loadManifest(doAppend);
  let idx = 0; const total = pairIdxs.length * condIdxs.length * seeds.length;
  for (const pi of pairIdxs) {
    const pr = pairings[pi];
    for (const ci of condIdxs) {
      const cond = CFG.factors.condition[ci];
      const config_id = `twoway_${pr.name}_${cond.name}`;
      for (const seed of seeds) {
        const env_seed = 42 + seed, t0 = Date.now();
        const { evalRowsIn, evalRowsOut } = runCell({ scenario: "twoway", pairIn: pr.in, pairOut: pr.out, seed, envSeed: env_seed,
          trainEpisodes: (pr.in === "RuleBased" && pr.out === "RuleBased") ? 0 : te, evalEpisodes: ee, totalStepsBudget: budget,
          noiseStd: cond.noise_std, packetErrorRate: cond.packet_error_rate, obstacleCount: 6 });
        const base = { config_id, pairing: pr.name, condition: cond.name, noise_std: cond.noise_std, packet_error_rate: cond.packet_error_rate, seed, env_seed };
        let nan = false;
        const rows = evalRowsIn.concat(evalRowsOut);
        const lines = rows.map((r) => { if (!isFinite(r.reward) || !isFinite(r.min_cpa)) nan = true; return line(WAY_COLS, Object.assign({}, base, r)); });
        fs.appendFileSync(WAY_CSV, lines.join("\n") + "\n");
        idx++; const ms = Date.now() - t0;
        manifest.cells.push({ scenario: "twoway", config_id, seed, ms, evalRows: rows.length });
        if (nan) manifest.nan_inf_flags.push({ config_id, seed });
        const vc = (evalRowsIn.reduce((a, r) => a + r.vessel_collisions, 0) / evalRowsIn.length).toFixed(2);
        const sI = (100 * evalRowsIn.reduce((a, r) => a + r.success, 0) / evalRowsIn.length).toFixed(0);
        const sO = (100 * evalRowsOut.reduce((a, r) => a + r.success, 0) / evalRowsOut.length).toFixed(0);
        console.log(`[twoway ${argCell || ""}][${idx}/${total}] ${config_id} seed=${seed} vColl/ep=${vc} succIn=${sI}% succOut=${sO}% (${ms}ms)`);
      }
    }
  }
  saveManifest(manifest);
}

function main() {
  const t0 = Date.now();
  const scn = (process.argv.find((a) => a.startsWith("--scn=")) || "").split("=")[1] || "twophase";
  const argCell = (process.argv.find((a) => a.startsWith("--cell=")) || "").split("=")[1] || null;
  const argSeedsRaw = (process.argv.find((a) => a.startsWith("--seeds=")) || "").split("=")[1] || null;
  const argSeeds = argSeedsRaw ? argSeedsRaw.split(",").map((s) => parseInt(s, 10)) : null;
  const doAppend = process.argv.includes("--append");
  if (scn === "twophase") runTwoPhase(argCell, argSeeds, doAppend);
  else runTwoWay(argCell, argSeeds, doAppend);
  console.log(`\nDONE (${scn}) in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
main();
