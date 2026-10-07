"use strict";
/* BPAN extension (e) — operational realism runner.
 *   twophase : single vessel inbound -> dock -> outbound round trip.
 *              factors = agent {RuleBased, DQN} x condition {clean,mid,harsh} x 6 seeds.
 *   twoway   : opposing inbound/outbound vessels sharing one access channel.
 *              factors = pairing {RB/RB, DQN/RB} x condition x 6 seeds.
 * Writes per-episode rows (twophase) and per-episode-per-direction rows (twoway)
 * to results_bintulu/raw_ops/. Each cell uses a seeded RNG so results are
 * byte-reproducible. Usage:
 *   node run_bintulu_ops.js                 # all cells, fresh files
 *   node run_bintulu_ops.js --scenario=twophase
 *   node run_bintulu_ops.js --cell=twoway:1:2 --append   # pairing idx 1, cond idx 2
 */
const fs = require("fs"), path = require("path");
const { runCell } = require("./harness_bintulu_ops.js");
const ROOT = path.join(__dirname, "..");
const RAW = path.join(ROOT, "results_bintulu/raw_ops"); fs.mkdirSync(RAW, { recursive: true });
const CSV_PHASE = path.join(RAW, "twophase_episodes.csv");
const CSV_WAY = path.join(RAW, "twoway_episodes.csv");
const MAN = path.join(RAW, "run_manifest.json");

const CONDS = [["clean", 0, 0], ["mid", 0.12, 0.2], ["harsh", 0.25, 0.4]];
const SEEDS = [0, 1, 2, 3, 4, 5];
const PHASE_AGENTS = ["RuleBased", "DQN"];
const WAY_PAIRS = [["RuleBased", "RuleBased"], ["DQN", "RuleBased"]];
const WAY_PNAME = ["RuleBased_vs_RuleBased", "DQN_vs_RuleBased"];

const TRAIN_PHASE = 220, EVAL_PHASE = 30, BUDGET_PHASE = 220 * 60;
const TRAIN_WAY = 200, EVAL_WAY = 30, BUDGET_WAY = 200 * 60;

const COLS_PHASE = ["config_id", "scenario", "agent", "condition", "noise_std", "packet_error_rate", "seed", "env_seed", "phase", "episode", "full_success", "dock_success", "reward", "dock_accuracy", "collisions", "cte_mean", "iala_violations"];
const COLS_WAY = ["config_id", "scenario", "pairing", "condition", "noise_std", "packet_error_rate", "seed", "env_seed", "phase", "episode", "direction", "agent", "success", "vessel_collisions", "vessel_collision_flag", "near_misses", "min_cpa", "head_on_events", "iala_violations"];
const line = (cols) => (o) => cols.map((c) => (o[c] === "" || o[c] == null ? "" : o[c])).join(",");
const linePhase = line(COLS_PHASE), lineWay = line(COLS_WAY);

function main() {
  const t0 = Date.now();
  const argScenario = (process.argv.find((a) => a.startsWith("--scenario=")) || "").split("=")[1] || null;
  const argCell = (process.argv.find((a) => a.startsWith("--cell=")) || "").split("=")[1] || null;
  const doAppend = process.argv.includes("--append");
  let man = { experiment_id: "bpan_operational_realism", cells: [] };
  if (doAppend && fs.existsSync(MAN)) { try { man = JSON.parse(fs.readFileSync(MAN, "utf8")); } catch (e) {} man.cells = man.cells || []; }

  // ---------------- TWO-PHASE ROUND TRIP ----------------
  const runPhase = (!argScenario || argScenario === "twophase") && (!argCell || argCell.startsWith("twophase"));
  if (runPhase) {
    let ais = PHASE_AGENTS.map((_, i) => i), cis = CONDS.map((_, i) => i);
    if (argCell && argCell.startsWith("twophase")) { const [, a, c] = argCell.split(":"); ais = [+a]; cis = [+c]; }
    if (!doAppend || !fs.existsSync(CSV_PHASE)) fs.writeFileSync(CSV_PHASE, COLS_PHASE.join(",") + "\n");
    let idx = 0, total = ais.length * cis.length * SEEDS.length;
    for (const ai of ais) for (const ci of cis) {
      const agent = PHASE_AGENTS[ai], [cn, ns, per] = CONDS[ci], config_id = `twophase_${agent}_${cn}`;
      for (const seed of SEEDS) {
        const env_seed = 42 + seed, c0 = Date.now();
        const { rows } = runCell({ scenario: "twophase", agent, seed, envSeed: env_seed,
          trainEpisodes: agent === "RuleBased" ? 0 : TRAIN_PHASE, evalEpisodes: EVAL_PHASE, budget: BUDGET_PHASE, noiseStd: ns, packetErrorRate: per });
        const base = { config_id, scenario: "twophase", agent, condition: cn, noise_std: ns, packet_error_rate: per, seed, env_seed };
        fs.appendFileSync(CSV_PHASE, rows.map((r) => linePhase(Object.assign({}, base, r))).join("\n") + "\n");
        idx++; man.cells.push({ config_id, seed, ms: Date.now() - c0 });
        const full = (100 * rows.reduce((a, r) => a + r.full_success, 0) / rows.length).toFixed(0);
        console.log(`[twophase ${idx}/${total}] ${config_id} seed=${seed} full=${full}% (${Date.now() - c0}ms)`);
      }
    }
  }

  // ---------------- TWO-WAY OPPOSING TRAFFIC ----------------
  const runWay = (!argScenario || argScenario === "twoway") && (!argCell || argCell.startsWith("twoway"));
  if (runWay) {
    let pis = WAY_PAIRS.map((_, i) => i), cis = CONDS.map((_, i) => i);
    if (argCell && argCell.startsWith("twoway")) { const [, p, c] = argCell.split(":"); pis = [+p]; cis = [+c]; }
    if (!doAppend || !fs.existsSync(CSV_WAY)) fs.writeFileSync(CSV_WAY, COLS_WAY.join(",") + "\n");
    let idx = 0, total = pis.length * cis.length * SEEDS.length;
    for (const pi of pis) for (const ci of cis) {
      const [pa, pb] = WAY_PAIRS[pi], [cn, ns, per] = CONDS[ci], config_id = `twoway_${WAY_PNAME[pi]}_${cn}`;
      for (const seed of SEEDS) {
        const env_seed = 42 + seed, c0 = Date.now();
        const rl = pa === "RuleBased" && pb === "RuleBased";
        const { rIn, rOut } = runCell({ scenario: "twoway", pairIn: pa, pairOut: pb, seed, envSeed: env_seed,
          trainEpisodes: rl ? 0 : TRAIN_WAY, evalEpisodes: EVAL_WAY, budget: BUDGET_WAY, noiseStd: ns, packetErrorRate: per });
        const base = { config_id, scenario: "twoway", pairing: WAY_PNAME[pi], condition: cn, noise_std: ns, packet_error_rate: per, seed, env_seed };
        fs.appendFileSync(CSV_WAY, rIn.concat(rOut).map((r) => lineWay(Object.assign({}, base, r))).join("\n") + "\n");
        idx++; man.cells.push({ config_id, seed, ms: Date.now() - c0 });
        const vc = (rIn.reduce((a, r) => a + r.vessel_collisions, 0) / rIn.length).toFixed(2);
        const ho = (rIn.reduce((a, r) => a + r.head_on_events, 0) / rIn.length).toFixed(2);
        console.log(`[twoway ${idx}/${total}] ${config_id} seed=${seed} vColl/ep=${vc} headOn/ep=${ho} (${Date.now() - c0}ms)`);
      }
    }
  }

  man.total_cells = man.cells.length; fs.writeFileSync(MAN, JSON.stringify(man, null, 2));
  console.log(`\nDONE in ${((Date.now() - t0) / 1000).toFixed(1)}s. Manifest cells: ${man.cells.length}.`);
}
main();
