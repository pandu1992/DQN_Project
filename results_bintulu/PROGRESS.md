# BPAN — Bintulu Port Autonomous Navigation — progress

**Status: IN PROGRESS** (saved before disconnect). Resume tonight.

## Done
- `assets/bintulu/bintulu_chart.png` — real Bintulu approach chart (copied from the uploaded `assets/schematics/map_3.png`), used as the live-sim background.
- `js/environmentBintulu.js` — `BintuluEnv`: waypoint graph on the chart's native 1536×1024 frame; North Access Channel (buoys NO1'–NO9'/N1'–N9') and South Access Channel (R2'–R10'/G2'–G10'), **extended into the harbour berths** (Southern Jetty / Container Terminal / Inner Harbour 1 & 2) so missions complete (fixes the chart's channel lines stopping in open water). Reuses the project's physics/sensing/comms + metric contract. 23 nodes, 20 buoys, all 6 start→goal missions solvable.
- `js/bintulu_main.js` + `bintulu.html` — live simulation rendering the vessel/waypoints/buoys/obstacles over the real chart background. Verified: loads, trains, telemetry advances, goal berth renders at Inner Harbour 2 / Southern Jetty. Separate from the Synthetic Port sim (`index.html`).
- `experiments/harness_bintulu.js`, `run_experiment_bintulu.js`, `results_bintulu/configs/experiment_config_bintulu.json` — 4 algorithms × 3 conditions × 8 seeds = 96 cells.

## Experiment data collected: 80/96 cells (2,400 rows), all complete, 0 NaN
Done (240 rows each): DQN {clean,mid,harsh}, DoubleDQN {clean,mid,harsh},
DuelingDQN {clean,mid,harsh}, DuelingDoubleDQN {clean}.

## Remaining to run tonight (2 shards)
```
NODE=$(ls /root/.nvm/versions/node/*/bin/node | head -1)
cd DQN_bintulu
$NODE experiments/run_experiment_bintulu.js --cell=3:1 --append   # DuelingDoubleDQN_mid
$NODE experiments/run_experiment_bintulu.js --cell=3:2 --append   # DuelingDoubleDQN_harsh
# (Dueling variants ~34s/cell -> ~5 min each shard; run one per call)
```
After both, each config should be 240 rows (96 cells, 2,880 rows total). Rebuild
the manifest, then: aggregate → stats → tables → figures → **write the BPAN paper**
(`bintulu_paper.html` + `manuscript`), add a landing link, PR + deploy + verify live.

## Early signal (self-test, DQN seed 0)
clean: success 40%, CTE|success 14.6, collisions 0.13/ep, IALA 0.0;
harsh: success 17%, CTE|success 19.8, collisions 1.20/ep, IALA 19.7 — degradation
clearly hurts safety/compliance on the real chart, same qualitative pattern as the
Synthetic Port studies.

## Notes
- Work on branch `bintulu/bpan-study` (fresh clone `DQN_bintulu`; the old
  `DQN_Project` checkout is corrupted — always start from a fresh clone).
- node: `$(ls /root/.nvm/versions/node/*/bin/node | head -1)`.
- Naming confirmed: this study is **"Bintulu Port Autonomous Navigation (BPAN)"**,
  kept separate from Synthetic Port Studies 1–5 (which stay rebranded).
