# BPAN — Bintulu Port Autonomous Navigation — COMPLETE

**Status: COMPLETE.** Separate chart-grounded study, distinct from the Synthetic
Port Studies 1–5.

## Delivered
- `assets/bintulu/bintulu_chart.png` — real Bintulu approach chart (background).
- `js/environmentBintulu.js` (`BintuluEnv`) — waypoint graph on the chart frame;
  North + South Access Channels extended into the harbour berths; full
  physics/sensing/comms + metric contract.
- `bintulu.html` + `js/bintulu_main.js` — live simulation over the real chart.
- Experiment: 4 algorithms × 3 conditions × 8 seeds = **96 cells / 2,880 episodes**
  (`experiments/{harness,run_experiment,analyze}_bintulu.js|.py`,
  `results_bintulu/{configs,raw,aggregated,statistics,tables,figures}`).
- **Full paper**: `results_bintulu/reports/BPAN_paper.md` (intro → conclusion) +
  dashboard `bintulu_paper.html`.
- Landing-page link added to the BPAN simulation.

## Headline results
- Algorithm choice is immaterial (partial η² ≈ 0.01–0.03 on every metric, n.s.).
- Degradation condition dominates safety/precision: collision-rate η²=0.64,
  IALA η²=0.63, docking-accuracy η²=0.79, success η²=0.18 (all p<0.001).
- Clean→harsh (mean over variants): IALA 0→22.7/ep, collisions 0.19→1.63/ep,
  CTE|success 2.9→25.2, dropped frames 0→19.8.

## Notes
- Branch `bintulu/bpan-study` (fresh clone). node: `$(ls /root/.nvm/versions/node/*/bin/node | head -1)`.
- Deployed Synthetic Port app unaffected; BPAN adds separate pages.
