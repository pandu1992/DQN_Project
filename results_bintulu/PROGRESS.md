# BPAN — Bintulu Port Autonomous Navigation — COMPLETE (+ GIFs + DQN improvement)

Separate chart-grounded study, distinct from the Synthetic Port Studies 1–5.

## Core study (merged earlier, PR #25)
- Real Bintulu chart background + live sim (`bintulu.html`), environment
  (`js/environmentBintulu.js`), 96-cell experiment (4 algos × 3 cond × 8 seeds,
  2,880 eps), full paper + dashboard. Finding: algorithm choice immaterial
  (η²≈0.01–0.03); degradation dominates (collision η²=0.64, IALA 0.63).

## This update
### 1) GIFs (`results_bintulu/gifs/`, via `experiments/record_bintulu_trajectories.js` + `make_bintulu_gifs.py`)
- `clean_success.gif` (2.7 MB) — clean transit, berths.
- `mid_transit.gif` (5.6 MB) — mid degradation, drifts but completes.
- `harsh_drift.gif` (5.8 MB) — harsh degradation, drift + collision.
- Rendered over the real chart at 614×409, ≤90 frames, 64-colour palette.
- Shown in a new **Animations** tab on `bintulu_paper.html`.

### 2) DQN improvement study (`results_bintulu/raw_improve/`, `aggregated/improve_*`, `statistics/improve_*`)
- Env variant `js/environmentBintuluPlus.js` (BintuluEnvPlus): toggle-able
  `chartAware` (Dijkstra charted-route prior → obs 28→32) and `shaping`
  (potential-based reward shaping, policy-invariant).
- 4 arms (baseline / +chart / +shaping / +chart+shaping) × 3 cond × 6 seeds =
  72 cells / 2,160 eps. `harness_bintulu_improve.js`, `run_experiment_improve.js`,
  `analyze_improve.py`.
- RESULT: **the improvement arm is the dominant factor** on success (partial
  η²=0.41, p<0.001 > condition 0.32). Success clean/mid/harsh:
  baseline 15/17/25%, +shaping 20/37/46%, +chart 36/43/56%, +chart+shaping
  30/50/65%. Chart prior is the single biggest lever (d_z up to 2.6 at harsh);
  n=6 so per-cell Holm p≈0.09 (pooled ANOVA + effect sizes carry inference).
- Figure `figBPAN_improvements`, table `table6_bpan_improvements` added to the
  paper (§4.4 animations, §4.5 improvements) + dashboard.

## Notes
- Branch `bintulu/gifs-and-dqn-improve` (fresh clone `DQN_bpan2`).
- node: `$(ls /root/.nvm/versions/node/*/bin/node | head -1)`; venv in experiments/.venv.
- Deployed Synthetic Port app unaffected.
