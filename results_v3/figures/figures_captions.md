# V3 figure captions

## figV3_success_vs_noise

Figure 1. Navigation success rate (%) vs sensor noise (averaged over packet-error levels), per algorithm. Mean over n=6 seeds; error bars/bands = 95% CI. Each seed = 30 eval episodes on VesselEnvV3. Nearly parallel curves indicate the algorithms degrade similarly (no algorithm x noise interaction).

## figV3_collision_vs_noise

Figure 2. Collision rate (%) vs sensor noise (averaged over packet-error levels), per algorithm. Mean over n=6 seeds; error bars/bands = 95% CI. Each seed = 30 eval episodes on VesselEnvV3. Nearly parallel curves indicate the algorithms degrade similarly (no algorithm x noise interaction).

## figV3_iala_vs_per

Figure 3. IALA violations per episode vs packet-error rate (averaged over noise levels), per algorithm. Mean over n=6 seeds; error bars/bands = 95% CI. Each seed = 30 eval episodes on VesselEnvV3.

## figV3_docking_vs_per

Figure 4. Docking error (distance) vs packet-error rate (averaged over noise levels), per algorithm. Mean over n=6 seeds; error bars/bands = 95% CI. Each seed = 30 eval episodes on VesselEnvV3.

## figV3_iala_heatmap

Figure 5. IALA violations / episode: mean over the 4 algorithms and 6 seeds across the noise x packet-error grid. Mean over n=6 seeds; error bars/bands = 95% CI. Each seed = 30 eval episodes on VesselEnvV3.

## figV3_collision_heatmap

Figure 6. Collision rate (%): mean over the 4 algorithms and 6 seeds across the noise x packet-error grid. Mean over n=6 seeds; error bars/bands = 95% CI. Each seed = 30 eval episodes on VesselEnvV3.

## figV3_effect_sizes

Figure. Partial $\eta^2$ from the three-way factorial ANOVA (y ~ algorithm*noise*PER + seed). Noise and packet-error dominate the safety/precision metrics (IALA, docking); the Algorithm effect and Algorithm x stressor interactions are small — algorithms degrade in parallel. Balanced design, 6 seeds/cell.

## figV3_robustness_ranking

Figure. Robustness ranking: mean absolute paired effect size (Cohen's $d_z$) of degradation from clean to the harshest condition (noise 0.25, PER 0.4), averaged over collision rate, IALA, docking error and CTE. Lower = more robust. Differences are modest and mostly non-significant (parallel degradation).

## figV3_collision_distributions

Figure. Per-seed distribution of collisions per episode (box = IQR, line = median, white diamond = mean, n=6 seeds), grouped by sensor-noise level, one box per algorithm (averaged over PER within each cell). Collisions rise with noise for all algorithms; spread reflects seed variance.

