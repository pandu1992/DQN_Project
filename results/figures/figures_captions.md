# Figure captions

## fig1_overall_performance

Figure 1. Overall navigation success rate and mean episodic reward for the four base value-based algorithms (PER off, Noisy off). Estimator: mean over n=10 seeds; error bars = 95% CI (Student-t). Each seed = 40 held-out greedy evaluation episodes on VesselEnvV2 (graph-navigation MDP). Real metrics only. Overlapping CIs indicate no significant algorithm effect (RM-ANOVA/Friedman, p>0.4).

## fig2_success_across_configs

Figure 2. Navigation success rate (%) across all 16 configurations (4 algorithms x PER x Noisy Nets). Estimator: mean over n=10 seeds; error bars = 95% CI (Student-t). Each seed = 40 held-out greedy evaluation episodes on VesselEnvV2 (graph-navigation MDP). Real metrics only.

## fig3_reward_across_configs

Figure 3. Mean episodic reward across all 16 configurations (4 algorithms x PER x Noisy Nets). Estimator: mean over n=10 seeds; error bars = 95% CI (Student-t). Each seed = 40 held-out greedy evaluation episodes on VesselEnvV2 (graph-navigation MDP). Real metrics only.

## fig4_per_seed_distributions

Figure 4. Distribution of per-seed performance for the four base algorithms (box = IQR, line = median, white diamond = mean, points = individual seeds). n=10 seeds; each seed = 40 eval episodes. The large spread (e.g. one DQN seed at 0% success) shows run-to-run variance dominates algorithm choice.

## fig5_ci_comparison

Figure 5. Forest plot of baseline success rate with 95% confidence intervals (n=10 seeds). All intervals overlap the grand mean, consistent with no significant algorithm effect. Estimator: mean over n=10 seeds; error bars = 95% CI (Student-t). Each seed = 40 held-out greedy evaluation episodes on VesselEnvV2 (graph-navigation MDP). Real metrics only.

## fig6_significance_matrix

Figure 6. Pairwise algorithm comparison on success rate: (a) paired Cohen's $d_z$ (row minus column), (b) Holm-adjusted paired-Wilcoxon p-values. All adjusted p = 1.00: no pair differs significantly after correction. n=10 seeds.

## fig7_enhancement_tradeoff

Figure 7. Effect of each enhancement on route optimality ratio (change vs the no-enhancement baseline; lower ratio = closer to Dijkstra optimum = better). Noisy Nets tends to reduce the ratio (better routes) for DQN and Dueling Double DQN, but no contrast survives Holm correction. Estimator: mean over n=10 seeds; error bars = 95% CI (Student-t). Each seed = 40 held-out greedy evaluation episodes on VesselEnvV2 (graph-navigation MDP). Real metrics only.

## fig8_variance_dominance

Figure 8. Partial $\eta^2$ (variance explained) from the factorial ANOVA (y ~ algorithm*PER*Noisy + seed). The Seed block explains the most variance for success and reward (partial $\eta^2\approx0.20$), exceeding all design factors: run-to-run variability dominates algorithm/enhancement choice. Balanced design, 10 seeds/cell.

