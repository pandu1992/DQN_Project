# Study 5 figure captions

## figS5_twophase_dock_vs_full

Figure. Scenario A. For each agent and condition, the inbound leg (docking, light) succeeds far more often than the FULL inbound→dock→outbound cycle (dark): the outbound leg roughly halves success for the rule-based controller and the learned DQN rarely completes it. Mean over n=8 seeds, 95% CI.

## figS5_twoway_collrate_vs_condition

Figure 2. Scenario B. Inter-vessel collision rate (%) across conditions for opposing-traffic pairings. Mean over n=8 seeds, 95% CI. Classical Rule×Rule in black (bold).

## figS5_twoway_success_vs_condition

Figure 3. Scenario B. Per-vessel success rate (%) across conditions for opposing-traffic pairings. Mean over n=8 seeds, 95% CI. Classical Rule×Rule in black (bold).

## figS5_twoway_tradeoff

Figure. Scenario B tradeoff. Each point is a (pairing, condition) cell; letters c/m/h = clean/mid/harsh. Because two vessels only encounter each other when they actually TRANSIT the shared lane, collision rate and success rise together: the classical Rule×Rule pairing (black star) both completes most and collides most; the learned pairings that barely transit sit near the origin — their low collisions are a stall artefact, not safe avoidance. Mean over n=8 seeds.

## figS5_twoway_cpa_distributions

Figure. Per-seed distribution of the episode closest-point-of-approach between the opposing vessels (box=IQR, line=median, white diamond=mean, n=8). The classical Rule×Rule pairing (left, black-edged) closes to the red collision line when it transits and meets head-on; the DQN pairings keep larger CPA largely because at least one vessel fails to transit.

## figS5_twoway_anova_eta2

Figure. Two-way factorial ANOVA partial $\eta^2$ (pairing, condition, interaction). The PAIRING factor dominates success, collision rate and closest-approach (large effects), confirming the choice of controller — not the degradation level — governs how two-way traffic resolves. Head-on events show a significant pairing×condition interaction.

