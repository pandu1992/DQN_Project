# Study 3 figure captions

## figS3_success_vs_condition

Figure 1. Navigation success rate (%) across degradation conditions, rule-based baseline vs DQN variants. Mean over n=15 seeds; error bars = 95% CI. Each seed = 30 eval episodes on VesselEnvV3. Rule-based drawn in black (bold).

## figS3_collisions_vs_condition

Figure 2. Collisions per episode across degradation conditions, rule-based baseline vs DQN variants. Mean over n=15 seeds; error bars = 95% CI. Each seed = 30 eval episodes on VesselEnvV3. Rule-based drawn in black (bold).

## figS3_iala_interaction

Figure. IALA channel-departure violations vs degradation, per agent. The lines FAN OUT (significant agent×condition interaction, partial η²=0.49): the DQN variants' violations rise far more steeply than the rule-based baseline's — the baseline keeps the channel better as sensing/comms degrade. Mean over n=15 seeds; error bars = 95% CI. Each seed = 30 eval episodes on VesselEnvV3. Rule-based drawn in black (bold).

## figS3_iala_harsh

Figure 4. IALA violations per episode under the HARSHEST condition (noise 0.25, packet-error 0.4). Rule-based baseline (hatched) vs DQN variants. Mean over n=15 seeds; error bars = 95% CI. Each seed = 30 eval episodes on VesselEnvV3. Rule-based drawn in black (bold).

## figS3_success_harsh

Figure 5. Navigation success rate (%) under the HARSHEST condition (noise 0.25, packet-error 0.4). Rule-based baseline (hatched) vs DQN variants. Mean over n=15 seeds; error bars = 95% CI. Each seed = 30 eval episodes on VesselEnvV3. Rule-based drawn in black (bold).

## figS3_effect_matrix

Figure. Paired effect size (Cohen's $d_z$) of the rule-based baseline minus each DQN variant, per metric and condition; * marks Holm-significant contrasts (n=15). For IALA (channel-keeping) the baseline is strongly and significantly better under mid/harsh degradation.

## figS3_iala_distributions

Figure. Per-seed distribution of IALA violations (box=IQR, line=median, white diamond=mean, n=15), grouped by condition, one box per agent. Under mid/harsh the rule-based baseline (black-edged, leftmost in each group) sits well below the DQN variants.

