## figMULTI_anova_eta2
Partial η² per factor for the multi-vessel study (two independent vessels sharing the Bintulu channels). The interaction **pairing** (Rule/Rule, DQN/Rule, DQN/DQN) explains the large majority of the variance in success, near-misses and inter-vessel collisions; the degradation **condition** and the pairing×condition interaction are comparatively small on the inter-vessel outcomes.

## figMULTI_success
Transit success by pairing across degradation conditions (6 seeds, 95% CI). The three pairings separate cleanly — Rule/Rule completes most, DQN/DQN least — and are essentially flat across degradation, showing the gap between them is the interaction structure rather than the noise.

## figMULTI_vcoll
Inter-vessel collisions per episode by pairing and condition. The classical Rule/Rule pair collides most (neither vessel yields); pairing a learner with the rule-follower cuts collisions; two learners collide most under harsh degradation when the sensed partner channel is noisiest.

## figMULTI_cpa
Mean minimum closest-point-of-approach (CPA, px) by pairing and condition. Larger CPA means the two vessels keep more water between them; the DQN/Rule pairing holds the largest separation.
