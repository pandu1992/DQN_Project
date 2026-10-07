## figOPS_twophase_full
Full round-trip success (inbound → dock → outbound) by agent across conditions (6 seeds, 95% CI). The classical rule-based planner closes the full cycle 100% of the time at every condition; the naive DQN docks but rarely completes the return leg (≈16–19%) — a control-architecture gap that is flat across degradation.

## figOPS_twophase_dock
Leg-1 docking success (reaching the inner berth) by agent and condition. Both controllers dock reliably in clean water; the gap in Figure "full" is therefore the *return* leg, not the initial approach.

## figOPS_twophase_coll
Static-obstacle collisions per episode during the round trip. Collisions rise with the degradation condition for both agents — the single-vessel degradation effect persists underneath the round-trip task.

## figOPS_twoway_success
Transit success for opposing inbound/outbound traffic sharing one access channel. The rule/rule pair always completes; the learned-inbound pairing takes a modest success hit under degradation (100% → 80%).

## figOPS_twoway_headon
Head-on encounter events per episode. The learned inbound vessel (blue) cuts head-on events below the rule/rule structural baseline (red ≈ 1.0) by learning a temporal give-way (holding to let the oncoming vessel pass); the residual inter-vessel collision is structural on a single centreline.

## figOPS_twoway_vcoll
Inter-vessel collisions per episode for two-way traffic. The count stays ≈1/episode regardless of pairing or condition — on a single shared centreline there is no lateral room to pass, so two opposing vessels that both insist on transiting must meet (a structural null the algorithm cannot remove).
