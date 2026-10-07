# Q1 Pre-Submission Validity Audit

**Purpose.** Before rewriting the manuscript to a Q1 standard, this document audits
the scientific validity of the experimental pipeline against the reviewer's
concerns. Every finding is grounded in the actual code (file:line), not assumption.
Findings are graded **[BLOCKER]** (must fix before submission),
**[FIX]** (should fix / re-report), or **[OK / DOCUMENT]** (correct, but must be
stated explicitly in the manuscript).

Audited files: `js/environment.js`, `js/environmentV3.js`, `js/environmentV4.js`,
`js/environmentV5.js`, `js/rulebased.js`, `experiments/harness_v3.js`,
`experiments/harness_study{3,4,5}.js`, `experiments/run_experiment_*.js`,
`experiments/aggregate_*.py`.

---

## A. Train/evaluation independence — **[BLOCKER]**

**What the code does.** For every cell the harness builds a *single* env instance
and runs the training episodes and then the evaluation episodes against it:

```js
// experiments/harness_v3.js
const env = new VesselEnvV3(envSeed, envCfg);           // one instance
for (ep in trainEpisodes) runEpisode(env, agent, false); // train
for (ep in evalEpisodes)  runEpisode(env, agent, true);  // eval
```

`runEpisode` calls `env.reset()`, and `reset()` draws the mission from a *single
continuous* RNG stream created once at construction:

```js
// environmentV3.js
this.missionRng = makeRNG(seed + 777);      // one stream per env
_randomMission() {
  start = startPool[floor(this.missionRng() * startPool.length)];  // next draw
  goal  = goalPool [floor(this.missionRng() * goalPool.length)];
}
```

**Consequence.** Evaluation missions are **not a held-out test set**. They are the
*next i.i.d. draws* from the same generative distribution the agent trained on. The
mission space is small — `startPool` ≈ 40 non-harbour waypoints × `goalPool` = 3
terminals ≈ **~120 distinct (start, goal) pairs** — so with 130 training + 30
evaluation draws, specific (start, goal) pairs seen in training almost certainly
recur in evaluation.

**Why it is a BLOCKER for Q1 wording (not for the science).** Evaluating on fresh
i.i.d. draws from the training distribution is a legitimate measure of
*within-distribution generalization* and is common in RL. **But the manuscript
currently calls this "held-out"** ("held-out greedy evaluation",
"held-out missions from the seeded stream"), which is inaccurate and a reviewer
will flag it.

**Required action.**
1. Correct the wording everywhere: evaluation is *greedy roll-outs on fresh draws
   from the same mission distribution*, i.e. within-distribution generalization —
   not a held-out split.
2. Preferably add a **true held-out** protocol: partition the (start, goal) space
   (or the missionRng stream) into disjoint train and test sets and re-report the
   headline metrics, so the generalization claim is defensible.

---

## B. Cross-track error (CTE) is a selection/composition artifact — **[BLOCKER]**

**What the code does.** CTE is accumulated per kinematic sub-step of every edge the
vessel traverses, as the min perpendicular distance to the planned path, and the
reported `cte_mean` is `cteSum / cteSamples` over **all** episodes (the aggregation
marks it `needs_solved_only = False`). Crucially, the lateral drift that creates
CTE is scaled by the degradation: `driftScale = 60·noiseStd + 30·packetErrorRate`,
so **in the clean condition there is no drift and a successful vessel rides the
plan exactly (CTE = 0).**

**Measured evidence (Study 2, DQN baseline).**

| Condition | Success | CTE (all eps) | CTE (successful) | CTE (failed) |
|---|---|---|---|---|
| clean | 17.2% | 28.26 | **0.00** | 34.14 |
| harsh | 36.7% | 27.72 | 17.49 | 33.64 |

**Consequence.** The "CTE decreases as the environment worsens" pattern the reviewer
spotted is **not** the vessel tracking better — it is a mixture artifact:
- In clean, successful episodes have CTE ≈ 0 (no drift), so the all-episode mean is
  essentially a *failure indicator* weighted by the failure rate.
- The all-episode mean barely moves clean→harsh (28.3 → 27.7) only because the
  success-rate composition of the two sub-populations shifts.
Reporting a single all-episode `cte_mean` therefore conflates three different
things (drift magnitude, failure rate, and success/failure mixing).

**Required action.** Report CTE decomposed and never as a single aggregate:
- CTE on **successful** episodes (tracking precision when the task is completed),
- CTE on **failed** episodes,
- CTE on **all** episodes, alongside the success rate that sets the mixture,
- and state that clean-condition CTE is ~0 by construction for successful runs.

---

## C. Precision metrics are conditional on success (survivorship) — **[FIX]**

**What the code does.** `docking_accuracy` is set only when the goal is reached
(`dockingAccuracy` is `null` otherwise, per `environmentV3.js` step()), and
`optimality_ratio` is likewise defined only for solved episodes. The aggregation
computes both **over solved episodes only** (`needs_solved_only = True`).

**Consequence.** Docking accuracy and optimality ratio describe *only the episodes
that succeeded*. When success rates differ across conditions/agents, these means
compare **different sub-populations** — a survivorship bias. E.g. an agent that
succeeds 15% of the time reports docking accuracy over a different (and likely
easier) 15% than an agent that succeeds 48%.

**Required action.** Keep the solved-only metric (it is meaningful) but (i) always
report it *next to the success rate*, (ii) add the denominator (n_solved), and
(iii) avoid cross-condition/agent claims about docking/optimality without noting
the differing success bases. Consider a success-adjusted or
conditional-on-survival presentation.

---

## D. "Success" does not exclude safety violations — **[OK / DOCUMENT]**

**What the code does.** Episodes terminate only on `reachedGoal` or, if
`endOnCollision` were true, on collision — but **`endOnCollision` defaults to
`false` in every study**, and there is **no off-channel/IALA termination**. So a
vessel can collide with obstacles and/or leave the buoyed channel repeatedly and
still be scored a "success" if it reaches the goal.

**Measured evidence.** In harsh Study 2, **40.9% of *successful* DQN episodes had
≥1 obstacle collision.** Success and safety violations genuinely co-occur.

**Assessment.** This is *correct and intended* — it is precisely why the paper
reports safety metrics separately from success, and it supports the paper's
"success conceals risk" thesis. **Action:** state explicitly in Methods that
success is goal-reaching only and does not imply a collision-free or channel-
compliant trajectory, and use the 40.9% figure as direct evidence for the
reporting-practice argument.

---

## E. IALA penalty is defined but never applied — **[OK / DOCUMENT]**

**What the code does.** `REWARD_CONFIG_V3.ialaPenalty = -3.0` exists, and IALA
violations are counted (`this.ialaViolations++`), but a search of `step()` shows
the penalty is **never added to the reward**. IALA compliance is therefore a pure
*evaluation* metric that the agents are **not trained to optimize**.

**Assessment.** Legitimate (an unoptimized safety metric is a fair robustness
probe), but it must be stated: the agents have no incentive to keep the channel, so
IALA violations measure emergent, not rewarded, behaviour. This actually
*strengthens* the "sensing/comms dominate safety" story (the metric is not
confounded by reward shaping) — but hiding it would be a validity gap.

---

## F. Inter-vessel geometry (Studies 4–5) — **[OK / DOCUMENT]**

**What the code does.** Collisions/near-misses are counted as **distinct events**
(on transition into the state) to avoid saturation from single-file locked pairs;
`collision_contact_steps` records per-step contact; `min_cpa` is the episode
closest-point-of-approach. The two hulls are advanced between their pre- and
post-step poses by **linear interpolation** across the shared sub-step grid to
measure separation.

**Assessment.** Definitions are sound and honestly documented. The one caveat to
state: the inter-vessel separation uses a *first-order linear interpolation* of
each hull between macro-step endpoints (the per-vessel trajectory itself is the
real V3 kinematics); this is an approximation of the intra-step path and should be
named as such. The Study-5 "DQN pairs barely transit → low collisions is a stall
artefact" caveat is already handled correctly.

---

## G. Reward exploitability — **[OK]**

**What the code does.** Reward = stepPenalty (−0.2) + per-edge cost term
(−0.1·nav_cost) + invalid (−5) + revisit (−1) + collision (−25) + a small CTE
term + goal (+100) + docking precision bonus (≤+20) + timeout (−30). Every step
costs something; there is no positive per-step term an idle agent could farm.

**Assessment.** No obvious reward hack: stalling accrues negative step penalties and
the timeout penalty; the only large positive is goal-reaching. The Study-5 stall
behaviour is a *training-failure* mode (the agent cannot find the long harbour
transit), not reward exploitation. OK.

---

## H. Seed / env_seed design — **[OK]**

**What the code does.** `env_seed = 42 + seed`; the same `env_seed` (hence the same
obstacle field and mission stream) is used across all conditions for a given seed,
and the agent-side RNG is seeded separately. Conditions are therefore compared on
the **same** environments and missions — a correct paired/randomized-block design.

**Assessment.** Sound. The one thing to make explicit: because `missionRng` depends
on `env_seed` and is shared across conditions, all conditions for a seed see the
*same* mission sequence — this is what makes the pairing valid, and it is correct.

---

## I. Inference unit and "31,150 episodes" — **[FIX — framing]**

The pipeline correctly treats the **per-seed scalar** as the unit of analysis, so
the effective sample size is the seed count (n = 6 for Study 2, 15 for Study 3,
8–10 elsewhere), **not** the ~31k episodes. The manuscript must sell 31,150 as
*computational replication / reproducibility*, and base all inference on seed-level
replication. (Also: complement the OLS factorial ANOVA with a mixed-effects model
that treats seed as a random effect — see Task 5.)

---

## Summary of required actions before writing

| # | Finding | Grade | Action |
|---|---|---|---|
| A | Eval not held out from training | BLOCKER | Fix wording; add a true held-out split |
| B | CTE is a selection/composition artifact | BLOCKER | Report CTE by success/failure/all + success rate |
| C | Docking/optimality are solved-only | FIX | Report with success base + n_solved |
| D | Success ≠ safe (40.9% collide) | DOCUMENT | State in Methods; use as evidence |
| E | IALA penalty defined but unused | DOCUMENT | State agents don't optimize IALA |
| F | Inter-vessel interpolation approximation | DOCUMENT | Name the first-order approximation |
| G | Reward exploitability | OK | — |
| H | Seed/env_seed paired design | OK | State pairing explicitly |
| I | 31,150 ≠ statistical power | FIX | Frame as reproducibility; add mixed model |

**Bottom line.** The experimental machinery is honest and the metrics are genuinely
computed. The two BLOCKERS are *reporting/validity framing* issues (held-out
language; single-aggregate CTE), both fixable without discarding data. Tasks 2–5
address the FIX items (CTE decomposition + failure taxonomy, chart-fairness,
external validity, mixed-effects inference) and Task 7 rewrites the manuscript with
the corrected framing.


---

## Addendum — Task 2 corrections produced (outcome decomposition)

The BLOCKER-B and FIX-C items are addressed by re-analyzing the existing raw CSVs
(no re-run needed; they already carry `terminated`, `truncated`, `collisions`,
`success`, `cte_mean`). New artifacts:

- `experiments/audit_outcome_decomposition.py` → per-study
  `aggregated/outcome_decomposition_{per_seed,summary}.csv` and
  `aggregated/failure_taxonomy.csv`.
- `results_audit/figures/figA_cte_outcome_correction.{png,svg}` — the misleading
  all-episode CTE vs the corrected CTE-on-success (monotone rise with degradation).
- `results_audit/figures/figB_success_hides_collisions.{png,svg}` — % of *successful*
  episodes that still collided.
- `results_audit/tables/tableAudit_*.csv`.

**Confirmed numbers.**
- Failure taxonomy: **0 non-timeout failures** in Study 2 (n=6,480) and Study 3
  (n=6,750). Timeout is the sole failure mode; collisions/IALA are orthogonal
  within-episode safety metrics.
- CTE-on-success rises **monotonically** with degradation for every controller
  (e.g. Study 3 Rule-based 0.0 → 5.9 → 13.2 clean/mid/harsh; DQN 0.4 → 7.3 → 17.8),
  whereas the all-episode aggregate is non-monotone (the artifact).
- Success hides safety: under harsh degradation ~35–47% of *successful* episodes
  had ≥1 collision.

These corrected, decomposed metrics — not the single all-episode aggregates — are
what the rewritten manuscript (Task 7) will report.
