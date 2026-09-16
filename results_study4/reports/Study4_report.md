# Study 4 — Two Independent Autonomous Vessels That Must Avoid Each Other

**A multi-agent extension of the Synthetic Port sensing/comms testbed: inter-vessel collision avoidance under Study-2 degradation**

> **Reproducibility statement.** Every quantitative claim below is regenerated
> from `results_study4/raw/eval_episodes.csv` by the scripts in `experiments/`
> (`aggregate_study4.py` → `stats_study4.py` → `make_tables_study4.py` →
> `make_figures_study4.py`). All inter-vessel geometry (collisions, near-misses,
> closest-point-of-approach) and every per-vessel metric are genuinely computed
> by `MultiVesselEnvV4` / `VesselEnvV3`; nothing is fabricated. The deployed web
> application (`environment.js`) is unaffected — this is an offline research build.

---

## 1. Motivation and design

Studies 1–3 evaluated a **single** vessel navigating an empty channel. A single
agent never has to negotiate a moving partner, so the safety-critical event was
"hull hits a static obstacle." Study 4 adds the essential multi-agent question:

> When **two independent autonomous vessels** — each with its own policy and its
> own mission, with **no central controller** — share the same port channel
> network, do they **avoid each other**, and does a *learned* policy avoid a
> moving partner as well as a *classical* COLREGs-aware controller?

**Environment (`js/environmentV4.js`, `MultiVesselEnvV4`).** Two `VesselEnvV3`
vessels (A and B) are composed over the **same** synthesized graph, obstacle
field and buoys. They step in **lockstep**; on every macro-step the real
Euclidean separation between the two hulls is measured across synchronized
continuous sub-steps. The new safety events are:

- **inter-vessel collision** — hull separation drops below 18 logical units;
- **near-miss / close-quarters** — separation below 40 (but above collision);
- **closest point of approach (CPA)** — the episode minimum separation.

Collisions and near-misses are counted as **distinct events** (on the transition
*into* the state), so two vessels locked single-file on the same channel are not
counted forty times; the reward penalty, by contrast, is applied every step the
hulls remain in contact, so a persistently unsafe situation keeps costing both
agents. Each vessel additionally observes the **sensed** relative position and
range of its partner (3 extra observation values), passed through the *same*
sensor-noise + packet-loss pipeline as the rest of its observation — so an agent
can perceive and react to the other vessel, but only through degraded sensing.
A COLREGs-style give-way / stand-on role is computed from real relative bearing
for attribution; it is recorded, not imposed.

Each vessel keeps **all** of its own Study-2 metrics (static-obstacle collisions,
cross-track error, IALA channel-keeping, docking accuracy, dropped comms frames).
Vessel B's mission is re-rolled at reset so the two vessels never share a berth
(a single dock cannot host both).

**Factors.** Four vessel **pairings** × three **degradation conditions**:

| Pairing | Vessel A | Vessel B | Meaning |
|---|---|---|---|
| `DQN_vs_DQN` | DQN | DQN | two independent learners (symmetric) |
| `DDQN_vs_DDQN` | Dueling Double DQN | Dueling Double DQN | two independent learners, best variant |
| `DQN_vs_RuleBased` | DQN | Rule-based (COLREGs) | mixed: learned vs classical |
| `RuleBased_vs_RuleBased` | Rule-based | Rule-based | two classical controllers (reference) |

Conditions: **clean** (noise 0, PER 0), **mid** (0.1, 0.2), **harsh** (0.25, 0.4).
**8 shared seeds** per cell (paired design), `env_seed = 42 + seed`. RL agents
train 150 two-vessel episodes **against the live partner** (self-play style),
then are evaluated greedily for 30 episodes; the rule-based controller never
learns. **4 × 3 × 8 = 96 cells**; each cell yields 30 episodes × 2 vessels = 60
per-vessel rows, for **5,760 per-vessel evaluation rows (2,880 encounters)**.

---

## 2. Headline result — a safety-vs-completion tradeoff

The central finding is a clean tradeoff between **inter-vessel safety** and
**mission completion**, and it is driven overwhelmingly by the *pairing*:

| Pairing | Collision rate (clean → harsh) | Min CPA (clean → harsh) | Success % (clean → harsh) |
|---|---|---|---|
| **Rule-based × Rule-based** | **40.8% → 37.9%** | **118 → 120** | **48.3% → 48.3%** |
| DQN × Rule-based | 23.3% → 25.8% | 186 → 192 | 26.9% → 27.3% |
| DQN × DQN | 15.8% → 24.6% | 224 → 199 | 15.6% → 21.7% |
| Dueling-Double × Dueling-Double | 13.8% → 29.6% | 233 → 194 | 13.5% → 17.5% |

Two classical vessels **complete the most missions** (~48%, invariant to
degradation) but **collide with each other most often** (38–41% of encounters)
and pass **closest** (CPA ≈ 118, barely above the 18-unit collision radius given
hull motion). This is expected and honest: the rule-based controller only avoids
*static* obstacles — it never models the moving partner — so two of them follow
their charted channels straight into each other.

The **learned** pairings do the opposite. Trained with the partner in their
observation and an inter-vessel collision penalty, they keep the vessels **much
farther apart** (CPA ≈ 190–230) and collide **less** (14–30% of encounters), but
at a **large cost to mission completion** (14–27%). The learned agents spend
policy capacity on avoidance and on the harder 31-dimensional multi-vessel task.

Figure `figS4_tradeoff_scatter` visualises this directly: the classical pairing
sits alone in the upper-right (complete but unsafe); the learned/mixed pairings
sit to the lower-left (safer but less complete).

---

## 3. Statistical analysis

**Design.** Two-way factorial: pairing (4) × condition (3), shared 8 seeds
(randomized-block/paired). Unit of analysis = per-seed cell scalar (n = 8).
Inter-vessel metrics are per-encounter; navigation/physical metrics are averaged
over both vessels. Assumptions checked with Shapiro–Wilk and Levene.

### 3.1 Omnibus — the pairing dominates inter-vessel safety

Two-way factorial ANOVA (`y ~ pairing*condition + seed`, Type-II, partial η²):

| Metric | Pairing η² (p) | Condition η² (p) | Pairing×Cond η² (p) |
|---|---|---|---|
| Min CPA | **0.70** (<0.001) | 0.05 (0.14) | 0.09 (0.30) |
| Inter-vessel collision rate | **0.52** (<0.001) | 0.11 (0.012) | 0.15 (0.052) |
| Collisions / encounter | **0.50** (<0.001) | 0.14 (0.003) | 0.15 (0.043) |
| Near-misses / encounter | **0.18** (0.001) | 0.02 (0.54) | 0.07 (0.47) |
| Mission success rate | **0.69** (<0.001) | 0.06 (0.077) | 0.04 (0.73) |

The **pairing** factor is the overwhelming driver of every inter-vessel safety
outcome (large effects: min-CPA η² = 0.70, collision-rate η² = 0.52,
collisions/encounter η² = 0.50) and of mission success (η² = 0.69). Sensing/comms
**condition** contributes a smaller but real effect on the collision metrics
(η² ≈ 0.11–0.14, p < 0.02) — degradation makes collisions somewhat more frequent
— but it does not move CPA or success significantly. The **interaction** is
modest and mostly non-significant, i.e. the pairings degrade roughly in parallel.

### 3.2 Primary contrast — classical reference vs each pairing

Paired Wilcoxon (exact) of the classical `RuleBased_vs_RuleBased` reference
against each learned/mixed pairing, per condition, with Holm and BH correction
within each (metric, condition) family, plus Cohen's *d_z* and Cliff's δ.
**49 of 90** contrasts are Holm-significant. At the **harshest** condition:

| Metric | Reference vs | Ref×2 | Pairing | *d_z* | Holm-sig? |
|---|---|---|---|---|---|
| Min CPA | DQN×DQN | 119.7 | 198.8 | −1.66 | **yes** |
| Min CPA | DuelDbl×DuelDbl | 119.7 | 193.7 | −1.49 | **yes** |
| Min CPA | DQN×Rule | 119.7 | 192.4 | −1.56 | **yes** |
| Success % | DQN×DQN | 48.3 | 21.7 | +2.00 | **yes** |
| Success % | DuelDbl×DuelDbl | 48.3 | 17.5 | +1.96 | **yes** |
| Success % | DQN×Rule | 48.3 | 27.3 | +1.99 | **yes** |

The two directions of the tradeoff are both **large and significant**: the
learned/mixed pairings keep a significantly larger CPA (safer separation), while
the classical pairing completes significantly more missions. Collision-rate
differences point the same way (learned pairings collide less) with large *d_z*
but sit just above the Holm threshold at n = 8 for some cells.

### 3.3 Robustness — clean vs harsh per pairing

Per-pairing clean→harsh paired tests show the expected direction (learned
pairings accrue somewhat more collisions and slightly smaller CPA under harsh
sensing; e.g. DuelDbl×DuelDbl collision rate 13.8% → 29.6%, *d_z* = 1.72), but
**none survive Holm correction at n = 8** — exactly the power limitation seen in
Study 2 (n = 6). The pooled omnibus, which has more degrees of freedom, *does*
detect the condition effect on collisions. The classical pairing is, as in
Study 3, essentially **invariant** to degradation (success 48.3% → 48.3%).

---

## 4. Interpretation

1. **Two independent learners genuinely avoid each other.** Given the partner in
   their (degraded) observation and a collision penalty, DQN/Dueling-Double
   vessels learn to keep large separation — roughly **doubling** the closest
   point of approach relative to two classical controllers, and cutting the
   inter-vessel collision rate. This is the multi-agent capability Study 4 set
   out to test, and it is real and statistically strong.

2. **The classical controller is blind to the moving partner.** Its channel-
   following policy is robust and completes missions reliably, but because it
   only reasons about static hazards it drives two vessels into repeated close-
   quarters situations. In a shared channel this is the least safe pairing.

3. **Avoidance costs completion.** The learned agents pay for their separation
   with a large drop in mission success. On this compact testbed, with a modest
   training budget and a harder 31-dimensional multi-vessel observation, learning
   *both* to complete the mission *and* to yield to a partner is hard; the agents
   prioritise avoidance (which the reward penalises most heavily).

4. **Degradation makes encounters somewhat more dangerous, in parallel.**
   Sensor noise and packet loss raise collision frequency (omnibus condition
   effect), and they do so for the learned pairings without a strong pairing×
   condition interaction — the safety ordering between pairings is preserved as
   the environment degrades.

---

## 5. Limitations (scope honesty)

- `MultiVesselEnvV4` is a **synthesized** multi-vessel testbed, not a validated
  hydrodynamic or COLREGs-certified simulator; the give-way role is a geometric
  heuristic for attribution, not a rule engine.
- Two vessels only; no n > 2 traffic (that is Study 5's two-way traffic case).
- n = 8 seeds gives the omnibus good power but leaves per-cell paired robustness
  tests floor-limited (as in Study 2).
- The inter-vessel replay uses a first-order linear interpolation of each hull
  between its pre- and post-step pose; the underlying per-vessel trajectory and
  all Study-2 metrics are unchanged by this measurement.
- The learned agents' low completion is partly a training-budget artefact; the
  qualitative safety-vs-completion tradeoff is the robust finding, not the
  absolute completion numbers.
- Deployed web application is unaffected; all Study 4 code is offline research.

---

## 6. Reproduce

```bash
# 1. run (sharded, resumable): --cell=pairingIdx:condIdx, optional --seeds=
node experiments/run_experiment_study4.js --cell=0:0 --append
# 2. analyse
experiments/.venv/bin/python experiments/aggregate_study4.py
experiments/.venv/bin/python experiments/stats_study4.py
experiments/.venv/bin/python experiments/make_tables_study4.py
experiments/.venv/bin/python experiments/make_figures_study4.py
```

Raw data: `results_study4/raw/eval_episodes.csv` (5,760 rows) +
`run_manifest.json`. Aggregates, statistics, tables and figures live under
`results_study4/`.
