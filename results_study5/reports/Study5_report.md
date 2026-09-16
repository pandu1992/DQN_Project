# Study 5 — Two-Phase Round-Trip Missions and Two-Way Opposing Traffic

**Two realism extensions of the Synthetic Port testbed: a full inbound→dock→outbound port call, and inbound/outbound vessels sharing one channel — both under Study-2 degradation**

> **Reproducibility statement.** Every number below regenerates from
> `results_study5/raw/{twophase,twoway}_episodes.csv` via `aggregate_study5.py`
> → `stats_study5.py` → `make_tables_study5.py` → `make_figures_study5.py`. All
> mission-cycle and inter-vessel metrics are genuinely computed by
> `TwoPhaseEnvV5` / `TwoWayEnvV5` on top of `VesselEnvV3`; nothing is fabricated.
> The deployed web application (`environment.js`) is unaffected — offline research.

---

## 1. Motivation and design

Studies 1–4 all use **one-way** missions (get from A to B). Two facts about real
port operations are still missing, and Study 5 adds both:

- **A port call is a round trip.** A vessel comes IN, DOCKS, and goes back OUT.
  *(Scenario A — two-phase mission.)*
- **Channels carry traffic in both directions.** Inbound and outbound vessels
  share the same water and must pass each other. *(Scenario B — two-way traffic.)*

Both are layered on the physical/sensing/comms base (`VesselEnvV3`) and combined
with the Study-2 degradation grid (**clean** 0/0, **mid** 0.1/0.2, **harsh**
0.25/0.4). **8 shared seeds** per cell (paired), `env_seed = 42 + seed`.

**Scenario A — TwoPhaseEnvV5 (single vessel).** One episode must complete BOTH
legs: inbound (entrance → harbour berth, dock) then outbound (berth → a channel
exit). On reaching the berth the environment automatically re-targets the vessel
to the outbound goal (continuous pose preserved) and flips a phase flag in the
observation (obs 28 → 30). "Success" = the FULL cycle. Agents: the COLREGs
rule-based controller and DQN. **2 × 3 × 8 = 48 cells**, 30 eval episodes each.

**Scenario B — TwoWayEnvV5 (two vessels, opposing flow).** One inbound and one
outbound vessel are placed on the SAME access lane with overlapping mid-lane
paths, so they meet head-on. The same real inter-vessel geometry as Study 4 is
measured (collision radius 18, near-miss 40, closest-point-of-approach, plus a
head-on/give-way role from real bearing); each vessel senses the partner
(obs 28 → 31) through the noise/packet-loss pipeline. Pairings: `Rule_vs_Rule`
(classical reference), `DQN_vs_Rule` (mixed), `DQN_vs_DQN` (learned). **3 × 3 ×
8 = 72 cells**, 30 episodes × 2 vessels each. Total across both scenarios:
**5,760 evaluation rows.**

> **An honest scope note, stated up front.** On this testbed the value-based DQN
> does **not** master the full harbour transit or the round trip within budget —
> it docks occasionally but rarely completes the outbound leg, and in two-way
> traffic it often fails to transit at all. We verified this is a genuine
> property of the harbour-goal mission (a plain `VesselEnvV3` DQN also fails
> these harbour transits), not a wiring bug, and it is consistent with Study 1
> (the environment is discriminative) and Study 3 (the classical controller is
> more robust). We therefore treat the **rule-based controller as the primary,
> competent navigator** and include the DQN as a learned comparator whose
> limitation is reported transparently rather than hidden.

---

## 2. Scenario A — the outbound leg is a large, real added burden

For every agent and condition, completing the FULL cycle succeeds far less often
than merely reaching the dock:

| Agent | Condition | Dock success | Full-cycle success |
|---|---|---|---|
| Rule-based | clean / mid / harsh | **43.3%** (all) | **21.7%** (all) |
| DQN | clean | 27.9% | 5.4% |
| DQN | mid | 7.9% | 1.7% |
| DQN | harsh | 41.7% | 0.8% |

The rule-based controller's full-cycle success is **exactly half** its docking
success (43.3% → 21.7%), and this holds identically across all three degradation
conditions — the deterministic channel-follower is invariant to sensing/comms
noise (as in Study 3), but the outbound leg roughly **doubles the failure rate**.
The learned DQN docks sometimes (up to 41.7% at harsh) yet almost never completes
the round trip (≤ 5.4%): re-targeting to the outbound goal is a distribution shift
it does not handle.

**Statistics (paired dock% vs full% per agent×condition, n = 8).** The drop is a
**large** effect everywhere — Cohen's *d_z* = 1.25 (rule-based, all conditions),
0.91 / 0.61 / 1.17 (DQN clean/mid/harsh) — but at n = 8 the paired Wilcoxon does
not clear Holm correction (adj *p* = 0.17). The effect is large and consistent;
the significance is power-limited, exactly as in Study 2. We report both honestly.

---

## 3. Scenario B — two-way traffic: the pairing governs everything

| Pairing | Collision rate (clean/mid/harsh) | Min CPA (clean/mid/harsh) | Success (clean/mid/harsh) |
|---|---|---|---|
| **Rule × Rule** | **42.5 / 42.5 / 42.5%** | **92 / 93 / 94** | **45.2 / 45.2 / 45.2%** |
| DQN × Rule | 32.9 / 25.4 / 30.8% | 165 / 153 / 163 | 23.1 / 31.2 / 28.8% |
| DQN × DQN | 0.0 / 7.5 / 21.7% | 222 / 205 / 184 | 2.1 / 1.9 / 9.6% |

The story is coherent and, importantly, **honest about the DQN**:

- **Rule × Rule** is competent two-way traffic. Both vessels transit and, because
  the rule-based controller models only *static* hazards, they meet **head-on**
  and collide in ~42% of encounters, passing close (CPA ≈ 92, near the 18-unit
  collision radius given hull motion). Genuine opposing-traffic behaviour.
- **DQN × DQN** shows a *deceptively* low collision rate at clean (0%) — but this
  is a **stall artefact**: the two learned vessels barely transit (success 2.1%),
  so they never reach the shared segment to meet. As degradation *perturbs* them
  into some motion, collisions actually **rise** (0% → 21.7% clean→harsh). Low
  collisions here are the absence of traffic, not safe avoidance.
- **DQN × Rule** is intermediate: the classical vessel carries the traffic while
  the learned vessel largely fails to transit.

**Omnibus (two-way factorial ANOVA, pairing × condition + seed).** The **pairing**
dominates every outcome: success partial η² = **0.70**, collision rate **0.55**,
min-CPA **0.56** (all *p* < 0.001). Sensing/comms **condition** is small and mostly
non-significant on these outcomes. **Head-on events** show a significant
**pairing × condition interaction** (η² = 0.20, *p* = 0.013) — the degradation
changes *when/whether* the pairings actually meet.

**Primary contrast (Rule×Rule reference vs each pairing, per condition; paired
Wilcoxon + Holm + BH).** **15 / 42** contrasts are Holm-significant. At harsh,
DQN×DQN keeps a significantly larger CPA (184 vs 92, *d_z* = −2.66, Holm-sig) and
significantly lower success (9.6% vs 45.2%, *d_z* = 1.63, Holm-sig) — the same
safety-vs-completion pattern as Study 4, but here the "safety" is inseparable from
the learned pairing's failure to move.

---

## 4. Interpretation

1. **A full port call is materially harder than a one-way transit.** Requiring the
   outbound leg roughly halves the rule-based controller's success and all but
   eliminates the DQN's. Papers that report only one-way navigation success
   overstate real end-to-end capability; Study 5 quantifies the gap.

2. **Classical control carries two-way traffic; it does not avoid the partner.**
   Two rule-based vessels reliably transit in both directions but, modelling only
   static hazards, they meet head-on and collide often — the same blind-spot
   Study 4 found, now in a guaranteed opposing-flow pattern.

3. **Learned value-based control is not the answer here — and we say so.** The DQN
   fails the harbour transit, the round trip, and two-way traffic on this testbed.
   Its low two-way collision counts are a stall artefact, not competence. This is a
   genuine negative result, consistent with Studies 1 and 3, and it motivates the
   classical controller as the sensible operational baseline for these missions.

4. **The controller — not the degradation level — governs two-way outcomes**
   (pairing η² ≈ 0.55–0.70 ≫ condition), mirroring Study 4's finding that the
   agent/pairing dominates multi-vessel safety.

---

## 5. Limitations (scope honesty)

- `TwoPhaseEnvV5` / `TwoWayEnvV5` are synthesized testbeds, not validated
  hydrodynamic or COLREGs-certified simulators.
- The DQN's failure on harbour/round-trip/two-way missions is a real property of
  this testbed at the given budget; it should not be read as a general claim about
  value-based RL for maritime autonomy. The qualitative findings (outbound leg is
  hard; classical control carries two-way traffic but collides head-on) are the
  robust results.
- n = 8 gives the two-way omnibus good power but floor-limits per-cell paired tests
  (as in Study 2); the two-phase dock-vs-full drops are large but not Holm-sig at
  n = 8.
- Two-way traffic uses a single opposing pair on one shared lane; higher-density
  traffic (n > 2) is future work.
- Deployed web application is unaffected; all Study 5 code is offline research.

---

## 6. Reproduce

```bash
# Scenario A (twophase): --scn=twophase --cell=agentIdx:condIdx
node experiments/run_experiment_study5.js --scn=twophase --cell=0:0 --append
# Scenario B (twoway):   --scn=twoway   --cell=pairIdx:condIdx  [--seeds=CSV]
node experiments/run_experiment_study5.js --scn=twoway --cell=0:0 --append
# analyse
experiments/.venv/bin/python experiments/aggregate_study5.py
experiments/.venv/bin/python experiments/stats_study5.py
experiments/.venv/bin/python experiments/make_tables_study5.py
experiments/.venv/bin/python experiments/make_figures_study5.py
```

Raw data: `results_study5/raw/twophase_episodes.csv` (1,440 rows) +
`twoway_episodes.csv` (4,320 rows) + `run_manifest.json`. Aggregates, statistics,
tables and figures live under `results_study5/`.
