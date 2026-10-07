# Literature Positioning & Contribution Reframe (Q1 audit, Task 6)

*Grounded in a targeted review of the maritime-DRL, robustness, and RL-methodology
literature. All sources paraphrased (≤30 words verbatim); inline links given.
Content was rephrased for compliance with licensing restrictions.*

## 1. What the literature already does

**Maritime DRL for navigation / COLREGs collision avoidance is a mature, active
area.** Numerous works learn COLREGs-aware collision-avoidance or path-following
policies for unmanned surface vehicles (USVs) and maritime autonomous surface
ships (MASS), spanning value-based, actor–critic, and distributional RL:
- COLREG-compliant USV collision avoidance with DRL ([Meyer et al., arXiv 2006.09540](https://arxiv.org/html/2006.09540v1)); risk-based COLREGs via DRL ([arXiv 2112.00115](https://arxiv.org/html/2112.00115v1)).
- Multi-vessel COLREGs-compliant DRL decision-making ([Zhang et al., JMSE 12(3):372, 2024](https://www.mdpi.com/2077-1312/12/3/372/xml)); COLREGs path planning for USV fleets ([JMSE 11(12):2334, 2023](https://www.mdpi.com/2077-1312/11/12/2334)); decentralized multi-ASV via distributional RL ([arXiv 2402.11799](https://arxiv.org/pdf/2402.11799v2)).
- Improved soft actor–critic for autonomous surface vessels ([JMSE 11(8):1554, 2023](https://www.mdpi.com/2077-1312/11/8/1554)); a safety-layer / dual-mode PPO for MASS ([Nature Sci. Rep. 2026](https://www.nature.com/articles/s41598-026-62506-2)).

**Robustness to imperfect perception is emerging but usually studied in
isolation.** Recent work benchmarks DRL navigation under sensor noise / denial
([arXiv 2410.14616](https://arxiv.org/html/2410.14616)), evaluates RL robustness for autonomous shipping in a simulator
([arXiv 2411.04915](https://arxiv.org/html/2411.04915v1)), and mitigates observational noise with distributionally robust
RL ([arXiv 2512.00030](https://arxiv.org/html/2512.00030)). These typically manipulate *one* stressor (noise) and rarely
report channel-compliance/docking safety metrics alongside success.

**Communication failure is almost never modelled in maritime RL.** Where it
appears at all, it is in connected-vehicle / platoon MARL (packet loss, delay)
([DCT-MARL, arXiv 2508.12633](https://arxiv.org/abs/2508.12633v1); [MTCC, arXiv 2311.11281](https://arxiv.org/pdf/2311.11281.pdf)), and a 2026 stress-test paper notes that cooperative
embodied-AI is "almost universally evaluated under idealized communication"
([arXiv 2603.20285](https://arxiv.org/html/2603.20285v1)). Communication packet-loss as a robustness factor for a *single*
learned vessel is essentially unaddressed.

**Classical priors improve learned navigation.** Outside maritime, guiding RL
with classical planners/heuristics improves sample efficiency, generalization,
and safety ([Regularized RL + classical planning, arXiv 2403.18524](https://arxiv.org/html/2403.18524v1); [heuristics as dense rewards, arXiv 2109.14830](https://arxiv.org/html/2109.14830v2); [hybrid classical/RL planner, arXiv 2410.03066](https://arxiv.org/html/2410.03066v1)). This motivates treating the *navigation prior* as a first-class factor.

**RL methodology work demands what most maritime papers omit.** The
reproducibility literature is explicit: report multiple seeds, significance
tests, and effect sizes, and treat the seed as a unit of variance
([Henderson et al., "Deep RL that Matters", arXiv 1709.06560](https://arxiv.org/html/1709.06560); [Colas et al., "How Many Random Seeds?", arXiv 1806.08295](https://arxiv.org/html/1806.08295v2); ["Hitchhiker's Guide to Statistical Comparisons of RL", arXiv 1904.06979](https://arxiv.org/html/1904.06979v2); [Patterson et al., "Empirical Design in RL", arXiv 2304.01315](https://arxiv.org/html/2304.01315v1)). A 2026 maritime path-planning review independently flags "inconsistent benchmarks, limited cross-scenario generalization, and insufficient full-scale validation" as persistent barriers ([JMSE 14(16):1477, 2026](https://www.mdpi.com/2077-1312/14/16/1477/xml)).

## 2. The gap

Existing maritime-DRL studies tend to (i) evaluate a **single stressor** under
otherwise nominal conditions, (ii) report **task success** without safety/
precision decomposition, (iii) use **few seeds without significance/effect-size
statistics**, and (iv) compare a learned policy against a classical baseline
**without controlling the information (chart/plan) each receives**. What is
missing is a *controlled, statistically disciplined* characterization of **how
much each level of the design stack — algorithmic refinement, perception/
communication quality, control architecture / navigation prior, multi-vessel
interaction, and end-to-end mission structure — actually contributes to safe,
complete autonomous vessel navigation**, on a common testbed with a fair
baseline.

> **Gap statement (for the manuscript).** Prior work establishes that DRL *can*
> navigate and avoid collisions, and that it *can* be brittle to isolated
> perturbations; it has not systematically quantified the **relative** influence
> of algorithm, perception/communication degradation, control architecture, and
> mission structure on safety and completion, nor disentangled a classical
> baseline's robustness from its information advantage.

## 3. Capability comparison (honest)

Legend: ✓ = a common/central feature of that line of work; ~ = present but
limited/rare; ✗ = essentially absent. Columns are *representative families* from
§1, not single papers.

| Dimension | Maritime DRL nav/COLREGs (e.g. JMSE 2023–24; arXiv 2006.09540, 2112.00115) | DRL robustness-to-noise (arXiv 2410.14616, 2411.04915, 2512.00030) | Connected-vehicle MARL comms (arXiv 2508.12633, 2603.20285) | RL-methodology (arXiv 1709.06560, 1806.08295, 1904.06979) | **This work** |
|---|---|---|---|---|---|
| DRL algorithm comparison | ✓ | ~ | ~ | ✓ | ✓ (diagnostic) |
| Sensor / observation noise | ✗ | ✓ | ~ | ✗ | ✓ |
| Communication packet-loss | ✗ | ✗ | ✓ (vehicles, not maritime) | ✗ | ✓ |
| Safety metrics beyond success (collision, channel/IALA, docking, CTE) | ~ | ~ | ~ | ✗ | ✓ (outcome-decomposed) |
| Classical baseline | ~ | ~ | ✗ | ✗ | ✓ |
| **Baseline information (chart) controlled as a factor** | ✗ | ✗ | ✗ | ✗ | **✓** |
| Multiple shared random seeds + significance + effect sizes | ~ | ~ | ~ | ✓ (prescribed) | ✓ |
| Mixed-effects / seed-as-random inference | ✗ | ✗ | ~ | ~ | ✓ |
| Multi-vessel interaction | ✓ | ✗ | ✓ | ✗ | ✓ |
| Round-trip / two-way-traffic mission structure | ~ | ✗ | ✗ | ✗ | ✓ |
| Cross-map external-validity check | ~ | ~ | ~ | ✗ | ✓ (3 maps) |
| Factor-hierarchy characterization across the stack | ✗ | ✗ | ✗ | ✗ | **✓** |

The distinctive combination is the **bottom rows**: a controlled, fair-baseline,
statistically-disciplined decomposition of *where* robustness and completion come
from — culminating in a factor hierarchy — rather than a new algorithm.

## 4. Reframed contribution & research questions

**Central question (revised).** *Under controlled degradation and increasing
operational complexity, which factors determine the safety, reliability, and
completion of DRL-based autonomous vessel navigation — and how do they rank?*

- **RQ1 (diagnostic).** Does algorithmic refinement within the value-based DRL
  family materially affect performance once seed variability is controlled?
- **RQ2 (perception/communication).** How do sensor noise and communication
  packet-loss affect safety and precision relative to algorithmic variation?
- **RQ3 (control architecture / navigation prior).** Is a classical baseline's
  robustness advantage inherent, or an artifact of the chart/plan information it
  is given — i.e. does supplying the learner the same navigation prior close the
  gap?
- **RQ4 (interaction & mission structure).** How do multi-vessel encounters and
  round-trip / two-way-traffic missions change conclusions drawn from
  single-vessel, one-way benchmarks?

**Headline finding — a factor hierarchy.** On this testbed the influence on safe,
complete navigation is ordered, from strongest to weakest:

> **mission structure & multi-vessel interaction ≳ control architecture /
> navigation prior ≳ perception & communication quality ≫ value-based algorithmic
> refinement.**

Supporting effect sizes (partial η²): algorithm ≈ 0.01–0.05 (Studies 1–2);
perception/comms up to 0.80–0.86 on channel compliance (Study 2); control
architecture / chart prior ≈ 0.30 on success (chart-fairness study); multi-vessel
pairing 0.52–0.70 (Study 4); mission structure (a full round trip roughly halves
success; Study 5). The chart-fairness result reframes the classical-vs-learned
debate: the baseline's advantage is **largely an information-access effect** — a
representation/architecture question — not an inherent property of classical
control.

## 5. Title candidates (phenomenon-first, not report-like)

1. **What Makes Autonomous Vessel Navigation Robust? A Controlled Factor-Hierarchy
   Study of Deep Reinforcement Learning under Degraded Sensing, Communication, and
   Multi-Vessel Interaction.**
2. **Beyond Navigation Success: Where Robustness, Safety, and Mission Completion
   Come From in Deep Reinforcement Learning for Autonomous Vessels.**
3. **The Chart, Not the Controller: Disentangling Information Priors from
   Learning-vs-Classical Control in Robust Autonomous Vessel Navigation.**

(Recommended: #1 — it names the factor-hierarchy contribution and avoids looking
like "another DQN comparison".)
