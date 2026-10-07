#!/usr/bin/env python3
"""
Audit figure: CTE by episode outcome (Study 3), showing the corrected,
survivorship-free view versus the misleading all-episode aggregate.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "results_audit", "figures")
os.makedirs(FIG, exist_ok=True)

s = pd.read_csv(os.path.join(ROOT, "results_study3", "aggregated", "outcome_decomposition_summary.csv"))
AGENTS = ["RuleBased", "DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
LAB = {"RuleBased": "Rule-based", "DQN": "DQN", "DoubleDQN": "Double", "DuelingDQN": "Dueling", "DuelingDoubleDQN": "Duel-Dbl"}
CONDS = ["clean", "mid", "harsh"]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.dpi": 320, "savefig.bbox": "tight"})


def cell(ag, c):
    r = s[(s.agent == ag) & (s.condition == c)]
    return r.iloc[0] if len(r) else None


def fig_cte_outcome():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # Left: the MISLEADING all-episode CTE (looks flat / non-monotone)
    # Right: the CORRECT CTE-on-success (monotone rise with degradation)
    for ax, key, title, ylab in [
        (axes[0], "cte_all_mean", "Misleading: all-episode CTE", "CTE (all episodes)"),
        (axes[1], "cte_success_mean", "Corrected: CTE on SUCCESSFUL episodes", "CTE | success"),
    ]:
        x = np.arange(len(CONDS))
        for i, ag in enumerate(AGENTS):
            ys = [cell(ag, c)[key] for c in CONDS]
            es = [cell(ag, c)[key.replace("_mean", "_ci95_hw")] for c in CONDS]
            lw = 3 if ag == "RuleBased" else 1.7
            ax.errorbar(x, ys, yerr=es, marker=["*","o","s","^","D"][i], lw=lw,
                        markersize=12 if ag == "RuleBased" else 6, capsize=3, label=LAB[ag],
                        color="#000" if ag == "RuleBased" else None,
                        zorder=5 if ag == "RuleBased" else 3)
        ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_xlabel("Degradation condition")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=12)
        ax.legend(frameon=False, fontsize=8.5, ncol=2)
    cap = ("Figure. The cross-track-error artifact and its correction (Study 3, n=15). "
           "LEFT: the all-episode CTE mean is non-monotone / roughly flat across degradation — a "
           "selection/composition artifact, because clean successful episodes have CTE=0 by "
           "construction (no drift) and the aggregate is a success-rate-weighted mixture of a ~0 "
           "successful sub-population and a large failed sub-population. RIGHT: CTE computed only on "
           "SUCCESSFUL episodes rises monotonically with degradation for every controller — the true "
           "tracking-precision loss. The single all-episode aggregate must not be reported alone.")
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=8.5, wrap=True)
    fig.savefig(os.path.join(FIG, "figA_cte_outcome_correction.png"))
    fig.savefig(os.path.join(FIG, "figA_cte_outcome_correction.svg"))
    plt.close(fig)
    print("  figA_cte_outcome_correction.png/.svg")


def fig_success_hides_collisions():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(CONDS))
    for i, ag in enumerate(AGENTS):
        ys = [cell(ag, c)["collision_rate_success_mean"] for c in CONDS]
        lw = 3 if ag == "RuleBased" else 1.7
        ax.plot(x, ys, marker=["*","o","s","^","D"][i], lw=lw,
                markersize=12 if ag == "RuleBased" else 6, label=LAB[ag],
                color="#000" if ag == "RuleBased" else None, zorder=5 if ag == "RuleBased" else 3)
    ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_xlabel("Degradation condition")
    ax.set_ylabel("% of SUCCESSFUL episodes with \u22651 collision"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("Success hides safety violations")
    cap = ("Figure. Among episodes scored a SUCCESS (goal reached), the fraction that nonetheless "
           "suffered at least one obstacle collision. Under harsh degradation ~35-47% of successful "
           "runs collided: 'success' (goal-reaching) does not imply a safe trajectory, motivating "
           "separate safety reporting. (endOnCollision=false: collisions never terminate the episode.)")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=8.5, wrap=True)
    fig.savefig(os.path.join(FIG, "figB_success_hides_collisions.png"))
    fig.savefig(os.path.join(FIG, "figB_success_hides_collisions.svg"))
    plt.close(fig)
    print("  figB_success_hides_collisions.png/.svg")


def main():
    print("Rendering audit figures:")
    fig_cte_outcome()
    fig_success_hides_collisions()


if __name__ == "__main__":
    main()
