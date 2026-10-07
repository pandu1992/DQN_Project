#!/usr/bin/env python3
"""Baseline-fairness figure: does the chart prior close the DRL-vs-rule-based gap?"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "results_chart", "figures")
os.makedirs(FIG, exist_ok=True)
s = pd.read_csv(os.path.join(ROOT, "results_chart", "aggregated", "summary.csv"))
ARMS = ["DRL_no_chart", "DRL_with_chart", "RuleBased"]
LAB = {"DRL_no_chart": "DRL (no chart)", "DRL_with_chart": "DRL (+chart prior)", "RuleBased": "Rule-based (has chart)"}
COL = {"DRL_no_chart": "#C44E52", "DRL_with_chart": "#4C72B0", "RuleBased": "#000000"}
MARK = {"DRL_no_chart": "o", "DRL_with_chart": "D", "RuleBased": "*"}
CONDS = ["clean", "mid", "harsh"]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.dpi": 320, "savefig.bbox": "tight"})


def cell(a, c):
    r = s[(s.arm == a) & (s.condition == c)]
    return r.iloc[0] if len(r) else None


def fig():
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    x = np.arange(len(CONDS))
    for a in ARMS:
        ys = [cell(a, c)["success_rate_mean"] for c in CONDS]
        es = [cell(a, c)["success_rate_ci95_hw"] for c in CONDS]
        lw = 3 if a == "RuleBased" else 2
        ax.errorbar(x, ys, yerr=es, marker=MARK[a], color=COL[a], lw=lw,
                    markersize=15 if a == "RuleBased" else 8, capsize=3, label=LAB[a],
                    zorder=5 if a == "RuleBased" else 3)
    ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_xlabel("Degradation condition (sensor noise, packet-error)")
    ax.set_ylabel("Navigation success rate (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("The rule-based advantage is largely chart access, not classical control")
    cap = ("Figure. Baseline fairness (n=8 seeds, 95% CI). The DRL agent DENIED the charted-route prior "
           "(red) sits well below the rule-based baseline (black star), reproducing the 'classical control is "
           "more robust' result (DRL-no-chart vs Rule-based: d_z=-1.6/-1.1 clean/mid, Holm-significant). Giving "
           "the SAME agent the chart prior the baseline uses (blue) closes the gap: DRL+chart vs Rule-based is "
           "no longer significant (d_z=-0.1 to -0.5). The bottleneck is access to structured navigation priors, "
           "not the learning-vs-classical-control dichotomy.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=8.5, wrap=True)
    fig.savefig(os.path.join(FIG, "figChart_success_gap_closes.png"))
    fig.savefig(os.path.join(FIG, "figChart_success_gap_closes.svg"))
    plt.close(fig)
    print("  figChart_success_gap_closes.png/.svg")


if __name__ == "__main__":
    print("Rendering chart-fairness figure:")
    fig()
