#!/usr/bin/env python3
"""
STUDY 4 — PUBLICATION FIGURES (>=300 DPI PNG + SVG).
Centerpiece: inter-vessel collision avoidance between two INDEPENDENT vessels
across pairings and degradation conditions, and the safety-vs-completion tradeoff.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study4", "aggregated")
STAT = os.path.join(ROOT, "results_study4", "statistics")
FIG = os.path.join(ROOT, "results_study4", "figures")
os.makedirs(FIG, exist_ok=True)

summary = pd.read_csv(os.path.join(AGG, "summary.csv"))
per_seed = pd.read_csv(os.path.join(AGG, "per_seed.csv"))
rvo = pd.read_csv(os.path.join(STAT, "reference_vs_pairings.csv"))
fac = pd.read_csv(os.path.join(STAT, "factorial_anova.csv"))

PAIRINGS = ["RuleBased_vs_RuleBased", "DQN_vs_RuleBased", "DQN_vs_DQN", "DDQN_vs_DDQN"]
LAB = {"DQN_vs_DQN": "DQN\u00d7DQN", "DDQN_vs_DDQN": "DuelDbl\u00d7DuelDbl",
       "DQN_vs_RuleBased": "DQN\u00d7Rule", "RuleBased_vs_RuleBased": "Rule\u00d7Rule"}
COND = ["clean", "mid", "harsh"]
COLORS = {"RuleBased_vs_RuleBased": "#000000", "DQN_vs_RuleBased": "#C44E52",
          "DQN_vs_DQN": "#4C72B0", "DDQN_vs_DDQN": "#8172B2"}
MARK = {"RuleBased_vs_RuleBased": "*", "DQN_vs_RuleBased": "^", "DQN_vs_DQN": "o", "DDQN_vs_DDQN": "D"}
N_SEEDS = 8

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.dpi": 320, "savefig.bbox": "tight"})
CAP = 8.5
CAPTIONS = []


def save(fig, name, caption):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".svg"))
    plt.close(fig)
    CAPTIONS.append((name, caption))
    print(f"  {name}.png/.svg")


def foot():
    return (f"Mean over n={N_SEEDS} seeds; error bars = 95% CI. Two independent vessels per episode on "
            "MultiVesselEnvV4. Classical Rule\u00d7Rule pairing drawn in black (bold).")


def series(pr, metric):
    ys, es = [], []
    for c in COND:
        r = summary[(summary.pairing == pr) & (summary.condition == c)].iloc[0]
        ys.append(r[f"{metric}_mean"]); es.append(r[f"{metric}_ci95_hw"])
    return ys, es


def line_fig(metric, ylabel, name, fignum, pct=False):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for pr in PAIRINGS:
        ys, es = series(pr, metric)
        lw = 3 if pr == "RuleBased_vs_RuleBased" else 1.8
        ms = 14 if pr == "RuleBased_vs_RuleBased" else 7
        ax.errorbar([0, 1, 2], ys, yerr=es, marker=MARK[pr], color=COLORS[pr], capsize=3,
                    lw=lw, markersize=ms, label=LAB[pr], zorder=5 if pr == "RuleBased_vs_RuleBased" else 3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["clean", "mid", "harsh"])
    ax.set_xlabel("Degradation condition (sensor noise, packet-error)")
    ax.set_ylabel(ylabel)
    if pct: ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, title="Vessel pairing")
    cap = (f"Figure {fignum}. {ylabel} across degradation conditions, by vessel pairing. " + foot())
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, name, cap)


def fig_tradeoff_scatter():
    """The core safety-vs-completion tradeoff: success vs collision rate."""
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for pr in PAIRINGS:
        xs, ys = [], []
        for c in COND:
            r = summary[(summary.pairing == pr) & (summary.condition == c)].iloc[0]
            xs.append(r["vessel_collision_rate_mean"]); ys.append(r["success_rate_mean"])
        ax.plot(xs, ys, "-", color=COLORS[pr], alpha=0.5, lw=1.4, zorder=2)
        for i, c in enumerate(COND):
            ax.scatter(xs[i], ys[i], color=COLORS[pr], marker=MARK[pr],
                       s=(230 if pr == "RuleBased_vs_RuleBased" else 90),
                       edgecolor="black", linewidth=0.7, zorder=4,
                       label=LAB[pr] if i == 0 else None)
            ax.annotate(c[0], (xs[i], ys[i]), fontsize=7, ha="center", va="center",
                        color="white" if pr == "RuleBased_vs_RuleBased" else "black", zorder=5)
    ax.set_xlabel("Inter-vessel collision rate (% of encounters)  \u2192 less safe")
    ax.set_ylabel("Mission success rate (%)  \u2192 more complete")
    ax.legend(frameon=False, fontsize=9, title="Vessel pairing")
    ax.set_title("Safety\u2013completion tradeoff between two independent vessels")
    cap = ("Figure. The core Study-4 tradeoff. Each point is a (pairing, condition) cell; letters c/m/h mark "
           "clean/mid/harsh. The classical Rule\u00d7Rule pairing (black star, upper-right) completes the most "
           "missions but collides most often (it never models the moving partner). The learned/mixed pairings "
           "sit to the LOWER-LEFT: fewer inter-vessel collisions at the cost of mission completion. " + foot())
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS4_tradeoff_scatter", cap)


def bar_metric_by_pairing(metric, ylabel, name, fignum, cond="harsh", pct=False):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    vals, errs, cols = [], [], []
    for pr in PAIRINGS:
        r = summary[(summary.pairing == pr) & (summary.condition == cond)].iloc[0]
        vals.append(r[f"{metric}_mean"]); errs.append(r[f"{metric}_ci95_hw"]); cols.append(COLORS[pr])
    x = np.arange(len(PAIRINGS))
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=cols, edgecolor="black", linewidth=0.7)
    bars[0].set_hatch("//")
    ax.set_xticks(x); ax.set_xticklabels([LAB[p] for p in PAIRINGS], rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    if pct: ax.set_ylim(0, 100)
    cap = (f"Figure {fignum}. {ylabel} under the {cond.upper()} condition, by vessel pairing "
           "(classical Rule\u00d7Rule hatched). " + foot())
    fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, name, cap)


def fig_cpa_distribution():
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    data, positions, colors = [], [], []
    pos = 1
    for cond in COND:
        for pr in PAIRINGS:
            d = per_seed[(per_seed.pairing == pr) & (per_seed.condition == cond)]["min_cpa"].to_numpy(float)
            data.append(d); positions.append(pos); colors.append(COLORS[pr]); pos += 1
        pos += 1
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(18, color="red", ls="--", lw=1, alpha=0.7)
    ax.text(positions[-1], 22, "collision radius (18)", color="red", fontsize=8, ha="right")
    centers = [2.5, 7.5, 12.5]
    ax.set_xticks(centers); ax.set_xticklabels(["clean", "mid", "harsh"])
    ax.set_ylabel("Closest point of approach (logical units)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[p], alpha=0.6, ec="black") for p in PAIRINGS]
    ax.legend(handles, [LAB[p] for p in PAIRINGS], frameon=False, fontsize=8.5, ncol=4, loc="upper center")
    ax.set_title("Closest point of approach between the two vessels, by pairing and condition")
    cap = ("Figure. Per-seed distribution of the episode closest-point-of-approach (CPA) between the two vessels "
           f"(box=IQR, line=median, white diamond=mean, n={N_SEEDS}), grouped by condition. The classical Rule\u00d7Rule "
           "pairing (leftmost, black-edged) hugs the red collision-radius line, while the learned/mixed pairings keep "
           "much larger separation.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS4_cpa_distributions", cap)


def fig_anova_bars():
    """partial eta^2 of pairing / condition / interaction across the key metrics."""
    metrics = ["vessel_collision_rate", "vessel_collisions_per_ep", "near_misses_per_ep", "min_cpa", "success_rate"]
    mlab = ["Coll %", "Coll/enc", "NearMiss/enc", "Min CPA", "Success %"]
    effects = ["pairing", "condition", "pairing:condition"]
    elab = {"pairing": "Pairing", "condition": "Condition", "pairing:condition": "Pairing\u00d7Cond"}
    ecol = {"pairing": "#4C72B0", "condition": "#DD8452", "pairing:condition": "#55A868"}
    data = {e: [] for e in effects}
    for m in metrics:
        for e in effects:
            r = fac[(fac.metric == m) & (fac.effect == e)]
            data[e].append(r.iloc[0]["partial_eta2"] if len(r) else np.nan)
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, e in enumerate(effects):
        ax.bar(x + (i - 1) * w, data[e], w, label=elab[e], color=ecol[e], edgecolor="black", linewidth=0.6)
    for thr, lab in [(0.14, "large"), (0.06, "medium")]:
        ax.axhline(thr, color="gray", ls=":", lw=1)
        ax.text(len(metrics) - 0.5, thr + 0.005, lab, fontsize=7.5, color="gray", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(mlab)
    ax.set_ylabel("Partial $\\eta^2$"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Variance explained: pairing dominates inter-vessel safety")
    cap = ("Figure. Two-way factorial ANOVA partial $\\eta^2$ (pairing, condition, and their interaction) for the "
           "key outcomes. The PAIRING factor dominates every inter-vessel safety metric (large effects for "
           "collision rate, collisions/encounter and especially min CPA), while sensing/comms condition contributes "
           "a smaller but real effect on the collision metrics.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS4_anova_eta2", cap)


def main():
    print("Rendering Study 4 figures:")
    line_fig("vessel_collision_rate", "Inter-vessel collision rate (%)", "figS4_collrate_vs_condition", 1, pct=True)
    line_fig("min_cpa", "Closest point of approach (units)", "figS4_cpa_vs_condition", 2)
    fig_tradeoff_scatter()
    bar_metric_by_pairing("vessel_collisions_per_ep", "Inter-vessel collisions per encounter", "figS4_collisions_harsh", 4)
    bar_metric_by_pairing("success_rate", "Mission success rate (%)", "figS4_success_harsh", 5, pct=True)
    fig_cpa_distribution()
    fig_anova_bars()
    with open(os.path.join(FIG, "figures_captions.md"), "w") as f:
        f.write("# Study 4 figure captions\n\n")
        for name, cap in CAPTIONS:
            f.write(f"## {name}\n\n{cap}\n\n")
    print(f"captions -> results_study4/figures/figures_captions.md ({len(CAPTIONS)} figures)")


if __name__ == "__main__":
    main()
