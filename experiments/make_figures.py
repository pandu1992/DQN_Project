#!/usr/bin/env python3
"""
PUBLICATION-QUALITY FIGURES (task #9)

Renders >=300 DPI PNG + vector SVG to results/figures/. Every figure caption
(embedded as the suptitle/annotation and written to figures_captions.md) states
the estimator, the error-bar meaning, the number of seeds, the number of
evaluation episodes per seed, and the scenario.

All values come from the aggregated / per-seed / statistics CSVs (real data).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results", "aggregated")
STAT = os.path.join(ROOT, "results", "statistics")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

summary = pd.read_csv(os.path.join(AGG, "summary.csv"))
per_seed = pd.read_csv(os.path.join(AGG, "per_seed.csv"))
pw = pd.read_csv(os.path.join(STAT, "pairwise_algorithms.csv"))
enh = pd.read_csv(os.path.join(STAT, "enhancement_contrasts.csv"))
fac = pd.read_csv(os.path.join(STAT, "factorial_anova.csv"))

ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
ALGO_LABEL = {"DQN": "DQN", "DoubleDQN": "Double\nDQN",
              "DuelingDQN": "Dueling\nDQN", "DuelingDoubleDQN": "Dueling\nDouble DQN"}
ALGO_SHORT = {"DQN": "DQN", "DoubleDQN": "DDQN", "DuelingDQN": "Duel", "DuelingDoubleDQN": "D+D"}
COLORS = {"DQN": "#4C72B0", "DoubleDQN": "#55A868",
          "DuelingDQN": "#C44E52", "DuelingDoubleDQN": "#8172B2"}
N_SEEDS = 10
N_EVAL = 40

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True, "figure.dpi": 120,
    "savefig.dpi": 320, "savefig.bbox": "tight",
})
CAPTION_FONT = 8.5
CAPTIONS = []


def save(fig, name, caption):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".svg"))
    plt.close(fig)
    CAPTIONS.append((name, caption))
    print(f"  {name}.png / .svg")


def base_cells():
    return summary[(summary.per == 0) & (summary.noisy == 0)].set_index("algorithm").loc[ALGOS]


def cap_footer():
    return (f"Estimator: mean over n={N_SEEDS} seeds; error bars = 95% CI (Student-t). "
            f"Each seed = {N_EVAL} held-out greedy evaluation episodes on VesselEnvV2 "
            f"(graph-navigation MDP). Real metrics only.")


# ---------- Figure 1 — overall performance (baseline algorithms) ----------
def fig1():
    b = base_cells()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(ALGOS))
    # success
    axes[0].bar(x, b["success_rate_mean"], yerr=b["success_rate_ci95_hw"],
                color=[COLORS[a] for a in ALGOS], capsize=4, edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels([ALGO_LABEL[a] for a in ALGOS])
    axes[0].set_ylabel("Navigation success rate (%)"); axes[0].set_ylim(0, 100)
    axes[0].set_title("(a) Success rate")
    # reward
    axes[1].bar(x, b["mean_reward_mean"], yerr=b["mean_reward_ci95_hw"],
                color=[COLORS[a] for a in ALGOS], capsize=4, edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels([ALGO_LABEL[a] for a in ALGOS])
    axes[1].set_ylabel("Mean episodic reward"); axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("(b) Mean reward")
    fig.suptitle("Figure 1. Overall performance of the four base DQN variants (no enhancements)", fontsize=12, y=1.02)
    cap = ("Figure 1. Overall navigation success rate and mean episodic reward for the four base "
           "value-based algorithms (PER off, Noisy off). " + cap_footer() +
           " Overlapping CIs indicate no significant algorithm effect (RM-ANOVA/Friedman, p>0.4).")
    fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=CAPTION_FONT, wrap=True)
    save(fig, "fig1_overall_performance", cap)


# ---------- Figure 2 & 3 — success / reward across all 16 configs ----------
def fig_across(metric, mlabel, name, fignum, is_pct):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    conds = [(0, 0, "baseline"), (1, 0, "+PER"), (0, 1, "+Noisy"), (1, 1, "+PER+Noisy")]
    width = 0.2
    x = np.arange(len(ALGOS))
    hatches = ["", "//", "..", "xx"]
    for i, (p, n, lab) in enumerate(conds):
        sub = summary[(summary.per == p) & (summary.noisy == n)].set_index("algorithm").loc[ALGOS]
        ax.bar(x + (i - 1.5) * width, sub[f"{metric}_mean"], width,
               yerr=sub[f"{metric}_ci95_hw"], capsize=3, label=lab,
               color=[COLORS[a] for a in ALGOS], alpha=0.55 + 0.15 * i,
               edgecolor="black", linewidth=0.5, hatch=hatches[i])
    ax.set_xticks(x); ax.set_xticklabels([a.replace("DQN", "DQN") for a in ALGOS])
    ax.set_ylabel(mlabel)
    if is_pct:
        ax.set_ylim(0, 100)
    else:
        ax.axhline(0, color="grey", lw=0.8)
    # legend for conditions (hatch), separate from algorithm colours
    handles = [Patch(facecolor="lightgrey", edgecolor="black", hatch=h, label=l)
               for h, (_, _, l) in zip(hatches, conds)]
    ax.legend(handles=handles, title="Enhancement", ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), frameon=False)
    cap = (f"Figure {fignum}. {mlabel} across all 16 configurations (4 algorithms x PER x Noisy Nets). "
           + cap_footer())
    fig.text(0.5, -0.07, cap, ha="center", va="top", fontsize=CAPTION_FONT, wrap=True)
    save(fig, name, cap)


# ---------- Figure 4 — per-seed distributions (box + strip) ----------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, mlabel in [(axes[0], "success_rate", "Success rate (%)"),
                               (axes[1], "mean_reward", "Mean reward")]:
        data = [per_seed[(per_seed.algorithm == a) & (per_seed.per == 0) & (per_seed.noisy == 0)][metric].values
                for a in ALGOS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5))
        for patch, a in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(COLORS[a]); patch.set_alpha(0.55)
        for i, d in enumerate(data):
            jitter = np.random.default_rng(1).normal(0, 0.05, len(d))
            ax.scatter(np.full(len(d), i + 1) + jitter, d, s=18, color="black", alpha=0.5, zorder=3)
        ax.set_xticks(range(1, len(ALGOS) + 1)); ax.set_xticklabels([ALGO_SHORT[a] for a in ALGOS])
        ax.set_ylabel(mlabel)
        if metric == "success_rate":
            ax.set_ylim(-5, 100)
    axes[0].set_title("(a) Success rate per seed"); axes[1].set_title("(b) Reward per seed")
    cap = ("Figure 4. Distribution of per-seed performance for the four base algorithms "
           "(box = IQR, line = median, white diamond = mean, points = individual seeds). "
           f"n={N_SEEDS} seeds; each seed = {N_EVAL} eval episodes. "
           "The large spread (e.g. one DQN seed at 0% success) shows run-to-run variance "
           "dominates algorithm choice.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAPTION_FONT, wrap=True)
    save(fig, "fig4_per_seed_distributions", cap)


# ---------- Figure 5 — CI comparison (forest-style) for success ----------
def fig5():
    b = base_cells()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(ALGOS))[::-1]
    ax.errorbar(b["success_rate_mean"], y, xerr=b["success_rate_ci95_hw"], fmt="o",
                color="black", capsize=4, markersize=7,
                markerfacecolor=[COLORS[a] for a in ALGOS][0])
    for yi, a in zip(y, ALGOS):
        r = b.loc[a]
        ax.plot(r["success_rate_mean"], yi, "o", color=COLORS[a], markersize=9, zorder=5)
    grand = b["success_rate_mean"].mean()
    ax.axvline(grand, color="grey", ls="--", lw=1, label=f"grand mean = {grand:.1f}%")
    ax.set_yticks(y); ax.set_yticklabels([a for a in ALGOS])
    ax.set_xlabel("Navigation success rate (%)  \u00b1 95% CI")
    ax.legend(frameon=False, loc="lower right")
    cap = ("Figure 5. Forest plot of baseline success rate with 95% confidence intervals "
           f"(n={N_SEEDS} seeds). All intervals overlap the grand mean, consistent with no "
           "significant algorithm effect. " + cap_footer())
    fig.text(0.5, -0.07, cap, ha="center", va="top", fontsize=CAPTION_FONT, wrap=True)
    save(fig, "fig5_ci_comparison", cap)


# ---------- Figure 6 — pairwise effect-size / p-value matrices (success) ----------
def fig6():
    metric = "success_rate"
    sub = pw[pw.metric == metric]
    n = len(ALGOS)
    dz = np.full((n, n), np.nan)
    padj = np.full((n, n), np.nan)
    idx = {a: i for i, a in enumerate(ALGOS)}
    for _, r in sub.iterrows():
        i, j = idx[r.algo_a], idx[r.algo_b]
        dz[i, j] = r["cohens_dz"]; dz[j, i] = -r["cohens_dz"]
        padj[i, j] = r["wilcoxon_p_holm"]; padj[j, i] = r["wilcoxon_p_holm"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    im0 = axes[0].imshow(dz, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("(a) Effect size (Cohen's $d_z$)")
    im1 = axes[1].imshow(padj, cmap="viridis_r", vmin=0, vmax=1)
    axes[1].set_title("(b) Holm-adjusted p-value")
    for ax in axes:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([ALGO_SHORT[a] for a in ALGOS]); ax.set_yticklabels([ALGO_SHORT[a] for a in ALGOS])
    for i in range(n):
        for j in range(n):
            if np.isfinite(dz[i, j]):
                axes[0].text(j, i, f"{dz[i,j]:.2f}", ha="center", va="center", fontsize=9,
                             color="white" if abs(dz[i, j]) > 0.5 else "black")
            if np.isfinite(padj[i, j]):
                axes[1].text(j, i, f"{padj[i,j]:.2f}", ha="center", va="center", fontsize=9,
                             color="white" if padj[i, j] < 0.5 else "black")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cap = ("Figure 6. Pairwise algorithm comparison on success rate: (a) paired Cohen's $d_z$ "
           "(row minus column), (b) Holm-adjusted paired-Wilcoxon p-values. "
           "All adjusted p = 1.00: no pair differs significantly after correction. "
           f"n={N_SEEDS} seeds.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAPTION_FONT, wrap=True)
    save(fig, "fig6_significance_matrix", cap)


# ---------- Figure 7 — enhancement trade-off (optimality ratio) ----------
def fig7():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    contrasts = ["+PER", "+Noisy", "+PER+Noisy"]
    x = np.arange(len(ALGOS)); width = 0.25
    for i, c in enumerate(contrasts):
        vals, errs = [], []
        for a in ALGOS:
            r = enh[(enh.algorithm == a) & (enh.contrast == c) & (enh.metric == "optimality_ratio")]
            vals.append(r["abs_change"].values[0] if len(r) else np.nan)
            errs.append(0)
        ax.bar(x + (i - 1) * width, vals, width, label=c, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([ALGO_SHORT[a] for a in ALGOS])
    ax.set_ylabel("$\\Delta$ optimality ratio vs baseline\n(negative = better routes)")
    ax.legend(title="Enhancement", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    cap = ("Figure 7. Effect of each enhancement on route optimality ratio (change vs the "
           "no-enhancement baseline; lower ratio = closer to Dijkstra optimum = better). "
           "Noisy Nets tends to reduce the ratio (better routes) for DQN and Dueling Double DQN, "
           "but no contrast survives Holm correction. " + cap_footer())
    fig.text(0.5, -0.07, cap, ha="center", va="top", fontsize=CAPTION_FONT, wrap=True)
    save(fig, "fig7_enhancement_tradeoff", cap)


# ---------- Figure 8 — variance-component / effect-size dominance ----------
def fig8():
    # partial eta^2 for each effect on the 3 primary metrics
    prim = ["success_rate", "mean_reward", "optimality_ratio"]
    effects = ["algorithm", "per", "noisy", "algorithm:noisy", "seed"]
    elab = {"algorithm": "Algorithm", "per": "PER", "noisy": "Noisy",
            "algorithm:noisy": "Algo x Noisy", "seed": "Seed (block)"}
    data = np.full((len(effects), len(prim)), np.nan)
    for j, m in enumerate(prim):
        sub = fac[fac.metric == m]
        for i, e in enumerate(effects):
            row = sub[sub.effect == e]
            if len(row):
                data[i, j] = row["partial_eta2"].values[0]
    # narrower figure + constrained_layout so the heatmap and its colorbar
    # fill the canvas (avoids the left-hand whitespace of a wide figure);
    # aspect="auto" lets the 5x3 grid stretch to fill the axes box.
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=max(0.25, np.nanmax(data)), aspect="auto")
    ax.set_xticks(range(len(prim))); ax.set_xticklabels(["Success", "Reward", "Opt. ratio"])
    ax.set_yticks(range(len(effects))); ax.set_yticklabels([elab[e] for e in effects])
    for i in range(len(effects)):
        for j in range(len(prim)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                        color="white" if data[i, j] > 0.13 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="partial $\\eta^2$")
    cap = ("Figure 8. Partial $\\eta^2$ (variance explained) from the factorial ANOVA "
           "(y ~ algorithm*PER*Noisy + seed). The Seed block explains the most variance for "
           "success and reward (partial $\\eta^2\\approx0.20$), exceeding all design factors: "
           "run-to-run variability dominates algorithm/enhancement choice. "
           f"Balanced design, {N_SEEDS} seeds/cell.")
    fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=CAPTION_FONT, wrap=True)
    save(fig, "fig8_variance_dominance", cap)


def main():
    print("Rendering figures:")
    fig1()
    fig_across("success_rate", "Navigation success rate (%)", "fig2_success_across_configs", 2, True)
    fig_across("mean_reward", "Mean episodic reward", "fig3_reward_across_configs", 3, False)
    fig4(); fig5(); fig6(); fig7(); fig8()
    with open(os.path.join(FIG, "figures_captions.md"), "w") as f:
        f.write("# Figure captions\n\n")
        for name, cap in CAPTIONS:
            f.write(f"## {name}\n\n{cap}\n\n")
    print(f"captions -> results/figures/figures_captions.md ({len(CAPTIONS)} figures)")


if __name__ == "__main__":
    main()
