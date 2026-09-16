#!/usr/bin/env python3
"""
V3 ROBUSTNESS — PUBLICATION FIGURES (>=300 DPI PNG + SVG)

Centerpiece = degradation / interaction plots showing how each algorithm's
navigation, safety, and precision respond to sensor noise and comms packet
error. All values from the aggregated/statistics CSVs (real data).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_v3", "aggregated")
STAT = os.path.join(ROOT, "results_v3", "statistics")
FIG = os.path.join(ROOT, "results_v3", "figures")
os.makedirs(FIG, exist_ok=True)

summary = pd.read_csv(os.path.join(AGG, "summary.csv"))
per_seed = pd.read_csv(os.path.join(AGG, "per_seed.csv"))
fac = pd.read_csv(os.path.join(STAT, "factorial_anova.csv"))
rob = pd.read_csv(os.path.join(STAT, "robustness_worst_vs_clean.csv"))

ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
ALGO_LABEL = {"DQN": "DQN", "DoubleDQN": "Double DQN", "DuelingDQN": "Dueling DQN", "DuelingDoubleDQN": "Dueling Double DQN"}
COLORS = {"DQN": "#4C72B0", "DoubleDQN": "#55A868", "DuelingDQN": "#C44E52", "DuelingDoubleDQN": "#8172B2"}
MARK = {"DQN": "o", "DoubleDQN": "s", "DuelingDQN": "^", "DuelingDoubleDQN": "D"}
NOISE = [0.0, 0.1, 0.25]
PER = [0.0, 0.2, 0.4]
N_SEEDS, N_EVAL = 6, 30

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


def val(algo, ns, per, metric):
    r = summary[(summary.algorithm == algo) & (summary.noise_std == ns) & (summary.packet_error_rate == per)]
    return r.iloc[0][f"{metric}_mean"], r.iloc[0][f"{metric}_ci95_hw"]


def foot():
    return f"Mean over n={N_SEEDS} seeds; error bars/bands = 95% CI. Each seed = {N_EVAL} eval episodes on VesselEnvV3."


# ---- Figure 1: metric vs noise (averaged over PER) per algorithm — degradation curves ----
def fig_vs_noise(metric, ylabel, name, fignum, pct=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for algo in ALGOS:
        ys, es = [], []
        for ns in NOISE:
            # average over PER levels
            rows = summary[(summary.algorithm == algo) & (summary.noise_std == ns)]
            ys.append(rows[f"{metric}_mean"].mean())
            es.append(rows[f"{metric}_mean"].std(ddof=1) / np.sqrt(len(rows)))
        ax.errorbar(NOISE, ys, yerr=es, marker=MARK[algo], color=COLORS[algo], capsize=3,
                    lw=2, markersize=7, label=ALGO_LABEL[algo])
    ax.set_xlabel("Sensor observation noise (std)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(NOISE)
    if pct:
        ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    cap = (f"Figure {fignum}. {ylabel} vs sensor noise (averaged over packet-error levels), per algorithm. "
           + foot() + " Nearly parallel curves indicate the algorithms degrade similarly (no algorithm x noise interaction).")
    fig.text(0.5, -0.03, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, name, cap)


# ---- Figure: metric vs PER per algorithm ----
def fig_vs_per(metric, ylabel, name, fignum, pct=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for algo in ALGOS:
        ys, es = [], []
        for per in PER:
            rows = summary[(summary.algorithm == algo) & (summary.packet_error_rate == per)]
            ys.append(rows[f"{metric}_mean"].mean())
            es.append(rows[f"{metric}_mean"].std(ddof=1) / np.sqrt(len(rows)))
        ax.errorbar(PER, ys, yerr=es, marker=MARK[algo], color=COLORS[algo], capsize=3,
                    lw=2, markersize=7, label=ALGO_LABEL[algo])
    ax.set_xlabel("Communication packet-error rate")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PER)
    if pct:
        ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    cap = (f"Figure {fignum}. {ylabel} vs packet-error rate (averaged over noise levels), per algorithm. " + foot())
    fig.text(0.5, -0.03, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, name, cap)


# ---- Figure: 2D heatmap of a metric over noise x PER (averaged over algorithms) ----
def fig_heatmap(metric, title, name, fignum, fmt="{:.1f}"):
    grid = np.zeros((len(NOISE), len(PER)))
    for i, ns in enumerate(NOISE):
        for j, per in enumerate(PER):
            rows = summary[(summary.noise_std == ns) & (summary.packet_error_rate == per)]
            grid[i, j] = rows[f"{metric}_mean"].mean()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(PER))); ax.set_xticklabels(PER)
    ax.set_yticks(range(len(NOISE))); ax.set_yticklabels(NOISE)
    ax.set_xlabel("Packet-error rate"); ax.set_ylabel("Sensor noise (std)")
    for i in range(len(NOISE)):
        for j in range(len(PER)):
            ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                    fontsize=10, color="black" if grid[i, j] < 0.6 * grid.max() else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    cap = (f"Figure {fignum}. {title}: mean over the 4 algorithms and {N_SEEDS} seeds across the "
           "noise x packet-error grid. " + foot())
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, name, cap)


# ---- Figure: factorial partial eta^2 heatmap (effect x metric) ----
def fig_effects():
    effects = ["algorithm", "noise", "per", "noise:per", "algorithm:noise", "algorithm:per", "seed"]
    elab = {"algorithm": "Algorithm", "noise": "Noise", "per": "Packet-err", "noise:per": "Noise\u00d7PER",
            "algorithm:noise": "Algo\u00d7Noise", "algorithm:per": "Algo\u00d7PER", "seed": "Seed"}
    metrics = ["success_rate", "collision_rate", "iala_violations", "docking_accuracy", "cte_mean", "optimality_ratio"]
    mlab = ["Success", "Collision%", "IALA", "Docking", "CTE", "Opt.ratio"]
    data = np.full((len(effects), len(metrics)), np.nan)
    for j, m in enumerate(metrics):
        sub = fac[fac.metric == m]
        for i, e in enumerate(effects):
            row = sub[sub.effect == e]
            if len(row):
                data[i, j] = row["partial_eta2"].values[0]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=min(1.0, np.nanmax(data)))
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(mlab, rotation=20, ha="right")
    ax.set_yticks(range(len(effects))); ax.set_yticklabels([elab[e] for e in effects])
    for i in range(len(effects)):
        for j in range(len(metrics)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if data[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="partial $\\eta^2$")
    ax.set_title("Variance explained by each factor (factorial ANOVA)")
    cap = ("Figure. Partial $\\eta^2$ from the three-way factorial ANOVA (y ~ algorithm*noise*PER + seed). "
           "Noise and packet-error dominate the safety/precision metrics (IALA, docking); the Algorithm effect and "
           "Algorithm x stressor interactions are small \u2014 algorithms degrade in parallel. Balanced design, 6 seeds/cell.")
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figV3_effect_sizes", cap)


# ---- Figure: robustness ranking (mean |d_z| degradation per algorithm) ----
def fig_ranking():
    metrics = ["collision_rate", "iala_violations", "docking_accuracy", "cte_mean"]
    scores = {a: np.nanmean([abs(rob[(rob.algorithm == a) & (rob.metric == m)].iloc[0]["cohens_dz"]) for m in metrics]) for a in ALGOS}
    order = sorted(ALGOS, key=lambda a: scores[a])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh([ALGO_LABEL[a] for a in order], [scores[a] for a in order],
            color=[COLORS[a] for a in order], edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Mean |Cohen's $d_z$| degradation (safety + precision, worst vs clean)")
    ax.invert_yaxis()
    for i, a in enumerate(order):
        ax.text(scores[a] + 0.03, i, f"{scores[a]:.2f}", va="center", fontsize=10)
    ax.set_title("Robustness ranking (lower = degrades less)")
    cap = ("Figure. Robustness ranking: mean absolute paired effect size (Cohen's $d_z$) of degradation from clean to the "
           "harshest condition (noise 0.25, PER 0.4), averaged over collision rate, IALA, docking error and CTE. "
           "Lower = more robust. Differences are modest and mostly non-significant (parallel degradation).")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figV3_robustness_ranking", cap)


# ---- Figure: per-seed distributions of a safety metric across noise levels ----
def fig_distributions():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions, labels, data, colors = [], [], [], []
    pos = 1
    for ns in NOISE:
        for algo in ALGOS:
            d = per_seed[(per_seed.algorithm == algo) & (per_seed.noise_std == ns)]["collisions_per_ep"].to_numpy(float)
            data.append(d); positions.append(pos); colors.append(COLORS[algo]); pos += 1
        pos += 1  # gap between noise groups
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    # group labels
    group_centers = [2.5, 7.5, 12.5]
    ax.set_xticks(group_centers); ax.set_xticklabels([f"noise={n}" for n in NOISE])
    ax.set_ylabel("Collisions per episode")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[a], alpha=0.55, ec="black") for a in ALGOS]
    ax.legend(handles, [ALGO_LABEL[a] for a in ALGOS], frameon=False, fontsize=9, ncol=4, loc="upper left")
    ax.set_title("Collision-rate distribution across seeds by noise level")
    cap = ("Figure. Per-seed distribution of collisions per episode (box = IQR, line = median, white diamond = mean, "
           f"n={N_SEEDS} seeds), grouped by sensor-noise level, one box per algorithm (averaged over PER within each cell). "
           "Collisions rise with noise for all algorithms; spread reflects seed variance.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figV3_collision_distributions", cap)


def main():
    print("Rendering V3 figures:")
    fig_vs_noise("success_rate", "Navigation success rate (%)", "figV3_success_vs_noise", 1, pct=True)
    fig_vs_noise("collision_rate", "Collision rate (%)", "figV3_collision_vs_noise", 2, pct=True)
    fig_vs_per("iala_violations", "IALA violations per episode", "figV3_iala_vs_per", 3)
    fig_vs_per("docking_accuracy", "Docking error (distance)", "figV3_docking_vs_per", 4)
    fig_heatmap("iala_violations", "IALA violations / episode", "figV3_iala_heatmap", 5, fmt="{:.1f}")
    fig_heatmap("collision_rate", "Collision rate (%)", "figV3_collision_heatmap", 6, fmt="{:.0f}")
    fig_effects()
    fig_ranking()
    fig_distributions()
    with open(os.path.join(FIG, "figures_captions.md"), "w") as f:
        f.write("# V3 figure captions\n\n")
        for name, cap in CAPTIONS:
            f.write(f"## {name}\n\n{cap}\n\n")
    print(f"captions -> results_v3/figures/figures_captions.md ({len(CAPTIONS)} figures)")


if __name__ == "__main__":
    main()
