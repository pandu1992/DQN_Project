#!/usr/bin/env python3
"""
STUDY 3 — PUBLICATION FIGURES (>=300 DPI PNG + SVG).
Centerpiece: rule-based COLREGs baseline vs DQN across degradation conditions,
with safety (collision/IALA) the key story.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study3", "aggregated")
STAT = os.path.join(ROOT, "results_study3", "statistics")
FIG = os.path.join(ROOT, "results_study3", "figures")
os.makedirs(FIG, exist_ok=True)

summary = pd.read_csv(os.path.join(AGG, "summary.csv"))
per_seed = pd.read_csv(os.path.join(AGG, "per_seed.csv"))
bvd = pd.read_csv(os.path.join(STAT, "baseline_vs_dqn.csv"))
fac = pd.read_csv(os.path.join(STAT, "factorial_anova.csv"))

AGENTS = ["RuleBased", "DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
LAB = {"RuleBased": "Rule-based", "DQN": "DQN", "DoubleDQN": "Double DQN",
       "DuelingDQN": "Dueling DQN", "DuelingDoubleDQN": "Dueling Dbl DQN"}
COND = ["clean", "mid", "harsh"]
CONDX = {"clean": 0, "mid": 1, "harsh": 2}
COLORS = {"RuleBased": "#000000", "DQN": "#4C72B0", "DoubleDQN": "#55A868",
          "DuelingDQN": "#C44E52", "DuelingDoubleDQN": "#8172B2"}
MARK = {"RuleBased": "*", "DQN": "o", "DoubleDQN": "s", "DuelingDQN": "^", "DuelingDoubleDQN": "D"}
N_SEEDS, N_EVAL = 15, 30

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
    return f"Mean over n={N_SEEDS} seeds; error bars = 95% CI. Each seed = {N_EVAL} eval episodes on VesselEnvV3. Rule-based drawn in black (bold)."


def series(ag, metric):
    ys, es = [], []
    for c in COND:
        r = summary[(summary.agent == ag) & (summary.condition == c)].iloc[0]
        ys.append(r[f"{metric}_mean"]); es.append(r[f"{metric}_ci95_hw"])
    return ys, es


def line_fig(metric, ylabel, name, fignum, pct=False):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for ag in AGENTS:
        ys, es = series(ag, metric)
        lw = 3 if ag == "RuleBased" else 1.8
        ms = 13 if ag == "RuleBased" else 7
        ax.errorbar([0, 1, 2], ys, yerr=es, marker=MARK[ag], color=COLORS[ag], capsize=3,
                    lw=lw, markersize=ms, label=LAB[ag], zorder=5 if ag == "RuleBased" else 3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["clean", "mid", "harsh"])
    ax.set_xlabel("Degradation condition (sensor noise, packet-error)")
    ax.set_ylabel(ylabel)
    if pct: ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    cap = (f"Figure {fignum}. {ylabel} across degradation conditions, rule-based baseline vs DQN variants. "
           + foot())
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, name, cap)


def bar_harsh(metric, ylabel, name, fignum, pct=False):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    vals, errs, cols = [], [], []
    for ag in AGENTS:
        r = summary[(summary.agent == ag) & (summary.condition == "harsh")].iloc[0]
        vals.append(r[f"{metric}_mean"]); errs.append(r[f"{metric}_ci95_hw"]); cols.append(COLORS[ag])
    x = np.arange(len(AGENTS))
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=cols, edgecolor="black", linewidth=0.7)
    bars[0].set_hatch("//")  # highlight rule-based
    ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in AGENTS], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if pct: ax.set_ylim(0, 100)
    cap = (f"Figure {fignum}. {ylabel} under the HARSHEST condition (noise 0.25, packet-error 0.4). "
           "Rule-based baseline (hatched) vs DQN variants. " + foot())
    fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, name, cap)


def fig_effect_matrix():
    # Cohen's d_z of RuleBased vs each DQN, per condition, for IALA + success
    metrics = ["iala_violations", "collision_rate", "success_rate", "docking_accuracy"]
    mlab = ["IALA", "Collision%", "Success%", "Docking"]
    dqn = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ci, cond in enumerate(COND):
        data = np.full((len(dqn), len(metrics)), np.nan)
        sig = np.zeros_like(data, dtype=bool)
        for j, m in enumerate(metrics):
            for i, dq in enumerate(dqn):
                r = bvd[(bvd.condition == cond) & (bvd.metric == m) & (bvd.vs == dq)]
                if len(r):
                    data[i, j] = r.iloc[0]["cohens_dz"]; sig[i, j] = r.iloc[0]["sig_holm"]
        ax = axes[ci]
        im = ax.imshow(data, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels(mlab, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(len(dqn))); ax.set_yticklabels(dqn if ci == 0 else [""] * len(dqn), fontsize=9)
        ax.set_title(cond, fontsize=11)
        for i in range(len(dqn)):
            for j in range(len(metrics)):
                if np.isfinite(data[i, j]):
                    txt = f"{data[i,j]:.1f}" + ("*" if sig[i, j] else "")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                            color="white" if abs(data[i, j]) > 1.5 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Cohen's $d_z$ (Rule-based \u2212 DQN)")
    fig.suptitle("Rule-based vs DQN effect size by metric and condition (* = Holm-significant)", fontsize=12, y=1.03)
    cap = ("Figure. Paired effect size (Cohen's $d_z$) of the rule-based baseline minus each DQN variant, "
           "per metric and condition; * marks Holm-significant contrasts (n=15). For IALA (channel-keeping) "
           "the baseline is strongly and significantly better under mid/harsh degradation.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS3_effect_matrix", cap)


def fig_interaction_iala():
    # the significant agent x condition interaction on IALA
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for ag in AGENTS:
        ys, es = series(ag, "iala_violations")
        lw = 3 if ag == "RuleBased" else 1.8
        ax.errorbar([0, 1, 2], ys, yerr=es, marker=MARK[ag], color=COLORS[ag], capsize=3,
                    lw=lw, markersize=13 if ag == "RuleBased" else 7, label=LAB[ag], zorder=5 if ag == "RuleBased" else 3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["clean", "mid", "harsh"])
    ax.set_xlabel("Degradation condition"); ax.set_ylabel("IALA violations per episode")
    ax.legend(frameon=False, fontsize=9)
    r = fac[(fac.metric == "iala_violations") & (fac.effect == "agent:condition")].iloc[0]
    ax.set_title(f"Agent \u00d7 Condition interaction on IALA (partial \u03b7\u00b2={r['partial_eta2']:.2f}, p<0.001)")
    cap = ("Figure. IALA channel-departure violations vs degradation, per agent. The lines FAN OUT (significant "
           f"agent\u00d7condition interaction, partial \u03b7\u00b2={r['partial_eta2']:.2f}): the DQN variants' violations rise "
           "far more steeply than the rule-based baseline's \u2014 the baseline keeps the channel better as sensing/comms degrade. "
           + foot())
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS3_iala_interaction", cap)


def fig_distributions():
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    data, positions, colors = [], [], []
    pos = 1
    for cond in COND:
        for ag in AGENTS:
            d = per_seed[(per_seed.agent == ag) & (per_seed.condition == cond)]["iala_violations"].to_numpy(float)
            data.append(d); positions.append(pos); colors.append(COLORS[ag]); pos += 1
        pos += 1
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    centers = [3, 9, 15]
    ax.set_xticks(centers); ax.set_xticklabels(["clean", "mid", "harsh"])
    ax.set_ylabel("IALA violations per episode")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[a], alpha=0.6, ec="black") for a in AGENTS]
    ax.legend(handles, [LAB[a] for a in AGENTS], frameon=False, fontsize=8.5, ncol=5, loc="upper left")
    ax.set_title("IALA violation distribution across 15 seeds, by condition")
    cap = ("Figure. Per-seed distribution of IALA violations (box=IQR, line=median, white diamond=mean, n=15), "
           "grouped by condition, one box per agent. Under mid/harsh the rule-based baseline (black-edged, leftmost in each group) "
           "sits well below the DQN variants.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS3_iala_distributions", cap)


def main():
    print("Rendering Study 3 figures:")
    line_fig("success_rate", "Navigation success rate (%)", "figS3_success_vs_condition", 1, pct=True)
    line_fig("collisions_per_ep", "Collisions per episode", "figS3_collisions_vs_condition", 2)
    fig_interaction_iala()
    bar_harsh("iala_violations", "IALA violations per episode", "figS3_iala_harsh", 4)
    bar_harsh("success_rate", "Navigation success rate (%)", "figS3_success_harsh", 5, pct=True)
    fig_effect_matrix()
    fig_distributions()
    with open(os.path.join(FIG, "figures_captions.md"), "w") as f:
        f.write("# Study 3 figure captions\n\n")
        for name, cap in CAPTIONS:
            f.write(f"## {name}\n\n{cap}\n\n")
    print(f"captions -> results_study3/figures/figures_captions.md ({len(CAPTIONS)} figures)")


if __name__ == "__main__":
    main()
