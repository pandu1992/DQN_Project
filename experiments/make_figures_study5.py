#!/usr/bin/env python3
"""
STUDY 5 — PUBLICATION FIGURES (>=300 DPI PNG + SVG).
Two-phase round trip (dock vs full-cycle) + two-way opposing traffic.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study5", "aggregated")
STAT = os.path.join(ROOT, "results_study5", "statistics")
FIG = os.path.join(ROOT, "results_study5", "figures")
os.makedirs(FIG, exist_ok=True)

phase_s = pd.read_csv(os.path.join(AGG, "twophase_summary.csv"))
phase_ps = pd.read_csv(os.path.join(AGG, "twophase_per_seed.csv"))
way_s = pd.read_csv(os.path.join(AGG, "twoway_summary.csv"))
way_ps = pd.read_csv(os.path.join(AGG, "twoway_per_seed.csv"))
wf = pd.read_csv(os.path.join(STAT, "twoway_factorial_anova.csv"))

CONDS = ["clean", "mid", "harsh"]
PHASE_AGENTS = ["RuleBased", "DQN"]
PLAB = {"RuleBased": "Rule-based", "DQN": "DQN"}
PCOL = {"RuleBased": "#000000", "DQN": "#4C72B0"}
WAY_PAIRINGS = ["Rule_vs_Rule", "DQN_vs_Rule", "DQN_vs_DQN"]
WLAB = {"Rule_vs_Rule": "Rule\u00d7Rule", "DQN_vs_Rule": "DQN\u00d7Rule", "DQN_vs_DQN": "DQN\u00d7DQN"}
WCOL = {"Rule_vs_Rule": "#000000", "DQN_vs_Rule": "#C44E52", "DQN_vs_DQN": "#4C72B0"}
WMARK = {"Rule_vs_Rule": "*", "DQN_vs_Rule": "^", "DQN_vs_DQN": "o"}
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


def pcell(ag, c): return phase_s[(phase_s.agent == ag) & (phase_s.condition == c)].iloc[0]
def wcell(pr, c): return way_s[(way_s.pairing == pr) & (way_s.condition == c)].iloc[0]


# Fig 1 — two-phase: dock vs full-cycle success (the headline for scenario A)
def fig_phase_dock_vs_full():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(PHASE_AGENTS) * len(CONDS))
    labels, dock, full, dockE, fullE = [], [], [], [], []
    for ag in PHASE_AGENTS:
        for c in CONDS:
            cc = pcell(ag, c)
            labels.append(f"{PLAB[ag]}\n{c}")
            dock.append(cc["dock_success_rate_mean"]); dockE.append(cc["dock_success_rate_ci95_hw"])
            full.append(cc["full_cycle_success_rate_mean"]); fullE.append(cc["full_cycle_success_rate_ci95_hw"])
    w = 0.38
    ax.bar(x - w / 2, dock, w, yerr=dockE, capsize=3, label="Dock (inbound leg)", color="#8Fb8de", edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, full, w, yerr=fullE, capsize=3, label="Full cycle (in+out)", color="#2a5d8f", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Two-phase mission: reaching the dock vs completing the FULL round trip")
    cap = ("Figure. Scenario A. For each agent and condition, the inbound leg (docking, light) succeeds far more often "
           "than the FULL inbound\u2192dock\u2192outbound cycle (dark): the outbound leg roughly halves success for the "
           f"rule-based controller and the learned DQN rarely completes it. Mean over n={N_SEEDS} seeds, 95% CI.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS5_twophase_dock_vs_full", cap)


# Fig 2 — two-way: collision rate vs condition by pairing
def way_line(metric, ylabel, name, fignum, pct=False):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for pr in WAY_PAIRINGS:
        ys, es = [], []
        for c in CONDS:
            r = wcell(pr, c); ys.append(r[f"{metric}_mean"]); es.append(r[f"{metric}_ci95_hw"])
        lw = 3 if pr == "Rule_vs_Rule" else 1.8
        ms = 14 if pr == "Rule_vs_Rule" else 7
        ax.errorbar([0, 1, 2], ys, yerr=es, marker=WMARK[pr], color=WCOL[pr], capsize=3, lw=lw, markersize=ms,
                    label=WLAB[pr], zorder=5 if pr == "Rule_vs_Rule" else 3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["clean", "mid", "harsh"])
    ax.set_xlabel("Degradation condition (sensor noise, packet-error)"); ax.set_ylabel(ylabel)
    if pct: ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, title="Two-way pairing")
    cap = (f"Figure {fignum}. Scenario B. {ylabel} across conditions for opposing-traffic pairings. " +
           f"Mean over n={N_SEEDS} seeds, 95% CI. Classical Rule\u00d7Rule in black (bold).")
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, name, cap)


# Fig 3 — two-way tradeoff scatter (success vs collision)
def fig_way_tradeoff():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for pr in WAY_PAIRINGS:
        xs, ys = [], []
        for c in CONDS:
            r = wcell(pr, c); xs.append(r["vessel_collision_rate_mean"]); ys.append(r["success_rate_mean"])
        ax.plot(xs, ys, "-", color=WCOL[pr], alpha=0.5, lw=1.4, zorder=2)
        for i, c in enumerate(CONDS):
            ax.scatter(xs[i], ys[i], color=WCOL[pr], marker=WMARK[pr], s=(230 if pr == "Rule_vs_Rule" else 95),
                       edgecolor="black", linewidth=0.7, zorder=4, label=WLAB[pr] if i == 0 else None)
            ax.annotate(c[0], (xs[i], ys[i]), fontsize=7, ha="center", va="center",
                        color="white" if pr == "Rule_vs_Rule" else "black", zorder=5)
    ax.set_xlabel("Inter-vessel collision rate (% of encounters)")
    ax.set_ylabel("Per-vessel mission success rate (%)")
    ax.legend(frameon=False, fontsize=9, title="Two-way pairing")
    ax.set_title("Two-way traffic: success and collisions rise together (only moving vessels meet)")
    cap = ("Figure. Scenario B tradeoff. Each point is a (pairing, condition) cell; letters c/m/h = clean/mid/harsh. "
           "Because two vessels only encounter each other when they actually TRANSIT the shared lane, collision rate and "
           "success rise together: the classical Rule\u00d7Rule pairing (black star) both completes most and collides most; "
           "the learned pairings that barely transit sit near the origin \u2014 their low collisions are a stall artefact, "
           f"not safe avoidance. Mean over n={N_SEEDS} seeds.")
    fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS5_twoway_tradeoff", cap)


# Fig 4 — two-way min CPA distribution
def fig_way_cpa():
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    data, positions, colors = [], [], []
    pos = 1
    for cond in CONDS:
        for pr in WAY_PAIRINGS:
            d = way_ps[(way_ps.pairing == pr) & (way_ps.condition == cond)]["min_cpa"].to_numpy(float)
            data.append(d); positions.append(pos); colors.append(WCOL[pr]); pos += 1
        pos += 1
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(18, color="red", ls="--", lw=1, alpha=0.7)
    ax.text(positions[-1], 26, "collision radius (18)", color="red", fontsize=8, ha="right")
    centers = [2, 6, 10]
    ax.set_xticks(centers); ax.set_xticklabels(["clean", "mid", "harsh"])
    ax.set_ylabel("Closest point of approach (logical units)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=WCOL[p], alpha=0.6, ec="black") for p in WAY_PAIRINGS]
    ax.legend(handles, [WLAB[p] for p in WAY_PAIRINGS], frameon=False, fontsize=8.5, ncol=3, loc="upper center")
    ax.set_title("Two-way: closest point of approach by pairing and condition")
    cap = ("Figure. Per-seed distribution of the episode closest-point-of-approach between the opposing vessels "
           f"(box=IQR, line=median, white diamond=mean, n={N_SEEDS}). The classical Rule\u00d7Rule pairing (left, black-edged) "
           "closes to the red collision line when it transits and meets head-on; the DQN pairings keep larger CPA "
           "largely because at least one vessel fails to transit.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS5_twoway_cpa_distributions", cap)


# Fig 5 — two-way ANOVA eta^2
def fig_way_anova():
    metrics = ["vessel_collision_rate", "min_cpa", "head_on_events", "success_rate"]
    mlab = ["Coll %", "Min CPA", "HeadOn/enc", "Success %"]
    effects = ["pairing", "condition", "pairing:condition"]
    elab = {"pairing": "Pairing", "condition": "Condition", "pairing:condition": "Pairing\u00d7Cond"}
    ecol = {"pairing": "#4C72B0", "condition": "#DD8452", "pairing:condition": "#55A868"}
    data = {e: [] for e in effects}
    for m in metrics:
        for e in effects:
            r = wf[(wf.metric == m) & (wf.effect == e)]
            data[e].append(r.iloc[0]["partial_eta2"] if len(r) else np.nan)
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for i, e in enumerate(effects):
        ax.bar(x + (i - 1) * w, data[e], w, label=elab[e], color=ecol[e], edgecolor="black", linewidth=0.6)
    for thr, lab in [(0.14, "large"), (0.06, "medium")]:
        ax.axhline(thr, color="gray", ls=":", lw=1); ax.text(len(metrics) - 0.5, thr + 0.005, lab, fontsize=7.5, color="gray", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(mlab); ax.set_ylabel("Partial $\\eta^2$"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Two-way traffic: the pairing dominates outcomes")
    cap = ("Figure. Two-way factorial ANOVA partial $\\eta^2$ (pairing, condition, interaction). The PAIRING factor "
           "dominates success, collision rate and closest-approach (large effects), confirming the choice of controller "
           "\u2014 not the degradation level \u2014 governs how two-way traffic resolves. Head-on events show a "
           "significant pairing\u00d7condition interaction.")
    fig.text(0.5, -0.05, cap, ha="center", va="top", fontsize=CAP, wrap=True)
    save(fig, "figS5_twoway_anova_eta2", cap)


def main():
    print("Rendering Study 5 figures:")
    fig_phase_dock_vs_full()
    way_line("vessel_collision_rate", "Inter-vessel collision rate (%)", "figS5_twoway_collrate_vs_condition", 2, pct=True)
    way_line("success_rate", "Per-vessel success rate (%)", "figS5_twoway_success_vs_condition", 3, pct=True)
    fig_way_tradeoff()
    fig_way_cpa()
    fig_way_anova()
    with open(os.path.join(FIG, "figures_captions.md"), "w") as f:
        f.write("# Study 5 figure captions\n\n")
        for name, cap in CAPTIONS:
            f.write(f"## {name}\n\n{cap}\n\n")
    print(f"captions -> results_study5/figures/figures_captions.md ({len(CAPTIONS)} figures)")


if __name__ == "__main__":
    main()
