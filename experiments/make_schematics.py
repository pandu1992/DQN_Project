#!/usr/bin/env python3
"""
Conceptual schematics contrasting Study 1 / 2 / 3, plus a workflow flowchart.
Dark-themed SVG (+PNG) matching the site palette. Purely illustrative
(conceptual), not data plots — they explain what each study adds.

Outputs -> assets/schematics/
  study_comparison.svg/.png   (3-panel: clean -> +physical/sensing/comms -> +baseline)
  workflow.svg/.png           (shared pipeline + per-study specialization)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "schematics")
os.makedirs(OUT, exist_ok=True)

# ---- site palette ----
BG = "#0a1220"; PANEL = "#0c1526"; LINE = "#22324f"; INK = "#e7eefc"; MUTED = "#8aa0c6"
ACCENT = "#4da3ff"; GREEN = "#37d67a"; ORANGE = "#ff9f43"; RED = "#ff5d5d"; GOLD = "#ffcf6b"; PURPLE = "#8172b2"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200})


def rbox(ax, x, y, w, h, fc, ec, text="", fs=9, tc=INK, bold=False, alpha=1.0, r=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                       fc=fc, ec=ec, lw=1.2, alpha=alpha, zorder=3)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=4, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, zorder=2))


def mini_channel(ax, cx, cy, scale=1.0, noise=False, obstacles=False, buoys=False,
                 dock=False, cte=False, two_agents=False):
    """A tiny conceptual channel scene inside a panel."""
    w = 3.0 * scale; h = 1.35 * scale
    x0 = cx - w / 2; y0 = cy - h / 2
    # water
    ax.add_patch(Rectangle((x0, y0), w, h, fc="#071324", ec=LINE, lw=1, zorder=1))
    # channel centreline (planned route) from start to goal
    sx, sy = x0 + 0.18 * w, y0 + 0.30 * h
    gx, gy = x0 + 0.86 * w, y0 + 0.72 * h
    ax.plot([sx, x0 + 0.5 * w, gx], [sy, y0 + 0.40 * h, gy], color=GREEN, lw=2, zorder=2, solid_capstyle="round")
    # start + goal
    ax.add_patch(Circle((sx, sy), 0.055 * w, fc=GREEN, ec="#04121f", lw=1, zorder=5))
    ax.text(sx, sy - 0.14 * h, "start", ha="center", fontsize=6.5, color=MUTED)
    ax.add_patch(Circle((gx, gy), 0.06 * w, fc=GOLD, ec="#04121f", lw=1, zorder=5))
    ax.text(gx, gy + 0.13 * h, "goal", ha="center", fontsize=6.5, color=MUTED)
    if dock:
        ax.add_patch(Rectangle((gx - 0.02 * w, gy - 0.11 * h), 0.14 * w, 0.05 * h, fc="none", ec=GOLD, lw=1, ls=":", zorder=4))
        ax.text(gx + 0.05 * w, gy - 0.15 * h, "dock", ha="center", fontsize=5.6, color=GOLD)
    # buoys (IALA channel markers)
    if buoys:
        for t in (0.35, 0.62):
            bx = x0 + t * w; by = y0 + (0.40 + 0.32 * (t - 0.35)) * h
            ax.add_patch(Circle((bx, by + 0.16 * h), 0.03 * w, fc=GREEN, ec="#04121f", lw=0.6, zorder=5))
            ax.add_patch(Circle((bx, by - 0.16 * h), 0.03 * w, fc=RED, ec="#04121f", lw=0.6, zorder=5))
    # obstacles
    if obstacles:
        for (ox, oy) in [(0.55, 0.62), (0.72, 0.32)]:
            ax.add_patch(Circle((x0 + ox * w, y0 + oy * h), 0.05 * w, fc="#3a2030", ec=RED, lw=1, zorder=4))
    # agent path(s)
    if two_agents:
        # rule-based follows channel (black), DQN deviates (blue)
        ax.plot([sx, x0 + 0.5 * w, gx], [sy, y0 + 0.40 * h, gy], color="#000000", lw=2.4, zorder=3, solid_capstyle="round")
        ax.plot([sx, x0 + 0.45 * w, x0 + 0.7 * w, gx], [sy, y0 + 0.66 * h, y0 + 0.30 * h, gy],
                color=ACCENT, lw=1.6, ls="--", zorder=3)
    else:
        col = ORANGE
        if cte:
            ax.plot([sx, x0 + 0.45 * w, x0 + 0.7 * w, gx], [sy, y0 + 0.62 * h, y0 + 0.28 * h, gy],
                    color=col, lw=1.8, zorder=3)
            ax.annotate("", xy=(x0 + 0.5 * w, y0 + 0.40 * h), xytext=(x0 + 0.47 * w, y0 + 0.60 * h),
                        arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1))
            ax.text(x0 + 0.40 * w, y0 + 0.52 * h, "CTE", fontsize=5.6, color=GOLD)
        else:
            ax.plot([sx, x0 + 0.5 * w, gx], [sy, y0 + 0.40 * h, gy], color=col, lw=1.8, zorder=3, solid_capstyle="round")
    # ship marker
    ax.add_patch(Polygon([[sx + 0.02 * w, sy], [sx - 0.03 * w, sy + 0.05 * h], [sx - 0.03 * w, sy - 0.05 * h]],
                         closed=True, fc=ACCENT, ec="#cfe6ff", lw=0.6, zorder=6))
    # sensor noise halo
    if noise:
        for rr in (0.09, 0.14):
            ax.add_patch(Circle((sx, sy), rr * w, fc="none", ec=ORANGE, lw=0.8, ls=":", alpha=0.7, zorder=2))
        ax.text(sx - 0.02 * w, sy + 0.22 * h, "noisy\nsensing", ha="center", fontsize=5.4, color=ORANGE)


def study_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.patch.set_facecolor(BG)
    titles = ["Study 1 — Clean comparison", "Study 2 — Degradation & safety", "Study 3 — Rule-based baseline"]
    subt = ["4 DQN variants x PER x Noisy\n(no noise, no obstacles)",
            "+ sensor noise, packet-loss,\ncollisions, IALA, docking, CTE",
            "COLREGs rule-based vs DQN\nconfirmatory, 15 seeds"]
    accents = [ACCENT, ORANGE, GOLD]
    for i, ax in enumerate(axes):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[i], color=accents[i], fontsize=12, fontweight="bold", pad=10)
        ax.text(5, 9.0, subt[i], ha="center", va="center", fontsize=8.2, color=MUTED)
        if i == 0:
            mini_channel(ax, 5, 5.0, scale=2.6)
            ax.text(5, 1.2, "Metric: navigation success, reward, route optimality", ha="center", fontsize=7.5, color=INK)
            ax.text(5, 0.5, "Finding: variants indistinguishable; SEED dominates variance", ha="center", fontsize=7.3, color=GREEN)
        elif i == 1:
            mini_channel(ax, 5, 5.0, scale=2.6, noise=True, obstacles=True, buoys=True, dock=True, cte=True)
            ax.text(5, 1.2, "+ collision rate, IALA violations, docking accuracy, CTE", ha="center", fontsize=7.5, color=INK)
            ax.text(5, 0.5, "Finding: sensing/comms dominate safety; variants degrade in parallel", ha="center", fontsize=7.3, color=ORANGE)
        else:
            mini_channel(ax, 5, 5.0, scale=2.6, noise=True, obstacles=True, buoys=True, dock=True, two_agents=True)
            # legend for two agents
            ax.plot([1.4, 2.1], [2.35, 2.35], color="#000000", lw=2.4)
            ax.text(2.25, 2.35, "rule-based (follows channel)", va="center", fontsize=6.6, color=INK)
            ax.plot([1.4, 2.1], [1.75, 1.75], color=ACCENT, lw=1.6, ls="--")
            ax.text(2.25, 1.75, "DQN (deviates under stress)", va="center", fontsize=6.6, color=INK)
            ax.text(5, 0.5, "Finding: rule-based ~2x fewer IALA violations (significant)", ha="center", fontsize=7.3, color=GOLD)
        # progression arrow between panels
    # big arrows between panels (in figure space)
    for xf in (0.345, 0.675):
        fig.text(xf, 0.52, "\u2794", fontsize=26, color=MUTED, ha="center", va="center")
    fig.suptitle("Three studies on one testbed: progressively richer, more realistic evaluation",
                 color=INK, fontsize=13.5, y=1.02, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, "study_comparison.svg"), facecolor=BG, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "study_comparison.png"), facecolor=BG, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  study_comparison.svg/.png")


def workflow():
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    # shared pipeline (top row)
    ax.text(50, 59, "Shared experimental pipeline", ha="center", color=INK, fontsize=13, fontweight="bold")
    steps = ["Seeded\nenvironment", "Train agent\n(RL) / rule-based", "Held-out\ngreedy eval",
             "Per-episode\nreal metrics", "Per-seed\naggregation", "Statistics\n(ANOVA, posthoc)", "Tables &\nfigures"]
    n = len(steps); w = 11.5; gap = (100 - n * w) / (n + 1)
    y = 45; xs = []
    for i, s in enumerate(steps):
        x = gap + i * (w + gap)
        xs.append(x + w / 2)
        col = PANEL
        rbox(ax, x, y, w, 9, col, ACCENT, s, fs=8.2, bold=False)
        if i > 0:
            arrow(ax, xs[i - 1] + w / 2 - 0.3, y + 4.5, x, y + 4.5, color=ACCENT)
    # per-study specialization (three lanes below)
    lanes = [
        ("Study 1", ACCENT, "VesselEnvV2 (clean)", "4 DQN x PER x Noisy", "10 seeds",
         "success, reward, optimality"),
        ("Study 2", ORANGE, "VesselEnvV3 (+physical/sensing/comms)", "4 DQN x noise x packet-error", "6 seeds",
         "+ collision, IALA, docking, CTE"),
        ("Study 3", GOLD, "VesselEnvV3", "rule-based + 4 DQN x 3 conditions", "15 seeds (confirmatory)",
         "same real safety/precision metrics"),
    ]
    ly = 30
    for k, (name, col, env, factors, seeds, metrics) in enumerate(lanes):
        yy = ly - k * 9.3
        rbox(ax, 1, yy, 12, 7.6, PANEL, col, name, fs=10, tc=col, bold=True)
        rbox(ax, 15, yy, 24, 7.6, "#0e1a2e", LINE, env, fs=7.6)
        rbox(ax, 40, yy, 25, 7.6, "#0e1a2e", LINE, factors, fs=7.6)
        rbox(ax, 66, yy, 14, 7.6, "#0e1a2e", LINE, seeds, fs=7.6)
        rbox(ax, 81, yy, 18, 7.6, "#0e1a2e", LINE, metrics, fs=7.2)
        arrow(ax, 13, yy + 3.8, 15, yy + 3.8, color=col)
        arrow(ax, 39, yy + 3.8, 40, yy + 3.8, color=MUTED)
        arrow(ax, 65, yy + 3.8, 66, yy + 3.8, color=MUTED)
        arrow(ax, 80, yy + 3.8, 81, yy + 3.8, color=MUTED)
    ax.text(50, 2.0, "Same environment graph, planner, agents and statistical protocol across all three studies \u2014 only the factors, seeds and metrics change.",
            ha="center", color=MUTED, fontsize=8.3)
    fig.savefig(os.path.join(OUT, "workflow.svg"), facecolor=BG, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "workflow.png"), facecolor=BG, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  workflow.svg/.png")


def main():
    print("Rendering schematics:")
    study_comparison()
    workflow()


if __name__ == "__main__":
    main()
