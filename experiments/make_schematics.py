#!/usr/bin/env python3
"""
Conceptual schematics contrasting Studies 1–5, plus a workflow flowchart.
Dark-themed SVG (+PNG) matching the site palette. Purely illustrative
(conceptual), not data plots — they explain what each study adds.

Outputs -> assets/schematics/
  study_comparison.svg/.png   (5-panel: clean -> +physical/sensing/comms -> +baseline
                               -> two vessels avoiding -> round trip + two-way traffic)
  workflow.svg/.png           (shared pipeline + per-study specialization, 5 lanes)
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
                 dock=False, cte=False, two_agents=False, multi_vessel=False,
                 round_trip=False, two_way=False):
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
    if multi_vessel:
        # TWO independent vessels on different missions that must avoid each other.
        # vessel A: start->goal (blue); vessel B: opposite corner crossing (green)
        ax.plot([sx, x0 + 0.5 * w, gx], [sy, y0 + 0.42 * h, gy], color=ACCENT, lw=2.0, zorder=3, solid_capstyle="round")
        bsx, bsy = x0 + 0.16 * w, y0 + 0.74 * h
        bgx, bgy = x0 + 0.86 * w, y0 + 0.28 * h
        ax.plot([bsx, x0 + 0.5 * w, bgx], [bsy, y0 + 0.55 * h, bgy], color=GREEN, lw=2.0, zorder=3, solid_capstyle="round")
        ax.add_patch(Polygon([[bsx + 0.02 * w, bsy], [bsx - 0.03 * w, bsy + 0.05 * h], [bsx - 0.03 * w, bsy - 0.05 * h]],
                             closed=True, fc=GREEN, ec="#cfe6ff", lw=0.6, zorder=6))
        # near-miss / CPA marker where the two cross
        ax.add_patch(Circle((x0 + 0.5 * w, y0 + 0.485 * h), 0.075 * w, fc="none", ec=RED, lw=1.4, ls="--", zorder=5))
        ax.text(x0 + 0.5 * w, y0 + 0.485 * h + 0.20 * h, "avoid!", ha="center", fontsize=5.8, color=RED)
    elif round_trip:
        # ONE vessel: inbound (solid) then outbound (dashed) — a full round trip
        ax.annotate("", xy=(gx, gy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.0,
                                    connectionstyle="arc3,rad=-0.25"), zorder=3)
        ax.annotate("", xy=(sx, sy + 0.02 * h), xytext=(gx, gy - 0.02 * h),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8, ls="--",
                                    connectionstyle="arc3,rad=-0.25"), zorder=3)
        ax.text(x0 + 0.5 * w, y0 + 0.86 * h, "inbound", ha="center", fontsize=5.8, color=ACCENT)
        ax.text(x0 + 0.5 * w, y0 + 0.14 * h, "outbound", ha="center", fontsize=5.8, color=GREEN)
    elif two_way:
        # two vessels, OPPOSING directions on the SAME lane -> head-on
        midy = y0 + 0.5 * h
        ax.annotate("", xy=(x0 + 0.82 * w, midy), xytext=(x0 + 0.18 * w, midy),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.0), zorder=3)
        ax.annotate("", xy=(x0 + 0.18 * w, midy - 0.10 * h), xytext=(x0 + 0.82 * w, midy - 0.10 * h),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0), zorder=3)
        ax.text(x0 + 0.5 * w, midy + 0.16 * h, "inbound \u2192", ha="center", fontsize=5.8, color=ACCENT)
        ax.text(x0 + 0.5 * w, midy - 0.24 * h, "\u2190 outbound", ha="center", fontsize=5.8, color=GREEN)
        ax.add_patch(Circle((x0 + 0.5 * w, midy - 0.05 * h), 0.07 * w, fc="none", ec=RED, lw=1.4, ls="--", zorder=5))
        ax.text(x0 + 0.5 * w, midy - 0.05 * h + 0.22 * h, "head-on", ha="center", fontsize=5.6, color=RED)
    elif two_agents:
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
    fig, axes = plt.subplots(1, 5, figsize=(20.5, 4.6))
    fig.patch.set_facecolor(BG)
    titles = ["Study 1 — Clean comparison", "Study 2 — Degradation & safety",
              "Study 3 — Rule-based baseline", "Study 4 — Two vessels avoiding",
              "Study 5 — Round trips & two-way"]
    subt = ["4 DQN variants x PER x Noisy\n(no noise, no obstacles)",
            "+ sensor noise, packet-loss,\ncollisions, IALA, docking, CTE",
            "COLREGs rule-based vs DQN\nconfirmatory, 15 seeds",
            "two INDEPENDENT vessels\nmust avoid each other",
            "round trip (in\u2192dock\u2192out)\n+ two-way opposing traffic"]
    accents = [ACCENT, ORANGE, GOLD, GREEN, PURPLE]
    findings = [
        ("Metric: navigation success, reward, route optimality",
         "Finding: variants indistinguishable; SEED dominates variance", GREEN),
        ("+ collision rate, IALA violations, docking accuracy, CTE",
         "Finding: sensing/comms dominate safety; variants degrade in parallel", ORANGE),
        ("+ COLREGs rule-based baseline, 15 seeds",
         "Finding: rule-based ~2x fewer IALA violations (significant)", GOLD),
        ("+ inter-vessel collision, near-miss, closest-point-of-approach",
         "Finding: pairing dominates; learned pair ~2x farther apart, completes less", GREEN),
        ("+ full-cycle success, head-on events, two-way CPA",
         "Finding: outbound leg ~halves success; classical carries two-way traffic", PURPLE),
    ]
    for i, ax in enumerate(axes):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[i], color=accents[i], fontsize=10.5, fontweight="bold", pad=8)
        ax.text(5, 9.1, subt[i], ha="center", va="center", fontsize=7.2, color=MUTED)
        if i == 0:
            mini_channel(ax, 5, 5.0, scale=2.5)
        elif i == 1:
            mini_channel(ax, 5, 5.0, scale=2.5, noise=True, obstacles=True, buoys=True, dock=True, cte=True)
        elif i == 2:
            mini_channel(ax, 5, 5.0, scale=2.5, noise=True, obstacles=True, buoys=True, dock=True, two_agents=True)
        elif i == 3:
            mini_channel(ax, 5, 5.4, scale=2.5, noise=True, obstacles=True, buoys=True, multi_vessel=True)
        else:
            mini_channel(ax, 5, 6.2, scale=2.2, buoys=True, dock=True, round_trip=True)
            mini_channel(ax, 5, 3.2, scale=2.2, buoys=True, two_way=True)
        ax.text(5, 1.2, findings[i][0], ha="center", fontsize=6.4, color=INK)
        ax.text(5, 0.5, findings[i][1], ha="center", fontsize=6.3, color=findings[i][2])
    # big arrows between panels (figure space): 5 panels -> 4 gaps
    for xf in (0.207, 0.405, 0.603, 0.801):
        fig.text(xf, 0.52, "\u2794", fontsize=20, color=MUTED, ha="center", va="center")
    fig.suptitle("Five studies on one testbed: progressively richer, more realistic evaluation",
                 color=INK, fontsize=13.5, y=1.03, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
        ("Study 4", GREEN, "MultiVesselEnvV4 (two vessels)", "4 pairings x 3 conditions", "8 seeds",
         "+ inter-vessel collision, near-miss, CPA"),
        ("Study 5", PURPLE, "TwoPhaseEnvV5 / TwoWayEnvV5", "round trip + two-way x 3 conditions", "8 seeds",
         "+ full-cycle success, head-on, two-way CPA"),
    ]
    ly = 34
    for k, (name, col, env, factors, seeds, metrics) in enumerate(lanes):
        yy = ly - k * 6.6
        rbox(ax, 1, yy, 12, 5.6, PANEL, col, name, fs=9.2, tc=col, bold=True)
        rbox(ax, 15, yy, 24, 5.6, "#0e1a2e", LINE, env, fs=7.0)
        rbox(ax, 40, yy, 25, 5.6, "#0e1a2e", LINE, factors, fs=7.0)
        rbox(ax, 66, yy, 14, 5.6, "#0e1a2e", LINE, seeds, fs=7.0)
        rbox(ax, 81, yy, 18, 5.6, "#0e1a2e", LINE, metrics, fs=6.7)
        arrow(ax, 13, yy + 2.8, 15, yy + 2.8, color=col)
        arrow(ax, 39, yy + 2.8, 40, yy + 2.8, color=MUTED)
        arrow(ax, 65, yy + 2.8, 66, yy + 2.8, color=MUTED)
        arrow(ax, 80, yy + 2.8, 81, yy + 2.8, color=MUTED)
    ax.text(50, 1.2, "Same environment graph, planner, agents and statistical protocol across all five studies \u2014 only the environment layer, factors, seeds and metrics change.",
            ha="center", color=MUTED, fontsize=8.0)
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
