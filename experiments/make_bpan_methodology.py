#!/usr/bin/env python3
"""
BPAN methodology-flow + factor-hierarchy illustrations (dark-themed, site palette).
Purely illustrative schematics that make the study design legible:

  figBPAN_methodology.png/.svg  — the end-to-end BPAN pipeline, left to right:
      Real chart -> waypoint/buoy/berth graph -> environment contract
      (physics, sensing noise, comms loss, obstacles, IALA, docking, CTE) ->
      four study blocks (core, DQN-improvement, multi-vessel (d),
      operational realism (e)) -> per-seed scalars -> statistics
      (factorial ANOVA + partial eta^2, paired Wilcoxon+Holm+d_z,
      mixed-effects (1|seed)+ICC) -> factor-hierarchy synthesis.

  figBPAN_factor_hierarchy.png/.svg — the design stack ranked by how much
      variance each level explains (mission/interaction >= control/prior >
      perception/comms >> algorithm), annotated with the measured partial eta^2
      ranges from this study.

Outputs -> results_bintulu/figures/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "results_bintulu", "figures")
os.makedirs(FIG, exist_ok=True)

BG = "#0a1220"; PANEL = "#0c1526"; LINE = "#22324f"; INK = "#e7eefc"; MUTED = "#8aa0c6"
ACCENT = "#4da3ff"; GREEN = "#37d67a"; ORANGE = "#ff9f43"; RED = "#ff5d5d"; GOLD = "#ffcf6b"; PURPLE = "#8172b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220})


def rbox(ax, x, y, w, h, fc, ec, text="", fs=9, tc=INK, bold=False, alpha=1.0, r=0.018):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                                fc=fc, ec=ec, lw=1.3, alpha=alpha, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=ACCENT, lw=1.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls, zorder=2))


def methodology():
    fig, ax = plt.subplots(figsize=(15.5, 9.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
    ax.text(2, 59.5, "BPAN — Bintulu Port Autonomous Navigation: methodology flow",
            fontsize=16, color=INK, fontweight="bold")
    ax.text(2, 56.6, "Every stage is sequential; all metrics are genuinely computed from the simulated kinematics (no hand-set numbers).",
            fontsize=10, color=MUTED)

    # ---- Stage 1: chart -> graph -> environment contract (left column) ----
    rbox(ax, 2, 46, 20, 7.5, PANEL, GREEN, "1. Real Bintulu approach chart\n(1536x1024 native pixel frame)", 10, bold=True)
    arrow(ax, 12, 46, 12, 43.2)
    rbox(ax, 2, 35.5, 20, 7.5, PANEL, ACCENT,
         "2. Chart-grounded graph\n23 waypoints (N/S channels + harbour),\n20 IALA buoys, berths at channel ends", 9)
    arrow(ax, 12, 35.5, 12, 32.7)
    rbox(ax, 2, 18.5, 20, 14, PANEL, LINE,
         "3. Environment contract\n(shared with Synthetic Port)\n\n- continuous kinematics, 6 sub-steps\n- Gaussian sensor noise\n- packet-loss / stale comms\n- static obstacles + collisions\n- IALA channel-keeping check\n- docking accuracy, cross-track error",
         8.3, tc=INK)
    arrow(ax, 22, 25.5, 27, 25.5)

    # ---- Stage 2: four study blocks (middle) ----
    studies = [
        (27, 46.5, GREEN, "A. Core study", "4 DRL variants x 3 conditions x 8 seeds\n= 96 cells / 2,880 episodes\nQ: does the algorithm matter?"),
        (27, 35.0, ACCENT, "B. DQN improvement", "3 arms: base / +chart prior / +shaping\nQ: which design lever lifts success?"),
        (27, 23.5, ORANGE, "C. Multi-vessel  (d)", "2 vessels, 3 pairings x 3 cond x 6 seeds\n= 54 cells; inter-vessel CPA/collision\nQ: does the interaction partner matter?"),
        (27, 12.0, PURPLE, "D. Operational realism  (e)", "round trip (dock+return) + two-way\nhead-on traffic, x3 cond x6 seeds\nQ: can the stack sustain real ops?"),
    ]
    for x, y, c, t, b in studies:
        rbox(ax, x, y, 25, 9.4, PANEL, c, "", 9)
        ax.text(x + 1, y + 7.9, t, fontsize=10.5, color=c, fontweight="bold")
        ax.text(x + 1, y + 3.3, b, fontsize=8.2, color=INK, va="center")
        arrow(ax, x + 25, y + 4.7, 54.5, 31, color=c, lw=1.5)

    # ---- Stage 3: per-seed scalars ----
    rbox(ax, 54.5, 27, 15, 8.5, PANEL, GOLD,
         "4. Per-seed scalars\n(unit of inference)\nmean per cell x seed;\nCTE on SUCCESSFUL\nepisodes only", 8.8, bold=False)
    arrow(ax, 69.5, 31.2, 74, 31.2)

    # ---- Stage 4: statistics stack ----
    rbox(ax, 74, 43, 24, 7.5, PANEL, ACCENT,
         "5a. Factorial ANOVA\ny ~ factor x condition + seed;\npartial eta^2 (effect size)", 9)
    rbox(ax, 74, 33.5, 24, 7.5, PANEL, GREEN,
         "5b. Paired contrasts\nclean->harsh Wilcoxon + Holm,\nCohen d_z; null-honest", 9)
    rbox(ax, 74, 24, 24, 7.5, PANEL, PURPLE,
         "5c. Mixed-effects  (f)\ny ~ fixed + (1 | seed);\nICC = seed variance share", 9)
    for yy in (46.7, 37.2, 27.7):
        arrow(ax, 72, 31.2, 74, yy if yy > 31 else yy, color=MUTED, lw=1.2)
    arrow(ax, 86, 24, 86, 20.5, color=MUTED)

    # ---- Stage 5: synthesis ----
    rbox(ax, 60, 6, 38, 12.5, "#101c33", GOLD,
         "\n6. Factor-hierarchy synthesis\n\nReframes 'which DRL variant wins?' into\n'what LEVEL of the design stack confers robust,\ncomplete, multi-vessel behaviour?'\n\nmission/interaction  >=  control-arch / nav-prior\n>  perception / comms  >>  algorithm",
         9.4, bold=False, tc=INK)
    ax.text(79, 20.0, "SYNTHESIS", fontsize=10.5, color=GOLD, fontweight="bold", ha="center")

    fig.savefig(os.path.join(FIG, "figBPAN_methodology.png"), facecolor=BG, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "figBPAN_methodology.svg"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("  figBPAN_methodology.png/.svg")


def factor_hierarchy():
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    ax.text(2, 57.5, "BPAN — the design-stack factor hierarchy", fontsize=16, color=INK, fontweight="bold")
    ax.text(2, 54.3, "Levels ordered by measured variance explained (partial \u03b7\u00b2). Higher levels dominate robustness; the algorithm barely moves it.",
            fontsize=9.8, color=MUTED)

    # pyramid-style stacked bands; width encodes influence
    levels = [
        ("Mission & interaction structure", "round-trip composite, head-on traffic, vessel pairing", GOLD,
         "\u03b7\u00b2 \u2248 0.37 - 0.90", 1.00),
        ("Control architecture & navigation prior", "rule-based vs learned; chart prior; reward shaping", ORANGE,
         "\u03b7\u00b2 \u2248 0.30 - 0.45", 0.78),
        ("Perception & communications", "sensor noise, packet-loss (degradation condition)", ACCENT,
         "\u03b7\u00b2 \u2248 0.06 - 0.80 (metric-dependent)", 0.56),
        ("Learning algorithm (DQN variant)", "DQN / Double / Dueling / Dueling-Double", MUTED,
         "\u03b7\u00b2 \u2248 0.01 - 0.05  (negligible)", 0.34),
    ]
    y = 42.5; h = 8.0; gap = 2.0; cx = 50
    for name, detail, col, eta, frac in levels:
        w = 86 * frac; x = cx - w / 2
        rbox(ax, x, y, w, h, PANEL, col, "", 9)
        ax.add_patch(Rectangle((x, y), 0.9, h, color=col, zorder=4))
        ax.text(cx, y + h * 0.63, name, ha="center", fontsize=11.5, color=col, fontweight="bold")
        ax.text(cx, y + h * 0.28, detail, ha="center", fontsize=8.6, color=INK)
        ax.text(x + w + 1.2, y + h / 2, eta, ha="left", va="center", fontsize=9, color=col, fontweight="bold")
        y -= (h + gap)

    ax.annotate("", xy=(5, y + 1.5), xytext=(5, 42.5 + h),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))
    ax.text(3.1, (42.5 + h + y) / 2, "increasing leverage on robust, complete behaviour",
            rotation=90, va="center", ha="center", fontsize=9, color=MUTED)
    ax.text(50, 2.2, "Takeaway: invest in mission framing, control architecture and the chart prior before tuning the DRL algorithm.",
            ha="center", fontsize=9.6, color=GOLD)

    fig.savefig(os.path.join(FIG, "figBPAN_factor_hierarchy.png"), facecolor=BG, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "figBPAN_factor_hierarchy.svg"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("  figBPAN_factor_hierarchy.png/.svg")


def main():
    print("Rendering BPAN methodology + factor-hierarchy illustrations:")
    methodology()
    factor_hierarchy()
    print("-> results_bintulu/figures/")


if __name__ == "__main__":
    main()
