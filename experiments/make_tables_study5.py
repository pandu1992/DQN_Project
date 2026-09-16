#!/usr/bin/env python3
"""
STUDY 5 — PUBLICATION TABLES (CSV + LaTeX + Excel, Interpretation columns).
Two-phase round trip + two-way opposing traffic; findings reported honestly.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study5", "aggregated")
STAT = os.path.join(ROOT, "results_study5", "statistics")
TBL = os.path.join(ROOT, "results_study5", "tables")
os.makedirs(TBL, exist_ok=True)

phase_s = pd.read_csv(os.path.join(AGG, "twophase_summary.csv"))
way_s = pd.read_csv(os.path.join(AGG, "twoway_summary.csv"))
dvf = pd.read_csv(os.path.join(STAT, "twophase_dock_vs_full.csv"))
wf = pd.read_csv(os.path.join(STAT, "twoway_factorial_anova.csv"))
rvo = pd.read_csv(os.path.join(STAT, "twoway_reference_vs_pairings.csv"))
wrob = pd.read_csv(os.path.join(STAT, "twoway_robustness.csv"))

CONDS = ["clean", "mid", "harsh"]
PHASE_AGENTS = ["RuleBased", "DQN"]
PLAB = {"RuleBased": "Rule-based (COLREGs)", "DQN": "DQN"}
WAY_PAIRINGS = ["Rule_vs_Rule", "DQN_vs_Rule", "DQN_vs_DQN"]
WLAB = {"Rule_vs_Rule": "Rule vs Rule", "DQN_vs_Rule": "DQN vs Rule", "DQN_vs_DQN": "DQN vs DQN"}
ARROW = "\u2192"
_w = {}


def emit(name, df):
    df.to_csv(os.path.join(TBL, name + ".csv"), index=False)
    with open(os.path.join(TBL, name + ".tex"), "w") as f:
        f.write(df.to_latex(index=False, escape=True, column_format="l" * len(df.columns)))
    _w[name] = df
    print(f"  {name}: {df.shape[0]}x{df.shape[1]}")


def pcell(ag, cond):
    return phase_s[(phase_s.agent == ag) & (phase_s.condition == cond)].iloc[0]

def wcell(pr, cond):
    return way_s[(way_s.pairing == pr) & (way_s.condition == cond)].iloc[0]

def pm(mean, hw, dp=1, unit=""):
    if not np.isfinite(hw): return f"{mean:.{dp}f}{unit}"
    return f"{mean:.{dp}f}\u00b1{hw:.{dp}f}{unit}"

def eff_lab(pe2):
    return "large" if pe2 >= 0.14 else "medium" if pe2 >= 0.06 else "small" if pe2 >= 0.01 else "negligible"


def table1():
    rows = [
        ["Scenario A", "Two-phase round trip (single vessel)", "inbound \u2192 dock \u2192 outbound in one episode", "Success = FULL cycle; a realistic port call, not one-way."],
        ["Scenario B", "Two-way opposing traffic (two vessels)", "one inbound + one outbound on a shared lane", "Forces head-on encounters in the same water."],
        ["Agents (A)", "Rule-based (COLREGs) + DQN", "RuleBased primary; DQN learned comparator", "DQN's round-trip limitation reported honestly."],
        ["Pairings (B)", "Rule\u00d7Rule, DQN\u00d7Rule, DQN\u00d7DQN", "classical / mixed / learned", "Rule\u00d7Rule is the competent reference."],
        ["Conditions", "clean (0,0), mid (0.1,0.2), harsh (0.25,0.4)", "(sensor noise, packet-error)", "Study-2 degradation grid."],
        ["Seeds", "8 shared seeds (0-7)", "paired; env_seed=42+seed", "Cells: A 2\u00d73\u00d78=48; B 3\u00d73\u00d78=72."],
    ]
    emit("table1_s5_configuration", pd.DataFrame(rows, columns=["Factor", "Levels / Value", "Detail", "Interpretation"]))


def table2_phase_success():
    rows = []
    for ag in PHASE_AGENTS:
        for cond in CONDS:
            c = pcell(ag, cond)
            rows.append({"Agent": PLAB[ag], "Condition": cond,
                         "Dock success %": pm(c["dock_success_rate_mean"], c["dock_success_rate_ci95_hw"], 1),
                         "Full-cycle success %": pm(c["full_cycle_success_rate_mean"], c["full_cycle_success_rate_ci95_hw"], 1),
                         "Interpretation": ("Outbound leg roughly HALVES success vs docking alone."
                                            if ag == "RuleBased" else "Learned agent rarely completes the round trip.")})
    emit("table2_s5_twophase_success", pd.DataFrame(rows))


def table3_dock_vs_full():
    rows = []
    for _, r in dvf.iterrows():
        interp = (f"Dock {r['dock_mean']:.0f}% {ARROW} full {r['full_mean']:.0f}% "
                  f"(drop {r['drop']:.0f} pts, d_z={r['cohens_dz']:.2f}, {r['effect_mag']}); "
                  + ("Holm-sig." if r["sig_holm"] else "large but not Holm-sig at n=8."))
        rows.append({"Agent": PLAB[r["agent"]], "Condition": r["condition"],
                     "Dock %": f"{r['dock_mean']:.1f}", "Full %": f"{r['full_mean']:.1f}",
                     "Drop (pts)": f"{r['drop']:.1f}", "adj p (Holm)": f"{r['wilcoxon_p_holm']:.3f}",
                     "Cohen d_z": f"{r['cohens_dz']:.2f}", "Interpretation": interp})
    emit("table3_s5_dock_vs_full", pd.DataFrame(rows))


def table4_twoway_safety():
    rows = []
    for pr in WAY_PAIRINGS:
        cr = {c: pm(wcell(pr, c)["vessel_collision_rate_mean"], wcell(pr, c)["vessel_collision_rate_ci95_hw"], 1, "%") for c in CONDS}
        interp = ("Competent classical traffic: vessels transit and meet head-on."
                  if pr == "Rule_vs_Rule" else
                  ("Mixed: the classical vessel carries the traffic."
                   if pr == "DQN_vs_Rule" else
                   "Learned pair barely transits (low collisions = a stall artefact, not safety)."))
        rows.append({"Pairing": WLAB[pr], "Coll% clean": cr["clean"], "Coll% mid": cr["mid"], "Coll% harsh": cr["harsh"],
                     "Interpretation": interp})
    emit("table4_s5_twoway_safety", pd.DataFrame(rows))


def table5_twoway_factorial():
    lab = {"pairing": "Pairing", "condition": "Condition", "pairing:condition": "Pairing\u00d7Condition", "seed": "Seed (block)"}
    rows = []
    for m in ["vessel_collision_rate", "min_cpa", "head_on_events", "success_rate"]:
        sub = wf[wf.metric == m]
        for eff in ["pairing", "condition", "pairing:condition", "seed"]:
            e = sub[sub.effect == eff]
            if not len(e): continue
            e = e.iloc[0]; p = e["p_value"]; pe2 = e["partial_eta2"]
            rows.append({"Metric": m, "Effect": lab[eff], "F": f"{e['F']:.2f}", "p-value": f"{p:.4f}",
                         "partial \u03b7\u00b2": f"{pe2:.3f}",
                         "Interpretation": f"{'sig' if p < 0.05 else 'n.s.'} (p={p:.3f}), {eff_lab(pe2)} effect"})
    emit("table5_s5_twoway_factorial", pd.DataFrame(rows))


def table6_reference_contrasts():
    rows = []
    metrics = ["vessel_collision_rate", "min_cpa", "success_rate"]
    mlab = {"vessel_collision_rate": "Collision %", "min_cpa": "Min CPA", "success_rate": "Success %"}
    h = rvo[rvo.condition == "harsh"]
    for m in metrics:
        for _, r in h[h.metric == m].iterrows():
            sig = r["sig_holm"]
            interp = (f"Rule\u00d7Rule {r['mean_x']:.1f} vs {WLAB[r['vs']]} {r['mean_y']:.1f} "
                      f"(d_z={r['cohens_dz']:.2f}, {r['effect_mag']}); "
                      + ("Holm-significant." if sig else f"n.s. after Holm (adj p={r['wilcoxon_p_holm']:.3f})."))
            rows.append({"Metric": mlab[m], "Reference vs": WLAB[r["vs"]],
                         "Rule\u00d7Rule": f"{r['mean_x']:.2f}", "Pairing": f"{r['mean_y']:.2f}", "\u0394": f"{r['mean_diff']:+.2f}",
                         "adj p (Holm)": f"{r['wilcoxon_p_holm']:.3f}", "Cohen d_z": f"{r['cohens_dz']:.2f}",
                         "Sig": "yes" if sig else "no", "Interpretation": interp})
    emit("table6_s5_reference_contrasts_harsh", pd.DataFrame(rows))


def table7_findings():
    rows = [
        ["Two-phase (RuleBased)", "Completes the full port call ~half as often as it docks",
         f"dock {pcell('RuleBased','clean')['dock_success_rate_mean']:.0f}% \u2192 full {pcell('RuleBased','clean')['full_cycle_success_rate_mean']:.0f}%",
         "The outbound leg is a real, large added burden; invariant to degradation."],
        ["Two-phase (DQN)", "Rarely completes the round trip",
         f"full-cycle \u2248 {pcell('DQN','clean')['full_cycle_success_rate_mean']:.0f}% (clean)",
         "Learned value-based control does not master the two-phase mission here."],
        ["Two-way (Rule\u00d7Rule)", "Competent opposing traffic with genuine head-on encounters",
         f"success {wcell('Rule_vs_Rule','clean')['success_rate_mean']:.0f}%, collision {wcell('Rule_vs_Rule','clean')['vessel_collision_rate_mean']:.0f}%",
         "The classical controller carries two-way traffic; it does not model the partner, so it collides head-on."],
        ["Two-way (DQN pairs)", "Barely transit \u2192 degenerate low collisions",
         f"DQN\u00d7DQN success {wcell('DQN_vs_DQN','clean')['success_rate_mean']:.0f}% (clean)",
         "Low collisions reflect a stall artefact, not learned avoidance (honest)."],
    ]
    emit("table7_s5_findings", pd.DataFrame(rows, columns=["Scenario", "Finding", "Key numbers", "Interpretation"]))


def main():
    print("Writing Study 5 tables:")
    table1(); table2_phase_success(); table3_dock_vs_full(); table4_twoway_safety()
    table5_twoway_factorial(); table6_reference_contrasts(); table7_findings()
    with pd.ExcelWriter(os.path.join(TBL, "all_tables_study5.xlsx"), engine="openpyxl") as xw:
        for name, df in _w.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"Excel: all_tables_study5.xlsx ({len(_w)} sheets)")


if __name__ == "__main__":
    main()
