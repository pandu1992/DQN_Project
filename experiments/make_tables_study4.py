#!/usr/bin/env python3
"""
STUDY 4 — PUBLICATION TABLES (CSV + LaTeX + Excel, Interpretation columns).
Two independent vessels that must avoid each other; findings reported honestly.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study4", "aggregated")
STAT = os.path.join(ROOT, "results_study4", "statistics")
TBL = os.path.join(ROOT, "results_study4", "tables")
os.makedirs(TBL, exist_ok=True)

summary = pd.read_csv(os.path.join(AGG, "summary.csv"))
fac = pd.read_csv(os.path.join(STAT, "factorial_anova.csv"))
rvo = pd.read_csv(os.path.join(STAT, "reference_vs_pairings.csv"))
rob = pd.read_csv(os.path.join(STAT, "robustness_clean_vs_harsh.csv"))

PAIRINGS = ["DQN_vs_DQN", "DDQN_vs_DDQN", "DQN_vs_RuleBased", "RuleBased_vs_RuleBased"]
LAB = {"DQN_vs_DQN": "DQN vs DQN", "DDQN_vs_DDQN": "Duel-Double vs Duel-Double",
       "DQN_vs_RuleBased": "DQN vs Rule-based", "RuleBased_vs_RuleBased": "Rule-based vs Rule-based"}
CONDS = ["clean", "mid", "harsh"]
ARROW = "\u2192"
_w = {}


def emit(name, df):
    df.to_csv(os.path.join(TBL, name + ".csv"), index=False)
    with open(os.path.join(TBL, name + ".tex"), "w") as f:
        f.write(df.to_latex(index=False, escape=True, column_format="l" * len(df.columns)))
    _w[name] = df
    print(f"  {name}: {df.shape[0]}x{df.shape[1]}")


def cell(pr, cond):
    return summary[(summary.pairing == pr) & (summary.condition == cond)].iloc[0]

def pm(mean, hw, dp=1, unit=""):
    if not np.isfinite(hw): return f"{mean:.{dp}f}{unit}"
    return f"{mean:.{dp}f}\u00b1{hw:.{dp}f}{unit}"

def eff_lab(pe2):
    return "large" if pe2 >= 0.14 else "medium" if pe2 >= 0.06 else "small" if pe2 >= 0.01 else "negligible"


def table1():
    rows = [
        ["Vessels", "Two INDEPENDENT autonomous vessels (A, B)", "no central controller", "Each has its own policy + mission; they must avoid each other."],
        ["Pairings", "DQN\u00d7DQN, DuelDouble\u00d7DuelDouble, DQN\u00d7Rule-based, Rule-based\u00d7Rule-based", "2 symmetric-learned, 1 mixed, 1 classical reference", "Learned-vs-learned, learned-vs-classical, classical-vs-classical."],
        ["Conditions", "clean (0,0), mid (0.1,0.2), harsh (0.25,0.4)", "(sensor noise std, packet-error rate)", "Study-2 degradation grid, layered on the encounter."],
        ["Seeds", "8 shared seeds (0-7)", "paired design; env_seed=42+seed", "Vessel B mission re-rolled so the two never share a berth."],
        ["Cells", "4 x 3 x 8 = 96", "5,760 per-vessel eval rows (2,880 encounters)", "Balanced factorial; 30 episodes/cell x 2 vessels."],
        ["Safety event", "inter-vessel collision (hull sep < 18) + near-miss (< 40)", "distinct events + min CPA", "Measured from synchronized continuous geometry."],
    ]
    emit("table1_s4_configuration", pd.DataFrame(rows, columns=["Factor", "Levels / Value", "Detail", "Interpretation"]))


def table2_intervessel_safety():
    rows = []
    for pr in PAIRINGS:
        cr = {c: pm(cell(pr, c)["vessel_collision_rate_mean"], cell(pr, c)["vessel_collision_rate_ci95_hw"], 1, "%") for c in CONDS}
        cpaH = cell(pr, "harsh")["min_cpa_mean"]
        interp = (f"Min CPA {cell(pr,'clean')['min_cpa_mean']:.0f}{ARROW}{cpaH:.0f} clean{ARROW}harsh. "
                  + ("Classical pair ignores the moving partner \u2192 highest collision rate." if pr == "RuleBased_vs_RuleBased"
                     else "Learned/mixed pair keeps vessels farther apart (larger CPA, fewer collisions)."))
        rows.append({"Pairing": LAB[pr], "Coll% clean": cr["clean"], "Coll% mid": cr["mid"], "Coll% harsh": cr["harsh"],
                     "Interpretation": interp})
    emit("table2_s4_intervessel_safety", pd.DataFrame(rows))


def table3_tradeoff():
    """The core safety-vs-completion tradeoff."""
    rows = []
    for pr in PAIRINGS:
        c, h = cell(pr, "clean"), cell(pr, "harsh")
        interp = (f"Success {c['success_rate_mean']:.0f}%; collisions/enc {c['vessel_collisions_per_ep_mean']:.2f}; "
                  f"CPA {c['min_cpa_mean']:.0f}. "
                  + ("High completion but blind to partner (most collisions, smallest CPA)." if pr == "RuleBased_vs_RuleBased"
                     else "Avoids the partner (fewer collisions, larger CPA) at a large completion cost."))
        rows.append({"Pairing": LAB[pr],
                     "Success % (clean)": f"{c['success_rate_mean']:.1f}",
                     "Coll/enc (clean)": f"{c['vessel_collisions_per_ep_mean']:.2f}",
                     "Near-miss/enc (clean)": f"{c['near_misses_per_ep_mean']:.2f}",
                     "Min CPA (clean)": f"{c['min_cpa_mean']:.0f}",
                     "Interpretation": interp})
    emit("table3_s4_tradeoff", pd.DataFrame(rows))


def table4_factorial():
    lab = {"pairing": "Pairing", "condition": "Condition", "pairing:condition": "Pairing\u00d7Condition", "seed": "Seed (block)"}
    rows = []
    for m in ["vessel_collision_rate", "vessel_collisions_per_ep", "near_misses_per_ep", "min_cpa", "success_rate", "docking_accuracy"]:
        sub = fac[fac.metric == m]
        for eff in ["pairing", "condition", "pairing:condition", "seed"]:
            e = sub[sub.effect == eff]
            if not len(e): continue
            e = e.iloc[0]; p = e["p_value"]; pe2 = e["partial_eta2"]
            rows.append({"Metric": m, "Effect": lab[eff], "F": f"{e['F']:.2f}", "p-value": f"{p:.4f}",
                         "partial \u03b7\u00b2": f"{pe2:.3f}",
                         "Interpretation": f"{'sig' if p < 0.05 else 'n.s.'} (p={p:.3f}), {eff_lab(pe2)} effect"})
    emit("table4_s4_factorial", pd.DataFrame(rows))


def table5_reference_contrasts():
    """Classical reference (Rule-based x2) vs each pairing at HARSH."""
    rows = []
    metrics = ["vessel_collision_rate", "min_cpa", "near_misses_per_ep", "success_rate"]
    mlab = {"vessel_collision_rate": "Collision %", "min_cpa": "Min CPA",
            "near_misses_per_ep": "Near-miss/enc", "success_rate": "Success %"}
    h = rvo[rvo.condition == "harsh"]
    for m in metrics:
        for _, r in h[h.metric == m].iterrows():
            sig = r["sig_holm"]
            interp = (f"Rule-based\u00d72 {r['mean_x']:.1f} vs {LAB[r['vs']]} {r['mean_y']:.1f} "
                      f"(d_z={r['cohens_dz']:.2f}, {r['effect_mag']}); "
                      + ("Holm-significant." if sig else f"n.s. after Holm (adj p={r['wilcoxon_p_holm']:.3f})."))
            rows.append({"Metric": mlab[m], "Reference vs": LAB[r["vs"]],
                         "RB\u00d72 mean": f"{r['mean_x']:.2f}", "Pairing mean": f"{r['mean_y']:.2f}", "\u0394": f"{r['mean_diff']:+.2f}",
                         "adj p (Holm)": f"{r['wilcoxon_p_holm']:.3f}", "Cohen d_z": f"{r['cohens_dz']:.2f}",
                         "Sig": "yes" if sig else "no", "Interpretation": interp})
    emit("table5_s4_reference_contrasts_harsh", pd.DataFrame(rows))


def table6_robustness():
    rows = []
    metrics = ["vessel_collision_rate", "min_cpa", "near_misses_per_ep", "success_rate"]
    mlab = {"vessel_collision_rate": "Collision %", "min_cpa": "Min CPA", "near_misses_per_ep": "Near-miss/enc", "success_rate": "Success %"}
    for pr in PAIRINGS:
        for m in metrics:
            r = rob[(rob.pairing == pr) & (rob.metric == m)].iloc[0]
            interp = (f"{r['clean_mean']:.1f}\u2192{r['harsh_mean']:.1f} (d_z={r['cohens_dz']:.2f}, {r['effect_mag']}); "
                      + ("Holm-sig." if r["sig_holm"] else "not Holm-sig at n=8 (see omnibus)."))
            rows.append({"Pairing": LAB[pr], "Metric": mlab[m], "Clean": f"{r['clean_mean']:.2f}",
                         "Harsh": f"{r['harsh_mean']:.2f}", "\u0394": f"{r['abs_change']:+.2f}",
                         "Cohen d_z": f"{r['cohens_dz']:.2f}", "Interpretation": interp})
    emit("table6_s4_robustness", pd.DataFrame(rows))


def table7_findings():
    rows = []
    for pr in PAIRINGS:
        c, h = cell(pr, "clean"), cell(pr, "harsh")
        if pr == "RuleBased_vs_RuleBased":
            strength = "highest mission completion"; weakness = "no partner modelling \u2192 most collisions, smallest CPA"
        elif pr == "DQN_vs_RuleBased":
            strength = "mixed: classical vessel completes, learned vessel yields"; weakness = "learned vessel completion low"
        else:
            strength = "largest CPA / fewest inter-vessel collisions"; weakness = "large drop in mission completion"
        interp = (f"Success {c['success_rate_mean']:.0f}%\u2192{h['success_rate_mean']:.0f}%, "
                  f"collision {c['vessel_collision_rate_mean']:.0f}%\u2192{h['vessel_collision_rate_mean']:.0f}%, "
                  f"CPA {c['min_cpa_mean']:.0f}\u2192{h['min_cpa_mean']:.0f} clean\u2192harsh.")
        rows.append({"Pairing": LAB[pr], "Strength": strength, "Weakness": weakness,
                     "Clean success": f"{c['success_rate_mean']:.0f}%", "Clean collision%": f"{c['vessel_collision_rate_mean']:.0f}%",
                     "Interpretation": interp})
    emit("table7_s4_findings", pd.DataFrame(rows))


def main():
    print("Writing Study 4 tables:")
    table1(); table2_intervessel_safety(); table3_tradeoff(); table4_factorial()
    table5_reference_contrasts(); table6_robustness(); table7_findings()
    with pd.ExcelWriter(os.path.join(TBL, "all_tables_study4.xlsx"), engine="openpyxl") as xw:
        for name, df in _w.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"Excel: all_tables_study4.xlsx ({len(_w)} sheets)")


if __name__ == "__main__":
    main()
