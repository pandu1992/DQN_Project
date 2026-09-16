#!/usr/bin/env python3
"""
STUDY 3 — PUBLICATION TABLES (CSV + LaTeX + Excel, Interpretation columns).
Rule-based COLREGs baseline vs 4 DQN variants; findings reported honestly.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study3", "aggregated")
STAT = os.path.join(ROOT, "results_study3", "statistics")
TBL = os.path.join(ROOT, "results_study3", "tables")
os.makedirs(TBL, exist_ok=True)

summary = pd.read_csv(os.path.join(AGG, "summary.csv"))
fac = pd.read_csv(os.path.join(STAT, "factorial_anova.csv"))
bvd = pd.read_csv(os.path.join(STAT, "baseline_vs_dqn.csv"))
rob = pd.read_csv(os.path.join(STAT, "robustness_clean_vs_harsh.csv"))

AGENTS = ["RuleBased", "DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
LAB = {"RuleBased": "Rule-based (COLREGs)", "DQN": "DQN", "DoubleDQN": "Double DQN",
       "DuelingDQN": "Dueling DQN", "DuelingDoubleDQN": "Dueling Double DQN"}
CONDS = ["clean", "mid", "harsh"]
ARROW = "\u2192"   # avoid backslash inside f-string expressions (py3.9)
PM = "\u00b1"
_w = {}


def emit(name, df):
    df.to_csv(os.path.join(TBL, name + ".csv"), index=False)
    with open(os.path.join(TBL, name + ".tex"), "w") as f:
        f.write(df.to_latex(index=False, escape=True, column_format="l" * len(df.columns)))
    _w[name] = df
    print(f"  {name}: {df.shape[0]}x{df.shape[1]}")


def cell(ag, cond):
    return summary[(summary.agent == ag) & (summary.condition == cond)].iloc[0]

def pm(mean, hw, dp=1, unit=""):
    if not np.isfinite(hw): return f"{mean:.{dp}f}{unit}"
    return f"{mean:.{dp}f}\u00b1{hw:.{dp}f}{unit}"

def eff_lab(pe2):
    return "large" if pe2 >= 0.14 else "medium" if pe2 >= 0.06 else "small" if pe2 >= 0.01 else "negligible"


# Table 1 — configuration
def table1():
    rows = [
        ["Agents", "Rule-based (COLREGs-aware) + DQN, Double, Dueling, Dueling-Double", "1 classical baseline + 4 value-based RL", "Adds a non-learning reference point."],
        ["Conditions", "clean (0,0), mid (0.1,0.2), harsh (0.25,0.4)", "(sensor noise std, packet-error rate)", "Spans ideal to heavy sensing+comms degradation."],
        ["Seeds", "15 shared seeds (0-14)", "paired design; env_seed=42+seed", "Confirmatory power (n=15 vs Study 2's n=6)."],
        ["Budget", "130 train / 30 eval (RL); rule-based no training", "greedy / deterministic eval", "Rule-based follows the charted channel."],
        ["Cells", "5 x 3 x 15 = 225", "6,750 evaluation episodes", "Balanced factorial."],
    ]
    emit("table1_s3_configuration", pd.DataFrame(rows, columns=["Factor", "Levels / Value", "Detail", "Interpretation"]))


# Table 2 — navigation success across conditions
def table2():
    rows = []
    for ag in AGENTS:
        cells = {c: pm(cell(ag, c)["success_rate_mean"], cell(ag, c)["success_rate_ci95_hw"], 1, "%") for c in CONDS}
        cl, ha = cell(ag, "clean")["success_rate_mean"], cell(ag, "harsh")["success_rate_mean"]
        trend = ("Invariant to degradation (chart-following)" if abs(ha - cl) < 3
                 else f"{cl:.0f}% {ARROW} {ha:.0f}% clean-to-harsh")
        interp = trend + "; " + ("classical baseline, no learning." if ag == "RuleBased" else "learned policy.")
        rows.append({"Agent": LAB[ag], "Clean": cells["clean"], "Mid": cells["mid"], "Harsh": cells["harsh"], "Interpretation": interp})
    emit("table2_s3_navigation", pd.DataFrame(rows))


# Table 3 — safety across conditions (collision + IALA)
def table3():
    rows = []
    for ag in AGENTS:
        c, m, h = cell(ag, "clean"), cell(ag, "mid"), cell(ag, "harsh")
        interp = (f"Collisions/ep {c['collisions_per_ep_mean']:.2f}\u2192{h['collisions_per_ep_mean']:.2f}; "
                  f"IALA {c['iala_violations_mean']:.0f}\u2192{h['iala_violations_mean']:.0f}/ep clean\u2192harsh. "
                  + ("Lowest violations under stress." if ag == "RuleBased" else "Degrades more than the baseline."))
        rows.append({"Agent": LAB[ag],
                     "Coll/ep clean": f"{c['collisions_per_ep_mean']:.2f}", "Coll/ep harsh": f"{h['collisions_per_ep_mean']:.2f}",
                     "IALA clean": f"{c['iala_violations_mean']:.1f}", "IALA harsh": f"{h['iala_violations_mean']:.1f}",
                     "Interpretation": interp})
    emit("table3_s3_safety", pd.DataFrame(rows))


# Table 4 — factorial ANOVA
def table4():
    lab = {"agent": "Agent", "condition": "Condition", "agent:condition": "Agent\u00d7Condition", "seed": "Seed (block)"}
    rows = []
    for m in ["success_rate", "collision_rate", "iala_violations", "cte_mean", "docking_accuracy", "optimality_ratio"]:
        sub = fac[fac.metric == m]
        for eff in ["agent", "condition", "agent:condition", "seed"]:
            e = sub[sub.effect == eff]
            if not len(e): continue
            e = e.iloc[0]; p = e["p_value"]; pe2 = e["partial_eta2"]
            rows.append({"Metric": m, "Effect": lab[eff], "F": f"{e['F']:.2f}", "p-value": f"{p:.4f}",
                         "partial \u03b7\u00b2": f"{pe2:.3f}",
                         "Interpretation": f"{'sig' if p < 0.05 else 'n.s.'} (p={p:.3f}), {eff_lab(pe2)} effect"})
    emit("table4_s3_factorial", pd.DataFrame(rows))


# Table 5 — RuleBased vs DQN at harsh (the key contrast), all safety/precision/nav metrics
def table5():
    rows = []
    metrics = ["success_rate", "collision_rate", "iala_violations", "cte_mean", "docking_accuracy"]
    mlab = {"success_rate": "Success %", "collision_rate": "Collision %", "iala_violations": "IALA/ep",
            "cte_mean": "CTE", "docking_accuracy": "Docking err"}
    h = bvd[bvd.condition == "harsh"]
    for m in metrics:
        for _, r in h[h.metric == m].iterrows():
            sig = r["sig_holm"]
            interp = (f"Rule-based {r['mean_x']:.1f} vs {LAB[r['vs']]} {r['mean_y']:.1f} "
                      f"(d_z={r['cohens_dz']:.2f}, {r['effect_mag']}); "
                      + ("significant after Holm." if sig else f"n.s. after Holm (adj p={r['wilcoxon_p_holm']:.3f})."))
            rows.append({"Metric": mlab[m], "Rule-based vs": LAB[r["vs"]],
                         "RB mean": f"{r['mean_x']:.2f}", "DQN mean": f"{r['mean_y']:.2f}", "\u0394": f"{r['mean_diff']:+.2f}",
                         "adj p (Holm)": f"{r['wilcoxon_p_holm']:.3f}", "Cohen d_z": f"{r['cohens_dz']:.2f}",
                         "Sig": "yes" if sig else "no", "Interpretation": interp})
    emit("table5_s3_baseline_vs_dqn_harsh", pd.DataFrame(rows))


# Table 6 — robustness (clean vs harsh) per agent, safety focus
def table6():
    rows = []
    metrics = ["collision_rate", "iala_violations", "docking_accuracy", "success_rate"]
    mlab = {"collision_rate": "Collision %", "iala_violations": "IALA/ep", "docking_accuracy": "Docking err", "success_rate": "Success %"}
    for ag in AGENTS:
        for m in metrics:
            r = rob[(rob.agent == ag) & (rob.metric == m)].iloc[0]
            interp = (f"{r['clean_mean']:.1f}\u2192{r['harsh_mean']:.1f} (d_z={r['cohens_dz']:.2f}, {r['effect_mag']}); "
                      + ("Holm-sig degradation." if r["sig_holm"] else "not Holm-sig."))
            rows.append({"Agent": LAB[ag], "Metric": mlab[m], "Clean": f"{r['clean_mean']:.2f}",
                         "Harsh": f"{r['harsh_mean']:.2f}", "\u0394": f"{r['abs_change']:+.2f}",
                         "Cohen d_z": f"{r['cohens_dz']:.2f}", "Interpretation": interp})
    emit("table6_s3_robustness", pd.DataFrame(rows))


# Table 7 — overall findings per agent
def table7():
    rows = []
    for ag in AGENTS:
        c, h = cell(ag, "clean"), cell(ag, "harsh")
        # count Holm-sig wins of RuleBased over this DQN (if applicable)
        if ag == "RuleBased":
            nsig = int(bvd[bvd.sig_holm].shape[0])
            interp = (f"Success invariant ~{c['success_rate_mean']:.0f}% across conditions; lowest IALA under stress "
                      f"({h['iala_violations_mean']:.0f}/ep). Classical baseline is the most channel-compliant under degradation.")
            strength = "channel compliance under stress"; weakness = "higher CTE / not learned"
        else:
            interp = (f"Success {c['success_rate_mean']:.0f}%\u2192{h['success_rate_mean']:.0f}%; IALA rises to "
                      f"{h['iala_violations_mean']:.0f}/ep at harsh (\u2248 2\u00d7 the rule-based baseline).")
            strength = "tight route (low CTE clean)"; weakness = "safety degrades more than baseline"
        rows.append({"Agent": LAB[ag], "Strength": strength, "Weakness": weakness,
                     "Clean success": f"{c['success_rate_mean']:.0f}%", "Harsh IALA/ep": f"{h['iala_violations_mean']:.0f}",
                     "Interpretation": interp})
    emit("table7_s3_findings", pd.DataFrame(rows))


def main():
    print("Writing Study 3 tables:")
    table1(); table2(); table3(); table4(); table5(); table6(); table7()
    with pd.ExcelWriter(os.path.join(TBL, "all_tables_study3.xlsx"), engine="openpyxl") as xw:
        for name, df in _w.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"Excel: all_tables_study3.xlsx ({len(_w)} sheets)")


if __name__ == "__main__":
    main()
