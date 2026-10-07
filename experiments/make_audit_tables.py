#!/usr/bin/env python3
"""
Audit tables: outcome-decomposed CTE + collision incidence + failure taxonomy,
for Study 2 and Study 3, as publication CSVs (with an Interpretation column).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TBL = os.path.join(ROOT, "results_audit", "tables")
os.makedirs(TBL, exist_ok=True)


def emit(name, df):
    df.to_csv(os.path.join(TBL, name + ".csv"), index=False)
    print(f"  {name}: {df.shape[0]}x{df.shape[1]}")


def study3_cte_table():
    s = pd.read_csv(os.path.join(ROOT, "results_study3", "aggregated", "outcome_decomposition_summary.csv"))
    AG = ["RuleBased", "DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
    LAB = {"RuleBased": "Rule-based", "DQN": "DQN", "DoubleDQN": "Double DQN",
           "DuelingDQN": "Dueling DQN", "DuelingDoubleDQN": "Dueling Double DQN"}
    rows = []
    for ag in AG:
        def g(c, k):
            r = s[(s.agent == ag) & (s.condition == c)]
            return float(r.iloc[0][k]) if len(r) else np.nan
        rows.append({
            "Agent": LAB[ag],
            "CTE|success clean": f"{g('clean','cte_success_mean'):.1f}",
            "CTE|success mid": f"{g('mid','cte_success_mean'):.1f}",
            "CTE|success harsh": f"{g('harsh','cte_success_mean'):.1f}",
            "CTE all-eps clean\u2192harsh": f"{g('clean','cte_all_mean'):.1f}\u2192{g('harsh','cte_all_mean'):.1f}",
            "Interpretation": (f"Tracking error on SUCCESS rises {g('clean','cte_success_mean'):.1f}\u2192"
                               f"{g('harsh','cte_success_mean'):.1f} clean\u2192harsh (true degradation); "
                               f"the all-episode aggregate ({g('clean','cte_all_mean'):.0f}\u2192{g('harsh','cte_all_mean'):.0f}) "
                               "is a misleading success/failure mixture."),
        })
    emit("tableAudit_cte_by_outcome_study3", pd.DataFrame(rows))


def success_hides_safety_table():
    rows = []
    for dir_, label, keycol, levels in [
        ("results_study3", "Study 3", "condition", ["clean", "mid", "harsh"]),
    ]:
        s = pd.read_csv(os.path.join(ROOT, dir_, "aggregated", "outcome_decomposition_summary.csv"))
        gcol = "agent" if "agent" in s.columns else "algorithm"
        for lvl in s[gcol].unique():
            for c in levels:
                r = s[(s[gcol] == lvl) & (s[keycol] == c)]
                if not len(r):
                    continue
                r = r.iloc[0]
                rows.append({
                    "Study": label, "Agent/Pairing": lvl, "Condition": c,
                    "Success %": f"{r['success_rate_mean']:.1f}",
                    "Collision% (all eps)": f"{r['collision_rate_all_mean']:.1f}",
                    "Collision% (successful eps)": f"{r['collision_rate_success_mean']:.1f}",
                    "Interpretation": ("Success is goal-reaching only; a large share of SUCCESSFUL episodes "
                                       "still collide, so success does not imply a safe trajectory."),
                })
    emit("tableAudit_success_hides_safety", pd.DataFrame(rows))


def failure_taxonomy_table():
    rows = []
    for dir_, label in [("results_v3", "Study 2"), ("results_study3", "Study 3")]:
        t = pd.read_csv(os.path.join(ROOT, dir_, "aggregated", "failure_taxonomy.csv"))
        rows.append({
            "Study": label,
            "Episodes": int(t["n_eval"].sum()),
            "% success (goal)": f"{100.0*t['n_success'].sum()/t['n_eval'].sum():.1f}",
            "% timeout failure": f"{100.0*t['n_fail_timeout'].sum()/t['n_eval'].sum():.1f}",
            "% other failure": f"{100.0*t['n_fail_other'].sum()/t['n_eval'].sum():.1f}",
            "Interpretation": ("endOnCollision=false and no off-channel termination: the ONLY failure mode "
                               "is timeout. Collisions/IALA are within-episode safety metrics, orthogonal to "
                               "the success/failure outcome."),
        })
    emit("tableAudit_failure_taxonomy", pd.DataFrame(rows))


def main():
    print("Writing audit tables:")
    study3_cte_table()
    success_hides_safety_table()
    failure_taxonomy_table()


if __name__ == "__main__":
    main()
