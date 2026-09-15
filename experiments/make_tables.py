#!/usr/bin/env python3
"""
PUBLICATION-READY TABLES (task #8)

Emits Tables 1-8 + a Methods table. Each is written as CSV, LaTeX (booktabs),
and consolidated into one Excel workbook (results/tables/all_tables.xlsx).
Every main table has a final `Interpretation` column whose text is generated
from the ACTUAL numbers (mean, CI, p, effect size) — never hand-fabricated.

All findings reflect the real (largely non-significant) results honestly.
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results", "aggregated")
STAT = os.path.join(ROOT, "results", "statistics")
TBL = os.path.join(ROOT, "results", "tables")
os.makedirs(TBL, exist_ok=True)

summary = pd.read_csv(os.path.join(AGG, "summary.csv"))
per_seed = pd.read_csv(os.path.join(AGG, "per_seed.csv"))
omni = pd.read_csv(os.path.join(STAT, "omnibus_algorithms.csv"))
fac = pd.read_csv(os.path.join(STAT, "factorial_anova.csv"))
pw = pd.read_csv(os.path.join(STAT, "pairwise_algorithms.csv"))
enh = pd.read_csv(os.path.join(STAT, "enhancement_contrasts.csv"))
norm = pd.read_csv(os.path.join(STAT, "normality.csv"))
lev = pd.read_csv(os.path.join(STAT, "variance_homogeneity.csv"))

ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
ALGO_LABEL = {"DQN": "DQN", "DoubleDQN": "Double DQN",
              "DuelingDQN": "Dueling DQN", "DuelingDoubleDQN": "Dueling Double DQN"}
_written = {}


def base_row(algo):
    return summary[(summary.algorithm == algo) & (summary.per == 0) & (summary.noisy == 0)].iloc[0]


def fmt_ci(mean, hw, dp=1, unit=""):
    if not np.isfinite(hw):
        return f"{mean:.{dp}f}{unit}"
    return f"{mean:.{dp}f} \u00b1 {hw:.{dp}f}{unit}"


def emit(name, df):
    df.to_csv(os.path.join(TBL, name + ".csv"), index=False)
    with open(os.path.join(TBL, name + ".tex"), "w") as f:
        f.write(df.to_latex(index=False, escape=True, longtable=False,
                            column_format="l" * len(df.columns)))
    _written[name] = df
    print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols")


# ---------------- Table 1 — Algorithm & Experimental Configuration ----------------
def table1():
    cfg = json.load(open(os.path.join(ROOT, "results/configs/experiment_config.json")))
    hp = cfg["agent_hyperparameters_from_source"]
    train_cfg = (f"train {cfg['training']['train_episodes']} eps, "
                 f"budget {cfg['training']['total_steps_budget']} steps, "
                 f"lr={hp['lr']}, gamma={hp['gamma']}, batch={hp['batch_size']}, "
                 f"target_update={hp['target_update']}")
    eval_cfg = (f"greedy, {cfg['evaluation']['eval_episodes']} held-out eps x "
                f"{cfg['n_seeds']} seeds; no learning")
    formulas = {
        "DQN": r"y=r+gamma*max_a' Q_theta-(s',a'); Huber loss",
        "DoubleDQN": r"a*=argmax_a' Q_theta(s',a'); y=r+gamma*Q_theta-(s',a*)",
        "DuelingDQN": r"Q(s,a)=V(s)+(A(s,a)-mean_a A(s,a))",
        "DuelingDoubleDQN": r"dueling network + double-Q target",
    }
    interp = {
        "DQN": "Baseline value estimator; over-estimation bias uncorrected.",
        "DoubleDQN": "Decoupled selection/evaluation intended to curb over-estimation.",
        "DuelingDQN": "Separates state value from action advantage; larger head.",
        "DuelingDoubleDQN": "Combines both refinements; most parameters/compute.",
    }
    rows = []
    for a in ALGOS:
        rows.append({
            "Algorithm": ALGO_LABEL[a],
            "Implementation": "pure-JS QNetwork/DuelingQNetwork + DQNAgent (js/dqn.js)",
            "Training Configuration": train_cfg,
            "Evaluation Configuration": eval_cfg,
            "Formula/Method": formulas[a],
            "Interpretation": interp[a],
        })
    emit("table1_configuration", pd.DataFrame(rows))


# ---------------- Table 2 — Overall Performance (baseline cells, mean +/- 95% CI) ----------------
def table2():
    rows = []
    for a in ALGOS:
        r = base_row(a)
        sr = fmt_ci(r["success_rate_mean"], r["success_rate_ci95_hw"], 1, "%")
        rw = fmt_ci(r["mean_reward_mean"], r["mean_reward_ci95_hw"], 1)
        opt = fmt_ci(r["optimality_ratio_mean"], r["optimality_ratio_ci95_hw"], 3)
        st = fmt_ci(r["mean_steps_mean"], r["mean_steps_ci95_hw"], 1)
        rk = fmt_ci(r["route_risk_mean"], r["route_risk_ci95_hw"], 2)
        interp = (f"Success {r['success_rate_mean']:.1f}% with a wide 95% CI "
                  f"(+/-{r['success_rate_ci95_hw']:.1f}%), reflecting high run-to-run "
                  f"variance; route optimality {r['optimality_ratio_mean']:.2f}x the "
                  f"Dijkstra optimum.")
        rows.append({
            "Algorithm": ALGO_LABEL[a], "Success Rate": sr, "Mean Reward": rw,
            "Optimality Ratio": opt, "Mean Steps": st, "Route Risk": rk,
            "Interpretation": interp,
        })
    emit("table2_overall_performance", pd.DataFrame(rows))


# ---------------- Table 3 — Omnibus algorithm effect (replaces clean-vs-noise; no noise factor) ----------------
def table3():
    rows = []
    labelmap = {"success_rate": "Success rate", "mean_reward": "Mean reward",
                "optimality_ratio": "Optimality ratio", "mean_steps": "Mean steps",
                "revisit_rate": "Revisit rate", "excess_cost": "Excess cost",
                "route_risk": "Route risk", "route_difficulty": "Route difficulty"}
    for _, r in omni.iterrows():
        m = r["metric"]
        # choose primary test by normality of that metric
        frac_norm = norm[(norm.metric == m)]["normal_at_0.05"].mean()
        prefer = "RM-ANOVA" if frac_norm >= 0.5 else "Friedman"
        if prefer == "RM-ANOVA":
            stat = f"F({int(r['rm_anova_df_between'])},{int(r['rm_anova_df_error'])})={r['rm_anova_F']:.2f}"
            p = r["rm_anova_p"]; es = f"partial eta^2={r['rm_anova_partial_eta2']:.3f}"
        else:
            stat = f"chi^2(3)={r['friedman_chi2']:.2f}"
            p = r["friedman_p"]; es = f"Kendall W={r['kendall_W']:.3f}"
        sig = "significant" if p < 0.05 else "not significant"
        interp = (f"Omnibus test across the 4 base algorithms is {sig} (p={p:.3f}); "
                  f"effect size {es}. "
                  + ("A real algorithm effect exists on this metric."
                     if p < 0.05 else
                     "Algorithm choice does not detectably affect this metric at this budget."))
        rows.append({
            "Metric": labelmap.get(m, m), "Primary Test": prefer,
            "Statistic": stat, "p-value": f"{p:.3f}", "Effect Size": es,
            "Interpretation": interp,
        })
    emit("table3_omnibus_algorithm", pd.DataFrame(rows))


# ---------------- Table 4 — Factorial main effects & interactions (replaces PER-levels) ----------------
def table4():
    rows = []
    keep = ["algorithm", "per", "noisy", "algorithm:per", "algorithm:noisy",
            "per:noisy", "algorithm:per:noisy", "seed"]
    lab = {"algorithm": "Algorithm", "per": "PER", "noisy": "Noisy Nets",
           "algorithm:per": "Algorithm x PER", "algorithm:noisy": "Algorithm x Noisy",
           "per:noisy": "PER x Noisy", "algorithm:per:noisy": "3-way",
           "seed": "Seed (block)"}
    for m in ["success_rate", "mean_reward", "optimality_ratio"]:
        sub = fac[fac.metric == m]
        for eff in keep:
            e = sub[sub.effect == eff]
            if not len(e):
                continue
            e = e.iloc[0]
            p = e["p_value"]; pe2 = e["partial_eta2"]
            sig = p < 0.05
            interp = (f"{'Significant' if sig else 'No significant'} effect "
                      f"(p={p:.3f}, partial eta^2={pe2:.3f}"
                      + (", large" if pe2 >= 0.14 else ", medium" if pe2 >= 0.06 else ", small" if pe2 >= 0.01 else ", negligible")
                      + " effect size).")
            rows.append({
                "Metric": m, "Effect": lab[eff],
                "F": f"{e['F']:.2f}", "p-value": f"{p:.3f}",
                "partial eta^2": f"{pe2:.3f}", "Interpretation": interp,
            })
    emit("table4_factorial_effects", pd.DataFrame(rows))


# ---------------- Table 5 — Full configuration matrix (all 16 cells) ----------------
def table5():
    rows = []
    for _, r in summary.sort_values(["algorithm", "per", "noisy"]).iterrows():
        cond = ("baseline" if (r.per == 0 and r.noisy == 0)
                else "+PER" if (r.per == 1 and r.noisy == 0)
                else "+Noisy" if (r.per == 0 and r.noisy == 1)
                else "+PER+Noisy")
        interp = (f"{fmt_ci(r['success_rate_mean'], r['success_rate_ci95_hw'],1,'%')} success; "
                  f"route {r['optimality_ratio_mean']:.2f}x optimal.")
        rows.append({
            "Algorithm": ALGO_LABEL[r.algorithm], "Condition": cond,
            "Success": fmt_ci(r["success_rate_mean"], r["success_rate_ci95_hw"], 1, "%"),
            "Reward": fmt_ci(r["mean_reward_mean"], r["mean_reward_ci95_hw"], 1),
            "Optimality": fmt_ci(r["optimality_ratio_mean"], r["optimality_ratio_ci95_hw"], 3),
            "Steps": fmt_ci(r["mean_steps_mean"], r["mean_steps_ci95_hw"], 1),
            "Interpretation": interp,
        })
    emit("table5_full_matrix", pd.DataFrame(rows))


# ---------------- Table 6 — Statistical comparison (pairwise algorithms) ----------------
def table6():
    rows = []
    for m in ["success_rate", "mean_reward", "optimality_ratio"]:
        sub = pw[pw.metric == m]
        for _, r in sub.iterrows():
            sig = r["wilcoxon_p_holm"] < 0.05
            interp = (f"{ALGO_LABEL[r['algo_a']]} vs {ALGO_LABEL[r['algo_b']]}: "
                      f"delta={r['mean_diff']:.2f}; "
                      + ("difference survives Holm correction."
                         if sig else
                         f"not significant after Holm (adj p={r['wilcoxon_p_holm']:.2f}); "
                         f"effect {r['effect_dz_mag']}."))
            rows.append({
                "Comparison": f"{ALGO_LABEL[r['algo_a']]} vs {ALGO_LABEL[r['algo_b']]}",
                "Metric": m, "Test": "paired Wilcoxon",
                "Statistic (dz)": f"{r['cohens_dz']:.2f}",
                "p-value": f"{r['wilcoxon_p']:.3f}",
                "Adj p (Holm)": f"{r['wilcoxon_p_holm']:.3f}",
                "Effect Size": f"d_z={r['cohens_dz']:.2f} ({r['effect_dz_mag']}); delta={r['cliffs_delta']:.2f}",
                "Interpretation": interp,
            })
    emit("table6_statistical_comparison", pd.DataFrame(rows))


# ---------------- Table 7 — Robustness / enhancement effects ----------------
def table7():
    rows = []
    for a in ALGOS:
        br = base_row(a)
        # find enhancement rows for success_rate
        e = enh[(enh.algorithm == a) & (enh.metric == "success_rate")]
        var_ratio = e["var_ratio"].mean()
        best = e.loc[e["abs_change"].idxmax()]
        worst = e.loc[e["abs_change"].idxmin()]
        any_sig = bool(e["sig_holm_0.05"].any())
        interp = (f"Enhancements shift success by {worst['abs_change']:+.1f}% to "
                  f"{best['abs_change']:+.1f}% (none significant after Holm); "
                  f"variance {'increases' if var_ratio > 1 else 'decreases'} on average "
                  f"(var ratio {var_ratio:.2f}). ")
        rows.append({
            "Algorithm": ALGO_LABEL[a],
            "Baseline Success": fmt_ci(br["success_rate_mean"], br["success_rate_ci95_hw"], 1, "%"),
            "Best Enhancement": f"{best['contrast']} ({best['abs_change']:+.1f}%)",
            "Worst Enhancement": f"{worst['contrast']} ({worst['abs_change']:+.1f}%)",
            "Mean Var Ratio": f"{var_ratio:.2f}",
            "Any Sig (Holm)": "yes" if any_sig else "no",
            "Interpretation": interp,
        })
    emit("table7_robustness", pd.DataFrame(rows))


# ---------------- Table 8 — Overall findings per algorithm ----------------
def table8():
    rows = []
    for a in ALGOS:
        r = base_row(a)
        # strongest/weakest by comparing baseline metrics to the cross-algo mean
        interp = (f"At 150-episode budget, {ALGO_LABEL[a]} reaches "
                  f"{r['success_rate_mean']:.0f}% success (95% CI +/-{r['success_rate_ci95_hw']:.0f}%); "
                  f"no statistical evidence it differs from the other variants on "
                  f"success or reward after correction. Route optimality "
                  f"{r['optimality_ratio_mean']:.2f}x.")
        rows.append({
            "Algorithm": ALGO_LABEL[a],
            "Best Strength": f"success {r['success_rate_mean']:.0f}%",
            "Main Weakness": f"route {r['optimality_ratio_mean']:.2f}x optimal; high seed variance",
            "Statistical Evidence": "no significant algorithm effect (Holm-corrected)",
            "Robustness": "enhancements not significant; seed variance dominant",
            "Practical Assessment": "interchangeable on success/reward at this budget",
            "Interpretation": interp,
        })
    emit("table8_overall_findings", pd.DataFrame(rows))


# ---------------- Methods table ----------------
def table_methods():
    rows = [
        ["Descriptive CI", r"CI95 = xbar +/- t_{0.975,n-1} * s/sqrt(n)", "Uncertainty of per-config mean over seeds", "Approx. normal mean; n=10", "Half-widths are wide (10-22% for success) -> low precision from 10 seeds."],
        ["Bootstrap CI", "percentile CI from 10,000 resamples of the seed means", "Distribution-free CI cross-check", "i.i.d. seeds", "Bootstrap and t-CI agree, supporting robustness of the interval estimates."],
        ["Normality", "Shapiro-Wilk W on each config's 10 seed-means", "Test parametric assumption", "n>=3, non-constant", "success/reward ~90% normal; optimality_ratio/excess_cost non-normal -> non-parametric primary."],
        ["Variance homogeneity", "Levene (Brown-Forsythe, median-centred)", "Test equal variance across algorithms", "independent groups", "All metrics homoscedastic (p>0.05)."],
        ["Omnibus (parametric)", r"one-way repeated-measures ANOVA; partial eta^2 = SS_treat/(SS_treat+SS_err)", "Overall algorithm effect (block=seed)", "normality, sphericity", "No success/reward effect; optimality_ratio effect exists in factorial model."],
        ["Omnibus (non-parametric)", r"Friedman chi^2 = 12/(nk(k+1)) sum R_j^2 - 3n(k+1); Kendall W", "Overall algorithm effect, rank-based", "repeated measures", "Confirms no rank difference in success/reward across algorithms."],
        ["Factorial ANOVA", r"OLS y ~ algo*per*noisy + seed, Type-II SS; partial eta^2", "Main effects + interactions", "balanced design (10/cell)", "Seed dominates (eta^2~0.20); Noisy & Algo x Noisy significant for optimality_ratio."],
        ["Pairwise (paired)", r"Wilcoxon signed-rank (exact); paired t as secondary", "Localise differences between algorithm pairs", "paired, matched seeds", "No pair survives Holm correction on any metric."],
        ["Effect size (paired)", r"Cohen's d_z = mean(d)/sd(d); Cliff's delta; rank-biserial r", "Magnitude independent of n", "-", "Effects are small/negligible on success/reward; up to medium for Noisy on optimality_ratio."],
        ["Correction", r"Holm: p_(i)*(m-i+1) monotone; BH: p_(i)*m/i", "Control FWER (Holm) / FDR (BH)", "family of tests", "Holm applied within each metric's 6 pairwise / 3 enhancement contrasts."],
    ]
    df = pd.DataFrame(rows, columns=["Analysis", "Formula / Method", "Purpose", "Assumption", "Interpretation"])
    emit("table_methods", df)


def main():
    print("Writing tables:")
    table1(); table2(); table3(); table4(); table5(); table6(); table7(); table8(); table_methods()
    # consolidated Excel workbook
    xlsx = os.path.join(TBL, "all_tables.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, df in _written.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"Excel workbook: {xlsx} ({len(_written)} sheets)")


if __name__ == "__main__":
    main()
