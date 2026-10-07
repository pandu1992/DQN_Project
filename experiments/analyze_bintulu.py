#!/usr/bin/env python3
"""
BPAN analysis — Bintulu Port Autonomous Navigation.

Aggregates the 4-algorithm x 3-condition x 8-seed experiment (per-seed scalars),
runs a two-way factorial ANOVA (algorithm x condition + seed, partial eta^2),
clean->harsh robustness contrasts (paired Wilcoxon + Holm, Cohen d_z), and emits
publication tables + figures. CTE is reported ON SUCCESSFUL episodes (audit-aware).
Outputs -> results_bintulu/{aggregated,statistics,tables,figures}/
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results_bintulu", "raw", "eval_episodes.csv")
AGG = os.path.join(ROOT, "results_bintulu", "aggregated")
STAT = os.path.join(ROOT, "results_bintulu", "statistics")
TBL = os.path.join(ROOT, "results_bintulu", "tables")
FIG = os.path.join(ROOT, "results_bintulu", "figures")
for d in (AGG, STAT, TBL, FIG):
    os.makedirs(d, exist_ok=True)
RNG = np.random.default_rng(12345)

ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
LAB = {"DQN": "DQN", "DoubleDQN": "Double DQN", "DuelingDQN": "Dueling DQN", "DuelingDoubleDQN": "Dueling Double DQN"}
CONDS = ["clean", "mid", "harsh"]
COL = {"DQN": "#4C72B0", "DoubleDQN": "#55A868", "DuelingDQN": "#C44E52", "DuelingDoubleDQN": "#8172B2"}
MARK = {"DQN": "o", "DoubleDQN": "s", "DuelingDQN": "^", "DuelingDoubleDQN": "D"}
METRICS = ["success_rate", "mean_reward", "collision_rate", "collisions_per_ep",
           "iala_violations", "cte_success", "docking_accuracy", "optimality_ratio", "dropped_frames"]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.dpi": 320, "savefig.bbox": "tight"})


def num(c, fr): return pd.to_numeric(fr[c], errors="coerce")

def per_seed(g):
    succ = g[g["success"] == 1]
    return {
        "success_rate": 100.0 * g["success"].mean(),
        "mean_reward": num("reward", g).mean(),
        "collision_rate": 100.0 * (num("collisions", g) > 0).mean(),
        "collisions_per_ep": num("collisions", g).mean(),
        "iala_violations": num("iala_violations", g).mean(),
        "cte_success": num("cte_mean", succ).mean() if len(succ) else np.nan,
        "docking_accuracy": num("docking_accuracy", succ).mean() if len(succ) else np.nan,
        "optimality_ratio": num("optimality_ratio", succ).mean() if len(succ) else np.nan,
        "dropped_frames": num("dropped_frames", g).mean(),
        "n_solved": int(len(succ)), "n_eval": int(len(g)),
    }

def t_hw(x, a=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    return stats.t.ppf(1 - a / 2, n - 1) * x.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan

def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    b = np.array([np.mean(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)])
    return (np.percentile(b, 2.5), np.percentile(b, 97.5))

def cohens_dz(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 2: return np.nan
    sd = d.std(ddof=1); return d.mean() / sd if sd > 0 else 0.0

def mag(d):
    ad = abs(d)
    return "n/a" if not np.isfinite(ad) else ("negligible" if ad < 0.2 else "small" if ad < 0.5 else "medium" if ad < 0.8 else "large")

def holm(p):
    p = np.asarray(p, float); m = len(p); v = np.where(np.isfinite(p))[0]; adj = np.full(m, np.nan)
    if not len(v): return adj
    order = v[np.argsort(p[v])]; run = 0.0
    for i, idx in enumerate(order): run = max(run, (len(v) - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def paired(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; d = x - y
    if len(x) >= 2 and np.any(d != 0):
        try: _, wp = stats.wilcoxon(x, y, method="exact")
        except Exception:
            try: _, wp = stats.wilcoxon(x, y)
            except Exception: wp = np.nan
    else: wp = 1.0 if (len(x) and np.allclose(d, 0)) else np.nan
    return {"n": len(x), "mean_x": np.nanmean(x) if len(x) else np.nan, "mean_y": np.nanmean(y) if len(y) else np.nan,
            "mean_diff": d.mean() if len(x) else np.nan, "wilcoxon_p": wp, "cohens_dz": cohens_dz(d)}


def main():
    df = pd.read_csv(RAW); df = df[df["phase"] == "eval"].copy()
    # per-seed
    rows = []
    for (al, cond, seed), g in df.groupby(["algorithm", "condition", "seed"], sort=True):
        rows.append({"algorithm": al, "condition": cond, "seed": int(seed), **per_seed(g)})
    ps = pd.DataFrame(rows); ps.to_csv(os.path.join(AGG, "per_seed.csv"), index=False)
    # summary
    srows = []
    for (al, cond), g in ps.groupby(["algorithm", "condition"], sort=True):
        e = {"algorithm": al, "condition": cond, "n_seeds": len(g)}
        for m in METRICS:
            x = g[m].to_numpy(float); e[f"{m}_mean"] = np.nanmean(x); e[f"{m}_ci95_hw"] = t_hw(x)
            lo, hi = boot_ci(x); e[f"{m}_boot_lo"] = lo; e[f"{m}_boot_hi"] = hi
        srows.append(e)
    summ = pd.DataFrame(srows); summ.to_csv(os.path.join(AGG, "summary.csv"), index=False)

    # factorial ANOVA
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    frows = []
    for m in METRICS:
        d = ps[["algorithm", "condition", "seed", m]].dropna().rename(columns={m: "y"}).copy()
        if d["y"].nunique() < 2: continue
        for c in ["algorithm", "condition", "seed"]: d[c] = d[c].astype("category")
        try: aov = anova_lm(ols("y ~ C(algorithm)*C(condition) + C(seed)", data=d).fit(), typ=2)
        except Exception: continue
        ssr = aov.loc["Residual", "sum_sq"]
        emap = {"C(algorithm)": "algorithm", "C(condition)": "condition", "C(algorithm):C(condition)": "algorithm:condition", "C(seed)": "seed"}
        for eff in aov.index:
            if eff == "Residual": continue
            ss = aov.loc[eff, "sum_sq"]
            frows.append({"metric": m, "effect": emap.get(eff, eff), "F": aov.loc[eff, "F"],
                          "p_value": aov.loc[eff, "PR(>F)"], "partial_eta2": ss / (ss + ssr) if (ss + ssr) > 0 else np.nan})
    fac = pd.DataFrame(frows); fac.to_csv(os.path.join(STAT, "factorial_anova.csv"), index=False)

    # robustness clean -> harsh per algorithm
    rob = []
    for al in ALGOS:
        base = ps[(ps.algorithm == al) & (ps.condition == "clean")].sort_values("seed")
        harsh = ps[(ps.algorithm == al) & (ps.condition == "harsh")].sort_values("seed")
        for m in METRICS:
            r = paired(harsh[m].to_numpy(float), base[m].to_numpy(float))
            rob.append({"algorithm": al, "metric": m, "clean_mean": r["mean_y"], "harsh_mean": r["mean_x"],
                        "abs_change": r["mean_diff"], "wilcoxon_p": r["wilcoxon_p"], "cohens_dz": r["cohens_dz"], "effect_mag": mag(r["cohens_dz"])})
    robdf = pd.DataFrame(rob)
    for m in METRICS:
        idx = robdf.index[robdf.metric == m]
        robdf.loc[idx, "wilcoxon_p_holm"] = holm(robdf.loc[idx, "wilcoxon_p"].to_numpy(float))
        robdf.loc[idx, "sig_holm"] = robdf.loc[idx, "wilcoxon_p_holm"] < 0.05
    robdf.to_csv(os.path.join(STAT, "robustness_clean_vs_harsh.csv"), index=False)

    # ---- tables (CSV with Interpretation) ----
    def cell(al, c): return summ[(summ.algorithm == al) & (summ.condition == c)].iloc[0]
    def pm(mean, hw, dp=1, u=""): return f"{mean:.{dp}f}" + ("" if not np.isfinite(hw) else f"\u00b1{hw:.{dp}f}") + u
    # T1 config
    pd.DataFrame([
        ["Testbed", "Bintulu Port approach chart (1536x1024 native frame)", "North + South Access Channels -> Inner Harbours / Southern Jetty", "Waypoints on the real chart; channels extended into berths."],
        ["Agents", "DQN, Double, Dueling, Dueling-Double", "value-based DRL", "Same four variants as the Synthetic Port studies, for comparability."],
        ["Conditions", "clean (0,0), mid (0.1,0.2), harsh (0.25,0.4)", "(sensor noise, packet-error)", "Graded sensing/communication degradation."],
        ["Seeds", "8 shared seeds (0-7)", "paired; env_seed=42+seed", "Cells: 4x3x8=96; 2,880 eval episodes."],
    ], columns=["Factor", "Levels / Value", "Detail", "Interpretation"]).to_csv(os.path.join(TBL, "table1_bpan_configuration.csv"), index=False)
    # T2 navigation success
    rows2 = []
    for al in ALGOS:
        cells = {c: pm(cell(al, c)["success_rate_mean"], cell(al, c)["success_rate_ci95_hw"], 1, "%") for c in CONDS}
        cl, ha = cell(al, "clean")["success_rate_mean"], cell(al, "harsh")["success_rate_mean"]
        rows2.append({"Agent": LAB[al], "Clean": cells["clean"], "Mid": cells["mid"], "Harsh": cells["harsh"],
                      "Interpretation": f"Success {cl:.0f}%\u2192{ha:.0f}% clean\u2192harsh on the Bintulu chart."})
    pd.DataFrame(rows2).to_csv(os.path.join(TBL, "table2_bpan_navigation.csv"), index=False)
    # T3 safety
    rows3 = []
    for al in ALGOS:
        c, h = cell(al, "clean"), cell(al, "harsh")
        rows3.append({"Agent": LAB[al],
                      "Coll/ep clean": f"{c['collisions_per_ep_mean']:.2f}", "Coll/ep harsh": f"{h['collisions_per_ep_mean']:.2f}",
                      "IALA clean": f"{c['iala_violations_mean']:.1f}", "IALA harsh": f"{h['iala_violations_mean']:.1f}",
                      "CTE|succ clean": f"{c['cte_success_mean']:.1f}", "CTE|succ harsh": f"{h['cte_success_mean']:.1f}",
                      "Interpretation": "Safety/compliance degrade sharply clean\u2192harsh; CTE on success rises."})
    pd.DataFrame(rows3).to_csv(os.path.join(TBL, "table3_bpan_safety.csv"), index=False)
    # T4 factorial
    lab = {"algorithm": "Algorithm", "condition": "Condition", "algorithm:condition": "Algorithm\u00d7Condition", "seed": "Seed (block)"}
    def eff_lab(pe): return "large" if pe >= 0.14 else "medium" if pe >= 0.06 else "small" if pe >= 0.01 else "negligible"
    rows4 = []
    for m in ["success_rate", "collision_rate", "iala_violations", "cte_success", "docking_accuracy"]:
        sub = fac[fac.metric == m]
        for eff in ["algorithm", "condition", "algorithm:condition", "seed"]:
            e = sub[sub.effect == eff]
            if not len(e): continue
            e = e.iloc[0]; p = e["p_value"]; pe2 = e["partial_eta2"]
            rows4.append({"Metric": m, "Effect": lab[eff], "F": f"{e['F']:.2f}", "p-value": f"{p:.4f}",
                          "partial \u03b7\u00b2": f"{pe2:.3f}", "Interpretation": f"{'sig' if p < 0.05 else 'n.s.'} (p={p:.3f}), {eff_lab(pe2)} effect"})
    pd.DataFrame(rows4).to_csv(os.path.join(TBL, "table4_bpan_factorial.csv"), index=False)
    # T5 robustness
    rows5 = []
    for al in ALGOS:
        for m in ["success_rate", "collision_rate", "iala_violations", "docking_accuracy"]:
            r = robdf[(robdf.algorithm == al) & (robdf.metric == m)].iloc[0]
            rows5.append({"Agent": LAB[al], "Metric": m, "Clean": f"{r['clean_mean']:.1f}", "Harsh": f"{r['harsh_mean']:.1f}",
                          "\u0394": f"{r['abs_change']:+.1f}", "Cohen d_z": f"{r['cohens_dz']:.2f}",
                          "adj p (Holm)": f"{r['wilcoxon_p_holm']:.3f}", "Sig": "yes" if r["sig_holm"] else "no",
                          "Interpretation": f"{r['effect_mag']} clean\u2192harsh change" + ("; Holm-sig." if r["sig_holm"] else "; n.s. at n=8.")})
    pd.DataFrame(rows5).to_csv(os.path.join(TBL, "table5_bpan_robustness.csv"), index=False)

    try:
        with pd.ExcelWriter(os.path.join(TBL, "all_tables_bpan.xlsx"), engine="openpyxl") as xw:
            for n in ["table1_bpan_configuration", "table2_bpan_navigation", "table3_bpan_safety", "table4_bpan_factorial", "table5_bpan_robustness"]:
                pd.read_csv(os.path.join(TBL, n + ".csv")).to_excel(xw, sheet_name=n[:31], index=False)
    except Exception as e:
        print("excel skip:", e)

    # ---- figures ----
    def line(metric, ylab, name, pct=False):
        fig, ax = plt.subplots(figsize=(7.4, 4.8)); x = np.arange(3)
        for al in ALGOS:
            ys = [cell(al, c)[f"{metric}_mean"] for c in CONDS]; es = [cell(al, c)[f"{metric}_ci95_hw"] for c in CONDS]
            ax.errorbar(x, ys, yerr=es, marker=MARK[al], color=COL[al], lw=1.9, markersize=7, capsize=3, label=LAB[al])
        ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_xlabel("Degradation condition (sensor noise, packet-error)")
        ax.set_ylabel(ylab)
        if pct: ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=9); ax.set_title(f"BPAN — {ylab} across conditions")
        fig.savefig(os.path.join(FIG, name + ".png")); fig.savefig(os.path.join(FIG, name + ".svg")); plt.close(fig)
    line("success_rate", "Navigation success rate (%)", "figBPAN_success", True)
    line("iala_violations", "IALA violations per episode", "figBPAN_iala")
    line("collisions_per_ep", "Collisions per episode", "figBPAN_collisions")

    # factorial eta2 bars
    metrics = ["success_rate", "collision_rate", "iala_violations", "cte_success", "docking_accuracy"]
    mlab = ["Success", "Collision%", "IALA", "CTE|succ", "Docking"]
    effects = ["algorithm", "condition", "algorithm:condition"]
    elab = {"algorithm": "Algorithm", "condition": "Condition", "algorithm:condition": "Algo\u00d7Cond"}
    ecol = {"algorithm": "#4C72B0", "condition": "#DD8452", "algorithm:condition": "#55A868"}
    data = {e: [fac[(fac.metric == m) & (fac.effect == e)]["partial_eta2"].iloc[0] if len(fac[(fac.metric == m) & (fac.effect == e)]) else np.nan for m in metrics] for e in effects}
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, e in enumerate(effects): ax.bar(x + (i - 1) * w, data[e], w, label=elab[e], color=ecol[e], edgecolor="black", linewidth=0.6)
    for thr, l in [(0.14, "large"), (0.06, "medium")]:
        ax.axhline(thr, color="gray", ls=":", lw=1); ax.text(len(metrics) - 0.5, thr + 0.01, l, fontsize=7.5, color="gray", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(mlab); ax.set_ylabel("Partial $\\eta^2$"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("BPAN — variance explained: degradation dominates, not the algorithm")
    fig.savefig(os.path.join(FIG, "figBPAN_anova_eta2.png")); fig.savefig(os.path.join(FIG, "figBPAN_anova_eta2.svg")); plt.close(fig)

    # ---- console headline ----
    print("=== BPAN success rate (%) ===")
    print(summ.pivot_table(index="algorithm", columns="condition", values="success_rate_mean").reindex(ALGOS)[CONDS].round(1).to_string())
    print("\n=== factorial eta^2 (key metrics) ===")
    for m in ["success_rate", "collision_rate", "iala_violations"]:
        for e in ["algorithm", "condition", "algorithm:condition"]:
            r = fac[(fac.metric == m) & (fac.effect == e)]
            if len(r): print(f"  {m:16s} {e:20s} eta2={r.iloc[0]['partial_eta2']:.3f} p={r.iloc[0]['p_value']:.4f}")
    print("\nwrote aggregated/, statistics/, 5 tables, 4 figures.")


if __name__ == "__main__":
    main()
