#!/usr/bin/env python3
"""
BPAN improvement analysis. Aggregates the 4-arm x 3-condition x 6-seed study,
runs a two-way factorial ANOVA (arm x condition + seed), and paired contrasts of
each improvement vs baseline per condition (Wilcoxon + Holm, Cohen d_z). Emits a
summary table and a figure. CTE on successful episodes.
Outputs -> results_bintulu/{aggregated,statistics,tables,figures}/ (improve_*)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results_bintulu", "raw_improve", "eval_episodes.csv")
AGG = os.path.join(ROOT, "results_bintulu", "aggregated")
STAT = os.path.join(ROOT, "results_bintulu", "statistics")
TBL = os.path.join(ROOT, "results_bintulu", "tables")
FIG = os.path.join(ROOT, "results_bintulu", "figures")
for d in (AGG, STAT, TBL, FIG):
    os.makedirs(d, exist_ok=True)

ARMS = ["baseline", "chart", "shaping", "chart_shaping"]
LAB = {"baseline": "Baseline DQN", "chart": "+ chart prior", "shaping": "+ reward shaping", "chart_shaping": "+ chart + shaping"}
COL = {"baseline": "#8a8a8a", "chart": "#4C72B0", "shaping": "#55A868", "chart_shaping": "#8172B2"}
MARK = {"baseline": "o", "chart": "D", "shaping": "s", "chart_shaping": "^"}
CONDS = ["clean", "mid", "harsh"]
METRICS = ["success_rate", "mean_reward", "collision_rate", "iala_violations", "cte_success", "docking_accuracy"]


def num(c, fr): return pd.to_numeric(fr[c], errors="coerce")

def per_seed(g):
    s = g[g["success"] == 1]
    return {
        "success_rate": 100.0 * g["success"].mean(),
        "mean_reward": num("reward", g).mean(),
        "collision_rate": 100.0 * (num("collisions", g) > 0).mean(),
        "iala_violations": num("iala_violations", g).mean(),
        "cte_success": num("cte_mean", s).mean() if len(s) else np.nan,
        "docking_accuracy": num("docking_accuracy", s).mean() if len(s) else np.nan,
    }

def t_hw(x, a=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    return stats.t.ppf(1 - a / 2, n - 1) * x.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan

def cohens_dz(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 2: return np.nan
    sd = d.std(ddof=1); return d.mean() / sd if sd > 0 else 0.0

def mag(d):
    ad = abs(d); return "n/a" if not np.isfinite(ad) else ("negligible" if ad < 0.2 else "small" if ad < 0.5 else "medium" if ad < 0.8 else "large")

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
    return {"mean_x": np.nanmean(x) if len(x) else np.nan, "mean_y": np.nanmean(y) if len(y) else np.nan,
            "mean_diff": d.mean() if len(x) else np.nan, "wilcoxon_p": wp, "cohens_dz": cohens_dz(d)}


def main():
    df = pd.read_csv(RAW); df = df[df["phase"] == "eval"].copy()
    rows = []
    for (arm, cond, seed), g in df.groupby(["arm", "condition", "seed"], sort=True):
        rows.append({"arm": arm, "condition": cond, "seed": int(seed), **per_seed(g)})
    ps = pd.DataFrame(rows); ps.to_csv(os.path.join(AGG, "improve_per_seed.csv"), index=False)
    srows = []
    for (arm, cond), g in ps.groupby(["arm", "condition"], sort=True):
        e = {"arm": arm, "condition": cond, "n_seeds": len(g)}
        for m in METRICS:
            x = g[m].to_numpy(float); e[f"{m}_mean"] = np.nanmean(x); e[f"{m}_ci95_hw"] = t_hw(x)
        srows.append(e)
    summ = pd.DataFrame(srows); summ.to_csv(os.path.join(AGG, "improve_summary.csv"), index=False)

    # factorial ANOVA on success
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    frows = []
    for m in ["success_rate", "collision_rate", "iala_violations"]:
        d = ps[["arm", "condition", "seed", m]].dropna().rename(columns={m: "y"}).copy()
        for c in ["arm", "condition", "seed"]: d[c] = d[c].astype("category")
        try: aov = anova_lm(ols("y ~ C(arm)*C(condition) + C(seed)", data=d).fit(), typ=2)
        except Exception: continue
        ssr = aov.loc["Residual", "sum_sq"]
        emap = {"C(arm)": "arm", "C(condition)": "condition", "C(arm):C(condition)": "arm:condition", "C(seed)": "seed"}
        for eff in aov.index:
            if eff == "Residual": continue
            ss = aov.loc[eff, "sum_sq"]
            frows.append({"metric": m, "effect": emap.get(eff, eff), "F": aov.loc[eff, "F"], "p_value": aov.loc[eff, "PR(>F)"], "partial_eta2": ss / (ss + ssr) if (ss + ssr) > 0 else np.nan})
    fac = pd.DataFrame(frows); fac.to_csv(os.path.join(STAT, "improve_factorial_anova.csv"), index=False)

    # each improvement vs baseline, per condition (success); Holm within condition
    con = []
    for cond in CONDS:
        sub = ps[ps.condition == cond]
        base = sub[sub.arm == "baseline"].sort_values("seed")
        recs = []
        for arm in ["chart", "shaping", "chart_shaping"]:
            a = sub[sub.arm == arm].sort_values("seed")
            r = paired(a["success_rate"].to_numpy(float), base["success_rate"].to_numpy(float))
            recs.append({"condition": cond, "arm": arm, "baseline_mean": r["mean_y"], "arm_mean": r["mean_x"],
                         "gain": r["mean_diff"], "wilcoxon_p": r["wilcoxon_p"], "cohens_dz": r["cohens_dz"], "effect_mag": mag(r["cohens_dz"])})
        h = holm([x["wilcoxon_p"] for x in recs])
        for i, x in enumerate(recs): x["wilcoxon_p_holm"] = h[i]; x["sig_holm"] = bool(np.isfinite(h[i]) and h[i] < 0.05); con.append(x)
    condf = pd.DataFrame(con); condf.to_csv(os.path.join(STAT, "improve_vs_baseline.csv"), index=False)

    # table (success by arm x condition + overall gain)
    def cell(arm, c): return summ[(summ.arm == arm) & (summ.condition == c)].iloc[0]
    trows = []
    for arm in ARMS:
        row = {"Arm": LAB[arm]}
        for c in CONDS: row[c.capitalize() + " %"] = f"{cell(arm, c)['success_rate_mean']:.1f}"
        if arm == "baseline":
            row["Interpretation"] = "Reference DQN with no improvement."
        else:
            gains = [condf[(condf.arm == arm) & (condf.condition == c)].iloc[0] for c in CONDS]
            avg = np.mean([g["gain"] for g in gains]); nsig = sum(g["sig_holm"] for g in gains)
            row["Interpretation"] = f"+{avg:.0f} pts success vs baseline on avg; {nsig}/3 conditions Holm-sig (d_z {gains[0]['cohens_dz']:.1f}-{gains[2]['cohens_dz']:.1f})."
        trows.append(row)
    pd.DataFrame(trows).to_csv(os.path.join(TBL, "table6_bpan_improvements.csv"), index=False)

    # figure: success by arm across conditions
    fig, ax = plt.subplots(figsize=(7.6, 4.8)); x = np.arange(3)
    for arm in ARMS:
        ys = [cell(arm, c)["success_rate_mean"] for c in CONDS]; es = [cell(arm, c)["success_rate_ci95_hw"] for c in CONDS]
        lw = 2.4 if arm == "chart" else 1.9
        ax.errorbar(x, ys, yerr=es, marker=MARK[arm], color=COL[arm], lw=lw, markersize=8, capsize=3, label=LAB[arm])
    ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_xlabel("Degradation condition")
    ax.set_ylabel("Navigation success rate (%)"); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=9)
    ax.set_title("BPAN — improving DQN on Bintulu: the chart prior is the dominant lever")
    cap = ("Figure. Navigation success by DQN improvement arm across degradation conditions (n=6 seeds, 95% CI). "
           "Appending the Dijkstra charted-route prior to the observation lifts success far above the baseline at every "
           "condition; potential-based reward shaping also helps; the two combine.")
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=8.5, wrap=True)
    fig.savefig(os.path.join(FIG, "figBPAN_improvements.png"), dpi=320, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "figBPAN_improvements.svg"), bbox_inches="tight"); plt.close(fig)

    print("=== success by arm x condition (%) ===")
    print(summ.pivot_table(index="arm", columns="condition", values="success_rate_mean").reindex(ARMS)[CONDS].round(1).to_string())
    print("\n=== arm effect on success (factorial) ===")
    print(fac[(fac.metric == "success_rate") & (fac.effect.isin(["arm", "condition", "arm:condition"]))][["effect", "F", "p_value", "partial_eta2"]].round(4).to_string(index=False))
    print("\n=== improvements vs baseline (success, Holm) ===")
    print(condf[["condition", "arm", "baseline_mean", "arm_mean", "gain", "wilcoxon_p_holm", "cohens_dz", "sig_holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
