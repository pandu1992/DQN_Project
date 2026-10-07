#!/usr/bin/env python3
"""
BPAN extension (d) — MULTI-VESSEL analysis.

Two independent vessels share the Bintulu approach channels, each with its own
policy and mission, and must avoid one another. We analyse the 3-pairing x
3-condition x 6-seed experiment (DQN/DQN, DQN/RuleBased, RuleBased/RuleBased).

Unit of inference = per-seed scalar (both vessels of a cell pooled). We report:
  * per-seed + summary aggregates (success, inter-vessel collisions, near-miss,
    min-CPA, static collisions, CTE on SUCCESSFUL episodes only);
  * a 2-way factorial ANOVA (pairing x condition + seed block, partial eta^2);
  * clean->harsh robustness contrasts (paired Wilcoxon + Holm, Cohen d_z);
  * publication tables + figures.
Outputs -> results_bintulu/multi/{aggregated,statistics,tables,figures}/
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results_bintulu", "raw_multi", "eval_episodes.csv")
BASE = os.path.join(ROOT, "results_bintulu", "multi")
AGG, STAT, TBL, FIG = (os.path.join(BASE, d) for d in ("aggregated", "statistics", "tables", "figures"))
for d in (AGG, STAT, TBL, FIG):
    os.makedirs(d, exist_ok=True)
RNG = np.random.default_rng(12345)

PAIRS = ["RuleBased_vs_RuleBased", "DQN_vs_RuleBased", "DQN_vs_DQN"]
PLAB = {"RuleBased_vs_RuleBased": "Rule / Rule", "DQN_vs_RuleBased": "DQN / Rule", "DQN_vs_DQN": "DQN / DQN"}
CONDS = ["clean", "mid", "harsh"]
PCOL = {"RuleBased_vs_RuleBased": "#C44E52", "DQN_vs_RuleBased": "#55A868", "DQN_vs_DQN": "#4C72B0"}
PMARK = {"RuleBased_vs_RuleBased": "^", "DQN_vs_RuleBased": "s", "DQN_vs_DQN": "o"}
METRICS = ["success_rate", "mean_reward", "vessel_collision_rate", "vessel_collisions_per_ep",
           "near_misses_per_ep", "min_cpa", "static_collisions_per_ep", "iala_violations", "cte_success"]
MLAB = {"success_rate": "Success rate (%)", "mean_reward": "Mean reward",
        "vessel_collision_rate": "Episodes with vessel collision (%)",
        "vessel_collisions_per_ep": "Inter-vessel collisions / episode",
        "near_misses_per_ep": "Near-misses / episode", "min_cpa": "Min closest-point-of-approach (px)",
        "static_collisions_per_ep": "Static-obstacle collisions / episode",
        "iala_violations": "IALA violations / episode", "cte_success": "Cross-track error | success (px)"}
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.dpi": 320, "savefig.bbox": "tight"})


def num(c, fr): return pd.to_numeric(fr[c], errors="coerce")

def per_seed(g):
    succ = g[g["success"] == 1]
    return {
        "success_rate": 100.0 * g["success"].mean(),
        "mean_reward": num("reward", g).mean(),
        "vessel_collision_rate": 100.0 * (num("vessel_collisions", g) > 0).mean(),
        "vessel_collisions_per_ep": num("vessel_collisions", g).mean(),
        "near_misses_per_ep": num("near_misses", g).mean(),
        "min_cpa": num("min_cpa", g).mean(),
        "static_collisions_per_ep": num("static_collisions", g).mean(),
        "iala_violations": num("iala_violations", g).mean(),
        "cte_success": num("cte_mean", succ).mean() if len(succ) else np.nan,
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
    rows = []
    for (pr, cond, seed), g in df.groupby(["pairing", "condition", "seed"], sort=True):
        rows.append({"pairing": pr, "condition": cond, "seed": int(seed), **per_seed(g)})
    ps = pd.DataFrame(rows); ps.to_csv(os.path.join(AGG, "per_seed.csv"), index=False)

    srows = []
    for (pr, cond), g in ps.groupby(["pairing", "condition"], sort=True):
        e = {"pairing": pr, "condition": cond, "n_seeds": len(g)}
        for m in METRICS:
            x = g[m].to_numpy(float); e[f"{m}_mean"] = np.nanmean(x); e[f"{m}_ci95_hw"] = t_hw(x)
            lo, hi = boot_ci(x); e[f"{m}_boot_lo"] = lo; e[f"{m}_boot_hi"] = hi
        srows.append(e)
    summ = pd.DataFrame(srows); summ.to_csv(os.path.join(AGG, "summary.csv"), index=False)

    # factorial ANOVA (pairing x condition + seed block)
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    frows = []
    for m in METRICS:
        d = ps[["pairing", "condition", "seed", m]].dropna().rename(columns={m: "y"}).copy()
        if d["y"].nunique() < 2: continue
        for c in ["pairing", "condition", "seed"]: d[c] = d[c].astype("category")
        try: aov = anova_lm(ols("y ~ C(pairing)*C(condition) + C(seed)", data=d).fit(), typ=2)
        except Exception: continue
        ssr = aov.loc["Residual", "sum_sq"]
        emap = {"C(pairing)": "pairing", "C(condition)": "condition", "C(pairing):C(condition)": "pairing:condition", "C(seed)": "seed"}
        for eff in aov.index:
            if eff == "Residual": continue
            ss = aov.loc[eff, "sum_sq"]
            frows.append({"metric": m, "effect": emap.get(eff, eff), "F": aov.loc[eff, "F"],
                          "p_value": aov.loc[eff, "PR(>F)"], "partial_eta2": ss / (ss + ssr) if (ss + ssr) > 0 else np.nan})
    fac = pd.DataFrame(frows); fac.to_csv(os.path.join(STAT, "factorial_anova.csv"), index=False)

    # robustness clean -> harsh per pairing
    rob = []
    for pr in PAIRS:
        base = ps[(ps.pairing == pr) & (ps.condition == "clean")].sort_values("seed")
        harsh = ps[(ps.pairing == pr) & (ps.condition == "harsh")].sort_values("seed")
        for m in METRICS:
            r = paired(harsh[m].to_numpy(float), base[m].to_numpy(float))
            rob.append({"pairing": pr, "metric": m, "clean_mean": r["mean_y"], "harsh_mean": r["mean_x"],
                        "abs_change": r["mean_diff"], "wilcoxon_p": r["wilcoxon_p"], "cohens_dz": r["cohens_dz"], "effect_mag": mag(r["cohens_dz"])})
    robdf = pd.DataFrame(rob)
    for m in METRICS:
        idx = robdf.index[robdf.metric == m]
        robdf.loc[idx, "wilcoxon_p_holm"] = holm(robdf.loc[idx, "wilcoxon_p"].to_numpy(float))
        robdf.loc[idx, "sig_holm"] = robdf.loc[idx, "wilcoxon_p_holm"] < 0.05
    robdf.to_csv(os.path.join(STAT, "robustness_clean_vs_harsh.csv"), index=False)

    # pairing contrast at harsh: DQN/DQN vs Rule/Rule, DQN/Rule vs Rule/Rule
    con = []
    for a, b in [("DQN_vs_DQN", "RuleBased_vs_RuleBased"), ("DQN_vs_RuleBased", "RuleBased_vs_RuleBased"), ("DQN_vs_DQN", "DQN_vs_RuleBased")]:
        for cond in CONDS:
            xa = ps[(ps.pairing == a) & (ps.condition == cond)].sort_values("seed")
            xb = ps[(ps.pairing == b) & (ps.condition == cond)].sort_values("seed")
            for m in ["success_rate", "vessel_collisions_per_ep", "near_misses_per_ep", "min_cpa"]:
                r = paired(xa[m].to_numpy(float), xb[m].to_numpy(float))
                con.append({"contrast": f"{PLAB[a]} - {PLAB[b]}", "condition": cond, "metric": m,
                            "mean_a": r["mean_x"], "mean_b": r["mean_y"], "diff": r["mean_diff"],
                            "wilcoxon_p": r["wilcoxon_p"], "cohens_dz": r["cohens_dz"], "effect_mag": mag(r["cohens_dz"])})
    condf = pd.DataFrame(con)
    for (cond, m), idx in condf.groupby(["condition", "metric"]).groups.items():
        condf.loc[idx, "wilcoxon_p_holm"] = holm(condf.loc[idx, "wilcoxon_p"].to_numpy(float))
    condf.to_csv(os.path.join(STAT, "pairing_contrasts.csv"), index=False)

    # ---- tables ----
    def cell(pr, c): return summ[(summ.pairing == pr) & (summ.condition == c)].iloc[0]
    def pm(mean, hw, dp=1, u=""): return f"{mean:.{dp}f}" + ("" if not np.isfinite(hw) else f"\u00b1{hw:.{dp}f}") + u
    # T1 config
    pd.DataFrame([
        ["Scenario", "Two independent vessels on the Bintulu chart", "each own policy + mission; must avoid each other", "Extension (d): multi-vessel interaction on the real chart."],
        ["Pairings", "Rule/Rule, DQN/Rule, DQN/DQN", "controller composition", "Isolates the effect of the interaction partner's control architecture."],
        ["Conditions", "clean (0,0), mid (0.1,0.2), harsh (0.25,0.4)", "(sensor+partner noise, packet-error)", "Degradation also corrupts the sensed partner channel (dx,dy,range)."],
        ["Seeds", "6 shared seeds (0-5)", "paired; env_seed=42+seed", "Cells: 3x3x6=54; 30 eval episodes/cell x 2 vessels."],
        ["Collision model", "hull radius 26 px, near-miss 60 px, 6 sub-steps", "continuous CPA between hulls", "Inter-vessel collisions/near-misses genuinely computed."],
    ], columns=["Factor", "Levels / Value", "Detail", "Interpretation"]).to_csv(os.path.join(TBL, "table_multi1_config.csv"), index=False)
    # T2 outcomes
    rows2 = []
    for pr in PAIRS:
        rows2.append({"Pairing": PLAB[pr],
                      "Success clean": pm(cell(pr, "clean")["success_rate_mean"], cell(pr, "clean")["success_rate_ci95_hw"], 0, "%"),
                      "Success harsh": pm(cell(pr, "harsh")["success_rate_mean"], cell(pr, "harsh")["success_rate_ci95_hw"], 0, "%"),
                      "vColl/ep clean": f"{cell(pr,'clean')['vessel_collisions_per_ep_mean']:.2f}",
                      "vColl/ep harsh": f"{cell(pr,'harsh')['vessel_collisions_per_ep_mean']:.2f}",
                      "minCPA clean": f"{cell(pr,'clean')['min_cpa_mean']:.0f}",
                      "Interpretation": "Rule/Rule completes most but collides most; learned pairs trade completion for separation."})
    pd.DataFrame(rows2).to_csv(os.path.join(TBL, "table_multi2_outcomes.csv"), index=False)
    # T3 factorial
    lab = {"pairing": "Pairing", "condition": "Condition", "pairing:condition": "Pairing\u00d7Condition", "seed": "Seed (block)"}
    def eff_lab(pe): return "large" if pe >= 0.14 else "medium" if pe >= 0.06 else "small" if pe >= 0.01 else "negligible"
    rows3 = []
    for m in ["success_rate", "vessel_collisions_per_ep", "near_misses_per_ep", "min_cpa"]:
        sub = fac[fac.metric == m]
        for eff in ["pairing", "condition", "pairing:condition", "seed"]:
            e = sub[sub.effect == eff]
            if not len(e): continue
            e = e.iloc[0]; p = e["p_value"]; pe2 = e["partial_eta2"]
            rows3.append({"Metric": MLAB[m], "Effect": lab[eff], "F": f"{e['F']:.2f}", "p-value": f"{p:.4f}",
                          "partial \u03b7\u00b2": f"{pe2:.3f}", "Interpretation": f"{'sig' if p < 0.05 else 'n.s.'} (p={p:.3f}), {eff_lab(pe2)} effect"})
    pd.DataFrame(rows3).to_csv(os.path.join(TBL, "table_multi3_factorial.csv"), index=False)
    # T4 robustness
    rows4 = []
    for pr in PAIRS:
        for m in ["success_rate", "vessel_collisions_per_ep", "near_misses_per_ep", "min_cpa"]:
            r = robdf[(robdf.pairing == pr) & (robdf.metric == m)].iloc[0]
            rows4.append({"Pairing": PLAB[pr], "Metric": MLAB[m], "Clean": f"{r['clean_mean']:.2f}", "Harsh": f"{r['harsh_mean']:.2f}",
                          "\u0394": f"{r['abs_change']:+.2f}", "Cohen d_z": f"{r['cohens_dz']:.2f}",
                          "adj p (Holm)": f"{r['wilcoxon_p_holm']:.3f}", "Sig": "yes" if r["sig_holm"] else "no",
                          "Interpretation": f"{r['effect_mag']} clean\u2192harsh change" + ("; Holm-sig." if r["sig_holm"] else "; n.s. at n=6.")})
    pd.DataFrame(rows4).to_csv(os.path.join(TBL, "table_multi4_robustness.csv"), index=False)

    try:
        with pd.ExcelWriter(os.path.join(TBL, "all_tables_multi.xlsx"), engine="openpyxl") as xw:
            for n in ["table_multi1_config", "table_multi2_outcomes", "table_multi3_factorial", "table_multi4_robustness"]:
                pd.read_csv(os.path.join(TBL, n + ".csv")).to_excel(xw, sheet_name=n[:31], index=False)
    except Exception as e:
        print("excel skip:", e)

    # ---- figures ----
    def line(metric, name, pct=False):
        fig, ax = plt.subplots(figsize=(7.4, 4.8)); x = np.arange(3)
        for pr in PAIRS:
            ys = [cell(pr, c)[f"{metric}_mean"] for c in CONDS]; es = [cell(pr, c)[f"{metric}_ci95_hw"] for c in CONDS]
            ax.errorbar(x, ys, yerr=es, marker=PMARK[pr], color=PCOL[pr], lw=1.9, markersize=7, capsize=3, label=PLAB[pr])
        ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_xlabel("Degradation condition (sensor+partner noise, packet-error)")
        ax.set_ylabel(MLAB[metric])
        if pct: ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=9, title="Pairing"); ax.set_title(f"BPAN multi-vessel — {MLAB[metric]}")
        fig.savefig(os.path.join(FIG, name + ".png")); fig.savefig(os.path.join(FIG, name + ".svg")); plt.close(fig)
    line("success_rate", "figMULTI_success", True)
    line("vessel_collisions_per_ep", "figMULTI_vcoll")
    line("min_cpa", "figMULTI_cpa")

    # factorial eta2 bars
    metrics = ["success_rate", "vessel_collisions_per_ep", "near_misses_per_ep", "min_cpa"]
    mlab = ["Success", "vColl/ep", "NearMiss/ep", "minCPA"]
    effects = ["pairing", "condition", "pairing:condition"]
    elab = {"pairing": "Pairing", "condition": "Condition", "pairing:condition": "Pair\u00d7Cond"}
    ecol = {"pairing": "#4C72B0", "condition": "#DD8452", "pairing:condition": "#55A868"}
    data = {e: [fac[(fac.metric == m) & (fac.effect == e)]["partial_eta2"].iloc[0] if len(fac[(fac.metric == m) & (fac.effect == e)]) else np.nan for m in metrics] for e in effects}
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, e in enumerate(effects): ax.bar(x + (i - 1) * w, data[e], w, label=elab[e], color=ecol[e], edgecolor="black", linewidth=0.6)
    for thr, l in [(0.14, "large"), (0.06, "medium")]:
        ax.axhline(thr, color="gray", ls=":", lw=1); ax.text(len(metrics) - 0.5, thr + 0.01, l, fontsize=7.5, color="gray", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(mlab); ax.set_ylabel("Partial $\\eta^2$"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("BPAN multi-vessel — variance explained: the interaction pairing dominates")
    fig.savefig(os.path.join(FIG, "figMULTI_anova_eta2.png")); fig.savefig(os.path.join(FIG, "figMULTI_anova_eta2.svg")); plt.close(fig)

    print("=== multi-vessel success (%) ===")
    print(summ.pivot_table(index="pairing", columns="condition", values="success_rate_mean").reindex(PAIRS)[CONDS].round(1).to_string())
    print("=== vessel collisions/ep ===")
    print(summ.pivot_table(index="pairing", columns="condition", values="vessel_collisions_per_ep_mean").reindex(PAIRS)[CONDS].round(2).to_string())
    print("\n=== factorial eta^2 ===")
    for m in ["success_rate", "vessel_collisions_per_ep", "near_misses_per_ep"]:
        for e in ["pairing", "condition", "pairing:condition"]:
            r = fac[(fac.metric == m) & (fac.effect == e)]
            if len(r): print(f"  {m:26s} {e:18s} eta2={r.iloc[0]['partial_eta2']:.3f} p={r.iloc[0]['p_value']:.4f}")
    print("\nwrote results_bintulu/multi/{aggregated,statistics,tables,figures}")


if __name__ == "__main__":
    main()
