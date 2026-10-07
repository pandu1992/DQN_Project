#!/usr/bin/env python3
"""
BPAN extension (e) — OPERATIONAL-REALISM analysis.

Two scenarios on the Bintulu chart:
  twophase : single vessel inbound -> dock -> outbound round trip
             (success = FULL cycle). Factors: agent {RuleBased, DQN} x
             condition {clean,mid,harsh} x 6 seeds.
  twoway   : an inbound and an outbound vessel share one access channel in
             opposing directions (head-on). Factors: pairing {Rule/Rule,
             DQN/Rule} x condition x 6 seeds.

Unit of inference = per-seed scalar. We report per-seed + summary aggregates,
2-way factorial ANOVA (partial eta^2), clean->harsh robustness (paired
Wilcoxon + Holm, Cohen d_z), tables and figures. CTE on SUCCESSFUL episodes.
Outputs -> results_bintulu/ops/{aggregated,statistics,tables,figures}/
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PHASE = os.path.join(ROOT, "results_bintulu", "raw_ops", "twophase_episodes.csv")
RAW_WAY = os.path.join(ROOT, "results_bintulu", "raw_ops", "twoway_episodes.csv")
BASE = os.path.join(ROOT, "results_bintulu", "ops")
AGG, STAT, TBL, FIG = (os.path.join(BASE, d) for d in ("aggregated", "statistics", "tables", "figures"))
for d in (AGG, STAT, TBL, FIG):
    os.makedirs(d, exist_ok=True)
RNG = np.random.default_rng(12345)
CONDS = ["clean", "mid", "harsh"]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.dpi": 320, "savefig.bbox": "tight"})


def num(c, fr): return pd.to_numeric(fr[c], errors="coerce")

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
    import warnings
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; d = x - y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(x) >= 2 and np.any(d != 0):
            try: _, wp = stats.wilcoxon(x, y, method="exact")
            except Exception:
                try: _, wp = stats.wilcoxon(x, y)
                except Exception: wp = np.nan
        else: wp = 1.0 if (len(x) and np.allclose(d, 0)) else np.nan
    return {"n": len(x), "mean_x": np.nanmean(x) if len(x) else np.nan, "mean_y": np.nanmean(y) if len(y) else np.nan,
            "mean_diff": d.mean() if len(x) else np.nan, "wilcoxon_p": wp, "cohens_dz": cohens_dz(d)}

def factorial(ps, factorA, metrics):
    """2-way ANOVA y ~ C(A)*C(condition) + C(seed)."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    frows = []
    for m in metrics:
        d = ps[[factorA, "condition", "seed", m]].dropna().rename(columns={m: "y"}).copy()
        if d["y"].nunique() < 2: continue
        for c in [factorA, "condition", "seed"]: d[c] = d[c].astype("category")
        try: aov = anova_lm(ols(f"y ~ C({factorA})*C(condition) + C(seed)", data=d).fit(), typ=2)
        except Exception: continue
        ssr = aov.loc["Residual", "sum_sq"]
        emap = {f"C({factorA})": factorA, "C(condition)": "condition", f"C({factorA}):C(condition)": f"{factorA}:condition", "C(seed)": "seed"}
        for eff in aov.index:
            if eff == "Residual": continue
            ss = aov.loc[eff, "sum_sq"]
            frows.append({"metric": m, "effect": emap.get(eff, eff), "F": aov.loc[eff, "F"],
                          "p_value": aov.loc[eff, "PR(>F)"], "partial_eta2": ss / (ss + ssr) if (ss + ssr) > 0 else np.nan})
    return pd.DataFrame(frows)

def eff_lab(pe): return "large" if pe >= 0.14 else "medium" if pe >= 0.06 else "small" if pe >= 0.01 else "negligible"


# =================== TWO-PHASE ===================
def analyze_twophase():
    df = pd.read_csv(RAW_PHASE); df = df[df["phase"] == "eval"].copy()
    AGENTS = ["RuleBased", "DQN"]
    ALAB = {"RuleBased": "Rule-based", "DQN": "DQN"}
    ACOL = {"RuleBased": "#C44E52", "DQN": "#4C72B0"}
    AMARK = {"RuleBased": "^", "DQN": "o"}
    METRICS = ["full_success_rate", "dock_success_rate", "mean_reward", "collisions_per_ep", "cte_success", "dock_accuracy_leg1"]
    MLAB = {"full_success_rate": "Full round-trip success (%)", "dock_success_rate": "Leg-1 docking success (%)",
            "mean_reward": "Mean reward", "collisions_per_ep": "Static collisions / episode",
            "cte_success": "Cross-track error | full success (px)", "dock_accuracy_leg1": "Leg-1 docking accuracy (px)"}

    def per_seed(g):
        full = g[g["full_success"] == 1]
        return {"full_success_rate": 100.0 * g["full_success"].mean(),
                "dock_success_rate": 100.0 * g["dock_success"].mean(),
                "mean_reward": num("reward", g).mean(),
                "collisions_per_ep": num("collisions", g).mean(),
                "cte_success": num("cte_mean", full).mean() if len(full) else np.nan,
                "dock_accuracy_leg1": num("dock_accuracy", g[g["dock_success"] == 1]).mean() if (g["dock_success"] == 1).any() else np.nan,
                "n_full": int(len(full)), "n_eval": int(len(g))}
    rows = []
    for (ag, cond, seed), g in df.groupby(["agent", "condition", "seed"], sort=True):
        rows.append({"agent": ag, "condition": cond, "seed": int(seed), **per_seed(g)})
    ps = pd.DataFrame(rows); ps.to_csv(os.path.join(AGG, "twophase_per_seed.csv"), index=False)
    srows = []
    for (ag, cond), g in ps.groupby(["agent", "condition"], sort=True):
        e = {"agent": ag, "condition": cond, "n_seeds": len(g)}
        for m in METRICS:
            x = g[m].to_numpy(float); e[f"{m}_mean"] = np.nanmean(x); e[f"{m}_ci95_hw"] = t_hw(x)
        srows.append(e)
    summ = pd.DataFrame(srows); summ.to_csv(os.path.join(AGG, "twophase_summary.csv"), index=False)
    fac = factorial(ps, "agent", METRICS); fac.to_csv(os.path.join(STAT, "twophase_factorial_anova.csv"), index=False)
    # robustness
    rob = []
    for ag in AGENTS:
        base = ps[(ps.agent == ag) & (ps.condition == "clean")].sort_values("seed")
        harsh = ps[(ps.agent == ag) & (ps.condition == "harsh")].sort_values("seed")
        for m in METRICS:
            r = paired(harsh[m].to_numpy(float), base[m].to_numpy(float))
            rob.append({"agent": ag, "metric": m, "clean_mean": r["mean_y"], "harsh_mean": r["mean_x"],
                        "abs_change": r["mean_diff"], "wilcoxon_p": r["wilcoxon_p"], "cohens_dz": r["cohens_dz"], "effect_mag": mag(r["cohens_dz"])})
    robdf = pd.DataFrame(rob)
    for m in METRICS:
        idx = robdf.index[robdf.metric == m]
        robdf.loc[idx, "wilcoxon_p_holm"] = holm(robdf.loc[idx, "wilcoxon_p"].to_numpy(float))
        robdf.loc[idx, "sig_holm"] = robdf.loc[idx, "wilcoxon_p_holm"] < 0.05
    robdf.to_csv(os.path.join(STAT, "twophase_robustness.csv"), index=False)

    def cell(ag, c): return summ[(summ.agent == ag) & (summ.condition == c)].iloc[0]
    # T outcomes
    rows2 = []
    for ag in AGENTS:
        rows2.append({"Agent": ALAB[ag],
                      "Dock clean": f"{cell(ag,'clean')['dock_success_rate_mean']:.0f}%",
                      "Full clean": f"{cell(ag,'clean')['full_success_rate_mean']:.0f}%",
                      "Full mid": f"{cell(ag,'mid')['full_success_rate_mean']:.0f}%",
                      "Full harsh": f"{cell(ag,'harsh')['full_success_rate_mean']:.0f}%",
                      "Coll/ep harsh": f"{cell(ag,'harsh')['collisions_per_ep_mean']:.2f}",
                      "Interpretation": "Rule-based completes the full cycle reliably; the naive learner docks but rarely closes the round trip."})
    pd.DataFrame(rows2).to_csv(os.path.join(TBL, "table_ops1_twophase_outcomes.csv"), index=False)
    # T factorial
    lab = {"agent": "Agent", "condition": "Condition", "agent:condition": "Agent\u00d7Condition", "seed": "Seed (block)"}
    rows3 = []
    for m in ["full_success_rate", "dock_success_rate", "collisions_per_ep"]:
        sub = fac[fac.metric == m]
        for eff in ["agent", "condition", "agent:condition", "seed"]:
            e = sub[sub.effect == eff]
            if not len(e): continue
            e = e.iloc[0]
            rows3.append({"Metric": MLAB[m], "Effect": lab[eff], "F": f"{e['F']:.2f}", "p-value": f"{e['p_value']:.4f}",
                          "partial \u03b7\u00b2": f"{e['partial_eta2']:.3f}", "Interpretation": f"{'sig' if e['p_value']<0.05 else 'n.s.'}, {eff_lab(e['partial_eta2'])} effect"})
    pd.DataFrame(rows3).to_csv(os.path.join(TBL, "table_ops2_twophase_factorial.csv"), index=False)

    # figures
    def line(metric, name, pct=False):
        fig, ax = plt.subplots(figsize=(7.0, 4.6)); x = np.arange(3)
        for ag in AGENTS:
            ys = [cell(ag, c)[f"{metric}_mean"] for c in CONDS]; es = [cell(ag, c)[f"{metric}_ci95_hw"] for c in CONDS]
            ax.errorbar(x, ys, yerr=es, marker=AMARK[ag], color=ACOL[ag], lw=1.9, markersize=7, capsize=3, label=ALAB[ag])
        ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_xlabel("Degradation condition")
        ax.set_ylabel(MLAB[metric])
        if pct: ax.set_ylim(0, 105)
        ax.legend(frameon=False, fontsize=9); ax.set_title(f"BPAN round trip — {MLAB[metric]}")
        fig.savefig(os.path.join(FIG, name + ".png")); fig.savefig(os.path.join(FIG, name + ".svg")); plt.close(fig)
    line("full_success_rate", "figOPS_twophase_full", True)
    line("dock_success_rate", "figOPS_twophase_dock", True)
    line("collisions_per_ep", "figOPS_twophase_coll")
    print("=== TWO-PHASE full-cycle success (%) ===")
    print(summ.pivot_table(index="agent", columns="condition", values="full_success_rate_mean").reindex(AGENTS)[CONDS].round(1).to_string())
    for m in ["full_success_rate", "collisions_per_ep"]:
        for e in ["agent", "condition", "agent:condition"]:
            r = fac[(fac.metric == m) & (fac.effect == e)]
            if len(r): print(f"  {m:20s} {e:16s} eta2={r.iloc[0]['partial_eta2']:.3f} p={r.iloc[0]['p_value']:.4f}")
    return summ, fac


# =================== TWO-WAY ===================
def analyze_twoway():
    df = pd.read_csv(RAW_WAY); df = df[df["phase"] == "eval"].copy()
    PAIRS = ["RuleBased_vs_RuleBased", "DQN_vs_RuleBased"]
    PLAB = {"RuleBased_vs_RuleBased": "Rule / Rule", "DQN_vs_RuleBased": "DQN / Rule"}
    PCOL = {"RuleBased_vs_RuleBased": "#C44E52", "DQN_vs_RuleBased": "#4C72B0"}
    PMARK = {"RuleBased_vs_RuleBased": "^", "DQN_vs_RuleBased": "o"}
    METRICS = ["success_rate", "vessel_collision_rate", "vessel_collisions_per_ep", "head_on_events_per_ep", "near_misses_per_ep", "min_cpa"]
    MLAB = {"success_rate": "Transit success (%)", "vessel_collision_rate": "Episodes with vessel collision (%)",
            "vessel_collisions_per_ep": "Inter-vessel collisions / episode", "head_on_events_per_ep": "Head-on encounter events / episode",
            "near_misses_per_ep": "Near-misses / episode", "min_cpa": "Min closest-point-of-approach (px)"}

    def per_seed(g):
        return {"success_rate": 100.0 * g["success"].mean(),
                "vessel_collision_rate": 100.0 * (num("vessel_collisions", g) > 0).mean(),
                "vessel_collisions_per_ep": num("vessel_collisions", g).mean(),
                "head_on_events_per_ep": num("head_on_events", g).mean(),
                "near_misses_per_ep": num("near_misses", g).mean(),
                "min_cpa": num("min_cpa", g).mean(), "n_eval": int(len(g))}
    rows = []
    # pool both directions per cell as the per-seed scalar
    for (pr, cond, seed), g in df.groupby(["pairing", "condition", "seed"], sort=True):
        rows.append({"pairing": pr, "condition": cond, "seed": int(seed), **per_seed(g)})
    ps = pd.DataFrame(rows); ps.to_csv(os.path.join(AGG, "twoway_per_seed.csv"), index=False)
    srows = []
    for (pr, cond), g in ps.groupby(["pairing", "condition"], sort=True):
        e = {"pairing": pr, "condition": cond, "n_seeds": len(g)}
        for m in METRICS:
            x = g[m].to_numpy(float); e[f"{m}_mean"] = np.nanmean(x); e[f"{m}_ci95_hw"] = t_hw(x)
        srows.append(e)
    summ = pd.DataFrame(srows); summ.to_csv(os.path.join(AGG, "twoway_summary.csv"), index=False)
    fac = factorial(ps, "pairing", METRICS); fac.to_csv(os.path.join(STAT, "twoway_factorial_anova.csv"), index=False)
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
    robdf.to_csv(os.path.join(STAT, "twoway_robustness.csv"), index=False)

    def cell(pr, c): return summ[(summ.pairing == pr) & (summ.condition == c)].iloc[0]
    rows2 = []
    for pr in PAIRS:
        rows2.append({"Pairing": PLAB[pr],
                      "Success clean": f"{cell(pr,'clean')['success_rate_mean']:.0f}%",
                      "Success harsh": f"{cell(pr,'harsh')['success_rate_mean']:.0f}%",
                      "vColl/ep": f"{cell(pr,'clean')['vessel_collisions_per_ep_mean']:.2f}",
                      "HeadOn/ep clean": f"{cell(pr,'clean')['head_on_events_per_ep_mean']:.2f}",
                      "HeadOn/ep harsh": f"{cell(pr,'harsh')['head_on_events_per_ep_mean']:.2f}",
                      "Interpretation": "Collision in a single shared channel is structural (~1/ep); a learned give-way vessel cuts head-on EVENTS, degraded by noise."})
    pd.DataFrame(rows2).to_csv(os.path.join(TBL, "table_ops3_twoway_outcomes.csv"), index=False)
    lab = {"pairing": "Pairing", "condition": "Condition", "pairing:condition": "Pairing\u00d7Condition", "seed": "Seed (block)"}
    rows3 = []
    for m in ["success_rate", "vessel_collisions_per_ep", "head_on_events_per_ep"]:
        sub = fac[fac.metric == m]
        for eff in ["pairing", "condition", "pairing:condition", "seed"]:
            e = sub[sub.effect == eff]
            if not len(e): continue
            e = e.iloc[0]
            rows3.append({"Metric": MLAB[m], "Effect": lab[eff], "F": f"{e['F']:.2f}", "p-value": f"{e['p_value']:.4f}",
                          "partial \u03b7\u00b2": f"{e['partial_eta2']:.3f}", "Interpretation": f"{'sig' if e['p_value']<0.05 else 'n.s.'}, {eff_lab(e['partial_eta2'])} effect"})
    pd.DataFrame(rows3).to_csv(os.path.join(TBL, "table_ops4_twoway_factorial.csv"), index=False)

    def line(metric, name, pct=False):
        fig, ax = plt.subplots(figsize=(7.0, 4.6)); x = np.arange(3)
        for pr in PAIRS:
            ys = [cell(pr, c)[f"{metric}_mean"] for c in CONDS]; es = [cell(pr, c)[f"{metric}_ci95_hw"] for c in CONDS]
            ax.errorbar(x, ys, yerr=es, marker=PMARK[pr], color=PCOL[pr], lw=1.9, markersize=7, capsize=3, label=PLAB[pr])
        ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_xlabel("Degradation condition")
        ax.set_ylabel(MLAB[metric])
        if pct: ax.set_ylim(0, 105)
        ax.legend(frameon=False, fontsize=9, title="Pairing"); ax.set_title(f"BPAN two-way traffic — {MLAB[metric]}")
        fig.savefig(os.path.join(FIG, name + ".png")); fig.savefig(os.path.join(FIG, name + ".svg")); plt.close(fig)
    line("success_rate", "figOPS_twoway_success", True)
    line("head_on_events_per_ep", "figOPS_twoway_headon")
    line("vessel_collisions_per_ep", "figOPS_twoway_vcoll")
    print("\n=== TWO-WAY head-on events/ep ===")
    print(summ.pivot_table(index="pairing", columns="condition", values="head_on_events_per_ep_mean").reindex(PAIRS)[CONDS].round(2).to_string())
    for m in ["success_rate", "head_on_events_per_ep", "vessel_collisions_per_ep"]:
        for e in ["pairing", "condition", "pairing:condition"]:
            r = fac[(fac.metric == m) & (fac.effect == e)]
            if len(r): print(f"  {m:24s} {e:18s} eta2={r.iloc[0]['partial_eta2']:.3f} p={r.iloc[0]['p_value']:.4f}")
    return summ, fac


def main():
    sp, fp = analyze_twophase()
    sw, fw = analyze_twoway()
    try:
        with pd.ExcelWriter(os.path.join(TBL, "all_tables_ops.xlsx"), engine="openpyxl") as xw:
            for n in ["table_ops1_twophase_outcomes", "table_ops2_twophase_factorial", "table_ops3_twoway_outcomes", "table_ops4_twoway_factorial"]:
                pd.read_csv(os.path.join(TBL, n + ".csv")).to_excel(xw, sheet_name=n[:31], index=False)
    except Exception as e:
        print("excel skip:", e)
    print("\nwrote results_bintulu/ops/{aggregated,statistics,tables,figures}")


if __name__ == "__main__":
    main()
