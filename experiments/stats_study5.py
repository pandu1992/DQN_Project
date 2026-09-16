#!/usr/bin/env python3
"""
STUDY 5 — STATISTICS (two-phase round trip + two-way opposing traffic; n=8)

Scenario A (twophase): agent(2) x condition(3), shared 8 seeds.
  - factorial ANOVA y ~ agent*condition + seed (partial eta^2)
  - PRIMARY paired contrast: dock vs FULL-cycle success per (agent, condition)
    (does completing the outbound leg significantly cut success? paired Wilcoxon)
  - robustness clean vs harsh per agent

Scenario B (twoway): pairing(3) x condition(3), shared 8 seeds.
  - factorial ANOVA on inter-vessel safety
  - PRIMARY contrast: Rule_vs_Rule reference vs each pairing per condition
    (paired Wilcoxon + Holm + BH; Cohen d_z + Cliff delta)
  - robustness clean vs harsh per pairing

Outputs -> results_study5/statistics/
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study5", "aggregated")
STAT = os.path.join(ROOT, "results_study5", "statistics")
os.makedirs(STAT, exist_ok=True)

phase = pd.read_csv(os.path.join(AGG, "twophase_per_seed.csv"))
way = pd.read_csv(os.path.join(AGG, "twoway_per_seed.csv"))

CONDS = ["clean", "mid", "harsh"]
PHASE_AGENTS = ["RuleBased", "DQN"]
WAY_PAIRINGS = ["Rule_vs_Rule", "DQN_vs_Rule", "DQN_vs_DQN"]
WAY_REF = "Rule_vs_Rule"
WAY_OTHERS = ["DQN_vs_Rule", "DQN_vs_DQN"]
PHASE_METRICS = ["full_cycle_success_rate", "dock_success_rate", "dock_accuracy", "cte_mean", "iala_violations"]
WAY_METRICS = ["vessel_collision_rate", "vessel_collisions_per_ep", "near_misses_per_ep",
               "min_cpa", "head_on_events", "success_rate", "cte_mean"]
ALPHA = 0.05


def cohens_dz(diff):
    diff = np.asarray(diff, float); diff = diff[~np.isnan(diff)]
    if len(diff) < 2: return np.nan
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0

def cliffs_delta(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]; b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0: return np.nan
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

def mag(d):
    ad = abs(d)
    if not np.isfinite(ad): return "n/a"
    return "negligible" if ad < 0.2 else "small" if ad < 0.5 else "medium" if ad < 0.8 else "large"

def holm(p):
    p = np.asarray(p, float); m = len(p); valid = np.where(np.isfinite(p))[0]; adj = np.full(m, np.nan)
    if len(valid) == 0: return adj
    order = valid[np.argsort(p[valid])]; run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(valid) - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def bh(p):
    p = np.asarray(p, float); m = len(p); valid = np.where(np.isfinite(p))[0]; adj = np.full(m, np.nan)
    if len(valid) == 0: return adj
    order = valid[np.argsort(p[valid])]; k = len(valid); prev = 1.0
    for rank in range(k - 1, -1, -1):
        idx = order[rank]; prev = min(prev, p[idx] * k / (rank + 1)); adj[idx] = min(1.0, prev)
    return adj

def paired(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y)); x, y = x[mask], y[mask]
    diff = x - y
    if len(x) >= 2 and np.any(diff != 0):
        try: _, wp = stats.wilcoxon(x, y, alternative="two-sided", method="exact")
        except Exception:
            try: _, wp = stats.wilcoxon(x, y, alternative="two-sided")
            except Exception: wp = np.nan
    else:
        wp = 1.0 if (len(x) and np.allclose(diff, 0)) else np.nan
    try: _, tp = stats.ttest_rel(x, y)
    except Exception: tp = np.nan
    return {"n": len(x), "mean_x": np.nanmean(x) if len(x) else np.nan, "mean_y": np.nanmean(y) if len(y) else np.nan,
            "mean_diff": diff.mean() if len(x) else np.nan, "wilcoxon_p": wp, "t_p": tp,
            "cohens_dz": cohens_dz(diff), "cliffs_delta": cliffs_delta(x, y)}


def factorial(ps, factor, levels, metrics, out):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    rows = []
    for m in metrics:
        d = ps[[factor, "condition", "seed", m]].dropna().rename(columns={m: "y", factor: "F1"}).copy()
        if d["y"].nunique() < 2: continue
        d["F1"] = d["F1"].astype("category"); d["condition"] = d["condition"].astype("category"); d["seed"] = d["seed"].astype("category")
        try:
            model = ols("y ~ C(F1)*C(condition) + C(seed)", data=d).fit()
            aov = anova_lm(model, typ=2)
        except Exception:
            continue
        ssr = aov.loc["Residual", "sum_sq"]
        emap = {"C(F1)": factor, "C(condition)": "condition", "C(F1):C(condition)": factor + ":condition", "C(seed)": "seed"}
        for eff in aov.index:
            if eff == "Residual": continue
            ss = aov.loc[eff, "sum_sq"]
            rows.append({"metric": m, "effect": emap.get(eff, eff), "F": aov.loc[eff, "F"],
                         "p_value": aov.loc[eff, "PR(>F)"], "partial_eta2": ss / (ss + ssr) if (ss + ssr) > 0 else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(STAT, out), index=False)
    return pd.DataFrame(rows)


def phase_dock_vs_full():
    """Does the outbound leg significantly cut success? paired dock% vs full% per (agent,cond)."""
    rows = []
    for ag in PHASE_AGENTS:
        for cond in CONDS:
            g = phase[(phase.agent == ag) & (phase.condition == cond)].sort_values("seed")
            r = paired(g["dock_success_rate"].to_numpy(float), g["full_cycle_success_rate"].to_numpy(float))
            rows.append({"agent": ag, "condition": cond, "dock_mean": r["mean_x"], "full_mean": r["mean_y"],
                         "drop": r["mean_diff"], "wilcoxon_p": r["wilcoxon_p"], "cohens_dz": r["cohens_dz"],
                         "effect_mag": mag(r["cohens_dz"]), "cliffs_delta": r["cliffs_delta"]})
    df = pd.DataFrame(rows)
    df["wilcoxon_p_holm"] = holm(df["wilcoxon_p"].to_numpy(float))
    df["sig_holm"] = df["wilcoxon_p_holm"] < ALPHA
    df.to_csv(os.path.join(STAT, "twophase_dock_vs_full.csv"), index=False)
    return df


def way_reference_vs_others():
    rows = []
    for cond in CONDS:
        sub = way[way.condition == cond]
        ref = sub[sub.pairing == WAY_REF].sort_values("seed")
        for m in WAY_METRICS:
            recs = []
            for pr in WAY_OTHERS:
                dv = sub[sub.pairing == pr].sort_values("seed")
                r = paired(ref[m].to_numpy(float), dv[m].to_numpy(float))
                recs.append({"condition": cond, "metric": m, "reference": WAY_REF, "vs": pr, **r})
            wp = [r["wilcoxon_p"] for r in recs]
            h, b = holm(wp), bh(wp)
            for i, r in enumerate(recs):
                r["wilcoxon_p_holm"] = h[i]; r["wilcoxon_p_bh"] = b[i]
                r["sig_holm"] = bool(np.isfinite(h[i]) and h[i] < ALPHA); r["effect_mag"] = mag(r["cohens_dz"])
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "twoway_reference_vs_pairings.csv"), index=False)
    return pd.DataFrame(rows)


def robustness(ps, factor, levels, metrics, out):
    rows = []
    for lvl in levels:
        base = ps[(ps[factor] == lvl) & (ps.condition == "clean")].sort_values("seed")
        harsh = ps[(ps[factor] == lvl) & (ps.condition == "harsh")].sort_values("seed")
        for m in metrics:
            r = paired(harsh[m].to_numpy(float), base[m].to_numpy(float))
            rows.append({factor: lvl, "metric": m, "clean_mean": r["mean_y"], "harsh_mean": r["mean_x"],
                         "abs_change": r["mean_diff"], "wilcoxon_p": r["wilcoxon_p"], "cohens_dz": r["cohens_dz"],
                         "effect_mag": mag(r["cohens_dz"]), "cliffs_delta": r["cliffs_delta"]})
    df = pd.DataFrame(rows)
    for m in metrics:
        idx = df.index[df.metric == m]
        df.loc[idx, "wilcoxon_p_holm"] = holm(df.loc[idx, "wilcoxon_p"].to_numpy(float))
        df.loc[idx, "sig_holm"] = df.loc[idx, "wilcoxon_p_holm"] < ALPHA
    df.to_csv(os.path.join(STAT, out), index=False)
    return df


def main():
    pf = factorial(phase, "agent", PHASE_AGENTS, PHASE_METRICS, "twophase_factorial_anova.csv")
    wf = factorial(way, "pairing", WAY_PAIRINGS, WAY_METRICS, "twoway_factorial_anova.csv")
    dvf = phase_dock_vs_full()
    rvo = way_reference_vs_others()
    robustness(phase, "agent", PHASE_AGENTS, PHASE_METRICS, "twophase_robustness.csv")
    robustness(way, "pairing", WAY_PAIRINGS, WAY_METRICS, "twoway_robustness.csv")

    json.dump({
        "design_twophase": "agent(2) x condition(3), shared 8 seeds (paired).",
        "design_twoway": "pairing(3) x condition(3), shared 8 seeds (paired).",
        "unit": "per-seed cell scalar (n=8).",
        "twophase_primary": "paired dock% vs full-cycle% per (agent,condition): does the outbound leg significantly cut success? Wilcoxon exact + Holm; d_z + Cliff's delta.",
        "twoway_primary": "Rule_vs_Rule reference vs each pairing per condition; paired Wilcoxon + Holm + BH; d_z + Cliff's delta.",
        "robustness": "clean vs harsh, paired, n=8.",
        "alpha": ALPHA,
    }, open(os.path.join(STAT, "methodology.json"), "w"), indent=2)

    print("=== two-phase: dock vs full-cycle drop (paired) ===")
    print(dvf[["agent", "condition", "dock_mean", "full_mean", "drop", "wilcoxon_p_holm", "cohens_dz", "sig_holm"]].round(3).to_string(index=False))
    print("\n=== two-way factorial ANOVA (inter-vessel safety) ===")
    show = wf[wf.effect.isin(["pairing", "condition", "pairing:condition"])]
    print(show[show.metric.isin(["vessel_collision_rate", "min_cpa", "head_on_events", "success_rate"])][
        ["metric", "effect", "F", "p_value", "partial_eta2"]].round(4).to_string(index=False))
    print(f"\nTwo-way reference-vs-pairing Holm-significant: {int(rvo['sig_holm'].sum())} / {len(rvo)}")


if __name__ == "__main__":
    main()
