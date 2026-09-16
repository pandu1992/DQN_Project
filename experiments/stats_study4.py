#!/usr/bin/env python3
"""
STUDY 4 — STATISTICS (two independent vessels; n=8 seeds per cell)

Design: pairing(4) x condition(3), shared 8 seeds (paired/randomized-block).

Pipeline:
  1. Shapiro-Wilk normality per cell (summary fraction per metric)
  2. Levene/Brown-Forsythe across pairings at each condition
  3. Two-way factorial ANOVA  y ~ pairing*condition + seed  (Type-II, partial eta^2)
  4. PRIMARY CONTRAST: the classical reference RuleBased_vs_RuleBased pairing vs
     each learned/mixed pairing, PER CONDITION — paired Wilcoxon (exact) + paired
     t; Holm + BH within (metric, condition); Cohen's d_z + Cliff's delta.
  5. Robustness: clean vs harsh per pairing (paired), n=8.

Focus metrics are the INTER-VESSEL safety outcomes (the point of Study 4) plus
the per-vessel navigation/precision metrics.

Outputs -> results_study4/statistics/
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study4", "aggregated")
STAT = os.path.join(ROOT, "results_study4", "statistics")
os.makedirs(STAT, exist_ok=True)
ps = pd.read_csv(os.path.join(AGG, "per_seed.csv"))

PAIRINGS = ["DQN_vs_DQN", "DDQN_vs_DDQN", "DQN_vs_RuleBased", "RuleBased_vs_RuleBased"]
REFERENCE = "RuleBased_vs_RuleBased"
OTHERS = ["DQN_vs_DQN", "DDQN_vs_DDQN", "DQN_vs_RuleBased"]
CONDS = ["clean", "mid", "harsh"]
METRICS = ["vessel_collision_rate", "vessel_collisions_per_ep", "near_misses_per_ep",
           "min_cpa", "give_way_events", "success_rate", "mean_reward",
           "cte_mean", "docking_accuracy", "optimality_ratio"]
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
    p = np.asarray(p, float); m = len(p)
    valid = np.where(np.isfinite(p))[0]
    adj = np.full(m, np.nan)
    if len(valid) == 0: return adj
    order = valid[np.argsort(p[valid])]; run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(valid) - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def bh(p):
    p = np.asarray(p, float); m = len(p)
    valid = np.where(np.isfinite(p))[0]
    adj = np.full(m, np.nan)
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
    return {"n": len(x), "mean_x": np.nanmean(x) if len(x) else np.nan,
            "mean_y": np.nanmean(y) if len(y) else np.nan,
            "mean_diff": diff.mean() if len(x) else np.nan,
            "wilcoxon_p": wp, "t_p": tp, "cohens_dz": cohens_dz(diff), "cliffs_delta": cliffs_delta(x, y)}


def normality():
    rows = []
    for (pr, cond), g in ps.groupby(["pairing", "condition"]):
        for m in METRICS:
            x = g[m].dropna().to_numpy(float)
            if len(x) >= 3 and np.ptp(x) > 0: W, p = stats.shapiro(x)
            else: W, p = np.nan, np.nan
            rows.append({"pairing": pr, "condition": cond, "metric": m, "shapiro_W": W, "p_value": p,
                         "normal": (p > 0.05) if np.isfinite(p) else np.nan})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(STAT, "normality.csv"), index=False)
    return df.dropna(subset=["p_value"]).groupby("metric")["normal"].mean().to_dict()


def levene_by_condition():
    rows = []
    for cond in CONDS:
        sub = ps[ps.condition == cond]
        for m in METRICS:
            samples = [sub[sub.pairing == p][m].dropna().to_numpy(float) for p in PAIRINGS]
            samples = [s for s in samples if len(s) > 1 and np.ptp(s) > 0]
            if len(samples) >= 2: W, p = stats.levene(*samples, center="median")
            else: W, p = np.nan, np.nan
            rows.append({"condition": cond, "metric": m, "levene_W": W, "p_value": p,
                         "homoscedastic": (p > 0.05) if np.isfinite(p) else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "variance_homogeneity.csv"), index=False)


def factorial():
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    rows = []
    for m in METRICS:
        d = ps[["pairing", "condition", "seed", m]].dropna().rename(columns={m: "y"}).copy()
        if d["y"].nunique() < 2: continue
        d["pairing"] = d["pairing"].astype("category"); d["condition"] = d["condition"].astype("category"); d["seed"] = d["seed"].astype("category")
        try:
            model = ols("y ~ C(pairing)*C(condition) + C(seed)", data=d).fit()
            aov = anova_lm(model, typ=2)
        except Exception:
            continue
        ssr = aov.loc["Residual", "sum_sq"]
        for eff in aov.index:
            if eff == "Residual": continue
            ss = aov.loc[eff, "sum_sq"]
            rows.append({"metric": m, "effect": eff.replace("C(", "").replace(")", ""),
                         "F": aov.loc[eff, "F"], "p_value": aov.loc[eff, "PR(>F)"],
                         "partial_eta2": ss / (ss + ssr) if (ss + ssr) > 0 else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "factorial_anova.csv"), index=False)
    return pd.DataFrame(rows)


def reference_vs_others():
    """RuleBased_vs_RuleBased (classical reference) vs each learned/mixed pairing,
    per condition. Holm/BH within (metric, condition)."""
    rows = []
    for cond in CONDS:
        sub = ps[ps.condition == cond]
        ref = sub[sub.pairing == REFERENCE].sort_values("seed")
        for m in METRICS:
            recs = []
            for pr in OTHERS:
                dv = sub[sub.pairing == pr].sort_values("seed")
                r = paired(ref[m].to_numpy(float), dv[m].to_numpy(float))
                recs.append({"condition": cond, "metric": m, "reference": REFERENCE, "vs": pr, **r})
            wp = [r["wilcoxon_p"] for r in recs]
            h, b = holm(wp), bh(wp)
            for i, r in enumerate(recs):
                r["wilcoxon_p_holm"] = h[i]; r["wilcoxon_p_bh"] = b[i]
                r["sig_holm"] = bool(np.isfinite(h[i]) and h[i] < ALPHA); r["effect_mag"] = mag(r["cohens_dz"])
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "reference_vs_pairings.csv"), index=False)
    return pd.DataFrame(rows)


def robustness():
    rows = []
    for pr in PAIRINGS:
        base = ps[(ps.pairing == pr) & (ps.condition == "clean")].sort_values("seed")
        harsh = ps[(ps.pairing == pr) & (ps.condition == "harsh")].sort_values("seed")
        for m in METRICS:
            r = paired(harsh[m].to_numpy(float), base[m].to_numpy(float))
            rows.append({"pairing": pr, "metric": m, "clean_mean": r["mean_y"], "harsh_mean": r["mean_x"],
                         "abs_change": r["mean_diff"], "wilcoxon_p": r["wilcoxon_p"],
                         "cohens_dz": r["cohens_dz"], "effect_mag": mag(r["cohens_dz"]), "cliffs_delta": r["cliffs_delta"]})
    df = pd.DataFrame(rows)
    for m in METRICS:
        idx = df.index[df.metric == m]
        df.loc[idx, "wilcoxon_p_holm"] = holm(df.loc[idx, "wilcoxon_p"].to_numpy(float))
        df.loc[idx, "sig_holm"] = df.loc[idx, "wilcoxon_p_holm"] < ALPHA
    df.to_csv(os.path.join(STAT, "robustness_clean_vs_harsh.csv"), index=False)
    return df


def main():
    frac = normality()
    levene_by_condition()
    fac = factorial()
    rvo = reference_vs_others()
    rob = robustness()

    json.dump({
        "design": "pairing(4) x condition(3), shared 8 seeds (paired).",
        "unit": "per-seed cell scalar (n=8). Inter-vessel metrics per encounter; navigation/physical per vessel (both) then averaged.",
        "normality_fraction_passing_shapiro": {k: round(float(v), 3) for k, v in frac.items()},
        "omnibus": "two-way factorial ANOVA y ~ pairing*condition + seed (Type-II, partial eta^2).",
        "primary_contrast": "RuleBased_vs_RuleBased (classical reference) vs each learned/mixed pairing per condition; paired Wilcoxon exact + paired t; Holm + BH within (metric,condition); Cohen's d_z + Cliff's delta.",
        "robustness": "clean vs harsh per pairing, paired, n=8.",
        "alpha": ALPHA,
    }, open(os.path.join(STAT, "methodology.json"), "w"), indent=2)

    print("=== factorial ANOVA — pairing & pairing:condition (inter-vessel safety) ===")
    show = fac[fac.effect.isin(["pairing", "condition", "pairing:condition"])]
    print(show[show.metric.isin(["vessel_collision_rate", "vessel_collisions_per_ep", "min_cpa", "near_misses_per_ep"])][
        ["metric", "effect", "F", "p_value", "partial_eta2"]].round(4).to_string(index=False))

    print("\n=== Reference (RuleBased x2) vs each pairing at HARSH — safety, Holm-corrected ===")
    h = rvo[(rvo.condition == "harsh") & (rvo.metric.isin(["vessel_collision_rate", "min_cpa", "success_rate"]))]
    print(h[["metric", "vs", "mean_x", "mean_y", "mean_diff", "wilcoxon_p_holm", "cohens_dz", "effect_mag", "sig_holm"]].round(3).to_string(index=False))
    nsig = int(rvo["sig_holm"].sum())
    print(f"\nTotal Holm-significant reference-vs-pairing contrasts: {nsig} / {len(rvo)}")


if __name__ == "__main__":
    main()
