#!/usr/bin/env python3
"""
STUDY 3 — STATISTICS (rule-based baseline vs DQN, n=15 seeds)

Design: agent(5) x condition(3), shared 15 seeds (paired/randomized-block).
n=15 gives the paired tests real power (unlike Study 2's n=6 floor).

Pipeline:
  1. Shapiro-Wilk normality per cell (summary fraction per metric)
  2. Levene/Brown-Forsythe across agents at each condition
  3. Two-way factorial ANOVA  y ~ agent*condition + seed  (Type-II, partial eta^2)
  4. PRIMARY CONTRAST: RuleBased vs each DQN variant, PER CONDITION —
     paired Wilcoxon (exact) + paired t; Holm + BH within (metric, condition);
     Cohen's d_z + Cliff's delta.
  5. Robustness: clean vs harsh per agent (paired), n=15.

Outputs -> results_study3/statistics/
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_study3", "aggregated")
STAT = os.path.join(ROOT, "results_study3", "statistics")
os.makedirs(STAT, exist_ok=True)
ps = pd.read_csv(os.path.join(AGG, "per_seed.csv"))

AGENTS = ["RuleBased", "DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
DQN_VARIANTS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
CONDS = ["clean", "mid", "harsh"]
METRICS = ["success_rate", "mean_reward", "collision_rate", "collisions_per_ep",
           "iala_violations", "cte_mean", "docking_accuracy", "optimality_ratio"]
ALPHA = 0.05


def cohens_dz(diff):
    diff = np.asarray(diff, float); diff = diff[~np.isnan(diff)]
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
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj

def bh(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]; prev = min(prev, p[idx] * m / (rank + 1)); adj[idx] = min(1.0, prev)
    return adj

def paired(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y)); x, y = x[mask], y[mask]
    diff = x - y
    if len(x) >= 2 and np.any(diff != 0):
        try: w, wp = stats.wilcoxon(x, y, alternative="two-sided", method="exact")
        except Exception:
            try: w, wp = stats.wilcoxon(x, y, alternative="two-sided")
            except Exception: wp = np.nan
    else:
        wp = 1.0 if np.allclose(diff, 0) else np.nan
    try: _, tp = stats.ttest_rel(x, y)
    except Exception: tp = np.nan
    return {"n": len(x), "mean_x": np.nanmean(x) if len(x) else np.nan,
            "mean_y": np.nanmean(y) if len(y) else np.nan, "mean_diff": diff.mean() if len(x) else np.nan,
            "wilcoxon_p": wp, "t_p": tp, "cohens_dz": cohens_dz(diff), "cliffs_delta": cliffs_delta(x, y)}


def normality():
    rows = []
    for (ag, cond), g in ps.groupby(["agent", "condition"]):
        for m in METRICS:
            x = g[m].dropna().to_numpy(float)
            if len(x) >= 3 and np.ptp(x) > 0: W, p = stats.shapiro(x)
            else: W, p = np.nan, np.nan
            rows.append({"agent": ag, "condition": cond, "metric": m, "shapiro_W": W, "p_value": p,
                         "normal": (p > 0.05) if np.isfinite(p) else np.nan})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(STAT, "normality.csv"), index=False)
    return df.dropna(subset=["p_value"]).groupby("metric")["normal"].mean().to_dict()


def levene_by_condition():
    rows = []
    for cond in CONDS:
        sub = ps[ps.condition == cond]
        for m in METRICS:
            samples = [sub[sub.agent == a][m].dropna().to_numpy(float) for a in AGENTS]
            samples = [s for s in samples if len(s) > 1]
            if len(samples) >= 2: W, p = stats.levene(*samples, center="median")
            else: W, p = np.nan, np.nan
            rows.append({"condition": cond, "metric": m, "levene_W": W, "p_value": p,
                         "homoscedastic": (p > 0.05) if np.isfinite(p) else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "variance_homogeneity.csv"), index=False)


def factorial():
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    rows = []
    for m in METRICS:
        d = ps[["agent", "condition", "seed", m]].dropna().rename(columns={m: "y"}).copy()
        if d["y"].nunique() < 2: continue
        d["agent"] = d["agent"].astype("category"); d["condition"] = d["condition"].astype("category"); d["seed"] = d["seed"].astype("category")
        model = ols("y ~ C(agent)*C(condition) + C(seed)", data=d).fit()
        aov = anova_lm(model, typ=2)
        ssr = aov.loc["Residual", "sum_sq"]
        for eff in aov.index:
            if eff == "Residual": continue
            ss = aov.loc[eff, "sum_sq"]
            rows.append({"metric": m, "effect": eff.replace("C(", "").replace(")", ""),
                         "F": aov.loc[eff, "F"], "p_value": aov.loc[eff, "PR(>F)"],
                         "partial_eta2": ss / (ss + ssr) if (ss + ssr) > 0 else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "factorial_anova.csv"), index=False)
    return pd.DataFrame(rows)


def baseline_vs_dqn():
    """RuleBased vs each DQN variant, per condition. Holm/BH within (metric,condition)."""
    rows = []
    for cond in CONDS:
        sub = ps[ps.condition == cond]
        rb = sub[sub.agent == "RuleBased"].sort_values("seed")
        for m in METRICS:
            recs = []
            for dq in DQN_VARIANTS:
                dv = sub[sub.agent == dq].sort_values("seed")
                r = paired(rb[m].to_numpy(float), dv[m].to_numpy(float))
                recs.append({"condition": cond, "metric": m, "baseline": "RuleBased", "vs": dq, **r})
            wp = [r["wilcoxon_p"] for r in recs]
            h, b = holm(wp), bh(wp)
            for i, r in enumerate(recs):
                r["wilcoxon_p_holm"] = h[i]; r["wilcoxon_p_bh"] = b[i]
                r["sig_holm"] = bool(h[i] < ALPHA); r["effect_mag"] = mag(r["cohens_dz"])
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "baseline_vs_dqn.csv"), index=False)
    return pd.DataFrame(rows)


def robustness():
    rows = []
    for ag in AGENTS:
        base = ps[(ps.agent == ag) & (ps.condition == "clean")].sort_values("seed")
        harsh = ps[(ps.agent == ag) & (ps.condition == "harsh")].sort_values("seed")
        for m in METRICS:
            r = paired(harsh[m].to_numpy(float), base[m].to_numpy(float))
            rows.append({"agent": ag, "metric": m, "clean_mean": r["mean_y"], "harsh_mean": r["mean_x"],
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
    bvd = baseline_vs_dqn()
    rob = robustness()

    json.dump({
        "design": "agent(5) x condition(3), shared 15 seeds (paired).",
        "unit": "per-seed scalar (n=15).",
        "normality_fraction_passing_shapiro": {k: round(float(v), 3) for k, v in frac.items()},
        "omnibus": "two-way factorial ANOVA y ~ agent*condition + seed (Type-II, partial eta^2).",
        "primary_contrast": "RuleBased vs each DQN per condition; paired Wilcoxon exact + paired t; Holm + BH within (metric,condition); Cohen's d_z + Cliff's delta.",
        "robustness": "clean vs harsh per agent, paired, n=15.",
        "alpha": ALPHA,
    }, open(os.path.join(STAT, "methodology.json"), "w"), indent=2)

    print("=== factorial ANOVA — agent & agent:condition effects (primary metrics) ===")
    show = fac[fac.effect.isin(["agent", "condition", "agent:condition"])]
    print(show[show.metric.isin(["success_rate", "collision_rate", "iala_violations", "cte_mean"])][
        ["metric", "effect", "F", "p_value", "partial_eta2"]].round(4).to_string(index=False))

    print("\n=== RuleBased vs DQN at HARSH — safety metrics, Holm-corrected ===")
    h = bvd[(bvd.condition == "harsh") & (bvd.metric.isin(["collision_rate", "iala_violations", "success_rate"]))]
    print(h[["metric", "vs", "mean_x", "mean_y", "mean_diff", "wilcoxon_p_holm", "cohens_dz", "effect_mag", "sig_holm"]].round(3).to_string(index=False))
    nsig = int(bvd["sig_holm"].sum())
    print(f"\nTotal Holm-significant RuleBased-vs-DQN contrasts: {nsig} / {len(bvd)}")


if __name__ == "__main__":
    main()
