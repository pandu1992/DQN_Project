#!/usr/bin/env python3
"""
Baseline-fairness analysis (Q1 audit Task 3).

Aggregates the 3-arm x 3-condition x 8-seed chart-fairness experiment and runs:
  - per-seed scalars (success, CTE|success, collision rate, docking|solved)
  - summary (mean +/- 95% CI)
  - two-way factorial ANOVA: arm x condition (+ seed), partial eta^2
  - key paired contrasts (Wilcoxon exact + Holm, Cohen d_z):
      * DRL_no_chart vs DRL_with_chart   (does the chart prior help?)
      * DRL_with_chart vs RuleBased      (does the gap to the baseline close?)
      * DRL_no_chart vs RuleBased        (the original, confounded comparison)
Outputs -> results_chart/{aggregated,statistics}/
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results_chart", "raw", "eval_episodes.csv")
AGG = os.path.join(ROOT, "results_chart", "aggregated")
STAT = os.path.join(ROOT, "results_chart", "statistics")
os.makedirs(AGG, exist_ok=True); os.makedirs(STAT, exist_ok=True)

ARMS = ["DRL_no_chart", "DRL_with_chart", "RuleBased"]
CONDS = ["clean", "mid", "harsh"]
METRICS = ["success_rate", "mean_reward", "cte_success", "collision_rate", "docking_accuracy", "optimality_ratio"]
ALPHA = 0.05


def num(c, fr):
    return pd.to_numeric(fr[c], errors="coerce")


def per_seed(g):
    succ = g[g["success"] == 1]
    return {
        "success_rate": 100.0 * g["success"].mean(),
        "mean_reward": num("reward", g).mean(),
        "cte_success": num("cte_mean", succ).mean() if len(succ) else np.nan,
        "collision_rate": 100.0 * (num("collisions", g) > 0).mean(),
        "docking_accuracy": num("docking_accuracy", succ).mean() if len(succ) else np.nan,
        "optimality_ratio": num("optimality_ratio", succ).mean() if len(succ) else np.nan,
        "n_solved": int(len(succ)), "n_eval": int(len(g)),
    }


def t_hw(x, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    return stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan


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
    for i, idx in enumerate(order):
        run = max(run, (len(v) - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def paired(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; d = x - y
    if len(x) >= 2 and np.any(d != 0):
        try: _, wp = stats.wilcoxon(x, y, method="exact")
        except Exception:
            try: _, wp = stats.wilcoxon(x, y)
            except Exception: wp = np.nan
    else:
        wp = 1.0 if (len(x) and np.allclose(d, 0)) else np.nan
    return {"n": len(x), "mean_x": np.nanmean(x) if len(x) else np.nan,
            "mean_y": np.nanmean(y) if len(y) else np.nan, "mean_diff": d.mean() if len(x) else np.nan,
            "wilcoxon_p": wp, "cohens_dz": cohens_dz(d)}


def main():
    df = pd.read_csv(RAW); df = df[df["phase"] == "eval"].copy()
    rows = []
    for (arm, cond, seed), g in df.groupby(["arm", "condition", "seed"], sort=True):
        rows.append({"arm": arm, "condition": cond, "seed": int(seed), **per_seed(g)})
    ps = pd.DataFrame(rows); ps.to_csv(os.path.join(AGG, "per_seed.csv"), index=False)

    srows = []
    for (arm, cond), g in ps.groupby(["arm", "condition"], sort=True):
        e = {"arm": arm, "condition": cond, "n_seeds": len(g)}
        for m in METRICS:
            x = g[m].to_numpy(float)
            e[f"{m}_mean"] = np.nanmean(x); e[f"{m}_ci95_hw"] = t_hw(x)
        srows.append(e)
    summ = pd.DataFrame(srows); summ.to_csv(os.path.join(AGG, "summary.csv"), index=False)

    # factorial ANOVA
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    frows = []
    for m in METRICS:
        d = ps[["arm", "condition", "seed", m]].dropna().rename(columns={m: "y"}).copy()
        if d["y"].nunique() < 2: continue
        d["arm"] = d["arm"].astype("category"); d["condition"] = d["condition"].astype("category"); d["seed"] = d["seed"].astype("category")
        try:
            aov = anova_lm(ols("y ~ C(arm)*C(condition) + C(seed)", data=d).fit(), typ=2)
        except Exception:
            continue
        ssr = aov.loc["Residual", "sum_sq"]
        emap = {"C(arm)": "arm", "C(condition)": "condition", "C(arm):C(condition)": "arm:condition", "C(seed)": "seed"}
        for eff in aov.index:
            if eff == "Residual": continue
            ss = aov.loc[eff, "sum_sq"]
            frows.append({"metric": m, "effect": emap.get(eff, eff), "F": aov.loc[eff, "F"],
                          "p_value": aov.loc[eff, "PR(>F)"], "partial_eta2": ss / (ss + ssr) if (ss + ssr) > 0 else np.nan})
    fac = pd.DataFrame(frows); fac.to_csv(os.path.join(STAT, "factorial_anova.csv"), index=False)

    # key paired contrasts, per condition, Holm within (contrast, metric) family across conditions
    contrasts = [("DRL_no_chart", "DRL_with_chart"), ("DRL_with_chart", "RuleBased"), ("DRL_no_chart", "RuleBased")]
    crows = []
    for (A, B) in contrasts:
        for m in METRICS:
            recs = []
            for cond in CONDS:
                a = ps[(ps.arm == A) & (ps.condition == cond)].sort_values("seed")
                b = ps[(ps.arm == B) & (ps.condition == cond)].sort_values("seed")
                r = paired(a[m].to_numpy(float), b[m].to_numpy(float))
                recs.append({"contrast": f"{A} vs {B}", "metric": m, "condition": cond, **r})
            h = holm([r["wilcoxon_p"] for r in recs])
            for i, r in enumerate(recs):
                r["wilcoxon_p_holm"] = h[i]; r["sig_holm"] = bool(np.isfinite(h[i]) and h[i] < ALPHA); r["effect_mag"] = mag(r["cohens_dz"])
                crows.append(r)
    con = pd.DataFrame(crows); con.to_csv(os.path.join(STAT, "contrasts.csv"), index=False)

    # ---- print headline ----
    print("=== success rate (%) by arm x condition ===")
    print(summ.pivot_table(index="arm", columns="condition", values="success_rate_mean").reindex(ARMS)[CONDS].round(1).to_string())
    print("\n=== arm effect on success (factorial) ===")
    print(fac[(fac.metric == "success_rate") & (fac.effect.isin(["arm", "condition", "arm:condition"]))][["effect", "F", "p_value", "partial_eta2"]].round(4).to_string(index=False))
    print("\n=== KEY contrasts on success rate (paired, Holm) ===")
    k = con[(con.metric == "success_rate")]
    print(k[["contrast", "condition", "mean_x", "mean_y", "mean_diff", "wilcoxon_p_holm", "cohens_dz", "sig_holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
