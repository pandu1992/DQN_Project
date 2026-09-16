#!/usr/bin/env python3
"""
V3 ROBUSTNESS — STATISTICS

Design: per-seed scalar is the unit (n=6 per cell). The SAME 6 seeds appear in
every cell (shared/paired). Factorial: Algorithm(4) x Noise(3) x PER(3).

Pipeline:
  1. normality (Shapiro-Wilk) per cell + summary fraction per metric
  2. variance homogeneity (Levene/Brown-Forsythe) across algorithms at clean
  3. three-way factorial ANOVA  y ~ algorithm*noise*PER + seed  (Type-II SS,
     partial eta^2) for each metric — MAIN EFFECTS + ALL INTERACTIONS.
     The Algorithm x Noise and Algorithm x PER interactions are the core
     robustness question ("do algorithms degrade differently?").
  4. robustness contrasts: for each algorithm, worst-condition vs clean,
     paired Wilcoxon + effect size (already have degradation.csv means).
  5. algorithm posthoc AT the harshest condition (noise=0.25,PER=0.4):
     pairwise paired Wilcoxon + Holm/BH + Cohen's d_z + Cliff's delta.

Outputs -> results_v3/statistics/
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results_v3", "aggregated")
STAT = os.path.join(ROOT, "results_v3", "statistics")
os.makedirs(STAT, exist_ok=True)

per_seed = pd.read_csv(os.path.join(AGG, "per_seed.csv"))
ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
PRIMARY = ["success_rate", "mean_reward", "collision_rate", "collisions_per_ep",
           "iala_violations", "cte_mean", "docking_accuracy", "dropped_frames", "optimality_ratio"]
ALPHA = 0.05


# ---------- helpers ----------
def cohens_dz(diff):
    diff = np.asarray(diff, float); diff = diff[~np.isnan(diff)]
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0

def cliffs_delta(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

def mag_d(d):
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
        except Exception: w, wp = np.nan, np.nan
    else:
        wp = 1.0 if np.allclose(diff, 0) else np.nan
    return {"n": len(x), "mean_diff": diff.mean() if len(x) else np.nan,
            "wilcoxon_p": wp, "cohens_dz": cohens_dz(diff), "cliffs_delta": cliffs_delta(x, y)}


# ---------- 1. normality ----------
def normality():
    rows = []
    for (algo, ns, per), g in per_seed.groupby(["algorithm", "noise_std", "packet_error_rate"]):
        for m in PRIMARY:
            x = g[m].dropna().to_numpy(float)
            if len(x) >= 3 and np.ptp(x) > 0:
                W, p = stats.shapiro(x)
            else:
                W, p = np.nan, np.nan
            rows.append({"algorithm": algo, "noise_std": ns, "packet_error_rate": per,
                         "metric": m, "shapiro_W": W, "p_value": p,
                         "normal": (p > 0.05) if np.isfinite(p) else np.nan})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(STAT, "normality.csv"), index=False)
    frac = df.dropna(subset=["p_value"]).groupby("metric")["normal"].mean()
    return frac.to_dict()


# ---------- 2. variance homogeneity across algorithms at clean ----------
def levene_clean():
    rows = []
    base = per_seed[(per_seed.noise_std == 0) & (per_seed.packet_error_rate == 0)]
    for m in PRIMARY:
        samples = [base[base.algorithm == a][m].dropna().to_numpy(float) for a in ALGOS]
        samples = [s for s in samples if len(s) > 1]
        if len(samples) >= 2:
            W, p = stats.levene(*samples, center="median")
        else:
            W, p = np.nan, np.nan
        rows.append({"metric": m, "levene_W": W, "p_value": p,
                     "homoscedastic": (p > 0.05) if np.isfinite(p) else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "variance_homogeneity.csv"), index=False)


# ---------- 3. three-way factorial ANOVA ----------
def factorial():
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    rows = []
    for m in PRIMARY:
        d = per_seed[["algorithm", "noise_std", "packet_error_rate", "seed", m]].dropna().copy()
        d = d.rename(columns={m: "y"})
        # need variation; skip if degenerate
        if d["y"].nunique() < 2:
            continue
        d["algorithm"] = d["algorithm"].astype("category")
        d["noise"] = d["noise_std"].astype("category")
        d["per"] = d["packet_error_rate"].astype("category")
        d["seed"] = d["seed"].astype("category")
        formula = "y ~ C(algorithm)*C(noise)*C(per) + C(seed)"
        model = ols(formula, data=d).fit()
        aov = anova_lm(model, typ=2)
        ss_resid = aov.loc["Residual", "sum_sq"]
        for eff in aov.index:
            if eff == "Residual":
                continue
            ss = aov.loc[eff, "sum_sq"]; F = aov.loc[eff, "F"]; p = aov.loc[eff, "PR(>F)"]
            pe2 = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else np.nan
            rows.append({"metric": m,
                         "effect": eff.replace("C(", "").replace(")", ""),
                         "F": F, "p_value": p, "partial_eta2": pe2})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(STAT, "factorial_anova.csv"), index=False)
    return df


# ---------- 4. robustness: worst condition vs clean, per algorithm ----------
def robustness():
    worst = (0.25, 0.4)
    rows = []
    for algo in ALGOS:
        base = per_seed[(per_seed.algorithm == algo) & (per_seed.noise_std == 0) & (per_seed.packet_error_rate == 0)].sort_values("seed")
        deg = per_seed[(per_seed.algorithm == algo) & (per_seed.noise_std == worst[0]) & (per_seed.packet_error_rate == worst[1])].sort_values("seed")
        for m in PRIMARY:
            r = paired(deg[m].to_numpy(float), base[m].to_numpy(float))
            rows.append({"algorithm": algo, "metric": m,
                         "clean_mean": np.nanmean(base[m].to_numpy(float)),
                         "worst_mean": np.nanmean(deg[m].to_numpy(float)),
                         "abs_change": r["mean_diff"], "wilcoxon_p": r["wilcoxon_p"],
                         "cohens_dz": r["cohens_dz"], "effect_mag": mag_d(r["cohens_dz"]),
                         "cliffs_delta": r["cliffs_delta"]})
    df = pd.DataFrame(rows)
    # Holm-correct within each metric (4 algorithms)
    for m in PRIMARY:
        idx = df.index[df.metric == m]
        p = df.loc[idx, "wilcoxon_p"].to_numpy(float)
        df.loc[idx, "wilcoxon_p_holm"] = holm(p)
        df.loc[idx, "sig_holm"] = df.loc[idx, "wilcoxon_p_holm"] < ALPHA
    df.to_csv(os.path.join(STAT, "robustness_worst_vs_clean.csv"), index=False)
    return df


# ---------- 5. algorithm posthoc at the harshest condition ----------
def posthoc_harsh():
    ns, per = 0.25, 0.4
    sub = per_seed[(per_seed.noise_std == ns) & (per_seed.packet_error_rate == per)]
    rows = []
    for m in PRIMARY:
        recs = []
        for a, b in combinations(ALGOS, 2):
            xa = sub[sub.algorithm == a].sort_values("seed")[m].to_numpy(float)
            xb = sub[sub.algorithm == b].sort_values("seed")[m].to_numpy(float)
            r = paired(xa, xb)
            recs.append({"metric": m, "algo_a": a, "algo_b": b, **r})
        wp = [r["wilcoxon_p"] for r in recs]
        h = holm(wp); b_ = bh(wp)
        for i, r in enumerate(recs):
            r["wilcoxon_p_holm"] = h[i]; r["wilcoxon_p_bh"] = b_[i]
            r["sig_holm"] = bool(h[i] < ALPHA); r["effect_mag"] = mag_d(r["cohens_dz"])
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(STAT, "posthoc_harsh_condition.csv"), index=False)


def main():
    frac = normality()
    levene_clean()
    fac = factorial()
    rob = robustness()
    posthoc_harsh()

    method = {
        "design": "Algorithm(4) x Noise(3) x PacketError(3) factorial, shared 6 seeds (paired/randomized-block).",
        "unit": "per-seed scalar (n=6 per cell).",
        "normality_fraction_passing_shapiro": {k: round(float(v), 3) for k, v in frac.items()},
        "omnibus": "three-way factorial ANOVA y ~ algorithm*noise*per + seed (Type-II SS, partial eta^2).",
        "robustness_definition": "worst condition (noise=0.25, PER=0.4) vs clean (0,0), paired Wilcoxon + Cohen's d_z, Holm-corrected within metric.",
        "posthoc": "pairwise algorithm comparison at the harshest condition; paired Wilcoxon (exact) + Holm + BH; Cohen's d_z + Cliff's delta.",
        "alpha": ALPHA,
    }
    json.dump(method, open(os.path.join(STAT, "methodology.json"), "w"), indent=2)

    print("=== normality (fraction of cells passing Shapiro) ===")
    for k, v in method["normality_fraction_passing_shapiro"].items():
        print(f"  {k:20s} {v:.2f}")
    print("\n=== factorial ANOVA — significant effects (p<0.05), primary metrics ===")
    sig = fac[fac.p_value < 0.05].sort_values(["metric", "partial_eta2"], ascending=[True, False])
    print(sig[["metric", "effect", "F", "p_value", "partial_eta2"]].round(4).to_string(index=False))
    print("\n=== robustness: worst-vs-clean (safety/precision), Holm-sig marked ===")
    show = rob[rob.metric.isin(["collision_rate", "collisions_per_ep", "iala_violations", "cte_mean", "docking_accuracy", "success_rate"])]
    print(show[["algorithm", "metric", "clean_mean", "worst_mean", "abs_change", "wilcoxon_p_holm", "cohens_dz", "effect_mag"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
