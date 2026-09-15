#!/usr/bin/env python3
"""
STATISTICAL METHODOLOGY — assumption checks + omnibus tests (task #5)

Design recap:
  - Unit of analysis = per-seed scalar (n=10 per config), shared seeds 0-9
    across all configs => a REPEATED-MEASURES / randomized-block design
    (seed = block). This justifies paired/repeated-measures tests.

Pipeline (per metric):
  1. Normality of residuals per group: Shapiro-Wilk on each config's 10 seeds.
  2. Variance homogeneity across the 4 algorithms (baseline, per=0,noisy=0):
     Levene (median-centered = Brown-Forsythe, robust).
  3. Omnibus over the 4 base algorithms (per=0,noisy=0), n=10 each:
       - parametric: one-way repeated-measures ANOVA (seed = block)
       - non-parametric: Friedman test (repeated measures)
     We report BOTH and select based on assumption checks.
  4. Factorial main effects + interactions on the full 4x2x2 design:
       - three-way repeated-measures ANOVA (within-subject factor = none;
         seed is the blocking/subject factor, all factors within-seed)
         via a mixed/OLS model with eta^2 / partial eta^2.
       - because RM-ANOVA with all-within factors == OLS on seed-demeaned
         data is fragile, we fit OLS with seed as a categorical block
         (type-II ANOVA) and report partial eta^2; assumptions noted.

Outputs:
  results/statistics/normality.csv
  results/statistics/variance_homogeneity.csv
  results/statistics/omnibus_algorithms.csv
  results/statistics/factorial_anova.csv
  results/statistics/methodology.json  (decisions + justification)
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results", "aggregated")
STATDIR = os.path.join(ROOT, "results", "statistics")
os.makedirs(STATDIR, exist_ok=True)

PER_SEED = pd.read_csv(os.path.join(AGG, "per_seed.csv"))

PRIMARY = ["success_rate", "mean_reward", "optimality_ratio"]
# invalid_rate is uniformly 0.0 under the greedy evaluation policy (the trained
# nets never select the non-existent 4th action slot), so it is a degenerate,
# zero-variance metric and is EXCLUDED from statistical tests (documented in
# methodology.json). revisit_rate carries real variance and is analysed instead.
SECONDARY = ["mean_steps", "revisit_rate", "excess_cost", "route_risk", "route_difficulty"]
METRICS = PRIMARY + SECONDARY
ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]


def shapiro_all():
    rows = []
    for cfg, g in PER_SEED.groupby("config_id"):
        for m in METRICS:
            x = g[m].dropna().to_numpy(float)
            if len(x) >= 3 and np.ptp(x) > 0:
                W, p = stats.shapiro(x)
            else:
                W, p = np.nan, np.nan
            rows.append({"config_id": cfg, "metric": m, "n": len(x),
                         "shapiro_W": W, "p_value": p,
                         "normal_at_0.05": (p > 0.05) if np.isfinite(p) else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATDIR, "normality.csv"), index=False)
    # summarise: fraction of groups normal per metric
    summ = df.dropna(subset=["p_value"]).groupby("metric")["normal_at_0.05"].mean()
    return df, summ.to_dict()


def levene_baseline():
    """Variance homogeneity across the 4 base algorithms (per=0,noisy=0)."""
    rows = []
    base = PER_SEED[(PER_SEED.per == 0) & (PER_SEED.noisy == 0)]
    for m in METRICS:
        samples = [base[base.algorithm == a][m].dropna().to_numpy(float) for a in ALGOS]
        samples = [s for s in samples if len(s) > 1]
        if len(samples) >= 2:
            W, p = stats.levene(*samples, center="median")  # Brown-Forsythe
        else:
            W, p = np.nan, np.nan
        rows.append({"metric": m, "levene_BF_W": W, "p_value": p,
                     "homoscedastic_at_0.05": (p > 0.05) if np.isfinite(p) else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATDIR, "variance_homogeneity.csv"), index=False)
    return df


def rm_anova_oneway(metric, subset):
    """One-way repeated-measures ANOVA over algorithm, block = seed.
    subset: DataFrame filtered to per=0,noisy=0 (4 algos x 10 seeds).
    Returns F, p, partial_eta2, df_between, df_error."""
    piv = subset.pivot_table(index="seed", columns="algorithm", values=metric)
    piv = piv.dropna(axis=0, how="any")
    if piv.shape[1] < 2 or piv.shape[0] < 2:
        return dict(F=np.nan, p=np.nan, eta2_partial=np.nan, df_b=np.nan, df_e=np.nan, n_blocks=piv.shape[0])
    data = piv.to_numpy(float)  # rows=seeds(blocks), cols=algorithms(treatments)
    n, k = data.shape
    grand = data.mean()
    row_means = data.mean(axis=1, keepdims=True)   # per seed
    col_means = data.mean(axis=0, keepdims=True)    # per algorithm
    SS_treat = n * np.sum((col_means - grand) ** 2)
    SS_block = k * np.sum((row_means - grand) ** 2)
    SS_total = np.sum((data - grand) ** 2)
    SS_error = SS_total - SS_treat - SS_block
    df_treat = k - 1
    df_error = (k - 1) * (n - 1)
    MS_treat = SS_treat / df_treat
    MS_error = SS_error / df_error if df_error > 0 else np.nan
    F = MS_treat / MS_error if MS_error and MS_error > 0 else np.nan
    p = stats.f.sf(F, df_treat, df_error) if np.isfinite(F) else np.nan
    eta2_partial = SS_treat / (SS_treat + SS_error) if (SS_treat + SS_error) > 0 else np.nan
    return dict(F=F, p=p, eta2_partial=eta2_partial, df_b=df_treat, df_e=df_error, n_blocks=n)


def friedman_oneway(metric, subset):
    piv = subset.pivot_table(index="seed", columns="algorithm", values=metric).dropna(axis=0, how="any")
    if piv.shape[1] < 3 or piv.shape[0] < 2:
        return dict(chi2=np.nan, p=np.nan, kendall_W=np.nan, n_blocks=piv.shape[0])
    cols = [piv[c].to_numpy(float) for c in piv.columns]
    chi2, p = stats.friedmanchisquare(*cols)
    n, k = piv.shape
    kendall_W = chi2 / (n * (k - 1))  # Kendall's W effect size
    return dict(chi2=chi2, p=p, kendall_W=kendall_W, n_blocks=n)


def omnibus_algorithms():
    base = PER_SEED[(PER_SEED.per == 0) & (PER_SEED.noisy == 0)]
    rows = []
    for m in METRICS:
        rm = rm_anova_oneway(m, base)
        fr = friedman_oneway(m, base)
        rows.append({
            "metric": m,
            "rm_anova_F": rm["F"], "rm_anova_df_between": rm["df_b"], "rm_anova_df_error": rm["df_e"],
            "rm_anova_p": rm["p"], "rm_anova_partial_eta2": rm["eta2_partial"],
            "friedman_chi2": fr["chi2"], "friedman_df": 3, "friedman_p": fr["p"], "kendall_W": fr["kendall_W"],
            "n_blocks": rm["n_blocks"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATDIR, "omnibus_algorithms.csv"), index=False)
    return df


def factorial_anova():
    """Full 4x2x2 factorial via OLS with seed as a categorical block.
    Type-II sums of squares; partial eta^2 per effect. Uses statsmodels if
    available, else a manual balanced-design ANOVA (design IS balanced:
    every cell has exactly 10 seeds)."""
    rows = []
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        have_sm = True
    except Exception:
        have_sm = False

    for m in METRICS:
        d = PER_SEED[["algorithm", "per", "noisy", "seed", m]].dropna().copy()
        d = d.rename(columns={m: "y"})
        if have_sm:
            d["algorithm"] = d["algorithm"].astype("category")
            d["per"] = d["per"].astype("category")
            d["noisy"] = d["noisy"].astype("category")
            d["seed"] = d["seed"].astype("category")
            formula = "y ~ C(algorithm)*C(per)*C(noisy) + C(seed)"
            model = ols(formula, data=d).fit()
            aov = anova_lm(model, typ=2)
            ss_resid = aov.loc["Residual", "sum_sq"]
            for eff in aov.index:
                if eff == "Residual":
                    continue
                ss = aov.loc[eff, "sum_sq"]
                F = aov.loc[eff, "F"]
                p = aov.loc[eff, "PR(>F)"]
                pe2 = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else np.nan
                rows.append({"metric": m, "effect": eff.replace("C(", "").replace(")", ""),
                             "sum_sq": ss, "F": F, "p_value": p, "partial_eta2": pe2})
        else:
            rows.append({"metric": m, "effect": "NA(statsmodels missing)",
                         "sum_sq": np.nan, "F": np.nan, "p_value": np.nan, "partial_eta2": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATDIR, "factorial_anova.csv"), index=False)
    return df, have_sm


def main():
    norm_df, norm_summ = shapiro_all()
    lev_df = levene_baseline()
    omni_df = omnibus_algorithms()
    fac_df, have_sm = factorial_anova()

    # methodology decisions
    frac_normal = {k: round(float(v), 3) for k, v in norm_summ.items()}
    methodology = {
        "design": "Randomized-block / repeated-measures: seed (0-9) is the block; the SAME 10 seeds appear in every configuration, so factor-level comparisons are within-block (paired).",
        "unit_of_analysis": "per-seed scalar (n=10 per configuration); evaluation episodes within a seed are not independent and are collapsed first.",
        "normality_fraction_groups_passing_shapiro_0.05": frac_normal,
        "excluded_metrics": {"invalid_rate": "uniformly 0.0 under greedy evaluation (zero variance); not testable and excluded from all statistics."},
        "omnibus_choice": "Report BOTH one-way repeated-measures ANOVA and Friedman for the 4 base algorithms. Prefer Friedman where Shapiro-Wilk indicates non-normality for that metric; prefer RM-ANOVA where normality holds. Effect sizes: partial eta^2 (ANOVA), Kendall's W (Friedman).",
        "factorial_model": ("OLS y ~ algorithm*per*noisy + seed (Type-II ANOVA), balanced design (10 per cell); "
                            "partial eta^2 per effect. statsmodels_available=" + str(have_sm)),
        "correction_note": "Pairwise post-hoc and multiple-comparison correction handled in the next stage (Holm + Benjamini-Hochberg).",
        "alpha": 0.05,
    }
    with open(os.path.join(STATDIR, "methodology.json"), "w") as f:
        json.dump(methodology, f, indent=2)

    print("=== normality: fraction of groups passing Shapiro (p>0.05) per metric ===")
    for k, v in frac_normal.items():
        print(f"  {k:<18} {v:.2f}")
    print("\n=== Levene (Brown-Forsythe) baseline homoscedasticity ===")
    print(lev_df[["metric", "levene_BF_W", "p_value", "homoscedastic_at_0.05"]].to_string(index=False))
    print("\n=== omnibus over 4 base algorithms (per=0,noisy=0) ===")
    print(omni_df[["metric", "rm_anova_F", "rm_anova_p", "rm_anova_partial_eta2",
                   "friedman_chi2", "friedman_p", "kendall_W"]].round(4).to_string(index=False))
    print("\n=== factorial ANOVA (main effects + interactions) — primary metrics ===")
    print(fac_df[fac_df.metric.isin(PRIMARY)][["metric", "effect", "F", "p_value", "partial_eta2"]].round(4).to_string(index=False))
    print(f"\nstatsmodels available: {have_sm}")


if __name__ == "__main__":
    main()
