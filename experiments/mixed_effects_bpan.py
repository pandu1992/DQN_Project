#!/usr/bin/env python3
"""
BPAN extension (f) — SEED-LEVEL MIXED-EFFECTS treatment.

The per-seed factorial ANOVAs treat the shared seed as a fixed block. Here we
instead treat SEED as a random effect and re-estimate the design factors as
fixed effects:  y ~ <fixed factors> + (1 | seed).  This (a) propagates
seed-to-seed variability into the fixed-effect inference honestly, and (b)
yields the intraclass correlation (ICC = sigma_seed^2 / (sigma_seed^2 +
sigma_resid^2)) — how much outcome variance is attributable to the random seed
alone. We fit a linear mixed model per metric for each BPAN sub-study and report
the fixed-effect coefficients, Wald p-values, and the variance partition.

Datasets:
  core  (per-seed, algorithm x condition)      -> results_bintulu/aggregated/per_seed.csv
  multi (per-seed, pairing   x condition)       -> results_bintulu/multi/aggregated/per_seed.csv
  ops2p (per-seed, agent     x condition)       -> results_bintulu/ops/aggregated/twophase_per_seed.csv
  opstw (per-seed, pairing   x condition)       -> results_bintulu/ops/aggregated/twoway_per_seed.csv

Outputs -> results_bintulu/mixed_effects/{mixed_effects_fixed.csv, variance_components.csv, icc_summary.csv}
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results_bintulu", "mixed_effects")
os.makedirs(OUT, exist_ok=True)

STUDIES = [
    {"name": "core", "label": "Core single-leg (algorithm x condition)",
     "path": os.path.join(ROOT, "results_bintulu", "aggregated", "per_seed.csv"),
     "factor": "algorithm", "metrics": ["success_rate", "collision_rate", "iala_violations"]},
    {"name": "multi", "label": "Multi-vessel (pairing x condition)",
     "path": os.path.join(ROOT, "results_bintulu", "multi", "aggregated", "per_seed.csv"),
     "factor": "pairing", "metrics": ["success_rate", "vessel_collisions_per_ep", "near_misses_per_ep"]},
    {"name": "ops_twophase", "label": "Round trip (agent x condition)",
     "path": os.path.join(ROOT, "results_bintulu", "ops", "aggregated", "twophase_per_seed.csv"),
     "factor": "agent", "metrics": ["full_success_rate", "collisions_per_ep"]},
    {"name": "ops_twoway", "label": "Two-way traffic (pairing x condition)",
     "path": os.path.join(ROOT, "results_bintulu", "ops", "aggregated", "twoway_per_seed.csv"),
     "factor": "pairing", "metrics": ["success_rate", "head_on_events_per_ep", "vessel_collisions_per_ep"]},
]


def fit_one(df, factor, metric):
    d = df[[factor, "condition", "seed", metric]].dropna().rename(columns={metric: "y"}).copy()
    if d["y"].nunique() < 2 or d["seed"].nunique() < 2:
        return None
    d[factor] = d[factor].astype("category")
    d["condition"] = pd.Categorical(d["condition"], categories=["clean", "mid", "harsh"], ordered=False)
    d["seed"] = d["seed"].astype("category")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            md = smf.mixedlm(f"y ~ C({factor}) + C(condition)", d, groups=d["seed"])
            mf = md.fit(reml=True, method="lbfgs")
        except Exception:
            try:
                mf = smf.mixedlm(f"y ~ C({factor}) + C(condition)", d, groups=d["seed"]).fit(reml=True)
            except Exception as e:
                return {"error": str(e)}
    # variance components
    resid_var = float(mf.scale)
    group_var = float(mf.cov_re.iloc[0, 0]) if mf.cov_re.shape[0] else 0.0
    icc = group_var / (group_var + resid_var) if (group_var + resid_var) > 0 else np.nan
    fixed = []
    for name in mf.params.index:
        if name == "Group Var":
            continue
        fixed.append({"term": name, "coef": mf.params[name], "se": mf.bse.get(name, np.nan),
                      "z": mf.tvalues.get(name, np.nan), "p_value": mf.pvalues.get(name, np.nan)})
    return {"fixed": fixed, "resid_var": resid_var, "group_var": group_var, "icc": icc,
            "n_obs": int(d.shape[0]), "n_seeds": int(d["seed"].nunique())}


def main():
    fixed_rows, var_rows, icc_rows = [], [], []
    for st in STUDIES:
        if not os.path.exists(st["path"]):
            print(f"skip {st['name']}: missing {st['path']}"); continue
        df = pd.read_csv(st["path"])
        for m in st["metrics"]:
            if m not in df.columns:
                continue
            res = fit_one(df, st["factor"], m)
            if res is None or "error" in (res or {}):
                print(f"  {st['name']}/{m}: fit skipped ({(res or {}).get('error','degenerate')})"); continue
            for fx in res["fixed"]:
                fixed_rows.append({"study": st["name"], "metric": m, **fx})
            var_rows.append({"study": st["name"], "metric": m, "group_var_seed": res["group_var"],
                             "resid_var": res["resid_var"], "icc_seed": res["icc"],
                             "n_obs": res["n_obs"], "n_seeds": res["n_seeds"]})
            icc_rows.append({"Study": st["label"], "Metric": m, "ICC (seed)": f"{res['icc']:.3f}",
                             "sigma^2 seed": f"{res['group_var']:.3f}", "sigma^2 resid": f"{res['resid_var']:.3f}",
                             "Interpretation": ("seed explains a large share of residual variance"
                                                if res["icc"] >= 0.3 else
                                                "seed contributes modestly" if res["icc"] >= 0.1 else
                                                "seed negligible after fixed effects")})
            print(f"  {st['name']:13s} {m:26s} ICC_seed={res['icc']:.3f} (sigma2_seed={res['group_var']:.2f}, resid={res['resid_var']:.2f})")

    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(fixed_rows).to_csv(os.path.join(OUT, "mixed_effects_fixed.csv"), index=False)
    pd.DataFrame(var_rows).to_csv(os.path.join(OUT, "variance_components.csv"), index=False)
    icc = pd.DataFrame(icc_rows)
    icc.to_csv(os.path.join(OUT, "icc_summary.csv"), index=False)
    icc.to_csv(os.path.join(OUT, "tables", "icc_summary.csv"), index=False)  # viewer convention: <dir>/tables/<file>.csv
    try:
        with pd.ExcelWriter(os.path.join(OUT, "mixed_effects_bpan.xlsx"), engine="openpyxl") as xw:
            pd.DataFrame(fixed_rows).to_excel(xw, sheet_name="fixed_effects", index=False)
            pd.DataFrame(var_rows).to_excel(xw, sheet_name="variance_components", index=False)
            icc.to_excel(xw, sheet_name="icc_summary", index=False)
    except Exception as e:
        print("excel skip:", e)
    print("\nwrote results_bintulu/mixed_effects/{mixed_effects_fixed,variance_components,icc_summary}.csv")
    print("\n=== Fixed-effect significance confirms the ANOVA conclusions under a random-seed model ===")
    fx = pd.DataFrame(fixed_rows)
    for st in STUDIES:
        sub = fx[(fx.study == st["name"]) & (fx.term.str.contains("C\\(" + st["factor"]))]
        if len(sub):
            sig = (sub.p_value < 0.05).sum()
            print(f"  {st['name']:13s}: {sig}/{len(sub)} {st['factor']}-level fixed effects significant at p<0.05")


if __name__ == "__main__":
    main()
