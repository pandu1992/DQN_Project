#!/usr/bin/env python3
"""
Mixed-effects inference (Q1 audit Task 5).

Complements the OLS factorial ANOVAs (which treat seed as a fixed blocking
factor) with a linear MIXED-EFFECTS model that treats SEED as a RANDOM effect
crossed with the design, i.e.

    y ~ <fixed design factors>  +  (1 | seed)

fit per study on the per-seed scalars. This is the more defensible inferential
model for repeated / shared-seed experimental data. We report:
  * the fixed-effect Wald tests for the design factors,
  * the SEED variance component vs the residual variance and the intraclass
    correlation (ICC = var_seed / (var_seed + var_resid)) — how much run-to-run
    variance the seed absorbs,
so the manuscript can (i) confirm the ANOVA conclusions under a random-seed
model and (ii) show that inference rests on seed-level replication (n = seeds),
not on the ~31k episodes.

Outputs -> results_audit/mixed_effects_summary.csv + a human note.
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results_audit")
os.makedirs(OUT, exist_ok=True)

# (label, per_seed_csv, fixed-effects formula RHS, n_seeds note)
STUDIES = [
    ("Study 2 (robustness, n=6)", "results_v3/aggregated/per_seed.csv",
     "C(algorithm) + noise_std + packet_error_rate", "success_rate", 6),
    ("Study 3 (rule-based + confirmatory, n=15)", "results_study3/aggregated/per_seed.csv",
     "C(agent) + C(condition)", "success_rate", 15),
    ("Study 4 (two vessels, n=8)", "results_study4/aggregated/per_seed.csv",
     "C(pairing) + C(condition)", "success_rate", 8),
    ("Chart fairness (n=8)", "results_chart/aggregated/per_seed.csv",
     "C(arm) + C(condition)", "success_rate", 8),
    ("External validity maps (n=6)", "results_maps/aggregated/per_seed.csv",
     "C(arm) + C(condition) + C(map_id)", "success_rate", 6),
]

rows = []
note = ["# Mixed-effects inference summary (Q1 audit, Task 5)\n",
        "Linear mixed model `y ~ <fixed design factors> + (1|seed)`, seed as a",
        "RANDOM effect, fit on per-seed scalars. ICC = var(seed)/(var(seed)+var(resid))",
        "is the share of variance absorbed by the random seed — a direct measure of",
        "run-to-run variability and a reminder that inference is seed-level (n=seeds),",
        "not episode-level (~31k episodes = reproducibility, not statistical power).\n"]


def main():
    for label, csv, rhs, dv, nseed in STUDIES:
        path = os.path.join(ROOT, csv)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df.dropna(subset=[dv]).copy()
        df["seed"] = df["seed"].astype(int)
        # REML can hit a singular matrix at small n with all-categorical
        # designs; fall back to ML and alternative optimizers. Record which fit.
        mf, fit_kind = None, None
        for reml, method in [(True, ["lbfgs"]), (False, ["lbfgs"]),
                             (False, ["powell"]), (False, ["cg"]), (False, ["nm", "lbfgs"])]:
            try:
                mf = smf.mixedlm(f"{dv} ~ {rhs}", df, groups=df["seed"]).fit(reml=reml, method=method)
                fit_kind = f"{'REML' if reml else 'ML'}/{'+'.join(method)}"
                break
            except Exception:
                continue
        if mf is None:
            note.append(f"\n## {label}\n- model failed to converge under all optimizers.")
            continue
        var_seed = float(mf.cov_re.iloc[0, 0]) if mf.cov_re.size else 0.0
        var_resid = float(mf.scale)
        icc = var_seed / (var_seed + var_resid) if (var_seed + var_resid) > 0 else np.nan
        # collect fixed effects (drop intercept)
        fe = mf.fe_params
        pvals = mf.pvalues
        sig_terms = []
        for term in fe.index:
            if term == "Intercept":
                continue
            p = pvals.get(term, np.nan)
            rows.append({"study": label, "dv": dv, "term": term,
                         "coef": round(float(fe[term]), 3), "p_value": round(float(p), 4),
                         "sig_0.05": bool(np.isfinite(p) and p < 0.05)})
            if np.isfinite(p) and p < 0.05:
                sig_terms.append(f"{term} (p={p:.3f})")
        rows.append({"study": label, "dv": dv, "term": "__variance__",
                     "coef": round(var_seed, 3), "p_value": round(var_resid, 3), "sig_0.05": ""})
        note.append(f"\n## {label}")
        note.append(f"- DV = {dv}; fixed = `{rhs}`; seeds n={nseed}; fit={fit_kind}.")
        note.append(f"- variance: seed={var_seed:.2f}, residual={var_resid:.2f}, **ICC={icc:.2f}**.")
        note.append(f"- significant fixed effects (p<0.05): {', '.join(sig_terms) if sig_terms else 'none'}.")
        print(f"{label}: ICC={icc:.2f} (seed var {var_seed:.1f} / resid {var_resid:.1f}); "
              f"sig FE: {sig_terms if sig_terms else 'none'}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mixed_effects_summary.csv"), index=False)
    with open(os.path.join(OUT, "mixed_effects_findings.md"), "w") as f:
        f.write("\n".join(note) + "\n")
    print("\n-> results_audit/mixed_effects_summary.csv + mixed_effects_findings.md")


if __name__ == "__main__":
    main()
