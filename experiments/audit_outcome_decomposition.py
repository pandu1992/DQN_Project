#!/usr/bin/env python3
"""
Q1 AUDIT FIX (Task 2) — outcome-decomposed metrics + failure taxonomy.

Addresses validity-audit findings B (CTE is a selection/composition artifact)
and C (docking/optimality are solved-only survivorship). Rebuilds the affected
metrics from the EXISTING raw CSVs (no re-run needed) split by episode outcome,
and documents the failure taxonomy.

Key facts established by the audit and re-verified here:
  * In these environments endOnCollision=false and there is no off-channel
    termination, so `terminated==1` iff the goal was reached (success) and
    `truncated==1` iff the episode timed out (the ONLY failure mode).
    Collisions are a within-episode safety metric, orthogonal to success.
  * CTE lateral drift scales with degradation (0 when clean), so a *successful*
    clean episode has CTE=0 by construction; the all-episode CTE mean is a
    success-rate-weighted mixture of a ~0 successful sub-population and a large
    failed sub-population -> must never be reported as a single aggregate.

Outputs (per study dir):
  <dir>/aggregated/outcome_decomposition_per_seed.csv
  <dir>/aggregated/outcome_decomposition_summary.csv
  <dir>/aggregated/failure_taxonomy.csv
and a combined human-readable note: results_audit/outcome_decomposition_findings.md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RNG = np.random.default_rng(12345)

# (dir, raw_csv_relative, group_cols) — single-vessel V3 studies with cte_mean+collisions
STUDIES = [
    ("results_v3", "raw/eval_episodes.csv", ["algorithm", "noise_std", "packet_error_rate", "seed"],
     ["algorithm", "noise_std", "packet_error_rate"], "Study 2 (robustness)"),
    ("results_study3", "raw/eval_episodes.csv", ["agent", "condition", "seed"],
     ["agent", "condition"], "Study 3 (rule-based + confirmatory)"),
]


def num(c, fr):
    return pd.to_numeric(fr[c], errors="coerce")


def t_hw(x, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    return stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)


def per_seed_outcome(g):
    """Decompose CTE + collision incidence by episode outcome for one seed-cell."""
    succ = g[g["success"] == 1]
    fail = g[g["success"] == 0]
    def cte(fr):
        return num("cte_mean", fr).mean() if len(fr) else np.nan
    return {
        "success_rate": 100.0 * g["success"].mean(),
        # --- CTE decomposed (audit B) ---
        "cte_all": cte(g),
        "cte_success": cte(succ),
        "cte_fail": cte(fail),
        # --- collision incidence decomposed (audit D) ---
        "collision_rate_all": 100.0 * (num("collisions", g) > 0).mean(),
        "collision_rate_success": (100.0 * (num("collisions", succ) > 0).mean()) if len(succ) else np.nan,
        "collision_rate_fail": (100.0 * (num("collisions", fail) > 0).mean()) if len(fail) else np.nan,
        "collisions_per_ep_all": num("collisions", g).mean(),
        # --- failure taxonomy (audit D): here the only failure mode is timeout ---
        "n_eval": int(len(g)),
        "n_success": int(len(succ)),
        "n_fail_timeout": int((g["truncated"] == 1).sum()),
        "n_fail_other": int(((g["success"] == 0) & (g["truncated"] != 1)).sum()),
    }


def summarise(ps, metrics, gcols):
    rows = []
    for key, gg in ps.groupby(gcols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        e = dict(zip(gcols, key))
        e["n_seeds"] = len(gg)
        for m in metrics:
            x = gg[m].to_numpy(float); xv = x[~np.isnan(x)]
            e[f"{m}_mean"] = np.nanmean(x) if len(xv) else np.nan
            e[f"{m}_sd"] = np.nanstd(x, ddof=1) if len(xv) > 1 else 0.0
            e[f"{m}_ci95_hw"] = t_hw(x)
        rows.append(e)
    return pd.DataFrame(rows)


DECOMP_METRICS = ["success_rate", "cte_all", "cte_success", "cte_fail",
                  "collision_rate_all", "collision_rate_success", "collision_rate_fail",
                  "collisions_per_ep_all"]

findings = ["# Outcome-decomposition findings (audit fix, Task 2)\n",
            "CTE and collision incidence, split by episode outcome, to remove the",
            "selection/composition artifact identified in the validity audit.",
            "In every study `endOnCollision=false` and there is no off-channel",
            "termination, so the **only failure mode is timeout** (`truncated==1`);",
            "collisions are a within-episode safety metric orthogonal to success.\n"]


def main():
    for dir_, raw_rel, gcols, sumcols, label in STUDIES:
        raw = os.path.join(ROOT, dir_, raw_rel)
        if not os.path.exists(raw):
            print(f"skip {dir_}: no raw csv"); continue
        agg = os.path.join(ROOT, dir_, "aggregated")
        os.makedirs(agg, exist_ok=True)
        df = pd.read_csv(raw)
        df = df[df["phase"] == "eval"].copy()

        rows = []
        for key, g in df.groupby(gcols, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            rows.append({**dict(zip(gcols, key)), **per_seed_outcome(g)})
        ps = pd.DataFrame(rows)
        ps.to_csv(os.path.join(agg, "outcome_decomposition_per_seed.csv"), index=False)

        summ = summarise(ps, DECOMP_METRICS, sumcols)
        summ.to_csv(os.path.join(agg, "outcome_decomposition_summary.csv"), index=False)

        # failure taxonomy at the study level
        tax = (ps.groupby(sumcols)[["n_eval", "n_success", "n_fail_timeout", "n_fail_other"]]
               .sum().reset_index())
        tax["pct_success"] = 100.0 * tax["n_success"] / tax["n_eval"]
        tax["pct_timeout"] = 100.0 * tax["n_fail_timeout"] / tax["n_eval"]
        tax["pct_other_failure"] = 100.0 * tax["n_fail_other"] / tax["n_eval"]
        tax.to_csv(os.path.join(agg, "failure_taxonomy.csv"), index=False)

        print(f"\n=== {label} ({dir_}) ===")
        print(f"  outcome_decomposition_per_seed.csv: {len(ps)} rows")
        print(f"  outcome_decomposition_summary.csv:  {len(summ)} rows")
        print(f"  failure_taxonomy.csv:               {len(tax)} rows")
        # show the headline CTE decomposition for the extreme conditions
        findings.append(f"\n## {label}\n")
        findings.append(f"- other (non-timeout) failures across all cells: "
                        f"{int(tax['n_fail_other'].sum())} (expected 0 — confirms timeout is the sole failure mode).")
        # print a compact view
        show_metrics = ["success_rate_mean", "cte_all_mean", "cte_success_mean", "cte_fail_mean",
                        "collision_rate_all_mean", "collision_rate_success_mean"]
        cols = sumcols + [c for c in show_metrics if c in summ.columns]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summ[cols].round(2).to_string(index=False))
        # capture a couple of concrete numbers into findings for the manuscript
        for _, r in summ.iterrows():
            tagvals = " ".join(str(r[c]) for c in sumcols)
            if ("clean" in tagvals) or ("0.0 0.0" in tagvals) or ("harsh" in tagvals) or ("0.25 0.4" in tagvals):
                findings.append(
                    f"- `{tagvals}`: success {r['success_rate_mean']:.1f}%, "
                    f"CTE all={r['cte_all_mean']:.2f} / success={r['cte_success_mean']:.2f} / fail={r['cte_fail_mean']:.2f}; "
                    f"collision-rate all={r['collision_rate_all_mean']:.1f}% / on-success={r['collision_rate_success_mean']:.1f}%.")

    os.makedirs(os.path.join(ROOT, "results_audit"), exist_ok=True)
    with open(os.path.join(ROOT, "results_audit", "outcome_decomposition_findings.md"), "w") as f:
        f.write("\n".join(findings) + "\n")
    print("\nfindings -> results_audit/outcome_decomposition_findings.md")


if __name__ == "__main__":
    main()
