#!/usr/bin/env python3
"""
AGGREGATION (task #4)

Reduces results/raw/eval_episodes.csv to:
  1. per-config per-SEED scalars  -> results/aggregated/per_seed.csv
     (each row = one independent sample; n = 10 seeds per config)
  2. across-seed summary          -> results/aggregated/summary.csv
     (mean, SD, SEM, 95% t-CI, and BCa/percentile bootstrap CI)

The per-seed scalar is the correct unit of analysis: episodes within a
seed are NOT independent (shared network), so we first collapse the 40
evaluation episodes of each seed into one scalar per metric, then treat
the 10 seed-scalars as the independent observations.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results", "raw", "eval_episodes.csv")
AGG = os.path.join(ROOT, "results", "aggregated")
os.makedirs(AGG, exist_ok=True)

RNG = np.random.default_rng(12345)  # fixed seed for reproducible bootstrap

# metrics computed per seed (one scalar per seed from its 40 eval episodes)
def per_seed_scalars(g: pd.DataFrame) -> dict:
    solved = g[g["success"] == 1]
    return {
        "success_rate": 100.0 * g["success"].mean(),          # %
        "mean_reward": g["reward"].mean(),
        "mean_steps": g["steps"].mean(),
        "invalid_rate": g["invalid"].mean(),                    # mean invalid actions/episode
        "revisit_rate": g["revisit"].mean(),
        # route-quality metrics defined only over SOLVED episodes
        "optimality_ratio": solved["optimality_ratio"].mean() if len(solved) else np.nan,
        "excess_cost": solved["excess_cost"].mean() if len(solved) else np.nan,
        "route_risk": solved["route_risk"].mean() if len(solved) else np.nan,
        "route_difficulty": solved["route_difficulty"].mean() if len(solved) else np.nan,
        "n_solved": int(len(solved)),
        "n_eval": int(len(g)),
    }


def bootstrap_ci(x, n_boot=10000, alpha=0.05):
    """Percentile bootstrap CI for the mean. Returns (lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    boot = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        boot[i] = np.mean(x[RNG.integers(0, n, n)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return (lo, hi)


def t_ci(x, alpha=0.05):
    """Student-t CI half-width and bounds for the mean."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    sd = x.std(ddof=1)
    sem = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    hw = tcrit * sem
    return (m - hw, m + hw, hw)


def main():
    df = pd.read_csv(RAW)
    df = df[df["phase"] == "eval"].copy()

    # ---- 1) per-seed scalars ----
    rows = []
    metrics = ["success_rate", "mean_reward", "mean_steps", "invalid_rate",
               "revisit_rate", "optimality_ratio", "excess_cost",
               "route_risk", "route_difficulty"]
    for (cfg, algo, per, noisy, seed), g in df.groupby(
        ["config_id", "algorithm", "per", "noisy", "seed"], sort=True
    ):
        sc = per_seed_scalars(g)
        rows.append({
            "config_id": cfg, "algorithm": algo, "per": int(per),
            "noisy": int(noisy), "seed": int(seed), **sc,
        })
    per_seed = pd.DataFrame(rows).sort_values(["algorithm", "per", "noisy", "seed"])
    per_seed.to_csv(os.path.join(AGG, "per_seed.csv"), index=False)
    print(f"per_seed.csv: {len(per_seed)} rows (should be 160)")

    # ---- 2) across-seed summary with t-CI and bootstrap CI ----
    srows = []
    for (cfg, algo, per, noisy), g in per_seed.groupby(
        ["config_id", "algorithm", "per", "noisy"], sort=True
    ):
        entry = {"config_id": cfg, "algorithm": algo, "per": int(per),
                 "noisy": int(noisy), "n_seeds": len(g)}
        for m in metrics:
            x = g[m].to_numpy(dtype=float)
            xv = x[~np.isnan(x)]
            mean = np.nanmean(x) if len(xv) else np.nan
            sd = np.nanstd(x, ddof=1) if len(xv) > 1 else 0.0
            sem = sd / np.sqrt(len(xv)) if len(xv) > 1 else 0.0
            tlo, thi, thw = t_ci(x)
            blo, bhi = bootstrap_ci(x)
            entry[f"{m}_mean"] = mean
            entry[f"{m}_sd"] = sd
            entry[f"{m}_sem"] = sem
            entry[f"{m}_ci95_lo"] = tlo
            entry[f"{m}_ci95_hi"] = thi
            entry[f"{m}_ci95_hw"] = thw
            entry[f"{m}_boot_lo"] = blo
            entry[f"{m}_boot_hi"] = bhi
        srows.append(entry)
    summary = pd.DataFrame(srows).sort_values(["algorithm", "per", "noisy"])
    summary.to_csv(os.path.join(AGG, "summary.csv"), index=False)
    print(f"summary.csv: {len(summary)} rows (should be 16)")

    # ---- 3) marginal (main-effect) aggregates for quick reference ----
    marg = {}
    for factor in ["algorithm", "per", "noisy"]:
        gb = per_seed.groupby(factor)
        rows_f = []
        for key, sub in gb:
            rec = {factor: key}
            for m in metrics:
                rec[f"{m}_mean"] = round(float(sub[m].mean()), 4)
                rec[f"{m}_sd"] = round(float(sub[m].std(ddof=1)), 4)
            rows_f.append(rec)
        marg[factor] = rows_f
    with open(os.path.join(AGG, "marginals.json"), "w") as f:
        json.dump(marg, f, indent=2, default=str)
    print("marginals.json written")

    # console preview: primary metrics per config, mean +/- t-CI hw
    print("\n=== eval success rate: mean +/- 95% t-CI (n=10 seeds) ===")
    for _, r in summary.sort_values("success_rate_mean", ascending=False).iterrows():
        print(f"  {r['config_id']:<28} {r['success_rate_mean']:5.1f} +/- {r['success_rate_ci95_hw']:4.1f} %   "
              f"reward {r['mean_reward_mean']:7.1f} +/- {r['mean_reward_ci95_hw']:5.1f}   "
              f"optR {r['optimality_ratio_mean']:.3f}")


if __name__ == "__main__":
    main()
