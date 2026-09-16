#!/usr/bin/env python3
"""
V3 ROBUSTNESS — AGGREGATION

Reduces results_v3/raw/eval_episodes.csv to:
  1. per-seed scalars (one row per algorithm x noise x PER x seed;
     the independent unit of analysis, n=6 per cell)
     -> results_v3/aggregated/per_seed.csv
  2. across-seed summary (mean, SD, SEM, 95% t-CI, percentile bootstrap CI)
     -> results_v3/aggregated/summary.csv
  3. degradation table: clean (noise=0,PER=0) vs each degraded condition,
     per algorithm, for the headline metrics (absolute + relative + variance)
     -> results_v3/aggregated/degradation.csv

Metrics:
  navigation : success_rate (%), mean_reward
  safety     : collision_rate (% eps with >=1 collision), collisions_per_ep,
               iala_violations (mean per ep)
  precision  : docking_accuracy (mean over SOLVED eps), cte_mean
  routing    : optimality_ratio (over SOLVED eps), mean_steps
  comms      : dropped_frames (mean per ep)
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results_v3", "raw", "eval_episodes.csv")
AGG = os.path.join(ROOT, "results_v3", "aggregated")
os.makedirs(AGG, exist_ok=True)
RNG = np.random.default_rng(12345)

ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]

# metric list: (name, needs_solved_only)
METRICS = [
    ("success_rate", False),
    ("mean_reward", False),
    ("collision_rate", False),
    ("collisions_per_ep", False),
    ("iala_violations", False),
    ("cte_mean", False),
    ("dropped_frames", False),
    ("docking_accuracy", True),
    ("optimality_ratio", True),
    ("mean_steps", False),
]


def per_seed_scalars(g):
    solved = g[g["success"] == 1]
    def num(col, frame):
        return pd.to_numeric(frame[col], errors="coerce")
    return {
        "success_rate": 100.0 * g["success"].mean(),
        "mean_reward": num("reward", g).mean(),
        "collision_rate": 100.0 * (g["collisions"] > 0).mean(),
        "collisions_per_ep": num("collisions", g).mean(),
        "iala_violations": num("iala_violations", g).mean(),
        "cte_mean": num("cte_mean", g).mean(),
        "dropped_frames": num("dropped_frames", g).mean(),
        "docking_accuracy": num("docking_accuracy", solved).mean() if len(solved) else np.nan,
        "optimality_ratio": num("optimality_ratio", solved).mean() if len(solved) else np.nan,
        "mean_steps": num("steps", g).mean(),
        "n_solved": int(len(solved)),
        "n_eval": int(len(g)),
    }


def boot_ci(x, n=10000, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    b = np.empty(n)
    for i in range(n):
        b[i] = np.mean(x[RNG.integers(0, len(x), len(x))])
    return (np.percentile(b, 100 * alpha / 2), np.percentile(b, 100 * (1 - alpha / 2)))


def t_ci(x, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (np.nan, np.nan, np.nan)
    m, sd = x.mean(), x.std(ddof=1)
    hw = stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n)
    return (m - hw, m + hw, hw)


def main():
    df = pd.read_csv(RAW)
    df = df[df["phase"] == "eval"].copy()

    # ---- per-seed scalars ----
    rows = []
    for (algo, ns, per, seed), g in df.groupby(["algorithm", "noise_std", "packet_error_rate", "seed"], sort=True):
        sc = per_seed_scalars(g)
        rows.append({"algorithm": algo, "noise_std": ns, "packet_error_rate": per, "seed": int(seed),
                     "condition": f"n{ns}_p{per}", **sc})
    per_seed = pd.DataFrame(rows).sort_values(["algorithm", "noise_std", "packet_error_rate", "seed"])
    per_seed.to_csv(os.path.join(AGG, "per_seed.csv"), index=False)
    print(f"per_seed.csv: {len(per_seed)} rows (expect 216)")

    metric_names = [m for m, _ in METRICS]

    # ---- across-seed summary ----
    srows = []
    for (algo, ns, per), g in per_seed.groupby(["algorithm", "noise_std", "packet_error_rate"], sort=True):
        e = {"algorithm": algo, "noise_std": ns, "packet_error_rate": per,
             "condition": f"n{ns}_p{per}", "n_seeds": len(g)}
        for m in metric_names:
            x = g[m].to_numpy(float)
            xv = x[~np.isnan(x)]
            e[f"{m}_mean"] = np.nanmean(x) if len(xv) else np.nan
            e[f"{m}_sd"] = np.nanstd(x, ddof=1) if len(xv) > 1 else 0.0
            e[f"{m}_sem"] = (np.nanstd(x, ddof=1) / np.sqrt(len(xv))) if len(xv) > 1 else 0.0
            lo, hi, hw = t_ci(x)
            e[f"{m}_ci95_hw"] = hw
            blo, bhi = boot_ci(x)
            e[f"{m}_boot_lo"] = blo; e[f"{m}_boot_hi"] = bhi
        srows.append(e)
    summary = pd.DataFrame(srows).sort_values(["algorithm", "noise_std", "packet_error_rate"])
    summary.to_csv(os.path.join(AGG, "summary.csv"), index=False)
    print(f"summary.csv: {len(summary)} rows (expect 36)")

    # ---- degradation: clean vs each degraded condition, per algorithm ----
    drows = []
    for algo in ALGOS:
        base = per_seed[(per_seed.algorithm == algo) & (per_seed.noise_std == 0) & (per_seed.packet_error_rate == 0)]
        for (ns, per), g in per_seed[per_seed.algorithm == algo].groupby(["noise_std", "packet_error_rate"]):
            if ns == 0 and per == 0:
                continue
            for m in metric_names:
                b = base[m].to_numpy(float); c = g[m].to_numpy(float)
                bm, cm = np.nanmean(b), np.nanmean(c)
                bv, cv = np.nanvar(b, ddof=1), np.nanvar(c, ddof=1)
                absd = cm - bm
                rel = 100.0 * absd / bm if (np.isfinite(bm) and bm != 0) else np.nan
                drows.append({
                    "algorithm": algo, "noise_std": ns, "packet_error_rate": per,
                    "condition": f"n{ns}_p{per}", "metric": m,
                    "clean_mean": bm, "degraded_mean": cm, "abs_change": absd,
                    "rel_change_pct": rel, "clean_var": bv, "degraded_var": cv,
                    "var_ratio": (cv / bv) if (np.isfinite(bv) and bv > 0) else np.nan,
                })
    degradation = pd.DataFrame(drows)
    degradation.to_csv(os.path.join(AGG, "degradation.csv"), index=False)
    print(f"degradation.csv: {len(degradation)} rows")

    # console preview: success rate grid per algorithm
    print("\n=== success rate (%) by algorithm x condition (mean of 6 seeds) ===")
    piv = summary.pivot_table(index="algorithm", columns="condition", values="success_rate_mean")
    piv = piv.reindex(ALGOS)
    print(piv.round(1).to_string())
    print("\n=== collisions/ep by algorithm x condition ===")
    piv2 = summary.pivot_table(index="algorithm", columns="condition", values="collisions_per_ep_mean").reindex(ALGOS)
    print(piv2.round(2).to_string())


if __name__ == "__main__":
    main()
