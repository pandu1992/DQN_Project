#!/usr/bin/env python3
"""
STUDY 3 — AGGREGATION (rule-based baseline vs DQN, 15 seeds)

Outputs (unit of analysis = per-seed scalar, n=15 per cell):
  results_study3/aggregated/per_seed.csv   (225 rows)
  results_study3/aggregated/summary.csv    (15 rows: 5 agents x 3 conditions)
  results_study3/aggregated/degradation.csv (clean vs harsh, per agent)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results_study3", "raw", "eval_episodes.csv")
AGG = os.path.join(ROOT, "results_study3", "aggregated")
os.makedirs(AGG, exist_ok=True)
RNG = np.random.default_rng(12345)

AGENTS = ["RuleBased", "DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
CONDS = ["clean", "mid", "harsh"]
METRICS = ["success_rate", "mean_reward", "collision_rate", "collisions_per_ep",
           "iala_violations", "cte_mean", "dropped_frames", "docking_accuracy",
           "optimality_ratio", "mean_steps"]


def per_seed(g):
    solved = g[g["success"] == 1]
    num = lambda c, fr: pd.to_numeric(fr[c], errors="coerce")
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
        "n_solved": int(len(solved)), "n_eval": int(len(g)),
    }


def boot_ci(x, n=10000, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    b = np.array([np.mean(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)])
    return (np.percentile(b, 2.5), np.percentile(b, 97.5))


def t_hw(x, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2: return np.nan
    return stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)


def main():
    df = pd.read_csv(RAW)
    df = df[df["phase"] == "eval"].copy()

    rows = []
    for (ag, cond, seed), g in df.groupby(["agent", "condition", "seed"], sort=True):
        rows.append({"agent": ag, "condition": cond, "seed": int(seed), **per_seed(g)})
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(AGG, "per_seed.csv"), index=False)
    print(f"per_seed.csv: {len(ps)} rows (expect 225)")

    srows = []
    for (ag, cond), g in ps.groupby(["agent", "condition"], sort=True):
        e = {"agent": ag, "condition": cond, "n_seeds": len(g)}
        for m in METRICS:
            x = g[m].to_numpy(float); xv = x[~np.isnan(x)]
            e[f"{m}_mean"] = np.nanmean(x) if len(xv) else np.nan
            e[f"{m}_sd"] = np.nanstd(x, ddof=1) if len(xv) > 1 else 0.0
            e[f"{m}_ci95_hw"] = t_hw(x)
            lo, hi = boot_ci(x)
            e[f"{m}_boot_lo"] = lo; e[f"{m}_boot_hi"] = hi
        srows.append(e)
    summary = pd.DataFrame(srows)
    summary.to_csv(os.path.join(AGG, "summary.csv"), index=False)
    print(f"summary.csv: {len(summary)} rows (expect 15)")

    # degradation: clean vs harsh, per agent
    drows = []
    for ag in AGENTS:
        base = ps[(ps.agent == ag) & (ps.condition == "clean")]
        deg = ps[(ps.agent == ag) & (ps.condition == "harsh")]
        for m in METRICS:
            bm, cm = np.nanmean(base[m]), np.nanmean(deg[m])
            drows.append({"agent": ag, "metric": m, "clean_mean": bm, "harsh_mean": cm,
                          "abs_change": cm - bm,
                          "rel_change_pct": (100.0 * (cm - bm) / bm) if (np.isfinite(bm) and bm != 0) else np.nan})
    pd.DataFrame(drows).to_csv(os.path.join(AGG, "degradation.csv"), index=False)
    print("degradation.csv written")

    # preview
    print("\n=== success rate (%) ===")
    print(summary.pivot_table(index="agent", columns="condition", values="success_rate_mean").reindex(AGENTS)[CONDS].round(1).to_string())
    print("\n=== collisions/ep ===")
    print(summary.pivot_table(index="agent", columns="condition", values="collisions_per_ep_mean").reindex(AGENTS)[CONDS].round(2).to_string())


if __name__ == "__main__":
    main()
