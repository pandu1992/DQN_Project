#!/usr/bin/env python3
"""
STUDY 4 — AGGREGATION (two independent vessels that must avoid each other)

Unit of analysis = per-seed cell scalar (n=8 per (pairing, condition)).

Two families of metrics:
  * INTER-VESSEL (per ENCOUNTER, shared by both vessels): computed from the
    per-episode encounter (deduplicated so the shared metric is not double
    counted across the two vessel rows).
  * PER-VESSEL navigation / physical: averaged over BOTH vessels' rows in the
    cell (success, reward, docking, cte, iala, static collisions, dropped
    frames, optimality, steps).

Outputs:
  results_study4/aggregated/per_seed.csv    (96 rows)
  results_study4/aggregated/summary.csv     (12 rows: 4 pairings x 3 conditions)
  results_study4/aggregated/degradation.csv (clean vs harsh, per pairing)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results_study4", "raw", "eval_episodes.csv")
AGG = os.path.join(ROOT, "results_study4", "aggregated")
os.makedirs(AGG, exist_ok=True)
RNG = np.random.default_rng(12345)

PAIRINGS = ["DQN_vs_DQN", "DDQN_vs_DDQN", "DQN_vs_RuleBased", "RuleBased_vs_RuleBased"]
CONDS = ["clean", "mid", "harsh"]

# per-encounter inter-vessel safety metrics
IV_METRICS = ["vessel_collision_rate", "vessel_collisions_per_ep", "near_misses_per_ep",
              "min_cpa", "collision_contact_steps", "give_way_events"]
# per-vessel navigation / physical metrics
PV_METRICS = ["success_rate", "mean_reward", "static_collisions", "cte_mean",
              "iala_violations", "dropped_frames", "docking_accuracy",
              "optimality_ratio", "mean_steps"]
METRICS = IV_METRICS + PV_METRICS


def num(c, fr):
    return pd.to_numeric(fr[c], errors="coerce")


def per_seed(g):
    """g = all rows for one (pairing, condition, seed): 60 rows (30 ep x 2 vessels)."""
    # inter-vessel metrics are per-encounter; dedupe to one row per episode
    enc = g.drop_duplicates(subset=["episode"])
    solved = g[g["success"] == 1]
    out = {
        # ---- inter-vessel (per encounter) ----
        "vessel_collision_rate": 100.0 * (num("vessel_collisions", enc) > 0).mean(),
        "vessel_collisions_per_ep": num("vessel_collisions", enc).mean(),
        "near_misses_per_ep": num("near_misses", enc).mean(),
        "min_cpa": num("min_cpa", enc).mean(),
        "collision_contact_steps": num("collision_contact_steps", enc).mean(),
        "give_way_events": num("give_way_events", enc).mean(),
        # ---- per-vessel navigation / physical (both vessels) ----
        "success_rate": 100.0 * g["success"].mean(),
        "mean_reward": num("reward", g).mean(),
        "static_collisions": num("static_collisions", g).mean(),
        "cte_mean": num("cte_mean", g).mean(),
        "iala_violations": num("iala_violations", g).mean(),
        "dropped_frames": num("dropped_frames", g).mean(),
        "docking_accuracy": num("docking_accuracy", solved).mean() if len(solved) else np.nan,
        "optimality_ratio": num("optimality_ratio", solved).mean() if len(solved) else np.nan,
        "mean_steps": num("steps", g).mean(),
        "n_encounters": int(len(enc)), "n_vessel_rows": int(len(g)), "n_solved": int(len(solved)),
    }
    return out


def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    b = np.array([np.mean(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)])
    return (np.percentile(b, 2.5), np.percentile(b, 97.5))


def t_hw(x, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    return stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)


def main():
    df = pd.read_csv(RAW)
    df = df[df["phase"] == "eval"].copy()

    rows = []
    for (pr, cond, seed), g in df.groupby(["pairing", "condition", "seed"], sort=True):
        rows.append({"pairing": pr, "condition": cond, "seed": int(seed), **per_seed(g)})
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(AGG, "per_seed.csv"), index=False)
    print(f"per_seed.csv: {len(ps)} rows (expect 96)")

    srows = []
    for (pr, cond), g in ps.groupby(["pairing", "condition"], sort=True):
        e = {"pairing": pr, "condition": cond, "n_seeds": len(g)}
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
    print(f"summary.csv: {len(summary)} rows (expect 12)")

    drows = []
    for pr in PAIRINGS:
        base = ps[(ps.pairing == pr) & (ps.condition == "clean")]
        deg = ps[(ps.pairing == pr) & (ps.condition == "harsh")]
        for m in METRICS:
            bm, cm = np.nanmean(base[m]), np.nanmean(deg[m])
            drows.append({"pairing": pr, "metric": m, "clean_mean": bm, "harsh_mean": cm,
                          "abs_change": cm - bm,
                          "rel_change_pct": (100.0 * (cm - bm) / bm) if (np.isfinite(bm) and bm != 0) else np.nan})
    pd.DataFrame(drows).to_csv(os.path.join(AGG, "degradation.csv"), index=False)
    print("degradation.csv written")

    print("\n=== inter-vessel collision rate (% of encounters) ===")
    print(summary.pivot_table(index="pairing", columns="condition", values="vessel_collision_rate_mean").reindex(PAIRINGS)[CONDS].round(1).to_string())
    print("\n=== vessel collisions / encounter ===")
    print(summary.pivot_table(index="pairing", columns="condition", values="vessel_collisions_per_ep_mean").reindex(PAIRINGS)[CONDS].round(2).to_string())
    print("\n=== min CPA (closest point of approach, logical units) ===")
    print(summary.pivot_table(index="pairing", columns="condition", values="min_cpa_mean").reindex(PAIRINGS)[CONDS].round(1).to_string())
    print("\n=== per-vessel success rate (%) ===")
    print(summary.pivot_table(index="pairing", columns="condition", values="success_rate_mean").reindex(PAIRINGS)[CONDS].round(1).to_string())


if __name__ == "__main__":
    main()
