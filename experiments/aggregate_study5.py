#!/usr/bin/env python3
"""
STUDY 5 — AGGREGATION (two-phase round trip + two-way opposing traffic)

Unit of analysis = per-seed cell scalar (n=8).

Scenario A (twophase): single vessel, per (agent, condition, seed).
Scenario B (twoway):   two vessels, per (pairing, condition, seed); inter-vessel
                       metrics per ENCOUNTER (deduped), navigation per vessel.

Outputs:
  results_study5/aggregated/twophase_per_seed.csv (48 rows)
  results_study5/aggregated/twophase_summary.csv  (6 rows)
  results_study5/aggregated/twoway_per_seed.csv    (72 rows)
  results_study5/aggregated/twoway_summary.csv     (9 rows)
  results_study5/aggregated/degradation.csv        (clean vs harsh, both scenarios)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "results_study5", "raw")
AGG = os.path.join(ROOT, "results_study5", "aggregated")
os.makedirs(AGG, exist_ok=True)
RNG = np.random.default_rng(12345)

CONDS = ["clean", "mid", "harsh"]
PHASE_AGENTS = ["RuleBased", "DQN"]
WAY_PAIRINGS = ["Rule_vs_Rule", "DQN_vs_Rule", "DQN_vs_DQN"]

PHASE_METRICS = ["full_cycle_success_rate", "dock_success_rate", "mean_reward",
                 "phase1_steps", "dock_accuracy", "collisions_per_ep",
                 "cte_mean", "iala_violations", "dropped_frames"]
WAY_IV = ["vessel_collision_rate", "vessel_collisions_per_ep", "near_misses_per_ep",
          "min_cpa", "head_on_events", "give_way_events"]
WAY_PV = ["success_rate", "mean_reward", "cte_mean", "iala_violations"]
WAY_METRICS = WAY_IV + WAY_PV


def num(c, fr):
    return pd.to_numeric(fr[c], errors="coerce")


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


def summarise(ps, metrics, groupcol, levels, out):
    srows = []
    for (lvl, cond), g in ps.groupby([groupcol, "condition"], sort=True):
        e = {groupcol: lvl, "condition": cond, "n_seeds": len(g)}
        for m in metrics:
            x = g[m].to_numpy(float); xv = x[~np.isnan(x)]
            e[f"{m}_mean"] = np.nanmean(x) if len(xv) else np.nan
            e[f"{m}_sd"] = np.nanstd(x, ddof=1) if len(xv) > 1 else 0.0
            e[f"{m}_ci95_hw"] = t_hw(x)
            lo, hi = boot_ci(x)
            e[f"{m}_boot_lo"] = lo; e[f"{m}_boot_hi"] = hi
        srows.append(e)
    df = pd.DataFrame(srows)
    df.to_csv(os.path.join(AGG, out), index=False)
    return df


# ---------------- Scenario A: two-phase ----------------
def phase_per_seed(g):
    solved_dock = g[g["dock_success"] == 1]
    return {
        "full_cycle_success_rate": 100.0 * g["full_success"].mean(),
        "dock_success_rate": 100.0 * g["dock_success"].mean(),
        "mean_reward": num("reward", g).mean(),
        "phase1_steps": num("phase1_steps", solved_dock).mean() if len(solved_dock) else np.nan,
        "dock_accuracy": num("dock_accuracy", solved_dock).mean() if len(solved_dock) else np.nan,
        "collisions_per_ep": num("collisions", g).mean(),
        "cte_mean": num("cte_mean", g).mean(),
        "iala_violations": num("iala_violations", g).mean(),
        "dropped_frames": num("dropped_frames", g).mean(),
        "n_eval": int(len(g)),
    }


def do_twophase():
    df = pd.read_csv(os.path.join(RAW, "twophase_episodes.csv"))
    df = df[df["phase"] == "eval"].copy()
    rows = []
    for (ag, cond, seed), g in df.groupby(["agent", "condition", "seed"], sort=True):
        rows.append({"agent": ag, "condition": cond, "seed": int(seed), **phase_per_seed(g)})
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(AGG, "twophase_per_seed.csv"), index=False)
    print(f"twophase_per_seed.csv: {len(ps)} rows (expect 48)")
    summ = summarise(ps, PHASE_METRICS, "agent", PHASE_AGENTS, "twophase_summary.csv")
    print(f"twophase_summary.csv: {len(summ)} rows (expect 6)")
    print("\n=== two-phase: dock vs FULL-cycle success (%) ===")
    for m in ["dock_success_rate", "full_cycle_success_rate"]:
        print(f"-- {m} --")
        print(summ.pivot_table(index="agent", columns="condition", values=f"{m}_mean").reindex(PHASE_AGENTS)[CONDS].round(1).to_string())
    return ps


# ---------------- Scenario B: two-way ----------------
def way_per_seed(g):
    enc = g.drop_duplicates(subset=["episode"])
    solved = g[g["success"] == 1]
    return {
        "vessel_collision_rate": 100.0 * (num("vessel_collisions", enc) > 0).mean(),
        "vessel_collisions_per_ep": num("vessel_collisions", enc).mean(),
        "near_misses_per_ep": num("near_misses", enc).mean(),
        "min_cpa": num("min_cpa", enc).mean(),
        "head_on_events": num("head_on_events", enc).mean(),
        "give_way_events": num("give_way_events", enc).mean(),
        "success_rate": 100.0 * g["success"].mean(),
        "mean_reward": num("reward", g).mean(),
        "cte_mean": num("cte_mean", g).mean(),
        "iala_violations": num("iala_violations", g).mean(),
        "n_encounters": int(len(enc)), "n_vessel_rows": int(len(g)),
    }


def do_twoway():
    df = pd.read_csv(os.path.join(RAW, "twoway_episodes.csv"))
    df = df[df["phase"] == "eval"].copy()
    rows = []
    for (pr, cond, seed), g in df.groupby(["pairing", "condition", "seed"], sort=True):
        rows.append({"pairing": pr, "condition": cond, "seed": int(seed), **way_per_seed(g)})
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(AGG, "twoway_per_seed.csv"), index=False)
    print(f"\ntwoway_per_seed.csv: {len(ps)} rows (expect 72)")
    summ = summarise(ps, WAY_METRICS, "pairing", WAY_PAIRINGS, "twoway_summary.csv")
    print(f"twoway_summary.csv: {len(summ)} rows (expect 9)")
    print("\n=== two-way: inter-vessel collision rate (% encounters) ===")
    print(summ.pivot_table(index="pairing", columns="condition", values="vessel_collision_rate_mean").reindex(WAY_PAIRINGS)[CONDS].round(1).to_string())
    print("\n=== two-way: head-on events / encounter ===")
    print(summ.pivot_table(index="pairing", columns="condition", values="head_on_events_mean").reindex(WAY_PAIRINGS)[CONDS].round(2).to_string())
    print("\n=== two-way: per-vessel success rate (%) ===")
    print(summ.pivot_table(index="pairing", columns="condition", values="success_rate_mean").reindex(WAY_PAIRINGS)[CONDS].round(1).to_string())
    return ps


def degradation(phase_ps, way_ps):
    drows = []
    for ag in PHASE_AGENTS:
        base = phase_ps[(phase_ps.agent == ag) & (phase_ps.condition == "clean")]
        deg = phase_ps[(phase_ps.agent == ag) & (phase_ps.condition == "harsh")]
        for m in PHASE_METRICS:
            bm, cm = np.nanmean(base[m]), np.nanmean(deg[m])
            drows.append({"scenario": "twophase", "level": ag, "metric": m, "clean_mean": bm, "harsh_mean": cm,
                          "abs_change": cm - bm, "rel_change_pct": (100.0 * (cm - bm) / bm) if (np.isfinite(bm) and bm != 0) else np.nan})
    for pr in WAY_PAIRINGS:
        base = way_ps[(way_ps.pairing == pr) & (way_ps.condition == "clean")]
        deg = way_ps[(way_ps.pairing == pr) & (way_ps.condition == "harsh")]
        for m in WAY_METRICS:
            bm, cm = np.nanmean(base[m]), np.nanmean(deg[m])
            drows.append({"scenario": "twoway", "level": pr, "metric": m, "clean_mean": bm, "harsh_mean": cm,
                          "abs_change": cm - bm, "rel_change_pct": (100.0 * (cm - bm) / bm) if (np.isfinite(bm) and bm != 0) else np.nan})
    pd.DataFrame(drows).to_csv(os.path.join(AGG, "degradation.csv"), index=False)
    print("\ndegradation.csv written")


def main():
    p = do_twophase()
    w = do_twoway()
    degradation(p, w)


if __name__ == "__main__":
    main()
