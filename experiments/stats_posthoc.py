#!/usr/bin/env python3
"""
PAIRWISE POST-HOC + ENHANCEMENT / ROBUSTNESS ANALYSIS (tasks #6, #7)

Paired design: the SAME seeds 0-9 appear in every configuration, so all
contrasts are WITHIN-seed (paired). Because most metrics are non-normal
(Shapiro), the PRIMARY test is the paired Wilcoxon signed-rank test; the
paired t-test is reported alongside for completeness.

Effect sizes (paired):
  - Cohen's d_z = mean(diff) / sd(diff)           (paired standardized mean)
  - matched-pairs rank-biserial r = Wilcoxon effect size
  - Cliff's delta (ordinal, direction-of-dominance)

Multiple-comparison correction: Holm and Benjamini-Hochberg (FDR), applied
within each (metric, family-of-comparisons).

Outputs:
  results/statistics/pairwise_algorithms.csv   (6 algo pairs, baseline cells)
  results/statistics/enhancement_contrasts.csv (PER / Noisy / PER+Noisy vs baseline, per algorithm)
  results/statistics/robustness_summary.csv    (variance change + degradation framing)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGG = os.path.join(ROOT, "results", "aggregated")
STATDIR = os.path.join(ROOT, "results", "statistics")
os.makedirs(STATDIR, exist_ok=True)

PER_SEED = pd.read_csv(os.path.join(AGG, "per_seed.csv"))
ALGOS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"]
PRIMARY = ["success_rate", "mean_reward", "optimality_ratio"]
SECONDARY = ["mean_steps", "revisit_rate", "excess_cost", "route_risk", "route_difficulty"]
METRICS = PRIMARY + SECONDARY
ALPHA = 0.05


# ---------------- effect sizes ----------------
def cohens_dz(diff):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def rank_biserial_from_wilcoxon(x, y):
    """Matched-pairs rank-biserial correlation for the paired Wilcoxon test."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, n
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    total = r_plus + r_minus
    if total == 0:
        return 0.0, n
    rb = (r_plus - r_minus) / total
    return rb, n


def mag_d(d):
    ad = abs(d)
    if not np.isfinite(ad): return "n/a"
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"


def mag_cliff(delta):
    ad = abs(delta)
    if not np.isfinite(ad): return "n/a"
    if ad < 0.147: return "negligible"
    if ad < 0.33: return "small"
    if ad < 0.474: return "medium"
    return "large"


# ---------------- corrections ----------------
def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        prev = min(prev, val)
        adj[idx] = min(1.0, prev)
    return adj


def paired_tests(x, y):
    """x,y aligned by seed. Returns dict with paired t and Wilcoxon results."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    diff = x - y
    # paired t
    if n >= 2 and diff.std(ddof=1) > 0:
        t_stat, t_p = stats.ttest_rel(x, y)
    else:
        t_stat, t_p = (0.0, 1.0) if np.allclose(diff, 0) else (np.nan, np.nan)
    # Wilcoxon signed-rank (needs some non-zero diffs)
    if np.any(diff != 0):
        try:
            # exact method is appropriate/available for small paired samples (n<=~25)
            w_stat, w_p = stats.wilcoxon(x, y, zero_method="wilcox",
                                         alternative="two-sided", method="exact")
        except Exception:
            try:
                w_stat, w_p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, w_p = np.nan, np.nan
    else:
        w_stat, w_p = 0.0, 1.0
    rb, _ = rank_biserial_from_wilcoxon(x, y)
    return {
        "n": n, "mean_x": x.mean() if n else np.nan, "mean_y": y.mean() if n else np.nan,
        "mean_diff": diff.mean() if n else np.nan,
        "t_stat": t_stat, "t_p": t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "cohens_dz": cohens_dz(diff), "rank_biserial": rb,
        "cliffs_delta": cliffs_delta(x, y),
    }


# ---------------- (task #6) pairwise across the 4 base algorithms ----------------
def pairwise_algorithms():
    base = PER_SEED[(PER_SEED.per == 0) & (PER_SEED.noisy == 0)]
    rows = []
    for m in METRICS:
        # collect the 6 pairwise contrasts for this metric, then correct together
        recs = []
        for a, b in combinations(ALGOS, 2):
            xa = base[base.algorithm == a].sort_values("seed")[m].to_numpy(float)
            xb = base[base.algorithm == b].sort_values("seed")[m].to_numpy(float)
            r = paired_tests(xa, xb)
            recs.append({"metric": m, "algo_a": a, "algo_b": b, **r})
        # corrections use the Wilcoxon p (primary) and t p (secondary)
        wp = [r["wilcoxon_p"] for r in recs]
        tp = [r["t_p"] for r in recs]
        wp_holm = holm(wp); wp_bh = benjamini_hochberg(wp)
        tp_holm = holm(tp); tp_bh = benjamini_hochberg(tp)
        for i, r in enumerate(recs):
            r["wilcoxon_p_holm"] = wp_holm[i]
            r["wilcoxon_p_bh"] = wp_bh[i]
            r["t_p_holm"] = tp_holm[i]
            r["t_p_bh"] = tp_bh[i]
            r["sig_holm_0.05"] = bool(wp_holm[i] < ALPHA)
            r["effect_dz_mag"] = mag_d(r["cohens_dz"])
            r["effect_cliff_mag"] = mag_cliff(r["cliffs_delta"])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATDIR, "pairwise_algorithms.csv"), index=False)
    return df


# ---------------- (task #7) enhancement contrasts + robustness ----------------
def enhancement_contrasts():
    """For each algorithm: compare +PER, +Noisy, +PER+Noisy vs the plain baseline
    (per=0,noisy=0), paired by seed. Also PER-vs-none and Noisy-vs-none marginals."""
    rows = []
    for algo in ALGOS:
        sub = PER_SEED[PER_SEED.algorithm == algo]
        baseline = sub[(sub.per == 0) & (sub.noisy == 0)].sort_values("seed")
        contrasts = {
            "+PER": sub[(sub.per == 1) & (sub.noisy == 0)],
            "+Noisy": sub[(sub.per == 0) & (sub.noisy == 1)],
            "+PER+Noisy": sub[(sub.per == 1) & (sub.noisy == 1)],
        }
        for m in METRICS:
            base_x = baseline[m].to_numpy(float)
            recs = []
            for name, cdf in contrasts.items():
                cx = cdf.sort_values("seed")[m].to_numpy(float)
                r = paired_tests(cx, base_x)  # contrast - baseline
                base_mean = np.nanmean(base_x)
                rel = 100.0 * r["mean_diff"] / base_mean if base_mean not in (0, np.nan) and np.isfinite(base_mean) and base_mean != 0 else np.nan
                var_base = np.nanvar(base_x, ddof=1)
                var_c = np.nanvar(cx, ddof=1)
                recs.append({
                    "algorithm": algo, "contrast": name, "metric": m,
                    "baseline_mean": base_mean, "enhanced_mean": np.nanmean(cx),
                    "abs_change": r["mean_diff"], "rel_change_pct": rel,
                    "var_baseline": var_base, "var_enhanced": var_c,
                    "var_ratio": (var_c / var_base) if var_base > 0 else np.nan,
                    "wilcoxon_p": r["wilcoxon_p"], "t_p": r["t_p"],
                    "cohens_dz": r["cohens_dz"], "rank_biserial": r["rank_biserial"],
                    "cliffs_delta": r["cliffs_delta"],
                })
            wp = [r["wilcoxon_p"] for r in recs]
            wp_holm = holm(wp); wp_bh = benjamini_hochberg(wp)
            for i, r in enumerate(recs):
                r["wilcoxon_p_holm"] = wp_holm[i]
                r["wilcoxon_p_bh"] = wp_bh[i]
                r["sig_holm_0.05"] = bool(wp_holm[i] < ALPHA)
                r["effect_dz_mag"] = mag_d(r["cohens_dz"])
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATDIR, "enhancement_contrasts.csv"), index=False)
    return df


def robustness_summary(enh):
    """Compact per-algorithm robustness view: variance ratio + whether any
    enhancement significantly helped/hurt each primary metric."""
    rows = []
    for algo in ALGOS:
        for m in PRIMARY:
            e = enh[(enh.algorithm == algo) & (enh.metric == m)]
            best = e.loc[e["abs_change"].idxmax()] if len(e) else None
            worst = e.loc[e["abs_change"].idxmin()] if len(e) else None
            any_sig = bool((e["sig_holm_0.05"]).any()) if len(e) else False
            mean_var_ratio = float(e["var_ratio"].mean()) if len(e) else np.nan
            rows.append({
                "algorithm": algo, "metric": m,
                "best_contrast": best["contrast"] if best is not None else "",
                "best_abs_change": best["abs_change"] if best is not None else np.nan,
                "worst_contrast": worst["contrast"] if worst is not None else "",
                "worst_abs_change": worst["abs_change"] if worst is not None else np.nan,
                "mean_var_ratio_vs_baseline": mean_var_ratio,
                "any_enhancement_sig_holm": any_sig,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STATDIR, "robustness_summary.csv"), index=False)
    return df


def main():
    pw = pairwise_algorithms()
    enh = enhancement_contrasts()
    rob = robustness_summary(enh)

    print("=== PAIRWISE (base algorithms) — significant after Holm (Wilcoxon) ===")
    sig = pw[pw["sig_holm_0.05"]]
    if len(sig):
        print(sig[["metric", "algo_a", "algo_b", "mean_diff", "wilcoxon_p", "wilcoxon_p_holm",
                   "cohens_dz", "effect_dz_mag", "cliffs_delta"]].round(4).to_string(index=False))
    else:
        print("  NONE — no pairwise algorithm difference survives Holm correction on any metric.")

    print("\n=== ENHANCEMENT CONTRASTS — significant after Holm (Wilcoxon), primary metrics ===")
    esig = enh[(enh["sig_holm_0.05"]) & (enh.metric.isin(PRIMARY))]
    if len(esig):
        print(esig[["algorithm", "contrast", "metric", "abs_change", "rel_change_pct",
                    "wilcoxon_p_holm", "cohens_dz", "effect_dz_mag"]].round(4).to_string(index=False))
    else:
        print("  NONE survive Holm on primary metrics.")

    print("\n=== optimality_ratio enhancement contrasts (all, raw Wilcoxon p) ===")
    optr = enh[enh.metric == "optimality_ratio"][["algorithm", "contrast", "abs_change",
             "rel_change_pct", "wilcoxon_p", "wilcoxon_p_holm", "cohens_dz", "effect_dz_mag"]]
    print(optr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
