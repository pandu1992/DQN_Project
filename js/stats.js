/* ============================================================
 * BINTULU PORT — Web DQN Simulation
 * stats.js — pure-JS statistical evaluation toolkit
 *
 * No external libraries. Everything is implemented from scratch
 * so it runs in the browser and from file://.
 *
 * Provided (via window.BintuluStats):
 *   Descriptive:  mean, variance, std (sample), sem, ci95
 *   Distributions: erf, normalCdf, normalPpf, logGamma, betacf,
 *                  regularizedIncompleteBeta, studentTCdf
 *   Tests:        welchTTest  (two-sample, unequal variance)
 *                 mannWhitneyU (rank-sum, normal approx w/ tie + CC)
 *   Effect size:  cohensD (pooled), hedgesG (bias-corrected)
 *   Helpers:      sigMarker, pairwise(), formatP
 *
 * All two-sample tests are TWO-SIDED.
 * ============================================================ */

(function (root) {
  "use strict";

  // ---------------- Descriptive statistics ----------------
  function mean(x) {
    if (!x || x.length === 0) return NaN;
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i];
    return s / x.length;
  }

  // sample variance (n-1 denominator, unbiased)
  function variance(x) {
    const n = x.length;
    if (n < 2) return 0;
    const m = mean(x);
    let s = 0;
    for (let i = 0; i < n; i++) {
      const d = x[i] - m;
      s += d * d;
    }
    return s / (n - 1);
  }

  function std(x) {
    return Math.sqrt(variance(x));
  }

  // standard error of the mean
  function sem(x) {
    const n = x.length;
    if (n < 1) return NaN;
    return std(x) / Math.sqrt(n);
  }

  // 95% CI half-width for the mean using the t-distribution
  function ci95(x) {
    const n = x.length;
    if (n < 2) return NaN;
    const t = tCritical(0.05, n - 1); // two-sided 95%
    return t * sem(x);
  }

  // ---------------- Special functions ----------------

  // Abramowitz & Stegun 7.1.26 approximation of the error function.
  // Max abs error ~1.5e-7 — plenty for reporting p-values.
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y =
      1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
        0.284496736) *
        t +
        0.254829592) *
        t *
        Math.exp(-x * x);
    return sign * y;
  }

  // standard normal CDF
  function normalCdf(z) {
    return 0.5 * (1 + erf(z / Math.SQRT2));
  }

  // inverse standard normal CDF (Acklam's algorithm)
  function normalPpf(p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    const a = [
      -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
      1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
    ];
    const b = [
      -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
      6.680131188771972e1, -1.328068155288572e1,
    ];
    const c = [
      -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
      -2.549732539343734, 4.374664141464968, 2.938163982698783,
    ];
    const d = [
      7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
      3.754408661907416,
    ];
    const plow = 0.02425;
    const phigh = 1 - plow;
    let q, r;
    if (p < plow) {
      q = Math.sqrt(-2 * Math.log(p));
      return (
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
      );
    } else if (p <= phigh) {
      q = p - 0.5;
      r = q * q;
      return (
        ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
          q) /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
      );
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return (
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
      );
    }
  }

  // log-gamma (Lanczos approximation)
  function logGamma(x) {
    const g = 7;
    const c = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if (x < 0.5) {
      // reflection formula
      return (
        Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x)
      );
    }
    x -= 1;
    let a = c[0];
    const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  // continued fraction for the incomplete beta function (Numerical Recipes)
  function betacf(a, b, x) {
    const MAXIT = 200;
    const EPS = 3e-12;
    const FPMIN = 1e-300;
    let qab = a + b;
    let qap = a + 1;
    let qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  // regularized incomplete beta function I_x(a,b)
  function regularizedIncompleteBeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const lbeta =
      logGamma(a + b) - logGamma(a) - logGamma(b) +
      a * Math.log(x) + b * Math.log(1 - x);
    const front = Math.exp(lbeta);
    if (x < (a + 1) / (a + b + 2)) {
      return (front * betacf(a, b, x)) / a;
    }
    return 1 - (front * betacf(b, a, 1 - x)) / b;
  }

  // Student-t CDF: P(T <= t) for df degrees of freedom
  function studentTCdf(t, df) {
    if (!isFinite(t)) return t > 0 ? 1 : 0;
    const x = df / (df + t * t);
    const ib = 0.5 * regularizedIncompleteBeta(x, df / 2, 0.5);
    return t > 0 ? 1 - ib : ib;
  }

  // two-sided t p-value
  function tTwoSidedP(t, df) {
    if (df <= 0) return NaN;
    const cdf = studentTCdf(Math.abs(t), df);
    return 2 * (1 - cdf);
  }

  // critical t value for a two-sided test at significance `alpha`
  // (i.e. the (1 - alpha/2) quantile). Uses bisection on studentTCdf.
  function tCritical(alpha, df) {
    if (df <= 0) return NaN;
    const target = 1 - alpha / 2;
    let lo = 0, hi = 1000;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (studentTCdf(mid, df) < target) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  }

  // ---------------- Two-sample tests ----------------

  // Welch's t-test (does NOT assume equal variances). Two-sided.
  // Returns { t, df, p, meanA, meanB, diff }.
  function welchTTest(a, b) {
    const na = a.length, nb = b.length;
    if (na < 2 || nb < 2) {
      return { t: NaN, df: NaN, p: NaN, meanA: mean(a), meanB: mean(b), diff: mean(a) - mean(b) };
    }
    const ma = mean(a), mb = mean(b);
    const va = variance(a), vb = variance(b);
    const sa = va / na, sb = vb / nb;
    const denom = Math.sqrt(sa + sb);
    if (denom === 0) {
      // no variance in either sample
      const equal = ma === mb;
      return { t: equal ? 0 : Infinity, df: na + nb - 2, p: equal ? 1 : 0, meanA: ma, meanB: mb, diff: ma - mb };
    }
    const t = (ma - mb) / denom;
    // Welch–Satterthwaite degrees of freedom
    const df =
      Math.pow(sa + sb, 2) /
      (Math.pow(sa, 2) / (na - 1) + Math.pow(sb, 2) / (nb - 1));
    const p = tTwoSidedP(t, df);
    return { t, df, p, meanA: ma, meanB: mb, diff: ma - mb };
  }

  // Mann–Whitney U test (Wilcoxon rank-sum), two-sided.
  // Normal approximation with tie correction and continuity correction.
  // Returns { U, z, p, nA, nB }.
  function mannWhitneyU(a, b) {
    const na = a.length, nb = b.length;
    if (na < 1 || nb < 1) return { U: NaN, z: NaN, p: NaN, nA: na, nB: nb };

    // combined, tagged, sorted for ranking
    const combined = [];
    for (let i = 0; i < na; i++) combined.push({ v: a[i], g: 0 });
    for (let i = 0; i < nb; i++) combined.push({ v: b[i], g: 1 });
    combined.sort((p, q) => p.v - q.v);

    // assign ranks with ties -> average rank
    const N = combined.length;
    const ranks = new Array(N);
    const tieGroups = [];
    let i = 0;
    while (i < N) {
      let j = i;
      while (j + 1 < N && combined[j + 1].v === combined[i].v) j++;
      const avgRank = (i + 1 + (j + 1)) / 2; // ranks are 1-based
      for (let k = i; k <= j; k++) ranks[k] = avgRank;
      const groupSize = j - i + 1;
      if (groupSize > 1) tieGroups.push(groupSize);
      i = j + 1;
    }

    // rank sum for group A
    let rankSumA = 0;
    for (let k = 0; k < N; k++) if (combined[k].g === 0) rankSumA += ranks[k];

    const Ua = rankSumA - (na * (na + 1)) / 2;
    const Ub = na * nb - Ua;
    const U = Math.min(Ua, Ub);

    const muU = (na * nb) / 2;
    // variance with tie correction
    let tieTerm = 0;
    for (const t of tieGroups) tieTerm += t * t * t - t;
    const sigmaU = Math.sqrt(
      (na * nb / 12) * (N + 1 - tieTerm / (N * (N - 1)))
    );

    if (sigmaU === 0) {
      return { U, z: 0, p: 1, nA: na, nB: nb };
    }
    // continuity correction
    const z = (Math.abs(U - muU) - 0.5) / sigmaU;
    const p = 2 * (1 - normalCdf(z));
    return { U, z: (Ua - muU) / sigmaU, p: Math.min(1, Math.max(0, p)), nA: na, nB: nb };
  }

  // ---------------- Effect size ----------------

  // Cohen's d with POOLED standard deviation.
  function cohensD(a, b) {
    const na = a.length, nb = b.length;
    if (na < 2 || nb < 2) return NaN;
    const va = variance(a), vb = variance(b);
    const pooled = Math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2));
    if (pooled === 0) return 0;
    return (mean(a) - mean(b)) / pooled;
  }

  // Hedges' g: Cohen's d with small-sample bias correction.
  function hedgesG(a, b) {
    const d = cohensD(a, b);
    if (!isFinite(d)) return NaN;
    const df = a.length + b.length - 2;
    if (df <= 0) return d;
    const J = 1 - 3 / (4 * df - 1); // correction factor
    return d * J;
  }

  // magnitude label for |d| (Cohen's conventions)
  function effectMagnitude(d) {
    const ad = Math.abs(d);
    if (!isFinite(ad)) return "—";
    if (ad < 0.2) return "negligible";
    if (ad < 0.5) return "small";
    if (ad < 0.8) return "medium";
    return "large";
  }

  // ---------------- Reporting helpers ----------------

  // significance marker from a p-value
  function sigMarker(p) {
    if (!isFinite(p)) return "—";
    if (p < 0.001) return "***";
    if (p < 0.01) return "**";
    if (p < 0.05) return "*";
    return "ns";
  }

  function formatP(p) {
    if (!isFinite(p)) return "—";
    if (p < 1e-4) return "<0.0001";
    if (p < 1e-3) return p.toFixed(5);
    return p.toFixed(4);
  }

  // Run all pairwise comparisons over a list of named groups.
  // groups: [{ key, label, samples:[...] }, ...]
  // Returns an array of pairwise result objects.
  function pairwise(groups) {
    const out = [];
    for (let i = 0; i < groups.length; i++) {
      for (let j = i + 1; j < groups.length; j++) {
        const A = groups[i], B = groups[j];
        const welch = welchTTest(A.samples, B.samples);
        const mw = mannWhitneyU(A.samples, B.samples);
        const d = cohensD(A.samples, B.samples);
        const g = hedgesG(A.samples, B.samples);
        out.push({
          aKey: A.key, aLabel: A.label,
          bKey: B.key, bLabel: B.label,
          nA: A.samples.length, nB: B.samples.length,
          meanA: mean(A.samples), meanB: mean(B.samples),
          diff: mean(A.samples) - mean(B.samples),
          welch,
          mannWhitney: mw,
          cohensD: d,
          hedgesG: g,
          magnitude: effectMagnitude(d),
          // primary p-value shown = Welch (fallback to MW if Welch NaN)
          pWelch: welch.p,
          pMann: mw.p,
          sig: sigMarker(welch.p),
        });
      }
    }
    return out;
  }

  // Bonferroni-adjusted alpha threshold given number of comparisons
  function bonferroniAlpha(nComparisons, alpha) {
    alpha = alpha || 0.05;
    return nComparisons > 0 ? alpha / nComparisons : alpha;
  }

  const API = {
    mean, variance, std, sem, ci95,
    erf, normalCdf, normalPpf, logGamma,
    betacf, regularizedIncompleteBeta, studentTCdf,
    tTwoSidedP, tCritical,
    welchTTest, mannWhitneyU,
    cohensD, hedgesG, effectMagnitude,
    sigMarker, formatP, pairwise, bonferroniAlpha,
  };

  root.BintuluStats = API;

  // also expose for Node (headless testing)
  if (typeof module !== "undefined" && module.exports) {
    module.exports = API;
  }
})(typeof window !== "undefined" ? window : globalThis);
