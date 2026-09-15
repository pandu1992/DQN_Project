# Reproducibility Guide — Q1 Experimental Analysis

Everything below regenerates the entire analysis from scratch. All randomness is
seeded; results are deterministic.

## Environment

- **Node.js** ≥ 18 (tested on v22) — runs the RL harness against the *actual*
  `js/environment.js`, `js/dqn.js`, and the offline research build
  `js/environmentV2.js`.
- **Python** 3.9 in a virtual environment for aggregation, statistics, tables,
  and figures.

```bash
# from the repo root
python3 -m venv experiments/.venv
experiments/.venv/bin/python -m pip install -r experiments/requirements.txt
```

## Pipeline (in order)

```bash
PY=experiments/.venv/bin/python

# 1. Run the full 4x2x2 x 10-seed experiment (160 cells).
#    Sharded per configuration to keep each invocation short; --append accumulates.
#    (Re-running a shard that partially wrote requires removing that config's rows first.)
for algo in DQN DoubleDQN DuelingDQN DuelingDoubleDQN; do
  for per in 0 1; do for noisy in 0 1; do
    node experiments/run_experiment.js --cell=$algo:$per:$noisy --append
  done; done
done
# -> results/raw/eval_episodes.csv (6400 rows), train_episodes.csv (24000), run_manifest.json

# 2. Aggregate to per-seed scalars + across-seed summary (t-CI + bootstrap CI)
$PY experiments/aggregate.py          # -> results/aggregated/

# 3. Assumption checks + omnibus + factorial ANOVA
$PY experiments/stats_omnibus.py      # -> results/statistics/ (normality, variance, omnibus, factorial, methodology)

# 4. Pairwise post-hoc + enhancement/robustness (Holm + BH, effect sizes)
$PY experiments/stats_posthoc.py      # -> results/statistics/ (pairwise, enhancement, robustness)

# 5. Publication tables (CSV + LaTeX + Excel)
$PY experiments/make_tables.py        # -> results/tables/

# 6. Publication figures (>=300 DPI PNG + SVG)
$PY experiments/make_figures.py       # -> results/figures/
```

## Determinism

- **Environment RNG** — `VesselEnv`/`VesselEnvV2` use a seeded mulberry32 stream;
  `env_seed = 42 + seed`.
- **Agent RNG** — `js/dqn.js` uses `Math.random`; the harness overrides it inside
  the module sandbox with `mulberry32(seed*1000003 + 12345)` so weight init,
  exploration, and replay sampling are reproducible per seed.
- **Bootstrap** — NumPy `default_rng(12345)`, 10,000 resamples.
- Verified: identical seed → byte-identical eval metrics; follow-Dijkstra policy →
  `optimality_ratio = 1.0000` on 100/100 episodes.

## results/ tree

```
results/
├── raw/            eval_episodes.csv, train_episodes.csv, run_manifest.json
├── aggregated/     per_seed.csv, summary.csv, marginals.json
├── statistics/     normality.csv, variance_homogeneity.csv, omnibus_algorithms.csv,
│                   factorial_anova.csv, pairwise_algorithms.csv,
│                   enhancement_contrasts.csv, robustness_summary.csv, methodology.json
├── tables/         table1..table8 + table_methods (.csv/.tex) + all_tables.xlsx
├── figures/        fig1..fig8 (.png 320dpi + .svg) + figures_captions.md
├── configs/        experiment_config.json  (frozen design)
└── reports/        Q1_experimental_report.md, SATURATED_BENCHMARK_FINDING.md,
                    REPRODUCIBILITY.md
```

## Quality-control checks performed

- Raw integrity: 6,400 eval rows, 0 NaN/Inf, `success ∈ {0,1}`, balanced cells
  (1,600 rows per PER/Noisy combination).
- Aggregation traced to raw (per-seed DQN success matches raw group means exactly).
- Statistics validated: normality (Shapiro), homoscedasticity (Levene),
  omnibus (RM-ANOVA + Friedman), factorial (Type-II, balanced), paired Wilcoxon
  (exact) + Holm/BH, effect sizes (Cohen's d_z, Cliff's δ, rank-biserial),
  t-CI vs bootstrap-CI agreement.
- `invalid_rate` excluded (zero variance under greedy eval; documented).
```
