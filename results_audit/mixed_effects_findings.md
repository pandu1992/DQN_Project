# Mixed-effects inference summary (Q1 audit, Task 5)

Linear mixed model `y ~ <fixed design factors> + (1|seed)`, seed as a
RANDOM effect, fit on per-seed scalars. ICC = var(seed)/(var(seed)+var(resid))
is the share of variance absorbed by the random seed — a direct measure of
run-to-run variability and a reminder that inference is seed-level (n=seeds),
not episode-level (~31k episodes = reproducibility, not statistical power).


## Study 2 (robustness, n=6)
- DV = success_rate; fixed = `C(algorithm) + noise_std + packet_error_rate`; seeds n=6; fit=REML/lbfgs.
- variance: seed=0.00, residual=294.92, **ICC=0.00**.
- significant fixed effects (p<0.05): C(algorithm)[T.DuelingDQN] (p=0.019), noise_std (p=0.000), packet_error_rate (p=0.000).

## Study 3 (rule-based + confirmatory, n=15)
- DV = success_rate; fixed = `C(agent) + C(condition)`; seeds n=15; fit=ML/lbfgs.
- variance: seed=62.81, residual=265.28, **ICC=0.19**.
- significant fixed effects (p<0.05): C(agent)[T.DuelingDQN] (p=0.013), C(agent)[T.RuleBased] (p=0.000), C(condition)[T.harsh] (p=0.031).

## Study 4 (two vessels, n=8)
- DV = success_rate; fixed = `C(pairing) + C(condition)`; seeds n=8; fit=ML/powell.
- variance: seed=44.70, residual=73.50, **ICC=0.38**.
- significant fixed effects (p<0.05): C(pairing)[T.DQN_vs_RuleBased] (p=0.000), C(pairing)[T.RuleBased_vs_RuleBased] (p=0.000), C(condition)[T.mid] (p=0.016).

## Chart fairness (n=8)
- DV = success_rate; fixed = `C(arm) + C(condition)`; seeds n=8; fit=ML/powell.
- variance: seed=24.97, residual=283.81, **ICC=0.08**.
- significant fixed effects (p<0.05): C(arm)[T.DRL_with_chart] (p=0.000), C(arm)[T.RuleBased] (p=0.000).

## External validity maps (n=6)
- DV = success_rate; fixed = `C(arm) + C(condition) + C(map_id)`; seeds n=6; fit=ML/powell.
- variance: seed=42.40, residual=367.10, **ICC=0.10**.
- significant fixed effects (p<0.05): C(arm)[T.RuleBased] (p=0.002), C(condition)[T.harsh] (p=0.002).
