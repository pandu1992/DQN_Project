# Appendix

This appendix collects the mathematical and statistical methods and the full
result tables for all five studies. The tables themselves are rendered from the
committed CSVs in the **Appendix — Tables** tab; this document provides the
formal methods reference and an index to those tables.

<a name="appendix-methods"></a>

## A.0 Mathematical and Statistical Methods

**Descriptive statistics.** For a per-seed sample $x_1,\dots,x_n$:

$$ \bar x = \frac1n\sum_i x_i, \qquad s = \sqrt{\frac{1}{n-1}\sum_i (x_i-\bar x)^2}, \qquad \mathrm{SE}=\frac{s}{\sqrt n}. $$

**Confidence interval** (Student-$t$; cross-checked by percentile bootstrap over
10,000 resamples):

$$ \mathrm{CI}_{95\%} = \bar x \pm t_{0.975,\,n-1}\,\frac{s}{\sqrt n}. $$

**Assumption checks.** Shapiro–Wilk $W$ for normality; Levene / Brown–Forsythe
(median-centred) for variance homogeneity.

**Factorial ANOVA.** Type-II sums of squares for the model
$y \sim \prod_k F_k + \text{seed}$ (main effects and interactions), with partial
$\eta^2$ effect size:

$$ \eta^2_p = \frac{\mathrm{SS}_{\text{effect}}}{\mathrm{SS}_{\text{effect}}+\mathrm{SS}_{\text{error}}}, \qquad F=\frac{\mathrm{MS}_{\text{effect}}}{\mathrm{MS}_{\text{error}}}. $$

Effect-size convention: $\eta^2_p \ge 0.14$ large, $\ge 0.06$ medium,
$\ge 0.01$ small.

**Paired post-hoc.** Wilcoxon signed-rank test (exact method for small $n$) as the
primary test given non-normality; paired $t$ as a secondary. Effect sizes: Cohen's

$$ d_z = \frac{\bar d}{s_d}, \quad d = x^{(A)} - x^{(B)} \text{ (paired by seed)}, $$

with magnitude bands $|d_z|<0.2$ negligible, $<0.5$ small, $<0.8$ medium,
$\ge 0.8$ large; and Cliff's $\delta$ for ordinal dominance.

**Multiple-comparison correction.** Within each family of contrasts:

$$ \text{Holm: } \tilde p_{(i)} = \max_{j\le i}\big[(m-j+1)\,p_{(j)}\big]; \qquad
\text{BH: } \tilde p_{(i)} = \min_{j\ge i}\Big[\tfrac{m}{j}\,p_{(j)}\Big]. $$

**Reinforcement-learning targets** (as implemented). DQN:
$y_t = r_t + \gamma(1-d_t)\max_{a'}Q_{\theta^-}(s_{t+1},a')$. Double DQN:
$a^\*=\arg\max_{a'}Q_\theta(s_{t+1},a')$, $y_t=r_t+\gamma(1-d_t)Q_{\theta^-}(s_{t+1},a^\*)$.
Dueling: $Q(s,a)=V(s)+(A(s,a)-\overline{A(s,\cdot)})$. PER priority
$p_i=|\delta_i|+\epsilon$, sample prob $\propto p_i^\alpha$, IS weight
$w_i=(N P(i))^{-\beta}/\max_j w_j$. Noisy Nets: $W=\mu_W+\sigma_W\odot\varepsilon$.

---

<a name="appendix-a1"></a>

## A.1 Study 1 Tables (clean-condition algorithm comparison)

Rendered in the **Appendix — Tables** tab under the *Study 1* group:

- **Table S1.1** — Algorithm and experimental configuration.
- **Table S1.2** — Overall performance (mean ± 95% CI) per algorithm.
- **Table S1.3** — Omnibus algorithm effect (RM-ANOVA + Friedman).
- **Table S1.4** — Factorial main effects and interactions (partial $\eta^2$).
- **Table S1.5** — Full 16-configuration matrix.
- **Table S1.6** — Pairwise statistical comparison (Holm-corrected, effect sizes).
- **Table S1.7** — Enhancement / robustness (PER, Noisy Nets).
- **Table S1.8** — Overall findings per algorithm.
- **Table S1.M** — Mathematical & statistical methods.

<a name="appendix-a2"></a>

## A.2 Study 2 Tables (robustness to sensing/communication degradation)

Under the *Study 2* group:

- **Table S2.1** — Study configuration.
- **Table S2.2** — Navigation performance across conditions.
- **Table S2.3** — Safety (collision rate & IALA), clean vs harsh.
- **Table S2.4** — Precision (docking accuracy & CTE).
- **Table S2.5** — Factorial ANOVA effects (partial $\eta^2$).
- **Table S2.6** — Robustness: worst vs clean (paired effect sizes).
- **Table S2.7** — Robustness ranking.
- **Table S2.M** — Mathematical & statistical methods.

<a name="appendix-a3"></a>

## A.3 Study 3 Tables (rule-based baseline, confirmatory)

Under the *Study 3* group:

- **Table S3.1** — Study configuration.
- **Table S3.2** — Navigation success across conditions.
- **Table S3.3** — Safety (collisions & IALA), clean vs harsh.
- **Table S3.4** — Factorial ANOVA (agent × condition, partial $\eta^2$).
- **Table S3.5** — Rule-based vs DQN at harsh (significance + effect size).
- **Table S3.6** — Robustness: clean vs harsh per agent.
- **Table S3.7** — Overall findings.
