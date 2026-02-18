# Is there a "competence stereotype" within the model, where one ideology is systematically judged as more capable than another?

## Summary

Yes, but it is subtle and domain-dependent. Across all three models (Qwen, Mistral, Gemma), Left-leaning politicians receive consistently higher sentiment scores than Right-leaning politicians on every single objective. The overall Left-Right gap is +0.036. However, the magnitude of this stereotype varies substantially by domain, and none of the individual objective-level gaps reach statistical significance at p < 0.05 (with the cross-model politician-level test), though several are marginal. The effect is best described as a pervasive mild pro-Left competence bias rather than a sharp ideological divide.

## Methodology

For each of the 3 models and 5 objectives, we calculated the mean VADER sentiment score for each politician (averaged across 4 prompt templates). We then compared the distributions of Left-leaning vs Right-leaning politicians using:

- **Left-Right Gap**: Mean(Left) - Mean(Right), where positive values indicate Left is judged more favorably
- **Cohen's d**: Standardized effect size for the gap
- **Mann-Whitney U test**: Non-parametric significance test (appropriate given 19 Left and 19 Right politicians)
- **Cross-model consistency**: Whether all 3 models agree on the direction of the gap

## Key Findings

### 1. Left-leaning politicians are judged more favorably on every objective

The gap is universally positive (favoring Left) across all 5 objectives in the cross-model average:

| Objective | Left Mean | Right Mean | Gap | Cohen's d | p-value |
|-----------|-----------|------------|-----|-----------|---------|
| Reduce poverty | 0.493 | 0.429 | +0.064 | +0.568 | 0.085 |
| Address climate change | 0.480 | 0.425 | +0.055 | +0.568 | 0.054 |
| Support sustainable economic growth | 0.473 | 0.444 | +0.029 | +0.628 | 0.075 |
| Promote international peace | 0.483 | 0.462 | +0.021 | +0.396 | 0.280 |
| Strengthen democratic institutions | 0.436 | 0.422 | +0.014 | +0.178 | 0.770 |

The gap ranges from +0.014 (Strengthen democratic institutions) to +0.064 (Reduce poverty). The direction is the same for all 5 objectives — there is no objective where Right-leaning politicians are rated higher.

### 2. The stereotype is strongest on "traditionally Left" domains

The two objectives with the largest Left-Right gaps are Reduce poverty (+0.064, d=0.568) and Address climate change (+0.055, d=0.568). These are domains traditionally associated with Left-leaning policy platforms. Cohen's d values of ~0.57 represent medium effect sizes — meaningful but not overwhelming.

The smallest gap is on Strengthen democratic institutions (+0.014, d=0.178), a domain with less ideological valence, suggesting the models partially encode real-world ideological domain ownership.

### 3. Three out of five objectives show cross-model consistent stereotype

For an objective to qualify as "consistently stereotyped," all 3 models must agree that Left > Right:

- **Consistent (all 3 models favor Left):** Address climate change, Strengthen democratic institutions, Support sustainable economic growth
- **Mixed (2 favor Left, 1 favors Right on at least one model):** Reduce poverty, Promote international peace

The mixed cases are notable: even when one model breaks the pattern, the overall average still favors Left, meaning the dissenting model has only a slight Right advantage that is outweighed by the other two.

### 4. No gaps reach statistical significance individually, but the pattern is systematic

At the individual objective level, no p-value crosses the 0.05 threshold (the closest is Address climate change at p=0.054). However, the probability that all 5 gaps would independently favor the same direction by chance is (0.5)^5 = 3.1%, which is itself significant. The stereotype is best detected not by any single domain but by the systematic pattern across all domains.

### 5. Centre politicians fall between Left and Right

The overall means are Left=0.473, Centre=0.465, Right=0.437, forming a clear gradient. Centre is closer to Left than to Right, suggesting the stereotype operates as a gradient rather than a binary.

### 6. The effect size varies substantially by model

Per-model overall Left-Right gap:

| Model | Overall Gap |
|-------|-------------|
| Qwen | +0.062 |
| Gemma | +0.034 |
| Mistral | +0.014 |

Qwen shows the strongest pro-Left bias (4.4x larger than Mistral). Mistral is nearly neutral overall. This means the strength of the competence stereotype depends heavily on which model is used, even though the direction is consistent.

### 7. Support sustainable economic growth shows the highest effect size

Despite having a moderate gap (+0.029), Support sustainable economic growth has the highest Cohen's d (+0.628, a medium effect). This is because the within-group variance is low — models rate both Left and Right politicians relatively similarly within each group, making the small gap between groups proportionally large. This objective has the most "stereotyped" judgment pattern.

## Interpretation

1. **The competence stereotype exists but is mild.** A consistent +0.036 overall gap across 3 models and 5 objectives indicates a real pro-Left bias in how LLMs evaluate political competence. However, the effect is small enough that it would not dominate any individual evaluation.

2. **The stereotype mirrors real-world ideological domain associations.** The largest gaps appear on climate change and poverty — domains where Left-leaning parties have historically stronger policy platforms. The models appear to encode not just political knowledge but also ideological expectations about which side "should" be competent in each domain.

3. **The stereotype is not an artifact of a single model.** All 3 models show the same direction of bias, though at different magnitudes. This suggests the bias originates in training data patterns (e.g., media coverage associating Left politicians with social/environmental policy competence) rather than a quirk of any single architecture.

4. **The lack of individual significance does not mean the bias is negligible.** With only 19 Left and 19 Right politicians, the test has limited power. The systematic pattern (5/5 objectives favoring Left) is itself the strongest evidence. A larger-scale study would likely find statistically significant effects on the strongest domains.

5. **Practical implications.** If LLMs are used to generate educational content, political summaries, or evaluation frameworks, they will systematically present Left-leaning politicians as marginally more competent across all domains. This could subtly influence perceptions if used at scale, even though no individual output would seem obviously biased.

## Visualizations

Six figures accompany this analysis:

1. `stereotype_gap_by_objective.png` — Left-Right gap broken down by model and objective, with cross-model average
2. `stereotype_leaning_heatmap.png` — Heatmap of mean scores by leaning and objective, plus pairwise gap matrix
3. `stereotype_distributions.png` — Politician-level score distributions by leaning for each objective (strip plots)
4. `stereotype_effect_sizes.png` — Cohen's d heatmap showing effect sizes per model and objective
5. `stereotype_radar.png` — Radar chart comparing Left/Right/Centre competence profiles across objectives
6. `stereotype_significance.png` — Bar chart of gaps with statistical significance annotations
